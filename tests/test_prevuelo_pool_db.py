"""Pool del store de Jacobs contra una MariaDB REAL (Task 15b, job
jacobs-gobernanza-db). Los tests puros (test_prevuelo_pool.py) prueban el
Pool de aiomysql con conexiones dobles; acá se ve lo que un doble no puede:
que el servidor ve la MISMA conexión entre pedidos, y que una conexión que el
servidor mata (KILL, o wait_timeout) no envenena el pedido siguiente.

No siembra ni borra filas: sólo lee y mata SUS propias conexiones.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os

# Base de tests de ESTA sesión (decisión de Fernando, 2026-09-17). Reemplaza
# la guarda vieja `if _db != "jax_memory_test": raise` + `setdefault`, que es
# anterior a `JAX_TEST_DB_SUFIJO` y rechazaba `jax_memory_test_<sufijo>`:
# protege lo mismo (nunca producción, nunca una base que no sea de tests) y
# además deja correr la base propia de la sesión.
from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import prevuelo_catalogo as pc  # noqa: E402
from jacobs import store  # noqa: E402
from jacobs.models import Step  # noqa: E402


async def _id(conn) -> int:
    async with conn.cursor() as cur:
        await cur.execute("SELECT CONNECTION_ID()")
        (valor,) = await cur.fetchone()
    return int(valor)


def _correr(cuerpo):
    async def envuelto():
        try:
            return await cuerpo()
        finally:
            await store.cerrar_pool()
    return asyncio.run(envuelto())


def test_dos_pedidos_seguidos_usan_la_misma_conexion_del_servidor():
    async def cuerpo():
        async with store.conexion_del_pool() as conn:
            primero = await _id(conn)
        async with store.conexion_del_pool() as conn:
            segundo = await _id(conn)
        return primero, segundo

    primero, segundo = _correr(cuerpo)
    assert primero == segundo


def test_una_conexion_que_el_servidor_mato_no_envenena_el_pedido_siguiente():
    async def cuerpo():
        async with store.conexion_del_pool() as conn:
            muerta = await _id(conn)
        verdugo = await store.conexion_dedicada()
        try:
            async with verdugo.cursor() as cur:
                await cur.execute("KILL CONNECTION %s", (muerta,))
        finally:
            verdugo.close()
        # El FIN del servidor llega al lector de la conexión ociosa: el Pool
        # la descarta al pedirla en vez de entregarla muerta.
        for _ in range(50):
            await asyncio.sleep(0.02)
        async with store.conexion_del_pool() as conn:
            nueva = await _id(conn)
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                (uno,) = await cur.fetchone()
        return muerta, nueva, uno

    muerta, nueva, uno = _correr(cuerpo)
    assert nueva != muerta
    assert uno == 1


def test_los_dos_lectores_del_catalogo_leen_por_una_sola_conexion():
    """resolver_motores (MotorCatalog.from_db) y leer_catalogo contra el
    esquema real, por la conexión prestada: ninguno la cierra ni abre otra."""
    async def cuerpo():
        async with store.conexion_del_pool() as conn:
            antes = await _id(conn)
            motores = await pc.resolver_motores(
                [Step(pipeline_id="p", step_index=0, facet="kimi", capability="generate", input={})],
                conexion=conn,
            )
            catalogo = await pc.leer_catalogo(
                conexion=conn, facetas={"jekyll"}, motores=[m for m in motores.values() if m],
                capabilities={"generate"}, ahora=0.0,
            )
            despues = await _id(conn)
            cerrada = conn.closed
        return antes, despues, cerrada, motores, catalogo

    antes, despues, cerrada, motores, catalogo = _correr(cuerpo)
    assert antes == despues
    assert not cerrada
    assert set(motores) == {0}
    assert isinstance(catalogo, pc.Catalogo)
