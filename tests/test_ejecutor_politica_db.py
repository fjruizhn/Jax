# tests/test_ejecutor_politica_db.py
"""La política que sale de la DB REAL (esquema y semilla de las migraciones de
jax-platform, que este job corre antes) es legible y TODAS sus reglas pasan su
autoprueba. Es donde se prueba la semilla de jax-platform con el evaluador de jax,
sin copiar ninguno de los dos."""
import asyncio

import pytest

from jacobs import store
from jax.ejecutor.contratos import exportar, politica

INVENTARIO = [("hall9000", "192.0.2.5", "hypervisor", 1), ("atemai", "192.0.2.11", "desarrollo", 0),
              ("prod", "192.0.2.10", "produccion", 0), ("bridge", "192.0.2.20", "clientes", 0)]


async def _con_inventario(accion):
    # store.conexion() y no una conexion suelta: get_conn() ya no existe (frente F,
    # pool de Jacobs). La limpieza va en su propia conexion para que un error de
    # `accion` (que descarta la primera) no deje filas de prueba.
    try:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                for nombre, ip, rol, local in INVENTARIO:
                    await cur.execute("INSERT IGNORE INTO ejecutor_host (nombre, ip, puerto, rol, es_local) "
                                      "VALUES (%s, %s, 58291, %s, %s)", (nombre, ip, rol, local))
                # Con la tabla vacía el optimizador puede no mostrar el índice: se mide con filas.
                for k in range(200):
                    await cur.execute(
                        "INSERT INTO ejecutor_punto_restauracion (host_nombre, referencia, metodo, "
                        "restaurado_y_verificado_at, verificado_por, evidencia) VALUES (%s, %s, 'prueba', "
                        "UTC_TIMESTAMP() - INTERVAL %s MINUTE, 'test', 'test')",
                        (INVENTARIO[k % 4][0], f"prueba-{k}", k))
            await conn.commit()
            return await accion(conn)
    finally:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM ejecutor_punto_restauracion WHERE referencia LIKE %s", ("prueba-%",))
                await cur.execute("DELETE FROM ejecutor_host WHERE ip LIKE %s", ("192.0.2.%",))
            await conn.commit()


def test_la_semilla_real_exporta_legible_y_pasa_su_autoprueba():
    async def accion(conn):
        filas = await exportar.leer(conn)
        return exportar.documento(*filas, "2026-09-17T12:00:00+00:00")

    doc = asyncio.run(_con_inventario(accion))
    p = politica.validar(doc)
    assert len(p.reglas) == 14
    assert politica.autoprueba(p) == ()


def test_la_consulta_de_respaldos_usa_su_indice():
    async def accion(conn):
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + exportar.SQL_RESPALDOS)
            return await cur.fetchall()

    filas = asyncio.run(_con_inventario(accion))
    assert any("idx_ejecutor_punto_host_fecha" in str(f) for f in filas), filas
