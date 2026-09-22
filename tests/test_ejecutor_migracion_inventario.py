# tests/test_ejecutor_migracion_inventario.py
"""La corrección de inventario del Ejecutor (2026-09-22): `ejecutor_host.sudo` y
`.machine_id` quedaron en 0/NULL después de que Fase 3 le dio sudo real a `axioma`
en las cuatro máquinas con clientes (`~/ejecutor-producto/LEDGER.md`, "SUDO EN LAS
CUATRO"). El propio ledger lo deja escrito: "se corrigen por migración en PR (regla
del ledger), no a mano". `ejecutor_host` la crea jax-platform; esto NO la duplica
-- sólo agrega el DATO sobre una tabla que ya existe, por el migrador compartido
que `jax/memory/db.py` ya corre en cada `connect()` de LAS MANOS, el memory worker
y la síntesis (job jacobs-gobernanza-db: el esquema lo siembran las migraciones de
jax-platform, sin `JAX_EJECUTOR_INVENTARIO`, así que `ejecutor_host` existe pero
vacía -- mismo patrón que test_ejecutor_politica_db.py / test_ejecutor_c5_db.py)."""
import asyncio

from jacobs import store
from jax.memory import migrations

# Nombres REALES (no un prefijo de prueba): la migración los usa como clave. IPs
# de documentación (RFC 5737), igual convención que test_ejecutor_politica_db.py.
INVENTARIO = [
    ("hall9000", "192.0.2.205", "hypervisor", 1),
    ("atemai", "192.0.2.211", "desarrollo", 0),
    ("prod", "192.0.2.210", "produccion", 0),
    ("bridge", "192.0.2.220", "clientes", 0),
    ("ejecutor-prueba", "192.0.2.250", "desarrollo", 0),
]


async def _con_inventario(accion):
    try:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                for nombre, ip, rol, local in INVENTARIO:
                    await cur.execute(
                        "INSERT IGNORE INTO ejecutor_host (nombre, ip, puerto, rol, es_local) "
                        "VALUES (%s, %s, 58291, %s, %s)", (nombre, ip, rol, local))
                    # Idempotencia real: la corrida parte de 0/NULL, no de lo que haya quedado.
                    await cur.execute(
                        "UPDATE ejecutor_host SET sudo=0, machine_id=NULL WHERE nombre=%s", (nombre,))
            await conn.commit()
            return await accion(conn)
    finally:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM ejecutor_host WHERE ip LIKE %s", ("192.0.2.2%",))
            await conn.commit()


def test_la_migracion_pone_sudo_y_machine_id_en_las_cuatro_y_no_toca_ejecutor_prueba():
    async def accion(_conn):
        pool = await store.obtener_pool()
        ok = await migrations.ensure_schema(pool)
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT nombre, sudo, machine_id FROM ejecutor_host WHERE nombre IN "
                    "('hall9000','atemai','prod','bridge','ejecutor-prueba')")
                return ok, {n: (bool(s), m) for n, s, m in await cur.fetchall()}

    ok, filas = asyncio.run(_con_inventario(accion))
    assert ok is True
    assert filas["atemai"] == (True, "95e56bf6da0d41f993a3e36869699af1")
    assert filas["bridge"] == (True, "ee090efa28cd46a7a0bff22d34e57eb4")
    assert filas["prod"] == (True, "da476dce01ea4c3e9e72a8078a3ffd48")
    assert filas["hall9000"] == (True, "37ce158242c649fa80804a8c17b83ca4")
    # ejecutor-prueba (VM desechable): el ledger la excluye explícitamente.
    assert filas["ejecutor-prueba"] == (False, None)


def test_la_migracion_es_idempotente():
    async def accion(_conn):
        pool = await store.obtener_pool()
        await migrations.ensure_schema(pool)
        segunda = await migrations.ensure_schema(pool)
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT sudo, machine_id FROM ejecutor_host WHERE nombre='prod'")
                return segunda, await cur.fetchone()

    segunda, fila = asyncio.run(_con_inventario(accion))
    assert segunda is True
    assert (bool(fila[0]), fila[1]) == (True, "da476dce01ea4c3e9e72a8078a3ffd48")


def test_las_claves_del_mapeo_son_exactamente_las_cuatro_con_clientes():
    assert set(migrations._EJECUTOR_HOST_MACHINE_ID) == {"hall9000", "atemai", "prod", "bridge"}
    assert "ejecutor-prueba" not in migrations._EJECUTOR_HOST_MACHINE_ID
