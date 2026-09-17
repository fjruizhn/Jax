# tests/test_ejecutor_c5_db.py
"""C5 contra el esquema y la semilla reales (job jacobs-gobernanza-db): la config
sembrada se lee, la compuerta nace CERRADA y, con ella, una misión que toca una
máquina con datos de clientes NO arranca. IPs de documentación (RFC 5737)."""
import asyncio

from jacobs.store import get_conn
from jax.ejecutor.contratos import eleccion_c5 as E
from jax.ejecutor.contratos.fallo import Fallo

INVENTARIO = [("c5-hall9000", "192.0.2.105", "hypervisor", 1, 0), ("c5-atemai", "192.0.2.111", "desarrollo", 0, 1),
              ("c5-prod", "192.0.2.110", "produccion", 0, 1), ("c5-bridge", "192.0.2.120", "clientes", 0, 1),
              ("c5-baja", "192.0.2.121", "clientes", 0, 0)]


async def _con_inventario(accion):
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            for nombre, ip, rol, local, clientes in INVENTARIO:
                await cur.execute("INSERT IGNORE INTO ejecutor_host (nombre, ip, puerto, rol, es_local, "
                                  "con_datos_de_clientes, activo) VALUES (%s, %s, 58291, %s, %s, %s, %s)",
                                  (nombre, ip, rol, local, clientes, 0 if nombre == "c5-baja" else 1))
        await conn.commit()
        return await accion(conn)
    finally:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM ejecutor_host WHERE nombre LIKE %s", ("c5-%",))
        await conn.commit()
        conn.close()


def test_la_config_sembrada_se_lee_y_la_compuerta_nace_cerrada():
    async def accion(conn):
        return await E.leer_config(conn), await E.es_local(conn, "ollama"), await E.es_local(conn, "openai")
    cfg, ollama_local, openai_local = asyncio.run(_con_inventario(accion))
    assert cfg.auditor_faceta == "thot" and cfg.admite_datos_de_clientes is False
    assert (ollama_local, openai_local) == (True, False)


def test_por_defecto_una_mision_sobre_maquinas_de_clientes_no_arranca():
    async def accion(conn):
        cfg = await E.leer_config(conn)
        de_clientes = await E.verificar_eleccion(conn, cfg=cfg, proveedor_cerebro="ollama", proveedor_auditor="openai",
                                                 hosts_mision=frozenset({"c5-atemai", "c5-prod", "c5-bridge"}))
        inactiva = await E.verificar_eleccion(conn, cfg=cfg, proveedor_cerebro="ollama", proveedor_auditor="openai",
                                              hosts_mision=frozenset({"c5-baja"}))
        propia = await E.verificar_eleccion(conn, cfg=cfg, proveedor_cerebro="ollama", proveedor_auditor="openai",
                                            hosts_mision=frozenset({"c5-hall9000"}))
        return de_clientes, inactiva, propia
    de_clientes, inactiva, propia = asyncio.run(_con_inventario(accion))
    assert de_clientes == (Fallo("c5", "auditor_no_admite_datos_de_clientes",
                                 (("hosts", ("c5-atemai", "c5-bridge", "c5-prod")),)),)
    assert inactiva[0].codigo == "auditor_no_admite_datos_de_clientes", "una máquina dada de baja cuenta como desconocida"
    assert propia == ()


def test_la_consulta_de_hosts_va_por_la_clave_primaria():
    async def accion(conn):
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + E.SQL_HOSTS.format("%s, %s"), ("c5-prod", "c5-bridge"))
            return await cur.fetchall()
    filas = asyncio.run(_con_inventario(accion))
    assert any("PRIMARY" in str(f) for f in filas), filas
