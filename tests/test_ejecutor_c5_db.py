# tests/test_ejecutor_c5_db.py
"""C5 contra el esquema y la semilla reales (job jacobs-gobernanza-db): la config
sembrada se lee, la compuerta nace CERRADA y, con ella, una misión que toca una
máquina con datos de clientes NO arranca. IPs de documentación (RFC 5737)."""
import asyncio

from jacobs import store
from jax.ejecutor.contratos import eleccion_c5 as E
from jax.ejecutor.contratos.fallo import Fallo

INVENTARIO = [("c5-hall9000", "192.0.2.105", "hypervisor", 1, 0), ("c5-atemai", "192.0.2.111", "desarrollo", 0, 1),
              ("c5-prod", "192.0.2.110", "produccion", 0, 1), ("c5-bridge", "192.0.2.120", "clientes", 0, 1),
              ("c5-baja", "192.0.2.121", "clientes", 0, 0)]


async def _con_inventario(accion):
    # store.conexion() y no una conexion suelta: get_conn() ya no existe (frente F,
    # pool de Jacobs). La limpieza va en su propia conexion para que un error de
    # `accion` (que descarta la primera) no deje filas de prueba.
    try:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                for nombre, ip, rol, local, clientes in INVENTARIO:
                    await cur.execute("INSERT IGNORE INTO ejecutor_host (nombre, ip, puerto, rol, es_local, "
                                      "con_datos_de_clientes, activo) VALUES (%s, %s, 58291, %s, %s, %s, %s)",
                                      (nombre, ip, rol, local, clientes, 0 if nombre == "c5-baja" else 1))
            await conn.commit()
            return await accion(conn)
    finally:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM ejecutor_host WHERE nombre LIKE %s", ("c5-%",))
            await conn.commit()


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


# --- spec 2026-09-18-auditor-local-opcion.md §4: el auditor se elige según los hosts -------

async def _con_auditor_local_de_prueba(accion):
    """Provider+model+binding sintéticos para 'auditor_local' -- la migración real de
    jax-platform sólo los siembra con JAX_OLLAMA_CPU_URL en el entorno (no seteada acá),
    igual que la fixture `auditor_local_bindeado` del lado de jax-platform. `model_ref` se
    fija a mano (no vía el backfill de _seed_models_and_backfill): resolve_facet real
    (facet_resolver._query_facet) hace JOIN contra `model` por esa columna."""
    from jacobs import store
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM facet_binding WHERE facet_key = 'auditor_local'")
            await cur.execute("DELETE FROM model WHERE provider_id = 't-c5db-auditor-local'")
            await cur.execute("DELETE FROM provider WHERE id = 't-c5db-auditor-local'")
            await cur.execute(
                "INSERT INTO provider (id, display_name, auth_type, is_local) "
                "VALUES ('t-c5db-auditor-local', 'auditor local de prueba', 'none', TRUE)")
            await cur.execute(
                "INSERT INTO model (provider_id, model_id, source, source_checked_at) "
                "VALUES ('t-c5db-auditor-local', 'modelo-cpu', 'manual', UTC_TIMESTAMP())")
            model_ref = cur.lastrowid
            await cur.execute(
                "INSERT INTO facet_binding (facet_key, provider_id, model_id, model_ref, role) "
                "VALUES ('auditor_local', 't-c5db-auditor-local', 'modelo-cpu', %s, 'primary')", (model_ref,))
        await conn.commit()
    try:
        return await _con_inventario(accion)
    finally:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM facet_binding WHERE facet_key = 'auditor_local'")
                await cur.execute("DELETE FROM model WHERE provider_id = 't-c5db-auditor-local'")
                await cur.execute("DELETE FROM provider WHERE id = 't-c5db-auditor-local'")
            await conn.commit()


def test_elegir_y_resolver_auditor_usa_el_local_con_datos_de_clientes():
    """El punto único de elección (mision_servicio.py, vigia_servicio.py y arranque.py lo
    llaman igual): una máquina con datos de clientes resuelve al proveedor local REAL; una
    sin datos, al de nube -- con el mismo resolve_facet, contra la DB real."""
    from facet_resolver import resolve_facet

    async def accion(conn):
        cfg = await E.leer_config(conn)
        de_clientes, _, _ = await E.elegir_y_resolver_auditor(
            conn, cfg=cfg, hosts_mision=frozenset({"c5-bridge"}), resolve_facet=resolve_facet)
        propia, _, _ = await E.elegir_y_resolver_auditor(
            conn, cfg=cfg, hosts_mision=frozenset({"c5-hall9000"}), resolve_facet=resolve_facet)
        return de_clientes, propia
    de_clientes, propia = asyncio.run(_con_auditor_local_de_prueba(accion))
    assert de_clientes.provider_id == "t-c5db-auditor-local"
    assert propia.provider_id == "openai"  # 'thot', el auditor de nube -- sin datos que proteger
