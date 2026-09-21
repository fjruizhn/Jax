# tests/test_ejecutor_c5_db.py
"""C5 contra el esquema y la semilla reales (job jacobs-gobernanza-db): la config
sembrada se lee, la compuerta nace CERRADA y, con ella, una misión que toca una
máquina con datos de clientes NO arranca. IPs de documentación (RFC 5737)."""
import asyncio
import uuid

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

async def _con_auditor_local_de_prueba(accion, *, is_local: bool = True):
    """Provider+model+binding sintéticos para LA CLAVE que `axioma_config` tenga hoy en
    `ejecutor.auditor_faceta_local` -- nunca 'auditor_local' a mano. Ese nombre es DATO,
    no una constante del código (eleccion_c5.py lo dice: "Los dos nombres de faceta salen
    de axioma_config -- ninguno hardcodeado acá"), y jax-platform (repo AJENO, clonado
    fresco en CADA corrida de este job) ya lo cambió una vez sin tocar jax: el 2026-09-20
    pasó de 'auditor_local' a 'el_juez' (jax-platform migrations.py, `_EJECUTOR_CONFIG_C5`).
    Este test rompió ese día -- no porque el código de jax cambiara, sino porque el
    fixture asumía un nombre que dejó de controlar. Leerlo de la config real en cada
    corrida hace que la próxima vez que cambie, el test lo siga sin arreglo.

    (Causa raíz completa, reportada aparte porque es de OTRO repo: `_seed_el_juez_facet`
    en jax-platform corre ANTES de `_seed_models_and_backfill` -- con `model_ref` de
    `jax_local` todavía NULL y la tabla `model` todavía sin la fila de `qwen3-coder:30b`,
    así que en una base NUEVA la semilla de 'el_juez' nunca llega a crear su
    `facet_binding`, aunque en producción sí exista porque se cargó a mano por la
    pantalla de administración. Este test no depende de que esa semilla funcione: siembra
    su propia fila para la clave que `axioma_config` diga, exactamente como ya hacía con
    'thot' más abajo.)

    La migración real de jax-platform sólo siembra un auditor local propio con
    JAX_OLLAMA_CPU_URL en el entorno (no seteada acá), igual que la fixture
    `auditor_local_bindeado` del lado de jax-platform. `model_ref` se fija a mano (no vía
    el backfill de _seed_models_and_backfill): resolve_facet real
    (facet_resolver._query_facet) hace JOIN contra `model` por esa columna. `is_local`
    parametrizable: el peor caso (spec §4) es el auditor local bindeado a un proveedor
    que NO es local de verdad."""
    from jacobs import store
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            cfg = await E.leer_config(conn)
            auditor_faceta_local = cfg.auditor_faceta_local
            # La configuración del job es la autoridad para esta clave (en CI es
            # `el_juez`, no el literal histórico `auditor_local`).  Si ya existe
            # un binding compartido, sólo sustituimos sus tres referencias por la
            # duración del test y las restauramos exactamente al terminar.
            await cur.execute(
                "SELECT provider_id, model_id, model_ref FROM facet_binding "
                "WHERE facet_key = %s AND role = 'primary'", (auditor_faceta_local,))
            binding_local_previo = await cur.fetchone()
            await cur.execute("SELECT status FROM facet WHERE `key` = %s", (auditor_faceta_local,))
            fila_faceta_local = await cur.fetchone()
            status_faceta_local_previo = fila_faceta_local[0] if fila_faceta_local else None
            await cur.execute("DELETE FROM model WHERE provider_id = 't-c5db-auditor-local'")
            await cur.execute("DELETE FROM provider WHERE id = 't-c5db-auditor-local'")
            await cur.execute(
                "INSERT INTO provider (id, display_name, auth_type, is_local) "
                "VALUES ('t-c5db-auditor-local', 'auditor local de prueba', 'none', %s)", (is_local,))
            await cur.execute(
                "INSERT INTO model (provider_id, model_id, source, source_checked_at) "
                "VALUES ('t-c5db-auditor-local', 'modelo-cpu', 'manual', UTC_TIMESTAMP())")
            model_ref = cur.lastrowid
            if binding_local_previo is None:
                await cur.execute(
                    "INSERT INTO facet_binding (facet_key, provider_id, model_id, model_ref, role) "
                    "VALUES (%s, 't-c5db-auditor-local', 'modelo-cpu', %s, 'primary')",
                    (auditor_faceta_local, model_ref))
            else:
                await cur.execute(
                    "UPDATE facet_binding SET provider_id = 't-c5db-auditor-local', "
                    "model_id = 'modelo-cpu', model_ref = %s "
                    "WHERE facet_key = %s AND role = 'primary'", (model_ref, auditor_faceta_local))
            await cur.execute("UPDATE facet SET status = 'active' WHERE `key` = %s", (auditor_faceta_local,))
            # El auditor de NUBE también se siembra acá. No alcanza con sembrar el local:
            # estos tests resuelven LOS DOS (una máquina con datos de clientes y una sin
            # ellos), y el job `jacobs-gobernanza-db` arma su base clonando jax-platform y
            # corriendo SUS migraciones -- que no dejan a 'thot' con binding resoluble.
            # En una máquina de desarrollo el test pasaba porque la base ya lo traía de
            # antes: es el mismo defecto que ya mordió con auditor_local/model_ref --
            # un test que depende de que OTRO haya sembrado lo que necesita.
            # Sólo se siembra si FALTA, y sólo se limpia lo que se haya agregado.
            # La primera versión de esto borraba el binding de 'thot' en el
            # teardown -- y en CI ese binding lo siembra la migración de
            # jax-platform: al borrarlo, reventaba OTRO test del mismo job
            # (_catalog_from_db_test, que espera ver a thot en el catálogo).
            # Un arnés que deja la base peor de como la encontró no es un arnés.
            await cur.execute(
                "SELECT 1 FROM facet_binding WHERE facet_key = 'thot' AND role = 'primary'")
            nube_sembrada_aca = (await cur.fetchone()) is None
            if nube_sembrada_aca:
                await cur.execute(
                    "INSERT IGNORE INTO provider (id, display_name, auth_type, is_local) "
                    "VALUES ('openai', 'OpenAI', 'bearer', 0)")
                await cur.execute(
                    "INSERT INTO model (provider_id, model_id, source, source_checked_at) "
                    "VALUES ('openai', 't-c5db-modelo-nube', 'manual', UTC_TIMESTAMP())")
                nube_ref = cur.lastrowid
                await cur.execute(
                    "INSERT INTO facet_binding (facet_key, provider_id, model_id, model_ref, role) "
                    "VALUES ('thot', 'openai', 't-c5db-modelo-nube', %s, 'primary')", (nube_ref,))
            # La faceta tiene que estar ACTIVA para que resolve_facet la vea
            # (facet_resolver._query_facet filtra por status='active').  Se guarda el
            # estado previo y se restaura en el teardown: dejar a 'thot' activa para
            # el resto del job es la MISMA falta que denuncia el bloque de arriba --
            # un arnes no deja la base distinta de como la encontro, ni siquiera
            # cuando el cambio parece inofensivo.
            await cur.execute("SELECT status FROM facet WHERE `key` = 'thot'")
            _fila_thot = await cur.fetchone()
            status_thot_previo = _fila_thot[0] if _fila_thot else None
            await cur.execute("UPDATE facet SET status = 'active' WHERE `key` = 'thot'")
            # Y ademas una credencial ACTIVA: resolve_facet la exige cuando el
            # transporte de la faceta no es ollama/subprocess (facet_resolver.py
            # :266-271), y 'thot' es de nube.  ESTA era la causa real del rojo de
            # jacobs-gobernanza-db -- CredentialUnavailableError, no el binding ni
            # el status; los dos intentos anteriores arreglaron sintomas.  Vale una
            # de mentira: sin FERNET_KEY decrypt_secret devuelve el valor tal cual
            # (crypto_secrets.py:60-62), mismo recurso que test_prevuelo_catalogo_db.
            await cur.execute(
                "SELECT provider_id FROM facet_binding "
                "WHERE facet_key = 'thot' AND role = 'primary'")
            _fila_prov = await cur.fetchone()
            proveedor_thot = _fila_prov[0] if _fila_prov else None
            credencial_sembrada_aca = False
            clave = f"ZZ_C5DB_{uuid.uuid4().hex[:12]}"  # marcador propio de ESTA corrida
            if proveedor_thot:
                await cur.execute(
                    "SELECT 1 FROM credential WHERE provider_id = %s AND state = 'active'",
                    (proveedor_thot,))
                credencial_sembrada_aca = (await cur.fetchone()) is None
                if credencial_sembrada_aca:
                    # `credential` es TABLA COMPARTIDA y el proveedor de 'thot' puede
                    # ser uno real ('openai'): borrar por provider_id en el teardown
                    # podria llevarse una fila AJENA -- es el incidente de R38, y el
                    # guardia estatico de tests/test_delete_de_tablas_compartidas.py
                    # lo rechaza con razon. La clave es unica por corrida y el borrado
                    # filtra por ella, asi que solo puede alcanzar lo que sembro ESTE test.
                    await cur.execute(
                        "INSERT INTO credential (provider_id, env_key, encrypted_value, state) "
                        "VALUES (%s, %s, 'no-se-descifra', 'active')",
                        (proveedor_thot, clave))
        await conn.commit()
    try:
        return await _con_inventario(accion)
    finally:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                if binding_local_previo is None:
                    # Alcance el binding que insertó esta corrida, no todos los
                    # bindings de una tabla compartida.
                    await cur.execute(
                        "DELETE FROM facet_binding WHERE facet_key = %s AND role = 'primary' "
                        "AND provider_id = 't-c5db-auditor-local' AND model_id = 'modelo-cpu'",
                        (auditor_faceta_local,))
                else:
                    await cur.execute(
                        "UPDATE facet_binding SET provider_id = %s, model_id = %s, model_ref = %s "
                        "WHERE facet_key = %s AND role = 'primary'",
                        (*binding_local_previo, auditor_faceta_local))
                if status_faceta_local_previo is not None:
                    await cur.execute("UPDATE facet SET status = %s WHERE `key` = %s",
                                      (status_faceta_local_previo, auditor_faceta_local))
                await cur.execute("DELETE FROM model WHERE provider_id = 't-c5db-auditor-local'")
                await cur.execute("DELETE FROM provider WHERE id = 't-c5db-auditor-local'")
                if nube_sembrada_aca:
                    await cur.execute("DELETE FROM facet_binding WHERE facet_key = 'thot' "
                                      "AND model_id = 't-c5db-modelo-nube'")
                    await cur.execute("DELETE FROM model WHERE model_id = 't-c5db-modelo-nube'")
                if status_thot_previo is not None and status_thot_previo != 'active':
                    await cur.execute("UPDATE facet SET status = %s WHERE `key` = 'thot'",
                                      (status_thot_previo,))
                if credencial_sembrada_aca:
                    await cur.execute(
                        "DELETE FROM credential WHERE env_key = %s", (clave,))  # marcador-propio: clave = ZZ_C5DB_<uuid> de esta corrida
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


# --- arranque.py::eleccion_del_auditor -- el freno que la revisión encontró sin prueba ------
#
# Hallazgo de la revisión 2026-09-18: "arranque.py:305-330 -- donde se elige la faceta, se
# consulta es_local(...) y se corre validar_eleccion -- no tiene test. El código está bien,
# pero si alguien cambia `auditor_es_local=` por una comparación de nombre, nada se pone
# rojo." Los tests de acá abajo ejercitan exactamente esa línea contra la DB real: el
# PEOR CASO (auditor_local bindeado a un proveedor que NO es local) tiene que seguir
# rechazando con la compuerta cerrada. Se verificó en rojo a mano (2026-09-18): reemplazar
# `await eleccion_c5.es_local(conn, auditor_f.provider_id)` por
# `auditor_f.key == cfg.auditor_faceta_local` en arranque.py hace que
# test_peor_caso_arranque_auditor_local_mal_bindeado_no_pasa_la_compuerta falle (el
# fallo esperado desaparece) -- exactamente la regresión que el hallazgo describe.

def _cfg_con_cerebro_seedeado(cfg):
    """`ejecutor.cerebro_faceta` vale 'ejecutor' en la config real, pero esa faceta no
    tiene binding en jax_memory_test (deuda preexistente, ajena a esta ronda -- por eso
    los tests de más arriba pasan `proveedor_cerebro` a mano en vez de resolverlo).
    `eleccion_del_auditor` SÍ resuelve el cerebro de verdad (es fiel a p_c5 real): se
    sustituye por 'jax_local' (sembrada, proveedor 'ollama', distinto de cualquier
    auditor de esta suite) sólo para tener un cerebro resoluble de verdad."""
    import dataclasses
    return dataclasses.replace(cfg, cerebro_faceta="jax_local")


def test_arranque_elige_el_auditor_local_con_datos_de_clientes_y_pasa_la_compuerta():
    from facet_resolver import resolve_facet
    from jax.ejecutor.contratos import arranque

    async def accion(conn):
        cfg = _cfg_con_cerebro_seedeado(await E.leer_config(conn))
        return await arranque.eleccion_del_auditor(
            conn, hosts_mision=frozenset({"c5-bridge"}), cfg=cfg, resolve_facet=resolve_facet)
    auditor_f, fallos = asyncio.run(_con_auditor_local_de_prueba(accion, is_local=True))
    assert auditor_f.provider_id == "t-c5db-auditor-local"
    assert fallos == ()


def test_peor_caso_arranque_auditor_local_mal_bindeado_no_pasa_la_compuerta():
    """auditor_es_local mira provider.is_local, NUNCA el nombre 'auditor_local'. Con la
    compuerta cerrada (semilla real) y el binding apuntando a un proveedor is_local=0, la
    misión contra c5-bridge (con datos de clientes) sigue rechazada. Verificado en rojo a
    mano (2026-09-18): cambiar `es_local(conn, auditor_f.provider_id)` por
    `auditor_f.key == cfg.auditor_faceta_local` en arranque.py hace que este test falle."""
    from facet_resolver import resolve_facet
    from jax.ejecutor.contratos import arranque

    async def accion(conn):
        cfg = _cfg_con_cerebro_seedeado(await E.leer_config(conn))
        return await arranque.eleccion_del_auditor(
            conn, hosts_mision=frozenset({"c5-bridge"}), cfg=cfg, resolve_facet=resolve_facet)
    auditor_f, fallos = asyncio.run(_con_auditor_local_de_prueba(accion, is_local=False))
    assert auditor_f.provider_id == "t-c5db-auditor-local"  # se resolvió -- el bloqueo es la compuerta, no la resolución
    assert fallos == (Fallo("c5", "auditor_no_admite_datos_de_clientes", (("hosts", ("c5-bridge",)),)),)


def test_arranque_sin_mision_usa_el_auditor_de_nube_por_defecto():
    """hosts_mision=None (arranque sin turno, plan 6): sin hosts que mirar, se valida sólo
    que cerebro y auditor sean proveedores distintos -- el mismo comportamiento de siempre."""
    from facet_resolver import resolve_facet
    from jax.ejecutor.contratos import arranque

    async def accion(conn):
        cfg = _cfg_con_cerebro_seedeado(await E.leer_config(conn))
        return await arranque.eleccion_del_auditor(conn, hosts_mision=None, cfg=cfg, resolve_facet=resolve_facet)
    auditor_f, fallos = asyncio.run(_con_auditor_local_de_prueba(accion, is_local=True))
    assert auditor_f.provider_id == "openai" and fallos == ()
