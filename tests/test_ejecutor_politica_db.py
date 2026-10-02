# tests/test_ejecutor_politica_db.py
"""La política que sale de la DB REAL (esquema y semilla de las migraciones de
jax-platform, que este job corre antes) es legible y TODAS sus reglas pasan su
autoprueba. Es donde se prueba la semilla de jax-platform con el evaluador de jax,
sin copiar ninguno de los dos."""
import asyncio
from datetime import datetime, timezone

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
                # jax-platform#149 (C2 «respaldo antes de la misión») agrega `respaldado_at` NOT NULL
                # y vuelve `metodo` un ENUM. El CI de jax clona el master de jax-platform, así que
                # este INSERT tiene que valer con el esquema de antes y con el de después:
                # 'recreacion' es un método válido en los dos, y la columna nueva va sólo si existe.
                await cur.execute("SHOW COLUMNS FROM ejecutor_punto_restauracion LIKE 'respaldado_at'")
                con_respaldado_at = bool(await cur.fetchall())
                columnas = "host_nombre, referencia, metodo, restaurado_y_verificado_at, verificado_por, evidencia"
                valores = "%s, %s, 'recreacion', UTC_TIMESTAMP() - INTERVAL %s MINUTE, 'test', 'test'"
                if con_respaldado_at:
                    columnas += ", respaldado_at"
                    valores += ", UTC_TIMESTAMP() - INTERVAL %s MINUTE"
                for k in range(200):
                    await cur.execute(
                        f"INSERT INTO ejecutor_punto_restauracion ({columnas}) VALUES ({valores})",
                        (INVENTARIO[k % 4][0], f"prueba-{k}", k) + ((k,) if con_respaldado_at else ()))
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
    assert {r.codigo for r in p.reglas} >= SEMILLA_V1 | ENVOLTORIOS
    assert politica.autoprueba(p) == ()


# Semilla v1 (14) y reglas contra envoltorios (7, jax-platform `ejecutor_reglas_envoltorios_v1`).
SEMILLA_V1 = {
    "canario_c1", "ssh_sin_tt", "apt_full_upgrade_bridge", "migrate_fresh_produccion", "sed_i_env",
    "pure_ftpd_parar_atemai", "respaldos_borrar", "ajustes_claude_code", "borrar_archivos",
    "sql_destructivo", "dns_correo", "parar_servicio", "quitar_paquetes", "disco",
}
ENVOLTORIOS = {
    "envoltorio_tmux_screen", "envoltorio_desacopla", "envoltorio_script_c", "envoltorio_at_batch",
    "envoltorio_systemd_run", "envoltorio_ssh_escondido", "sandbox_desactivado",
}
# La evasión vista en la misión real (registro de C3, 2026-09-17 ~08:35), con la VM
# cambiada por atemai (por nombre: el inventario de prueba se inserta con IGNORE); y lo que corre la misión de humo.
EVASION_REAL = ("tmux new-session -d -s ssh-test 'ssh -p 58291 axioma@atemai hostname; sleep 2' "
                "&& tmux capture-pane -t ssh-test -p && tmux kill-session -t ssh-test")
HUMO = tuple(f"ssh -tt -p 58291 axioma@atemai {c}" for c in ("hostname", "df -h /", "free -h"))


def test_la_politica_real_bloquea_la_evasion_con_tmux_y_deja_la_mision_de_humo():
    async def accion(conn):
        filas = await exportar.leer(conn)
        return exportar.documento(*filas, "2026-09-17T12:00:00+00:00")

    p = politica.validar(asyncio.run(_con_inventario(accion)))
    ahora = datetime.now(timezone.utc)
    casos = {
        EVASION_REAL: "envoltorio_tmux_screen",
        "setsid ssh -tt -p 58291 axioma@atemai hostname": "envoltorio_desacopla",
        "ssh -tt -p 58291 axioma@atemai hostname": None,
        "ssh -tt -p 58291 axioma@atemai 'tmux ls'": None,
    }
    for comando, regla in casos.items():
        d = politica.evaluar(p, "Bash", {"command": comando}, ahora)
        assert (d.permitir, d.regla) == (regla is None, regla), (comando, d)
    for comando in HUMO:
        assert politica.evaluar(p, "Bash", {"command": comando}, ahora).permitir, comando
    d = politica.evaluar(p, "Bash", {"command": HUMO[0], "dangerouslyDisableSandbox": True}, ahora)
    assert (d.permitir, d.regla) == (False, "sandbox_desactivado")


def test_la_consulta_de_respaldos_usa_su_indice():
    async def accion(conn):
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + exportar.SQL_RESPALDOS)
            return await cur.fetchall()

    filas = asyncio.run(_con_inventario(accion))
    # X-3: la edad del respaldo se mide por respaldado_at (LEDGER:247), así que la consulta
    # agrupa por ese índice, no por el de «cuándo se restauró y verificó».
    assert any("idx_ejecutor_punto_host_respaldo" in str(f) for f in filas), filas


COMANDO_DESTRUCTIVO = "ssh -tt -p 58291 axioma@atemai sudo systemctl stop nginx"


async def _un_solo_punto(conn, host, respaldado_hace_min, verificado_hace_min):
    """Deja UN punto de restauración (referencia `prueba-...`, que la limpieza borra) para `host`
    y ninguno para los demás: lo que se mide es la edad, no el relleno del índice."""
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM ejecutor_punto_restauracion WHERE referencia LIKE %s", ("prueba-%",))
        await cur.execute(
            "INSERT INTO ejecutor_punto_restauracion (host_nombre, referencia, metodo, respaldado_at, "
            "restaurado_y_verificado_at, verificado_por, evidencia) VALUES "
            "(%s, 'prueba-edad', 'recreacion', UTC_TIMESTAMP() - INTERVAL %s MINUTE, "
            "UTC_TIMESTAMP() - INTERVAL %s MINUTE, 'test', 'test')",
            (host, respaldado_hace_min, verificado_hace_min))
    await conn.commit()


def _decision_con_punto(respaldado_hace_min, verificado_hace_min):
    async def accion(conn):
        await _un_solo_punto(conn, "atemai", respaldado_hace_min, verificado_hace_min)
        filas = await exportar.leer(conn)
        return exportar.documento(*filas, "2026-09-17T12:00:00+00:00")

    p = politica.validar(asyncio.run(_con_inventario(accion)))
    return politica.evaluar(p, "Bash", {"command": COMANDO_DESTRUCTIVO}, datetime.now(timezone.utc))


def test_un_snapshot_viejo_reverificado_hoy_no_cuenta_como_vigente():
    """X-3: C2 mide la edad por `respaldado_at` (la hora del SNAPSHOT), no por cuándo se lo
    volvió a restaurar y verificar. Un snapshot de hace 3 días que el verificador probó hace
    1 minuto es un respaldo de hace 3 días: la misión destructiva NO arranca."""
    d = _decision_con_punto(respaldado_hace_min=3 * 24 * 60, verificado_hace_min=1)
    assert (d.permitir, d.codigo) == (False, politica.DESTRUCTIVO_SIN_RESPALDO), d


def test_un_snapshot_reciente_sigue_contando_como_vigente():
    """Control positivo del anterior: sin él, «siempre deniega» pasaría el test de arriba."""
    d = _decision_con_punto(respaldado_hace_min=60, verificado_hace_min=1)
    assert d.permitir, d
