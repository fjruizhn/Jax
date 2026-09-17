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
    assert any("idx_ejecutor_punto_host_fecha" in str(f) for f in filas), filas
