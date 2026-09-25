#!/usr/bin/env python3
"""JAX_AUDIT_LOG_PATH deja de tener default silencioso (2026-09-25,
ops/rutas-de-produccion).

Hasta hoy, `las_manos/server.py` resolvía la ruta de auditoría con
`os.getenv("JAX_AUDIT_LOG_PATH", SERVER_CFG["audit_log"])`, y
`las_manos/config.toml` traía ese default hardcodeado apuntando al checkout
de TRABAJO (`/home/fruiz/jax/las_manos/logs/audit.jsonl`) -- el mismo
checkout que las otras rutas de producción de `/etc/jax/.env`
(JAX_CONFIG_PATH, JAX_REPO_BASE) usaban por error, en vez de
`/srv/jax-prod/jax`. Sin la variable puesta, LAS MANOS arrancaba igual y
escribía auditoría ahí, en silencio -- exactamente lo que
`ops/rutas-de-produccion.sh --verificar` existe para detectar en las
DEMÁS rutas, pero esta lo hacía por un camino distinto: no apuntando bajo
`/home/fruiz/jax/` desde `/etc/jax/.env` (eso SÍ lo atrapa el guion), sino
cayendo ahí desde un default hardcodeado en `config.toml` cuando la
variable de entorno faltaba.

Verificado contra el código viejo (2026-09-25, antes de este commit): un
`import server` con `JAX_AUDIT_LOG_PATH` ausente del entorno pero con el
resto de las variables requeridas presentes (mismo patrón que
`conftest.py` de la raíz) terminaba en `RC=0` y
`server.AUDIT_LOG_PATH == "/home/fruiz/jax/las_manos/logs/audit.jsonl"` --
ningún error, ninguna señal. Ese es el escenario ROJO de este archivo.

El fix: `server.py` ahora resuelve `AUDIT_LOG_PATH` con
`config_entorno.ruta_absoluta_requerida("JAX_AUDIT_LOG_PATH")` -- mismo
patrón fail-closed que `JAX_KILL_SWITCH_PATH` en `interruptor.py`: sin la
variable, `EntornoInvalido`, nunca un default conocido reintroducido con
otro nombre. `config.toml` ya no trae `audit_log` en `[server]`.

Los dos tests de import corren en un SUBPROCESO limpio, no en el proceso
de pytest: `server.py` fija módulo-global en tiempo de import (igual que
`interruptor.ruta_del_interruptor()`, `auth_servicio.cargar_credenciales()`
y el resto de la cadena), y el propio `conftest.py` de la raíz ya fija
`JAX_AUDIT_LOG_PATH` ANTES de que nada pueda importar `server` -- para
poner a prueba el fallback hay que importarlo en un proceso que no haya
pasado por ese conftest. El resto de las variables requeridas (kill
switch, URLs de LAS_MANOS/Ollama, credenciales de servicio, sello de
facetas, REPO_BASE) se fijan acá con el mismo criterio que usa
`conftest.py`: valores de prueba, nunca los de `/etc/jax/.env`. Medido
2026-09-25: `import server` con todas esas variables presentes NO toca
ninguna base de datos ni red -- las importaciones que sí la necesitan
(`motor_registry.routes`, `jacobs.routes`, `jacobs.store`,
`procesamiento_routes`) están MÁS ABAJO en el archivo que la resolución de
`AUDIT_LOG_PATH`, así que el caso rojo (variable faltante) revienta antes
de llegar ahí y el caso verde (variable presente) las atraviesa sin
necesitar DB para completar el import.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest las_manos/_server_audit_log_entorno_test.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LAS_MANOS = RAIZ / "las_manos"


def _entorno_minimo_para_importar_server(tmp_path: Path) -> dict[str, str]:
    """Las variables que `server.py` necesita ANTES de la sección que
    importa `motor_registry.routes`/`jacobs.routes`/DB -- ver el docstring
    de este archivo. Valores de prueba, nunca los reales de
    `/etc/jax/.env` (mismo criterio que `conftest.py` de la raíz)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{RAIZ}{os.pathsep}{LAS_MANOS}"
    env["JAX_KILL_SWITCH_PATH"] = str(tmp_path / "interruptor" / "PAUSE")
    env["LAS_MANOS_URL"] = "http://las-manos.invalid:7777"
    env["JAX_OLLAMA_URL"] = "http://ollama.invalid:11434"
    env["JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA"] = secrets.token_urlsafe(32)
    env["JAX_LAS_MANOS_CREDENCIAL_JACOBS"] = secrets.token_urlsafe(32)
    env["JAX_FACET_SEAL_PATH"] = str(tmp_path / "facet-cache-seal")
    env["JAX_REPO_BASE"] = str(tmp_path / "repo")
    return env


def _importar_server_en_subproceso(env: dict[str, str]) -> subprocess.CompletedProcess:
    codigo = 'import server; print("AUDIT_LOG_PATH=" + str(server.AUDIT_LOG_PATH))'
    return subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=str(LAS_MANOS), env=env, capture_output=True, text=True, timeout=60,
    )


def test_sin_JAX_AUDIT_LOG_PATH_server_no_arranca_y_no_cae_a_home_fruiz(tmp_path):
    env = _entorno_minimo_para_importar_server(tmp_path)
    env.pop("JAX_AUDIT_LOG_PATH", None)

    resultado = _importar_server_en_subproceso(env)

    assert resultado.returncode != 0, (
        "import server tendria que fallar sin JAX_AUDIT_LOG_PATH -- en vez de "
        f"eso salio 0. stdout: {resultado.stdout!r}")
    assert "JAX_AUDIT_LOG_PATH" in resultado.stderr, resultado.stderr
    assert "EntornoInvalido" in resultado.stderr, resultado.stderr
    assert "/home/fruiz" not in resultado.stdout, (
        "el import llego a imprimir una ruta bajo /home/fruiz -- el fallback "
        f"silencioso sigue vivo. stdout: {resultado.stdout!r}")


def test_con_JAX_AUDIT_LOG_PATH_server_usa_esa_ruta_sin_caer_a_home_fruiz(tmp_path):
    env = _entorno_minimo_para_importar_server(tmp_path)
    ruta_de_prueba = tmp_path / "audit-log-de-prueba" / "audit.jsonl"
    env["JAX_AUDIT_LOG_PATH"] = str(ruta_de_prueba)

    resultado = _importar_server_en_subproceso(env)

    assert resultado.returncode == 0, (
        f"import server fallo con JAX_AUDIT_LOG_PATH presente.\n"
        f"stdout: {resultado.stdout}\nstderr: {resultado.stderr}")
    assert f"AUDIT_LOG_PATH={ruta_de_prueba}" in resultado.stdout, resultado.stdout
    assert "/home/fruiz" not in resultado.stdout, resultado.stdout


def test_config_toml_ya_no_trae_audit_log_bajo_home_fruiz():
    """El default hardcodeado en `[server]` -- la otra mitad del bug -- ya
    no está: sin la clave, no hay nada de dónde caer."""
    with open(LAS_MANOS / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    assert "audit_log" not in cfg["server"], (
        "config.toml todavia trae 'audit_log' en [server] -- el default "
        "hardcodeado que este archivo prueba que ya no existe sigue ahi: "
        f"{cfg['server'].get('audit_log')!r}")


def test_server_py_resuelve_audit_log_path_con_ruta_absoluta_requerida():
    """Chequeo de forma (igual criterio que
    `test_ningun_modulo_de_servicio_tiene_una_url_escrita` en
    tests/test_config_entorno.py): la línea que resuelve `AUDIT_LOG_PATH`
    tiene que llamar a `ruta_absoluta_requerida`, no a `os.getenv` con un
    segundo argumento -- que es exactamente el patrón del bug que este
    archivo cierra."""
    fuente = (LAS_MANOS / "server.py").read_text(encoding="utf-8")
    assert 'AUDIT_LOG_PATH = ruta_absoluta_requerida("JAX_AUDIT_LOG_PATH")' in fuente, (
        "server.py ya no tiene la línea esperada -- ¿cambió la forma de "
        "resolver AUDIT_LOG_PATH sin actualizar este test?")
    assert 'os.getenv("JAX_AUDIT_LOG_PATH"' not in fuente, (
        "server.py todavia tiene un os.getenv(\"JAX_AUDIT_LOG_PATH\", ...) -- "
        "el patron de default silencioso que este archivo prueba que se fue.")
