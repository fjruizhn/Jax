#!/usr/bin/env python3
"""Golden de `hyde_sandbox.wrap_hyde_command` (D-4, spec facetas-por-suscripcion,
adenda §F). Congela el argv de bwrap y el env EXACTOS que Hyde produce HOY,
antes de que `hyde_sandbox` pase a apoyarse en el nucleo comun `cli_sandbox`.

Dos piezas, y las dos importan:

1. `test_wrap_hyde_command_coincide_con_la_golden`: el codigo vigente produce,
   byte a byte, lo que dice la fixture `tests/fixtures/hyde_wrap_golden.json`.
2. `test_la_fixture_no_cambio_desde_su_commit`: la fixture misma esta fijada por
   el hash de su blob (el mismo que calcula git). Si alguien la regenerara con el
   codigo NUEVO, el test 1 pasaria por la razon equivocada; este test lo atrapa
   porque para regenerarla hay que tocar la constante `BLOB_SHA1_GOLDEN`, y ese
   cambio es visible en el diff de la revision. No usa `git`: funciona igual en
   un checkout superficial de CI.

El entorno del host se SIMULA (que existan /lib64, ~/.nvm, los repos, etc.):
sin eso la golden dependeria de la maquina donde corre y fallaria en CI.

Corre con:
  cd <repo> && python -m pytest _hyde_wrap_golden_test.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import hyde_sandbox

FIXTURE = Path(__file__).resolve().parent / "tests" / "fixtures" / "hyde_wrap_golden.json"

# Hash de blob de git de la fixture (sha1("blob <n>\0" + contenido)). Fijado en
# el commit que crea la fixture, ANTES del refactor de Hyde.
BLOB_SHA1_GOLDEN = "b3806394ced7731d6e2b1d85ef69c15ff4f245fb"

_TOKEN_CENTINELA = "tok-centinela-golden-NO-ES-REAL"
_WORKSPACE = "/golden/workspace"
_BWRAP_FIJO = "/golden/bin/bwrap"
_CMD = ["claude", "-p", "--model", "modelo-golden", "--output-format", "text"]

# Rutas que "existen" en el host simulado, por escenario.
_BASE_EXISTE = {
    _BWRAP_FIJO, "/lib64", "/bin", "/sbin",
    "/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/hosts",
    "/etc/ssl", "/etc/passwd", "/etc/group",
}
_OPCIONALES = {hyde_sandbox.REAL_NVM_DIR, hyde_sandbox.REAL_JAX_PLATFORM_REPO}

ESCENARIOS = {
    # token OAuth presente, host completo
    "con_token": dict(token=_TOKEN_CENTINELA, existe=_BASE_EXISTE | _OPCIONALES),
    # sin token, archivo de credenciales legible, host completo
    "sin_token_con_archivo": dict(
        token=None,
        existe=_BASE_EXISTE | _OPCIONALES | {hyde_sandbox.REAL_CREDENTIALS},
    ),
    # token, sin ~/.nvm ni jax-platform ni /lib64 /bin /sbin (rama de rutas opcionales)
    "con_token_host_minimo": dict(
        token=_TOKEN_CENTINELA,
        existe={_BWRAP_FIJO, "/etc/resolv.conf"},
    ),
}


def _capturar(escenario: str) -> dict:
    cfg = ESCENARIOS[escenario]
    existe = set(cfg["existe"])
    real_isfile, real_isdir = os.path.isfile, os.path.isdir
    real_exists, real_islink, real_access = os.path.exists, os.path.islink, os.access

    def _wrap(real, solo_si_existe=True):
        def f(p, *a, **k):
            p = os.fspath(p)
            if p in existe:
                return True
            if p in _TODAS_SIMULADAS:
                return False
            return real(p, *a, **k)
        return f

    _TODAS_SIMULADAS = (
        _BASE_EXISTE | _OPCIONALES | {hyde_sandbox.REAL_CREDENTIALS, _BWRAP_FIJO, _WORKSPACE}
    )

    def _access(p, mode, *a, **k):
        p = os.fspath(p)
        if p in existe:
            return True
        if p in _TODAS_SIMULADAS:
            return False
        return real_access(p, mode, *a, **k)

    with tempfile.TemporaryDirectory() as tmp:
        tpl = Path(tmp) / "tpl"
        env_proceso = {"PATH": "/usr/bin", "SECRETO_DEL_PADRE": "no-debe-aparecer"}
        if cfg["token"]:
            env_proceso[hyde_sandbox.HYDE_OAUTH_TOKEN_ENV] = cfg["token"]
        with patch.object(hyde_sandbox, "_BWRAP_BIN", _BWRAP_FIJO), \
             patch.object(hyde_sandbox, "_TEMPLATE_DIR", tpl), \
             patch.dict(os.environ, env_proceso, clear=True), \
             patch("os.path.isfile", _wrap(real_isfile)), \
             patch("os.path.isdir", _wrap(real_isdir)), \
             patch("os.path.exists", _wrap(real_exists)), \
             patch("os.path.islink", _wrap(real_islink)), \
             patch("os.access", _access), \
             patch("os.makedirs", lambda *a, **k: None):
            argv, env = hyde_sandbox.wrap_hyde_command(list(_CMD), _WORKSPACE)
        plantilla = {
            rel: (tpl / rel).read_text(encoding="utf-8")
            for rel in (".claude.json", ".claude/settings.json")
        }
    norm = lambda s: s.replace(str(tpl), "<TEMPLATE_DIR>")
    return {
        "argv": [norm(x) for x in argv],
        "env": dict(env),
        "plantilla_home": plantilla,
    }


def _blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _canonico(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_la_fixture_no_cambio_desde_su_commit():
    assert BLOB_SHA1_GOLDEN and len(BLOB_SHA1_GOLDEN) == 40, "la constante no se fijo"
    real = _blob_sha1(FIXTURE.read_bytes())
    assert real == BLOB_SHA1_GOLDEN, (
        "tests/fixtures/hyde_wrap_golden.json cambio desde el commit que la creo "
        f"(blob {real} != {BLOB_SHA1_GOLDEN}). Esa fixture se genero con el "
        "wrap_hyde_command ANTERIOR al refactor: regenerarla con el codigo nuevo "
        "haria pasar la golden por la razon equivocada."
    )


def test_la_fixture_cubre_los_tres_escenarios():
    assert set(json.loads(FIXTURE.read_text(encoding="utf-8"))) == set(ESCENARIOS)


def test_wrap_hyde_command_coincide_con_la_golden():
    esperado = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = {nombre: _capturar(nombre) for nombre in ESCENARIOS}
    # comparacion de la forma serializada: lo que se congela es el texto, byte a byte
    assert _canonico(actual) == _canonico(esperado)
    # y el argv sigue sin llevar nunca el token (B-1)
    for nombre, cap in actual.items():
        assert _TOKEN_CENTINELA not in json.dumps(cap["argv"]), nombre
