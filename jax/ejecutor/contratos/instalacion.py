# jax/ejecutor/contratos/instalacion.py
"""Qué se instala en /opt/ejecutor/lib y cómo se renderiza (C1/C2; C4 agrega el freno).

`INSTALABLES` es la única lista: la usan el instalador, el test de biblioteca
estándar y el arranque (plan 6) para comparar sha256 instalado contra repo.

`python -m jax.ejecutor.contratos.instalacion <etapa>` escribe en <etapa>:
gancho.sh, managed-settings.json, settings-usuario.json, ejecutor-freno.service (C4),
instalables.txt y
manifiesto.sha256, leyendo JAX_EJECUTOR_LIB, JAX_EJECUTOR_POLITICA y
JAX_EJECUTOR_GANCHO_TOPE_S (sin defaults).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
PLANTILLA_GANCHO = RAIZ / "ops" / "ejecutor" / "gancho.sh.plantilla"
PLANTILLA_FRENO = RAIZ / "ops" / "ejecutor" / "ejecutor-freno.service.plantilla"

INSTALABLES = (
    "jax/__init__.py",
    "jax/core/__init__.py",
    "jax/core/interruptor.py",
    "jax/ejecutor/__init__.py",
    "jax/ejecutor/contratos/__init__.py",
    "jax/ejecutor/contratos/formato.py",
    "jax/ejecutor/contratos/destinos.py",
    "jax/ejecutor/contratos/politica.py",
    "jax/ejecutor/contratos/gancho.py",
    "jax/ejecutor/contratos/pausa.py",
    "jax/ejecutor/contratos/freno.py",
)

# settings de usuario de la cuenta DENTRO de la jaula: vacíos y de solo lectura.
SETTINGS_USUARIO = "{}\n"

_RUTA_SEGURA = re.compile(r"^/[A-Za-z0-9_./-]+$")
_MARGEN_TIMEOUT_S = 5


def _ruta(valor: str) -> str:
    if not _RUTA_SEGURA.match(valor or "") or "/../" in valor:
        raise ValueError("ruta_insegura")
    return valor


def _tope(valor: int) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int) or not 1 <= valor <= 60:
        raise ValueError("tope_invalido")
    return valor


def renderizar_gancho(lib: str, tope_s: int) -> str:
    return (PLANTILLA_GANCHO.read_text(encoding="utf-8")
            .replace("@LIB@", _ruta(lib)).replace("@TOPE_S@", str(_tope(tope_s))))


def renderizar_managed_settings(lib: str, politica: str, tope_s: int) -> str:
    doc = {
        "allowManagedHooksOnly": True,
        "disableAllHooks": False,
        "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{
            "type": "command",
            "command": f"{_ruta(lib)}/gancho.sh {_ruta(politica)}",
            "timeout": _tope(tope_s) + _MARGEN_TIMEOUT_S,
        }]}]},
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def renderizar_unidad_freno(lib: str) -> str:
    return PLANTILLA_FRENO.read_text(encoding="utf-8").replace("@LIB@", _ruta(lib))


def manifiesto(archivos: dict) -> str:
    return "".join(f"{hashlib.sha256(archivos[n]).hexdigest()}  {n}\n" for n in sorted(archivos))


def principal(argv, env=None) -> int:
    env = os.environ if env is None else env
    etapa = Path(argv[0])
    lib = env["JAX_EJECUTOR_LIB"]
    tope = int(env["JAX_EJECUTOR_GANCHO_TOPE_S"])
    renderizados = {
        "gancho.sh": renderizar_gancho(lib, tope).encode(),
        "managed-settings.json": renderizar_managed_settings(lib, env["JAX_EJECUTOR_POLITICA"], tope).encode(),
        "settings-usuario.json": SETTINGS_USUARIO.encode(),
        "ejecutor-freno.service": renderizar_unidad_freno(lib).encode(),
    }
    for nombre, contenido in renderizados.items():
        (etapa / nombre).write_bytes(contenido)
    (etapa / "instalables.txt").write_text("".join(f"{r}\n" for r in INSTALABLES), encoding="utf-8")
    todos = {**renderizados, **{r: (RAIZ / r).read_bytes() for r in INSTALABLES}}
    (etapa / "manifiesto.sha256").write_text(manifiesto(todos), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
