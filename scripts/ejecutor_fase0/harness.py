"""Corre Claude Code COMO EL USUARIO axioma (sin sudo) por SSH a 127.0.0.1.

El kernel acota las dos puntas: local (hall9000) y remota (.10/.11/.20).
La llave del cerebro viaja por stdin, nunca por argv: `ps` la vería.
Spec: 2026-09-15-ejecutor-design.md §8, Fase 0.
RETIRADO para lanzar (2026-09-17): ver Fase0Retirada. `preparar` queda (documenta que la
llave viaja por stdin) y no lanza nada.
"""
from __future__ import annotations

import pathlib
import shlex

NODE_BIN = "/opt/ejecutor/node-v24.16.0/bin"
CLAUDE = f"{NODE_BIN}/claude"
CONTROLADOR = [
    "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
    "-p", "58291", "-i", str(pathlib.Path("~/.ssh/id_ejecutor_controlador").expanduser()),
    "axioma@127.0.0.1",
]


def preparar(base_url, modelo, prompt, llave, auto_compact=None, plugin_dirs=(), formato="stream-json"):
    env = [
        f"ANTHROPIC_BASE_URL={shlex.quote(base_url)}",
        'ANTHROPIC_AUTH_TOKEN="$K"',
        f"PATH={NODE_BIN}:/usr/bin:/bin",
    ]
    if auto_compact:
        env.append(f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={int(auto_compact)}")
    args = [CLAUDE, "-p", prompt, "--output-format", formato, "--model", modelo,
            "--allowedTools", "Bash,Read"]
    if formato == "stream-json":
        args.append("--verbose")
    for d in plugin_dirs:
        args += ["--plugin-dir", d]
    remoto = ("read -r K; cd ~ && env " + " ".join(env) + " timeout 900 "
              + " ".join(shlex.quote(a) for a in args))
    return CONTROLADOR + [remoto], llave + "\n"


class Fase0Retirada(RuntimeError):
    """Retirado el 2026-09-17 (Ejecutor SP1, plan 6): lanzaba el arnés como la cuenta sin
    jaula ni gancho (C1) y directo a Ollama (C3). El lanzamiento vive en el transporte de SP2,
    detrás de jax/ejecutor/contratos/arranque.py::exigir_contratos."""


def correr(base_url, modelo, prompt, llave="ollama", **kw):
    raise Fase0Retirada("fase0_retirada")
