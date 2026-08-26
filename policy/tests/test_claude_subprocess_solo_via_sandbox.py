#!/usr/bin/env python3
"""Gobernanza de sub-agentes de Claude Code (DEUDA.md, "Sub-agentes de
Claude Code sin gobernanza real"): el UNICO punto de entrada aprobado
para lanzar `claude` como subproceso es
hyde_sandbox.py::run_sandboxed_claude() -- aplica el sandbox de bwrap y
el semaforo compartido (ver Task 1/2 del plan que agregó este scanner).

Enforcement mecanico y acotado (mismo espiritu que
test_no_fail_open_except.py): un archivo que (a) llama a
asyncio.create_subprocess_exec o asyncio.create_subprocess_shell, Y (b)
contiene la palabra "claude" en algun lugar de su codigo fuente, DEBE ser
hyde_sandbox.py -- cualquier otro archivo con esa combinacion esta
lanzando `claude` (o algo que lo referencia) por fuera del sandbox
compartido. Verificado 2026-08-25 contra el codigo real de ambos repos:
cero falsos positivos (ningun otro call site de subprocess async
-- las_manos/workers/ssh_worker.py, jax/voice/tts.py, jax/voice/ears.py,
jax-platform/backend/api/command.py -- menciona "claude").

No detecta un lanzamiento de `claude` disfrazado (sin la palabra literal
"claude" en el archivo, ej. leida de una variable de entorno con otro
nombre) -- eso queda como residuo conocido, mismo criterio que P10 con
las formas mas sutiles del patron fail-open.

Corre con:
  python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_roots() -> list[Path]:
    roots = [_THIS_REPO_ROOT]
    env_root = os.environ.get("JAX_PLATFORM_REPO_ROOT")
    roots.append(Path(env_root) if env_root else _THIS_REPO_ROOT.parent / "jax-platform")
    return roots


REPO_ROOTS = _repo_roots()

EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build",
}

ALLOWED_FILENAME = "hyde_sandbox.py"

_SUBPROCESS_CALL_NAMES = {"create_subprocess_exec", "create_subprocess_shell"}


def _iter_python_files():
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            yield path


def _calls_create_subprocess(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        if name in _SUBPROCESS_CALL_NAMES:
            return True
    return False


def find_naked_claude_subprocess_files() -> list[str]:
    violations = []
    for path in _iter_python_files():
        if path.name == ALLOWED_FILENAME:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "claude" not in source.lower():
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        if _calls_create_subprocess(tree):
            violations.append(str(path))
    return violations


def test_no_naked_claude_subprocess() -> None:
    violations = find_naked_claude_subprocess_files()
    assert not violations, (
        f"{len(violations)} archivo(s) lanzan un subprocess async y "
        "mencionan 'claude', fuera de hyde_sandbox.py::run_sandboxed_claude() "
        "-- cualquier invocacion de `claude` como subproceso debe pasar por "
        "ese wrapper (sandbox + semaforo compartido):\n" + "\n".join(violations)
    )


def main() -> int:
    violations = find_naked_claude_subprocess_files()
    if violations:
        print(f"FAIL — {len(violations)} archivo(s):")
        for v in violations:
            print(f"  {v}")
        return 1
    print("OK — ningun subprocess de 'claude' fuera de hyde_sandbox.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
