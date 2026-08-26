#!/usr/bin/env python3
"""Gobernanza de sub-agentes de Claude Code (DEUDA.md, "Sub-agentes de
Claude Code sin gobernanza real"): el UNICO punto de entrada aprobado
para lanzar `claude` como subproceso es
hyde_sandbox.py::run_sandboxed_claude() -- aplica el sandbox de bwrap y
el lock cross-proceso (flock(2)) que serializa las invocaciones entre el
proceso de las_manos y el del REPL (ver Task 1/2 del plan que agregó este
scanner).

Enforcement mecanico y acotado (mismo espiritu que
test_no_fail_open_except.py): un archivo que (a) lanza un subproceso por
CUALQUIERA de las formas conocidas (asyncio.create_subprocess_exec/_shell,
loop.subprocess_exec, subprocess.run/Popen/call/check_call/check_output,
os.system, os.exec*), Y (b) contiene la palabra "claude" en algun lugar de
su codigo fuente, DEBE ser el hyde_sandbox.py del root del repo --
cualquier otro archivo con esa combinacion esta lanzando `claude` (o algo
que lo referencia) por fuera del sandbox compartido.

La lista de formas se amplio 2026-08-25 (review final de rama): antes solo
cubria las dos async, asi que un `subprocess.run(["claude", ...])` pasaba
limpio. El item de DEUDA.md que este scanner cierra habla de "cualquier
OTRO músculo/automatización que dispare `claude`", y este repo ya usa la
API sincrona en 9+ archivos para otras cosas (git, etc.) -- es un idioma
vivo aca, no una hipotesis.

No detecta un lanzamiento de `claude` disfrazado (sin la palabra literal
"claude" en el archivo, ej. leida de una variable de entorno con otro
nombre) -- eso queda como residuo conocido, mismo criterio que P10 con
las formas mas sutiles del patron fail-open.

ALCANCE REAL EN CI (misma limitacion honesta que
test_no_fail_open_except.py, ver tambien el header de
.github/workflows/policy.yml): _repo_roots() incluye un fallback a
jax-platform, pero en el runner de GitHub Actions ese directorio NO existe
(repo privado separado, sin checkout cruzado configurado -- requeriria un
PAT/secret nuevo), asi que `if not root.exists(): continue` hace que en CI
solo se escanee el arbol de jax. La verificacion manual 2026-08-25 contra
los dos repos (cero falsos positivos: ningun otro call site de subprocess
-- las_manos/workers/ssh_worker.py, jax/voice/tts.py, jax/voice/ears.py,
jax-platform/backend/api/command.py -- menciona "claude") fue eso, manual;
no es una garantia que este job sostenga corrida a corrida para
jax-platform.

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

# Exencion por RUTA RELATIVA, no por nombre pelado: tiene que ser el
# hyde_sandbox.py que vive en el root de uno de los repos escaneados. Con
# `path.name == ALLOWED_FILENAME` un hipotetico tools/hyde_sandbox.py
# quedaba exento tambien, sin ser el modulo real y aprobado.
#
# Se compara sobre la ruta RESUELTA (readlink) a proposito: el modulo real
# vive en el root del repo y se alcanza tambien como
# las_manos/hyde_sandbox.py, que es un SYMLINK a ../hyde_sandbox.py (mismo
# patron que facet_resolver.py / credential_resolver.py). Es el mismo
# archivo, no una copia -- exentarlo no afloja nada: un tools/hyde_sandbox.py
# que fuera un archivo DISTINTO resuelve a tools/hyde_sandbox.py y sigue
# siendo violacion.
ALLOWED_FILENAME = "hyde_sandbox.py"

# Todas las formas de lanzar un subproceso que este repo podria usar. Las
# dos async eran las unicas cubiertas hasta la review final de rama
# (2026-08-25) -- un subprocess.run(["claude", ...]) pasaba limpio.
# os.exec* se matchea por prefijo (execl/execle/execlp/execlpe/execv/
# execve/execvp/execvpe). loop.subprocess_exec se matchea solo por nombre
# de atributo, sin verificar que el objeto sea un event loop: imprecision
# aceptada a proposito, igual que el resto de este scanner de nivel AST.
_SUBPROCESS_CALL_NAMES = {
    "create_subprocess_exec", "create_subprocess_shell", "subprocess_exec",
    "run", "Popen", "call", "check_call", "check_output",
    "system",
}

_SUBPROCESS_CALL_PREFIXES = ("exec",)

# Nombres genericos que SOLO cuentan como lanzamiento de subproceso si se
# llaman como atributo de `subprocess`/`os` (subprocess.run(...),
# os.system(...)). Sin esto, cualquier `x.run(...)` o `self.call(...)` del
# codebase daria un falso positivo masivo.
_QUALIFIED_ONLY_NAMES = {"run", "call", "check_call", "check_output", "system"}
_SUBPROCESS_MODULES = {"subprocess", "os"}


def _iter_python_files():
    """Rinde pares (root, path) -- el root hace falta para poder exentar
    por ruta relativa, no por nombre de archivo pelado."""
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel_parts = path.relative_to(root).parts
            if any(part in EXCLUDE_DIR_NAMES for part in rel_parts):
                continue
            yield root, path


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_subprocess_launch(node: ast.Call) -> bool:
    name = _call_name(node.func)
    if name is None:
        return False
    if name not in _SUBPROCESS_CALL_NAMES and not name.startswith(_SUBPROCESS_CALL_PREFIXES):
        return False
    if name in _QUALIFIED_ONLY_NAMES or name.startswith(_SUBPROCESS_CALL_PREFIXES):
        # Solo cuenta como subprocess.run / os.system / os.execv, etc.
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return False
        if func.value.id not in _SUBPROCESS_MODULES:
            return False
    return True


def _calls_create_subprocess(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_subprocess_launch(node):
            return True
    return False


def _is_approved_sandbox_module(root: Path, path: Path) -> bool:
    """El modulo aprobado es el hyde_sandbox.py del ROOT de un repo
    escaneado -- no cualquier archivo que se llame asi. Se resuelve el
    symlink primero (las_manos/hyde_sandbox.py -> ../hyde_sandbox.py es el
    mismo archivo)."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return False
    return rel == Path(ALLOWED_FILENAME)


def find_naked_claude_subprocess_files() -> list[str]:
    violations = []
    for root, path in _iter_python_files():
        if _is_approved_sandbox_module(root, path):
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
        f"{len(violations)} archivo(s) lanzan un subproceso y "
        "mencionan 'claude', fuera de hyde_sandbox.py::run_sandboxed_claude() "
        "-- cualquier invocacion de `claude` como subproceso debe pasar por "
        "ese wrapper (sandbox de bwrap + lock cross-proceso via flock):\n"
        + "\n".join(violations)
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
