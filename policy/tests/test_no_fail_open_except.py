#!/usr/bin/env python3
"""P10 — ningún validador o gate puede fallar abierto ante error o
ausencia de señal, incluyendo vía excepción sin capturar (REFORMAS-v3.1.md,
Apendice C-bis).

Enforcement mecanico y acotado, no un analizador general: un bloque
`except` cuyo cuerpo es unicamente `pass` (o `pass` + comentarios) traga
el error sin propagarlo, sin loguearlo y sin dejar ningun rastro -- es la
forma mas pura del patron fail-open que el Apendice C-bis documenta en
7 casos reales. Esta prueba escanea el codigo fuente real (no vendored,
no venvs, no tests) de ambos repos y falla si encuentra una instancia
nueva.

No detecta las formas mas sutiles del patron (un fallback que retorna
default_ok=True, un validador que no re-lanza) -- eso queda documentado
como residuo conocido, igual que Q13 en REFORMAS-v3.1.md para el barrido
de vocabulario. Esta prueba cierra la instancia mas barata de detectar
mecanicamente, no el patron completo.

Marcado, no allowlist (triage 2026-08-19, 32 casos revisados uno por uno):
un except-pass legitimo (fail-soft real: nadie depende de que esa
operacion haya funcionado) se marca con un comentario en la MISMA linea
del `except`, formato `# fail-soft: <razon especifica de este sitio>`.
Sin esa marca, es una violacion -- incluye a proposito el except-pass
nuevo que alguien escriba mañana sin marcarlo. La razon tiene que
explicar por que nada aguas abajo cree que la operacion salio bien; una
marca generica ("# fail-soft: ok") pasaria el test pero no cumple el
espiritu de la regla -- responsabilidad de quien revisa el PR, el test
solo puede verificar que la marca existe, no que sea honesta.

Corre con:
  python3 policy/tests/test_no_fail_open_except.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import linecache
import os
import sys
from pathlib import Path

# Ronda 4 (2026-08-20, T3): ANTES estas rutas eran absolutas y especificas
# de la maquina de Fernando (/home/fruiz/...). Confirmado con evidencia real
# (gh run view del CI wireado en ronda 3) que eso hacia de .github/workflows/
# policy.yml un NO-OP TOTAL, no cobertura parcial: en el runner de GitHub
# Actions ninguna de las dos rutas existe (checkout va a /home/runner/work/
# Jax/Jax), asi que `root.exists()` era False para AMBAS y el scanner
# recorria cero archivos -- "1 passed in 0.01s" en el log real, imposible
# para un scan real. El enforcement se sentia completo y no escaneaba nada.
#
# jax se resuelve ahora relativo a este archivo (portable a cualquier
# checkout: CI, otra maquina, otro path). jax-platform sigue sin checkout
# cruzado en la CI de jax (repo privado separado, requeriria un PAT/secret
# nuevo -- decision de infraestructura fuera de esta sesion) pero se
# resuelve por env var si algun dia se configura, o por el sibling local
# (conveniencia de Fernando en hall9000, no usado en CI).
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

FAIL_SOFT_MARKER = "# fail-soft:"


def _iter_python_files():
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            yield path


def _is_bare_pass_except(node: ast.ExceptHandler) -> bool:
    body = [
        stmt for stmt in node.body
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Constant)
        or not isinstance(stmt.value.value, str)  # descarta docstrings/comentarios-como-string
    ]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def find_fail_open_excepts() -> list[str]:
    linecache.clearcache()
    violations = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _is_bare_pass_except(node):
                source_line = linecache.getline(str(path), node.lineno)
                if FAIL_SOFT_MARKER in source_line:
                    continue
                violations.append(f"{path}:{node.lineno}")
    return violations


def test_no_fail_open_except() -> None:
    """Entry point pytest -- mismo escaneo que main(), vía assert en vez de
    exit code, para que CI (y `pytest` local) lo descubran solos."""
    violations = find_fail_open_excepts()
    assert not violations, (
        f"{len(violations)} except-pass (fail-open) sin marcar '# fail-soft: <razón>':\n"
        + "\n".join(violations)
    )


def main() -> int:
    violations = find_fail_open_excepts()
    if violations:
        print(f"FAIL — {len(violations)} except-pass (fail-open) encontrados:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("OK — cero except-pass silenciosos en el codigo fuente escaneado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
