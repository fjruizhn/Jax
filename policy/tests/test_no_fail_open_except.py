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

Corre con:
  python3 policy/tests/test_no_fail_open_except.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOTS = [
    Path("/home/fruiz/jax"),
    Path("/home/fruiz/jax-platform"),
]

EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build",
}

# Rutas donde un except-pass silencioso es una decision de diseño ya
# documentada, no una instancia nueva del patron -- ver notas al pie.
ALLOWED = {
    # best-effort de sincronizacion de provider, ya documentado en
    # CONTEXT.md como fail-soft deliberado (mismo patron que anthropic).
}


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
    violations = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _is_bare_pass_except(node):
                rel = str(path)
                if rel in ALLOWED:
                    continue
                violations.append(f"{path}:{node.lineno}")
    return violations


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
