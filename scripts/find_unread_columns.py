#!/usr/bin/env python3
"""
Cruza el schema real de jax_memory contra el código de los dos repos (jax,
jax-platform) para encontrar columnas que el schema declara pero ningún
código real consume -- el mismo patrón encontrado 3 veces en 6 rondas
(reasoning_default_visibility, max_execution_minutes, y el bug de la
clave 'reasoning_content' que era una lectura equivocada, no ausente).

Método: por cada (tabla, columna) real, busca el NOMBRE de la columna como
palabra completa en todo el código Python/JS/JSX de ambos repos, EXCLUYENDO
los archivos donde solo aparecería por definir el schema (migrations.py,
CREATE TABLE, seeds) -- si aparece SOLO ahí, se marca candidata.

Falsos positivos conocidos, y por qué no los filtra el script (declarado a
propósito, no automatizado -- ver T1 de la sesión 2026-08-19):
  - Acceso dinámico (row["col"], **kwargs desempaquetado sin nombrar la
    columna): el nombre igual aparece LITERAL en el SELECT que trae la
    columna (mismo archivo que consume el resultado) -- grep lo encuentra.
    Riesgo real solo si el SELECT usa `SELECT *` y el nombre nunca se
    escribe en ningún lado -- no se filtra automático, se revisa a mano.
  - Columnas usadas SOLO en migraciones/seeds: el script las EXCLUYE de la
    cuenta a propósito (ver EXCLUDED_FILES) -- si el único match real es
    justamente ahí, cuenta como candidata, no como falso positivo.
  - Columnas de auditoría escritas pero nunca leídas (ej. created_at,
    approved_at): el script las sigue reportando si nada las LEE -- pero
    eso puede ser correcto (auditoría forense, no deuda). Veredicto manual.

NO borra nada. Diagnóstico puro -- ver la lista con veredicto en el commit
de esta sesión (docs/columnas-sin-consumidor-2026-08-19.md o el mensaje del
commit, según lo que exista al momento de leer esto).

Uso:
  set -a; source /etc/jax/.env; set +a
  /home/fruiz/jax/las_manos/.venv/bin/python scripts/find_unread_columns.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPOS = [
    "/home/fruiz/jax",
    "/home/fruiz/jax-platform",
]

# Directorios/archivos que solo definen schema (DDL/seed) -- un match SOLO
# acá no cuenta como "consumido", cuenta como candidata.
EXCLUDED_PATH_FRAGMENTS = [
    "/backend/db/migrations.py",
    "/.venv/", "/.git/", "/node_modules/", "/__pycache__/",
    "/.worktrees/", "/scripts/find_unread_columns.py",
    "/docs/", "/missions/", "/six-impossible-things.html",
    "_test.py", "test_", "/workspace/",  # tests y outputs de pipeline, no código de produccion
]

# Nombres tan genéricos que CUALQUIER cruce es ruido -- se listan aparte,
# no se investigan campo por campo (declarado, no oculto).
GENERIC_NAMES = {
    "id", "name", "key", "status", "created_at", "updated_at", "type",
    "value", "content", "reason", "path", "role", "message", "error",
    "description", "data", "config", "enabled",
}


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.stdout


def get_schema_columns() -> list[tuple[str, str]]:
    import pymysql
    conn = pymysql.connect(
        host=os.environ.get("JAX_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("JAX_DB_PORT", "3308")),
        user=os.environ["JAX_DB_USER"],
        password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ["JAX_DB_NAME"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME, ORDINAL_POSITION",
                (os.environ["JAX_DB_NAME"],),
            )
            return list(cur.fetchall())
    finally:
        conn.close()


def grep_column(column: str) -> list[str]:
    """Devuelve las rutas (fuera de EXCLUDED_PATH_FRAGMENTS) que mencionan
    `column` como palabra completa, en cualquiera de los dos repos."""
    pattern = rf"\b{re.escape(column)}\b"
    hits: list[str] = []
    for repo in REPOS:
        out = _run([
            "grep", "-rlP", "--include=*.py", "--include=*.js", "--include=*.jsx",
            "--include=*.toml", pattern, repo,
        ])
        for line in out.splitlines():
            if not any(frag in line for frag in EXCLUDED_PATH_FRAGMENTS):
                hits.append(line)
    return hits


def main() -> None:
    columns = get_schema_columns()
    print(f"{len(columns)} columnas reales en el schema.\n")

    candidates: list[tuple[str, str]] = []
    for table, column in columns:
        if column in GENERIC_NAMES:
            continue
        hits = grep_column(column)
        if not hits:
            candidates.append((table, column))

    print(f"=== {len(candidates)} candidatas (sin match fuera de migrations/tests/docs) ===\n")
    for table, column in candidates:
        print(f"{table}.{column}")

    print(f"\n({len(GENERIC_NAMES)} nombres genéricos excluidos sin investigar: {sorted(GENERIC_NAMES)})")


if __name__ == "__main__":
    main()
