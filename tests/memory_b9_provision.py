"""Provision the fresh MariaDB database that `memory_b9_regression_driver.py` needs.

Run as a Python file, never through pytest/conftest:

    JAX_DB_HOST=127.0.0.1 JAX_DB_PORT=<port> JAX_DB_USER=jax_test \\
    JAX_DB_PASSWORD=... JAX_DB_NAME=jax_memory_test_memb9_<sufijo> \\
    PYTHONPATH=. python tests/memory_b9_provision.py

What it builds, in this order (same order the production deploy follows):

1. `jax_memory_schema.sql` WITHOUT its first two statements. That file opens with
   `CREATE DATABASE IF NOT EXISTS jax_memory` and `USE jax_memory;`: loading it
   verbatim would point the rest of the script at the production name. They are
   dropped here and the connection is already bound to the test database.
2. `jax_tenants` and `jax_users` (minimal copy of the production shape). They
   belong to jax-platform, not to this repo, and migration 003 has a FK to
   `jax_users`, so they must exist before it.
3. Migrations 001 -> 004 (`jax/memory/b9_migrations/*.sql`).
4. Migration 005, which is Python only: `apply_project_authority_migration`.
5. Migrations 006 -> 013.
6. Fixture rows the driver asserts on: `jax_tenants` 1 and 2; `jax_users` user 1
   (tenant 1) and user 2 (tenant 2), both `active` and role `admin` (the legacy
   adoption case needs a real administrator, `memory:admin`); and user 4
   (tenant 1, `active`, role `operator`): the non-administrator member whose legacy
   import must be denied with AuthorizationDenied and whose retrieval must not see
   user 1's private memory. Without user 4 the denial would be a ScopeDenied
   ("not a tenant member"), which is a different, weaker control.

Fail closed: the user must be `jax_test`, the database name must start with
`jax_memory_test_memb9_`, and the database must be EMPTY (no tables). It never
drops anything: a database that is not empty is refused, not cleaned.
"""
import asyncio
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiomysql

from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.project_authority_migrations import apply_project_authority_migration

RAIZ = Path(__file__).resolve().parents[1]
ESQUEMA = RAIZ / "jax_memory_schema.sql"
MIGRACIONES = RAIZ / "jax/memory/b9_migrations"
PREFIJO_BASE = "jax_memory_test_memb9_"
USUARIO_DE_PRUEBA = "jax_test"

# Production's shape, minus columns nobody here reads.
DDL_JAX_PLATAFORMA = (
    "CREATE TABLE jax_tenants (tenant_id INT NOT NULL AUTO_INCREMENT, name VARCHAR(100) NOT NULL, "
    "plan VARCHAR(20) DEFAULT 'personal', status VARCHAR(20) DEFAULT 'active', "
    "created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (tenant_id)) ENGINE=InnoDB",
    "CREATE TABLE jax_users (user_id INT NOT NULL AUTO_INCREMENT, tenant_id INT NOT NULL, "
    "email VARCHAR(320) NOT NULL, role VARCHAR(20) DEFAULT 'operator', status VARCHAR(20) DEFAULT 'active', "
    "created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (user_id), UNIQUE KEY email (email), "
    "KEY tenant_id (tenant_id), CONSTRAINT jax_users_ibfk_1 FOREIGN KEY (tenant_id) REFERENCES jax_tenants (tenant_id)) "
    "ENGINE=InnoDB",
)
FIXTURES = (
    "INSERT INTO jax_tenants (tenant_id, name) VALUES (1, 'b9 fixture tenant 1'), (2, 'b9 fixture tenant 2')",
    "INSERT INTO jax_users (user_id, tenant_id, email, role, status) VALUES "
    "(1, 1, 'b9-user-1@example.invalid', 'admin', 'active'), "
    "(2, 2, 'b9-user-2@example.invalid', 'admin', 'active'), "
    "(4, 1, 'b9-user-4@example.invalid', 'operator', 'active')",
)
_PREAMBULO_DE_ESQUEMA = re.compile(r"^\s*(CREATE\s+DATABASE\b|USE\s)", re.IGNORECASE)


def dividir_sentencias(texto):
    """Split a .sql file into statements. Honors `DELIMITER` (migration 001's
    triggers) and drops `--` comment lines. A statement ends at a line whose
    stripped end is the current delimiter."""
    delimitador = ";"
    sentencias, actual = [], []
    for linea in texto.splitlines():
        base = linea.strip()
        if base.upper().startswith("DELIMITER "):
            delimitador = base.split(None, 1)[1].strip()
            continue
        if not base or base.startswith("--"):
            continue
        actual.append(linea)
        if base.endswith(delimitador):
            sentencia = "\n".join(actual).rstrip()
            sentencias.append(sentencia[: -len(delimitador)].rstrip())
            actual = []
    if actual:
        sentencias.append("\n".join(actual).strip())
    return [s for s in sentencias if s]


def sentencias_de_esquema():
    """The schema file minus `CREATE DATABASE jax_memory` / `USE jax_memory`."""
    todas = dividir_sentencias(ESQUEMA.read_text())
    utiles = [s for s in todas if not _PREAMBULO_DE_ESQUEMA.match(s)]
    if len(todas) - len(utiles) != 2:
        raise RuntimeError("schema_preamble_changed")  # fail closed: the strip rule no longer matches the file
    if any(re.search(r"\bjax_memory\b", s) for s in utiles):
        raise RuntimeError("schema_names_production")  # a leftover statement still names the production database
    return utiles


def guarda(env):
    if env.get("JAX_DB_USER") != USUARIO_DE_PRUEBA or not env.get("JAX_DB_NAME", "").startswith(PREFIJO_BASE):
        raise RuntimeError("test_database_guard")
    for nombre in ("JAX_DB_HOST", "JAX_DB_PORT", "JAX_DB_PASSWORD"):
        if not env.get(nombre):
            raise RuntimeError("test_configuration_missing")


async def _ejecutar(cur, sentencias):
    for sentencia in sentencias:
        await cur.execute(sentencia)


async def provisionar(env):
    guarda(env)
    conn = await aiomysql.connect(
        host=env["JAX_DB_HOST"], port=int(env["JAX_DB_PORT"]), user=env["JAX_DB_USER"],
        password=env["JAX_DB_PASSWORD"], db=env["JAX_DB_NAME"], autocommit=True, charset="utf8mb4",
        connect_timeout=db_connect_timeout_seconds())
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE()")
            if (await cur.fetchone())[0]:
                raise RuntimeError("fresh_database_required")
            await _ejecutar(cur, sentencias_de_esquema())
            await _ejecutar(cur, DDL_JAX_PLATAFORMA)
            migraciones = {p.name[:3]: p for p in sorted(MIGRACIONES.glob("[0-9][0-9][0-9]_*.sql"))}
            if sorted(migraciones) != ["001", "002", "003", "004", "006", "007", "008", "009", "010", "011", "012", "013"]:
                raise RuntimeError("migration_set_changed")  # a new migration must be placed in this order on purpose
            for numero in ("001", "002", "003", "004"):
                await _ejecutar(cur, dividir_sentencias(migraciones[numero].read_text()))
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await apply_project_authority_migration(cur)  # 005, Python only
        async with conn.cursor() as cur:
            for numero in ("006", "007", "008", "009", "010", "011", "012", "013"):
                await _ejecutar(cur, dividir_sentencias(migraciones[numero].read_text()))
            await _ejecutar(cur, FIXTURES)
    finally:
        conn.close()


def main():
    try:
        asyncio.run(provisionar(os.environ))
    except Exception as error:  # fail-closed: sanitized type only, rc=1; never echo connection data
        print(f"PROVISION_FAIL {type(error).__name__}: {error}")
        return 1
    print(f"PROVISION_OK {os.environ['JAX_DB_NAME']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
