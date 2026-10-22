"""Tests del dry-run de `scripts/b9_revertir_005.py` (MINOR 6, ronda 3 de
PR-J1, 2026-09-26).

POR QUE EXISTEN. El script se corrigio para dejar de prometer una
transaccion que MariaDB no puede dar (DDL hace commit implicito por
sentencia) -- el hallazgo real era de documentacion/expectativa, no de
comportamiento del dry-run en si, pero el dry-run es la unica parte de este
script que se puede probar sin ejecutar DDL real contra una base. Corre
contra una MariaDB real (la misma convencion que el resto de la suite):
sin JAX_DB_HOST, se salta.
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys
import uuid
from pathlib import Path

import aiomysql
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_de_test import es_base_de_test  # noqa: E402
from jax.core.db_connect_config import db_connect_timeout_seconds  # noqa: E402
from jax.memory.project_authority_migrations import apply_project_authority_migration  # noqa: E402
from b9_revertir_005 import _dry_run  # noqa: E402

_DB = os.getenv("JAX_DB_NAME", "")
requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB),
    reason="necesita una MariaDB real y JAX_DB_NAME en una base de tests",
)


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


def _conn_params() -> dict:
    return dict(
        host=os.environ.get("JAX_DB_HOST", ""), port=int(os.environ.get("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
    )


def _projects_ddl() -> str:
    import re
    schema = Path(__file__).resolve().parents[1] / "jax_memory_schema.sql"
    m = re.search(r"CREATE TABLE `projects` \(.*?\n\)[^;\n]*", schema.read_text(encoding="utf-8"), re.S)
    assert m
    return m.group(0)


async def _sql(query, args=(), fetch=False):
    conn = await aiomysql.connect(
        db=_DB, autocommit=True, cursorclass=aiomysql.DictCursor,
        connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            if fetch:
                return await cur.fetchall()
            return cur.lastrowid
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def _esquema():
    if not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB):
        return

    async def _ensure():
        conn = await aiomysql.connect(
            db=_DB, autocommit=True, connect_timeout=db_connect_timeout_seconds(),
            **_conn_params())
        try:
            async with conn.cursor() as cur:
                await cur.execute("SHOW TABLES LIKE 'projects'")
                if not await cur.fetchone():
                    await cur.execute(_projects_ddl())
                await apply_project_authority_migration(cur)
        finally:
            conn.close()

    asyncio.run(_ensure())


@requiere_db_de_prueba
@asincrono
async def test_dry_run_sin_filas_bloqueantes_da_cero_y_no_escribe():
    pool = await aiomysql.create_pool(
        db=_DB, autocommit=True, minsize=1, maxsize=2,
        connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        antes = await _sql("SELECT COUNT(*) AS n FROM jax_project_scope", fetch=True)
        resultado = await _dry_run(pool)
        despues = await _sql("SELECT COUNT(*) AS n FROM jax_project_scope", fetch=True)
        assert despues[0]["n"] == antes[0]["n"]
        # 0 solo si de verdad no hay filas ARCHIVED/HIDDEN en la sesion; con
        # otros tests corriendo en la misma base podria haber alguna --
        # entonces el propio dry-run debe dar 1, nunca reventar.
        assert resultado in (0, 1)
    finally:
        pool.close()
        await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_dry_run_con_fila_archivada_da_uno_y_no_escribe():
    tenant_id = await _sql("INSERT INTO jax_tenants (name,plan,status) VALUES ('t','personal','active')")
    project_id = await _sql(
        "INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),%s,'archived')", (f"p-{uuid.uuid4().hex[:6]}",))
    await _sql(
        "INSERT INTO jax_project_scope (project_id,tenant_id,status,created_at,created_by,updated_at) "
        "VALUES (%s,%s,'ARCHIVED',NOW(6),'test',NOW(6))", (project_id, tenant_id))
    pool = await aiomysql.create_pool(
        db=_DB, autocommit=True, minsize=1, maxsize=2,
        connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        resultado = await _dry_run(pool)
        assert resultado == 1
        rows = await _sql(
            "SELECT status FROM jax_project_scope WHERE project_id=%s", (project_id,), fetch=True)
        assert rows[0]["status"] == "ARCHIVED"  # sin tocar
    finally:
        pool.close()
        await pool.wait_closed()
