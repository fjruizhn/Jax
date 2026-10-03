"""E2a / T1: migracion 006a, tabla `project_documents`, contra MariaDB real.

Reusa los ayudantes de `test_project_authority_mariadb.py` (el repo ya importa
entre tests) y su fixture de esquema, que aplica la migracion REAL.
"""
from __future__ import annotations

import uuid

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.project_authority_migrations import (
    apply_project_authority_migration, revert_project_lifecycle_migration,
)
from test_project_authority_mariadb import (  # noqa: F401  (el fixture se registra por importacion)
    _DB, _conn_params, _crear_proyecto_activo, _crear_tenant, _crear_usuario, _ensure_schema,
    _esquema_de_proyectos, _projects_ddl, _sql, asincrono, requiere_db_de_prueba,
)

pytestmark = requiere_db_de_prueba


@asincrono
async def test_006a_crea_project_documents_con_indices_y_fks():
    await _ensure_schema(_DB)
    cols = {r["COLUMN_NAME"]: r["COLUMN_TYPE"] for r in await _sql(
        "SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='project_documents'", fetch=True)}
    assert cols["project_id"] == "int(11)"
    assert cols["sha256"] == "char(64)"
    assert cols["subido_por"] == "int(11)"
    assert cols["estado"].startswith(
        "enum('en_cola','pendiente','procesando','listo','parcial','error','sin_extractor','cancelado')")
    idx = {r["INDEX_NAME"] for r in await _sql(
        "SELECT INDEX_NAME FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='project_documents'", fetch=True)}
    assert {"uq_project_documents_sha", "idx_project_documents_lista", "idx_project_documents_despacho"} <= idx
    fks = {r["CONSTRAINT_NAME"] for r in await _sql(
        "SELECT CONSTRAINT_NAME FROM information_schema.REFERENTIAL_CONSTRAINTS "
        "WHERE CONSTRAINT_SCHEMA=DATABASE() AND TABLE_NAME='project_documents'", fetch=True)}
    assert fks == {"fk_project_documents_project", "fk_project_documents_subido_por",
                   "fk_project_documents_oculto_por"}


@asincrono
async def test_006a_es_idempotente():
    await _ensure_schema(_DB)
    await _ensure_schema(_DB)   # segunda pasada: sin error, sin cambios


@asincrono
async def test_006a_mismo_contenido_dos_veces_en_un_proyecto_choca():
    await _ensure_schema(_DB)
    t = await _crear_tenant("d1"); u = await _crear_usuario(t); p = await _crear_proyecto_activo(t, name="D")
    ins = ("INSERT INTO project_documents (project_id, sha256, nombre_original, bytes, tipo, subido_por) "
           "VALUES (%s, %s, 'a.pdf', 10, 'pdf', %s)")
    await _sql(ins, (p, "a" * 64, u))
    with pytest.raises(Exception, match="1062"):
        await _sql(ins, (p, "a" * 64, u))


@asincrono
async def test_006a_revert_borra_si_vacia_y_falla_cerrado_con_filas():
    """Base propia: la reversion de 005 arranca columnas de las que dependen
    las demas pruebas de la base compartida."""
    mig_db = f"{_DB}_mig{uuid.uuid4().hex[:8]}"
    assert es_base_de_test(mig_db)
    setup_conn = await aiomysql.connect(
        autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        async with setup_conn.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{mig_db}`")
    finally:
        setup_conn.close()
    try:
        conn = await aiomysql.connect(
            db=mig_db, autocommit=True, cursorclass=aiomysql.DictCursor,
            connect_timeout=db_connect_timeout_seconds(), **_conn_params())
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "CREATE TABLE jax_tenants (tenant_id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) "
                    "NOT NULL, plan VARCHAR(20), status VARCHAR(20), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
                await cur.execute(
                    "CREATE TABLE jax_users (user_id INT AUTO_INCREMENT PRIMARY KEY, tenant_id INT NOT NULL, "
                    "email VARCHAR(320) UNIQUE NOT NULL, password_hash VARCHAR(255) NOT NULL, "
                    "role VARCHAR(20) DEFAULT 'operator', status VARCHAR(20) DEFAULT 'active')")
                await cur.execute(_projects_ddl())
                await apply_project_authority_migration(cur)

                # Vacia: la reversion la quita y volver a aplicar la recrea.
                await revert_project_lifecycle_migration(cur)
                await cur.execute("SHOW TABLES LIKE 'project_documents'")
                assert await cur.fetchone() is None
                await apply_project_authority_migration(cur)
                await cur.execute("SHOW TABLES LIKE 'project_documents'")
                assert await cur.fetchone() is not None

                # Con filas: falla cerrado, y NO toca nada de 005.
                await cur.execute("INSERT INTO jax_tenants (name,plan,status) VALUES ('t','personal','active')")
                tenant_id = cur.lastrowid
                await cur.execute(
                    "INSERT INTO jax_users (tenant_id,email,password_hash) VALUES (%s,'u@test.invalid','x')",
                    (tenant_id,))
                user_id = cur.lastrowid
                await cur.execute("INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),'p','active')")
                project_id = cur.lastrowid
                await cur.execute(
                    "INSERT INTO project_documents (project_id, sha256, nombre_original, bytes, tipo, subido_por) "
                    "VALUES (%s, %s, 'a.pdf', 10, 'pdf', %s)", (project_id, "b" * 64, user_id))
                with pytest.raises(RuntimeError, match="project_documents"):
                    await revert_project_lifecycle_migration(cur)
                await cur.execute("SHOW TABLES LIKE 'project_documents'")
                assert await cur.fetchone() is not None
                await cur.execute("SHOW TABLES LIKE 'jax_project_creation_request'")
                assert await cur.fetchone() is not None  # fallo cerrado ANTES de bajar 005
        finally:
            conn.close()
    finally:
        drop_conn = await aiomysql.connect(
            autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
        async with drop_conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{mig_db}`")
        drop_conn.close()
