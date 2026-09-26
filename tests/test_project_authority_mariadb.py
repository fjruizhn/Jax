"""§3-bis / D1-D5 project authority, against the REAL resolver and MariaDB.

PR-J1 (plan `2026-09-25-proyectos-d1d5-plan.md`, section 3). Every test here
talks to a real MariaDB 12.3.3 -- never a resolver double -- per
`test_no_trusted_admin_resolver_double_reappears`
(`test_project_scope_authority.py`).

Isolation: like every other MariaDB-backed test in this repo, this file only
runs against a base whose name is the shared test template or carries its
`_test_<suffix>` prefix (`es_base_de_test`), and skips outright without
`JAX_DB_HOST`. It creates its own `tenant`/`user`/`projects` rows with
AUTO_INCREMENT ids (no fixed literal id can collide with another test file in
the same session database) and, for the two tests that need a schema
completely of their own (the migration apply/apply/revert/apply cycle, and
the reserved-id-range test that would otherwise poison the whole session's
`projects.AUTO_INCREMENT` forever), a throw-away sibling database it drops
when done.
"""
from __future__ import annotations

import asyncio
import functools
import os
import uuid
from pathlib import Path

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.b9 import MutationAuthorizationRequest, ScopeContext, ScopeDenied, Visibility
from jax.memory.b9_mariadb import MariaDBB9Store
from jax.memory.project_authority import (
    AlreadyMember, IdempotencyKeyConflict, LastOwnerRequired, MemberNotFound,
    ProjectAuthorityAdmin, ProjectAuthorityError, ProjectNotVisible, ProjectRoleInsufficient,
    ProjectStateConflict, ReservedProjectIdRange, TargetUserNotEligible, TenantAdminMembershipProtected,
)
from jax.memory.project_authority_migrations import (
    apply_project_authority_migration, revert_project_lifecycle_migration,
)
from jax.memory.scope_authority import (
    MariaDBScopeAuthorityResolver, ProjectLifecycle, ProjectRole, TENANT_ADMIN_ROLES,
)

_DB = os.getenv("JAX_DB_NAME", "")
requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB),
    reason="necesita una MariaDB real y JAX_DB_NAME en una base de tests",
)

_SCHEMA_SQL = Path(__file__).resolve().parents[1] / "jax_memory_schema.sql"


def asincrono(fn):
    """Same convention as `test_memory_scope_denormalized.py`: one fresh
    event loop per test via `asyncio.run`, not `@pytest.mark.asyncio` --
    several of these tests open more than one pool/connection and a shared
    session-scoped loop is not worth the bookkeeping here."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


def _conn_params() -> dict:
    return dict(
        host=os.environ.get("JAX_DB_HOST", ""), port=int(os.environ.get("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        connect_timeout=db_connect_timeout_seconds(),
    )


def _projects_ddl() -> str:
    import re
    match = re.search(r"CREATE TABLE `projects` \(.*?\n\)[^;\n]*", _SCHEMA_SQL.read_text(encoding="utf-8"), re.S)
    assert match, "projects no esta en jax_memory_schema.sql: el test no podria crearla"
    return match.group(0)


async def _ensure_schema(db_name: str) -> None:
    """`projects` (from the repo's schema file, matching what CI's mysql-client
    step applies) + jax_project_scope/membership/event/creation_request via
    the REAL migration (003 DDL + 005), never a hand-rolled copy."""
    conn = await aiomysql.connect(db=db_name, autocommit=True, **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute("SHOW TABLES LIKE 'projects'")
            if not await cur.fetchone():
                await cur.execute(_projects_ddl())
            await apply_project_authority_migration(cur)
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def _esquema_de_proyectos():
    if not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB):
        return
    asyncio.run(_ensure_schema(_DB))


async def _sql(query: str, args: tuple = (), *, db: str | None = None, fetch: bool = False):
    conn = await aiomysql.connect(db=db or _DB, autocommit=True, cursorclass=aiomysql.DictCursor, **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            if fetch:
                return await cur.fetchall()
            return cur.lastrowid
    finally:
        conn.close()


async def _crear_tenant(name: str = "t") -> int:
    return await _sql("INSERT INTO jax_tenants (name,plan,status) VALUES (%s,'personal','active')", (name,))


async def _crear_usuario(tenant_id: int, *, role: str = "operator", status: str = "active",
                         email: str | None = None) -> int:
    email = email or f"u{uuid.uuid4().hex[:20]}@test.invalid"
    return await _sql(
        "INSERT INTO jax_users (tenant_id,email,password_hash,role,status) VALUES (%s,%s,'x',%s,%s)",
        (tenant_id, email, role, status))


async def _crear_proyecto_activo(tenant_id: int, *, name: str = "p", status: str = "active") -> int:
    return await _sql("INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),%s,%s)", (name, status))


async def _crear_scope(project_id: int, tenant_id: int, status: str = "ACTIVE") -> None:
    await _sql(
        "INSERT INTO jax_project_scope (project_id,tenant_id,status,created_at,created_by,updated_at) "
        "VALUES (%s,%s,%s,NOW(6),'test',NOW(6))", (project_id, tenant_id, status))


async def _crear_membresia(project_id: int, tenant_id: int, user_id: int, *, role: str = "OWNER",
                           status: str = "ACTIVE", origin: str = "EXPLICIT") -> None:
    await _sql(
        "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,status,"
        "grant_origin,created_at,created_by,updated_at) VALUES (UUID(),%s,%s,%s,%s,%s,%s,NOW(6),'test',NOW(6))",
        (project_id, tenant_id, user_id, role, status, origin))


async def _pool(maxsize: int = 6):
    return await aiomysql.create_pool(
        db=_DB, autocommit=True, minsize=1, maxsize=maxsize, cursorclass=aiomysql.DictCursor, **_conn_params())


def _scope(user_id: int, tenant_id: int, project_id: int | None = None) -> ScopeContext:
    return ScopeContext(f"user:{user_id}", "USER", str(user_id), str(tenant_id),
                        str(project_id) if project_id is not None else None)


def _request(scope: ScopeContext, operation: str) -> MutationAuthorizationRequest:
    return MutationAuthorizationRequest(scope, operation, Visibility.PROJECT_SHARED)


def _admin(pool) -> ProjectAuthorityAdmin:
    return ProjectAuthorityAdmin(MariaDBB9Store(pool), MariaDBScopeAuthorityResolver(pool))


async def _admins_sin_ownership(tenant_id: int):
    """§6.5's invariant, checked with a query (D2's own decision text): every
    ACTIVE admin of T has an ACTIVE OWNER membership on every project of T.
    Zero rows == the invariant holds. (The actual `proyectos_detector.py`
    script lives in jax-platform, PR-P1, out of this PR's scope.)"""
    return await _sql(
        "SELECT u.user_id, s.project_id FROM jax_users u "
        "JOIN jax_project_scope s ON s.tenant_id=u.tenant_id "
        "LEFT JOIN jax_project_membership m ON m.project_id=s.project_id AND m.user_id=u.user_id "
        "AND m.status='ACTIVE' AND m.project_role='OWNER' "
        "WHERE u.tenant_id=%s AND u.status='active' "
        "AND u.role IN ('admin','superadmin','super_admin') AND m.membership_id IS NULL",
        (tenant_id,), fetch=True)


async def _espejo_desincronizado(tenant_id: int):
    """D5's mirror invariant: `projects.status` always matches
    `jax_project_scope.status` for the same project."""
    return await _sql(
        "SELECT s.project_id FROM jax_project_scope s JOIN projects p ON p.id=s.project_id "
        "WHERE s.tenant_id=%s AND ("
        "(s.status='ACTIVE' AND p.status<>'active') OR (s.status='ARCHIVED' AND p.status<>'archived') OR "
        "(s.status='HIDDEN' AND p.status<>'hidden') OR (s.status='DISABLED' AND p.status<>'disabled'))",
        (tenant_id,), fetch=True)


class _Recorder:
    """Wraps a real aiomysql cursor and records every SQL statement, in
    order, so test 13 can assert the FIRST statement of a transaction is
    the tenant-row lock -- without a resolver/store double (H4's lesson)."""
    def __init__(self, real_cur, calls: list[str]):
        self._real, self._calls = real_cur, calls

    async def execute(self, sql, args=None):
        self._calls.append(sql)
        return await (self._real.execute(sql, args) if args is not None else self._real.execute(sql))

    async def fetchone(self):
        return await self._real.fetchone()

    async def fetchall(self):
        return await self._real.fetchall()


class _RecordingStore:
    """Same `.mutation()` contract as `MariaDBB9Store`, against the REAL
    pool/transaction -- it only adds the SQL-order tap of `_Recorder`. Each
    `mutation()` call gets its OWN fresh list in `self.mutations`, so the
    lock-free idempotency pre-read `create_project` does before opening its
    write transaction (decision 6) never gets confused with that write
    transaction's own first statement."""
    def __init__(self, pool):
        self._pool = pool
        self.mutations: list[list[str]] = []

    async def mutation(self, operation):
        calls: list[str] = []
        self.mutations.append(calls)
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as real_cur:
                    value = await operation(_Recorder(real_cur, calls))
                await conn.commit()
                return value
            except Exception:
                await conn.rollback()
                raise


async def _promote_to_admin(pool, actor_scope: ScopeContext, tenant_id: int, user_id: int) -> None:
    """Mirrors what the platform's own `jax_users` admin-role transaction is
    specified to do (plan section 4, `backend/api/admin/users.py`): lock the
    tenant, then the admin set, then the target row, flip the DB role, and
    call `sync_tenant_admin_memberships_in_transaction` inside that same
    transaction -- all before this PR's own `create_project`/etc. can even
    start (the tenant row is a per-tenant mutex, decision 1)."""
    admin_api = _admin(pool)
    async with pool.acquire() as conn:
        await conn.begin()
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT tenant_id FROM jax_tenants WHERE tenant_id=%s FOR UPDATE", (tenant_id,))
                await cur.execute(
                    "SELECT user_id FROM jax_users WHERE tenant_id=%s AND role IN "
                    "('admin','superadmin','super_admin') AND status='active' ORDER BY user_id FOR UPDATE",
                    (tenant_id,))
                await cur.fetchall()
                await cur.execute("SELECT role,status FROM jax_users WHERE user_id=%s FOR UPDATE", (user_id,))
                before = await cur.fetchone()
                was_admin = (str(before["role"]).lower() in TENANT_ADMIN_ROLES
                            and str(before["status"]).lower() == "active")
                await cur.execute("UPDATE jax_users SET role='admin' WHERE user_id=%s", (user_id,))
                await admin_api.sync_tenant_admin_memberships_in_transaction(
                    cur, actor_scope=actor_scope, user_id=user_id, tenant_id=tenant_id,
                    was_active_admin=was_admin, is_active_admin=True)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def _demote_from_admin(pool, actor_scope: ScopeContext, tenant_id: int, user_id: int) -> None:
    """The descent mirror of `_promote_to_admin`: same lock order (tenant ->
    admin set -> target row), flips the DB role back, then calls
    `sync_tenant_admin_memberships_in_transaction` with
    `was_active_admin=True, is_active_admin=False` inside the SAME
    transaction. A `LastOwnerRequired` raised by the sync call rolls back
    EVERYTHING in this transaction -- the role flip included -- because the
    `except` here does `conn.rollback()` before re-raising."""
    admin_api = _admin(pool)
    async with pool.acquire() as conn:
        await conn.begin()
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT tenant_id FROM jax_tenants WHERE tenant_id=%s FOR UPDATE", (tenant_id,))
                await cur.execute(
                    "SELECT user_id FROM jax_users WHERE tenant_id=%s AND role IN "
                    "('admin','superadmin','super_admin') AND status='active' ORDER BY user_id FOR UPDATE",
                    (tenant_id,))
                await cur.fetchall()
                await cur.execute("SELECT role,status FROM jax_users WHERE user_id=%s FOR UPDATE", (user_id,))
                before = await cur.fetchone()
                was_admin = (str(before["role"]).lower() in TENANT_ADMIN_ROLES
                            and str(before["status"]).lower() == "active")
                await cur.execute("UPDATE jax_users SET role='operator' WHERE user_id=%s", (user_id,))
                await admin_api.sync_tenant_admin_memberships_in_transaction(
                    cur, actor_scope=actor_scope, user_id=user_id, tenant_id=tenant_id,
                    was_active_admin=was_admin, is_active_admin=False)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def _contar_eventos_de_sincronizacion(project_id: int, user_id: int) -> int:
    """Cuenta solo los eventos que produce el sync de administración
    (GRANT_TENANT_ADMIN/REVOKE_TENANT_ADMIN) para un usuario en un proyecto --
    nunca CREATE_PROJECT ni otros, que son de otras operaciones."""
    rows = await _sql(
        "SELECT COUNT(*) AS n FROM jax_project_membership_event WHERE project_id=%s AND target_user_id=%s "
        "AND operation IN ('GRANT_TENANT_ADMIN','REVOKE_TENANT_ADMIN')", (project_id, user_id), fetch=True)
    return rows[0]["n"] if rows else 0


# --------------------------------------------------------------------------
# 1. Migration: apply -> apply (idempotent) -> revert -> apply. The bajada
#    fails closed with an ARCHIVED row present. Own throw-away database:
#    running this against the shared session DB would rip out columns the
#    other 12 tests depend on.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_migracion_apply_apply_revert_apply_y_falla_cerrado_con_fila_archivada():
    mig_db = f"{_DB}_mig{uuid.uuid4().hex[:8]}"
    assert es_base_de_test(mig_db)
    setup_conn = await aiomysql.connect(autocommit=True, **_conn_params())
    try:
        async with setup_conn.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{mig_db}`")
    finally:
        setup_conn.close()
    try:
        conn = await aiomysql.connect(db=mig_db, autocommit=True, cursorclass=aiomysql.DictCursor, **_conn_params())
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

                async def _snapshot() -> dict:
                    snap = {}
                    for table in ("jax_project_scope", "jax_project_membership",
                                 "jax_project_membership_event", "jax_project_creation_request", "projects"):
                        await cur.execute(f"SHOW CREATE TABLE `{table}`")
                        row = await cur.fetchone()
                        snap[table] = row["Create Table"]
                    return snap

                await apply_project_authority_migration(cur)
                snap1 = await _snapshot()
                await apply_project_authority_migration(cur)  # apply -> apply, idempotent
                assert await _snapshot() == snap1

                await revert_project_lifecycle_migration(cur)
                await cur.execute("SHOW TABLES LIKE 'jax_project_creation_request'")
                assert await cur.fetchone() is None
                await cur.execute(
                    "SELECT COUNT(*) AS n FROM information_schema.CHECK_CONSTRAINTS "
                    "WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME='chk_jax_project_scope_status'")
                assert (await cur.fetchone())["n"] == 1

                await apply_project_authority_migration(cur)  # revert -> apply
                assert await _snapshot() == snap1

                await cur.execute("INSERT INTO jax_tenants (name,plan,status) VALUES ('t','personal','active')")
                tenant_id = cur.lastrowid
                await cur.execute("INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),'p','archived')")
                project_id = cur.lastrowid
                await cur.execute(
                    "INSERT INTO jax_project_scope (project_id,tenant_id,status,created_at,created_by,updated_at) "
                    "VALUES (%s,%s,'ARCHIVED',NOW(6),'test',NOW(6))", (project_id, tenant_id))
                with pytest.raises(RuntimeError):
                    await revert_project_lifecycle_migration(cur)
        finally:
            conn.close()
    finally:
        drop_conn = await aiomysql.connect(autocommit=True, **_conn_params())
        async with drop_conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{mig_db}`")
        drop_conn.close()


# --------------------------------------------------------------------------
# 2. Section 3-bis.
# --------------------------------------------------------------------------

def test_bootstrap_no_acepta_tenant_id_como_parametro():
    import inspect
    params = inspect.signature(ProjectAuthorityAdmin.bootstrap_existing_project).parameters
    assert "tenant_id" not in params


@requiere_db_de_prueba
@asincrono
async def test_seccion_3bis_bootstrap_reactivacion_disabled_y_anti_lavado():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        admin_user = await _crear_usuario(tenant_id, role="admin")
        owner_user = await _crear_usuario(tenant_id, role="operator")

        legacy_project_id = await _crear_proyecto_activo(tenant_id, name="legacy")
        bootstrap_scope = _scope(admin_user, tenant_id, legacy_project_id)
        created = await admin_api.bootstrap_existing_project(
            _request(bootstrap_scope, "BOOTSTRAP_PROJECT"), legacy_project_id, owner_user_id=owner_user)
        assert created is True
        rows = await _sql("SELECT status FROM jax_project_scope WHERE project_id=%s", (legacy_project_id,), fetch=True)
        assert rows[0]["status"] == "ACTIVE"

        await _sql("UPDATE jax_project_scope SET status='DISABLED' WHERE project_id=%s", (legacy_project_id,))
        ok = await admin_api.set_project_lifecycle(
            _request(bootstrap_scope, "SET_PROJECT_LIFECYCLE"), legacy_project_id, ProjectLifecycle.ACTIVE)
        assert ok is True
        rows = await _sql("SELECT status FROM jax_project_scope WHERE project_id=%s", (legacy_project_id,), fetch=True)
        assert rows[0]["status"] == "ACTIVE"

        other_project_id = await _crear_proyecto_activo(tenant_id, name="other")
        await _crear_scope(other_project_id, tenant_id)
        # owner_user is an OWNER of legacy_project_id (P1) only. Its own
        # resolved scope points at P1; passing P2 (other_project_id) as the
        # ARGUMENT must not grant anything (H4's laundering closed by D2).
        laundering_scope = _scope(owner_user, tenant_id, legacy_project_id)
        with pytest.raises(ProjectNotVisible):
            await admin_api.grant_member(
                _request(laundering_scope, "GRANT_MEMBER"), other_project_id,
                email="whoever@test.invalid", role=ProjectRole.VIEWER)
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 3. Cross-tenant.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_tenant_cruzado_niega_lectura_lifecycle_y_grant_y_usuario_no_elegible():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        resolver = MariaDBScopeAuthorityResolver(pool)
        tenant1 = await _crear_tenant("t1")
        tenant2 = await _crear_tenant("t2")
        owner1 = await _crear_usuario(tenant1, role="operator")
        admin2 = await _crear_usuario(tenant2, role="admin")
        outsider2_email = f"outsider-{uuid.uuid4().hex[:8]}@test.invalid"
        await _crear_usuario(tenant2, role="operator", email=outsider2_email)

        project1 = await _crear_proyecto_activo(tenant1, name="p1")
        await _crear_scope(project1, tenant1)
        await _crear_membresia(project1, tenant1, owner1, role="OWNER", origin="CREATOR")

        cross_scope = _scope(admin2, tenant2, project1)
        with pytest.raises(ScopeDenied):
            await resolver.resolve_project_read(cross_scope)
        with pytest.raises(ProjectNotVisible):
            await admin_api.set_project_lifecycle(
                _request(cross_scope, "SET_PROJECT_LIFECYCLE"), project1, ProjectLifecycle.ARCHIVED)
        with pytest.raises(ProjectNotVisible):
            await admin_api.grant_member(
                _request(cross_scope, "GRANT_MEMBER"), project1, email="whoever@test.invalid", role=ProjectRole.VIEWER)

        owner_scope = _scope(owner1, tenant1, project1)
        with pytest.raises(TargetUserNotEligible):
            await admin_api.grant_member(
                _request(owner_scope, "GRANT_MEMBER"), project1, email=outsider2_email, role=ProjectRole.VIEWER)
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 4. An OWNER cannot demote or revoke an active admin; no event row either.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_owner_no_puede_rebajar_ni_revocar_a_un_admin_activo_y_no_deja_evento():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        owner_user = await _crear_usuario(tenant_id, role="operator")
        admin_user = await _crear_usuario(tenant_id, role="admin")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, owner_user, role="OWNER", origin="CREATOR")
        await _crear_membresia(project_id, tenant_id, admin_user, role="OWNER", origin="TENANT_ADMIN")

        events_before = await _sql(
            "SELECT COUNT(*) AS n FROM jax_project_membership_event WHERE project_id=%s", (project_id,), fetch=True)
        owner_scope = _scope(owner_user, tenant_id, project_id)
        with pytest.raises(TenantAdminMembershipProtected):
            await admin_api.change_project_role(
                _request(owner_scope, "CHANGE_PROJECT_ROLE"), project_id, admin_user, ProjectRole.VIEWER)
        with pytest.raises(TenantAdminMembershipProtected):
            await admin_api.revoke_member(_request(owner_scope, "REVOKE_MEMBER"), project_id, admin_user)
        events_after = await _sql(
            "SELECT COUNT(*) AS n FROM jax_project_membership_event WHERE project_id=%s", (project_id,), fetch=True)
        assert events_before[0]["n"] == events_after[0]["n"]
        rows = await _sql(
            "SELECT project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s",
            (project_id, admin_user), fetch=True)
        assert rows[0]["project_role"] == "OWNER" and rows[0]["status"] == "ACTIVE"
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 5. ARCHIVED: memory paths denied, project-read allowed to a member,
#    grant_member -> proyecto_no_activo.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_archived_niega_memoria_pero_permite_lectura_de_miembro():
    pool = await _pool()
    try:
        resolver = MariaDBScopeAuthorityResolver(pool)
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        owner_user = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id, status="ARCHIVED")
        await _crear_membresia(project_id, tenant_id, owner_user, role="OWNER", origin="CREATOR")
        scope = _scope(owner_user, tenant_id, project_id)

        with pytest.raises(ScopeDenied):
            await resolver.resolve_scope(scope)

        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    with pytest.raises(ScopeDenied):
                        await resolver.resolve_mutation_in_transaction(cur, scope, "RETRIEVE", Visibility.PROJECT_SHARED)
            finally:
                await conn.rollback()

        service_scope = ScopeContext("service:memory-extraction", "SERVICE", str(owner_user), str(tenant_id),
                                     str(project_id), calling_component="memory-extraction")
        with pytest.raises(ScopeDenied):
            await resolver.resolve_service_mutation(service_scope, "CREATE", Visibility.PROJECT_SHARED)

        read = await resolver.resolve_project_read(scope)
        assert read.lifecycle is ProjectLifecycle.ARCHIVED

        with pytest.raises(ProjectStateConflict) as excinfo:
            await admin_api.grant_member(
                _request(scope, "GRANT_MEMBER"), project_id, email="whoever@test.invalid", role=ProjectRole.VIEWER)
        assert excinfo.value.code == "proyecto_no_activo"
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 6. HIDDEN: denied to a plain member, allowed to an admin member.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_hidden_niega_a_miembro_comun_y_permite_a_admin_miembro():
    pool = await _pool()
    try:
        resolver = MariaDBScopeAuthorityResolver(pool)
        tenant_id = await _crear_tenant()
        member_user = await _crear_usuario(tenant_id, role="operator")
        admin_user = await _crear_usuario(tenant_id, role="admin")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id, status="HIDDEN")
        await _crear_membresia(project_id, tenant_id, member_user, role="OWNER", origin="EXPLICIT")
        await _crear_membresia(project_id, tenant_id, admin_user, role="OWNER", origin="TENANT_ADMIN")

        with pytest.raises(ScopeDenied):
            await resolver.resolve_project_read(_scope(member_user, tenant_id, project_id))
        read = await resolver.resolve_project_read(_scope(admin_user, tenant_id, project_id))
        assert read.lifecycle is ProjectLifecycle.HIDDEN and read.is_tenant_admin is True
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 7. Reserved id range: create_project rejects and leaves 0 new rows
#    anywhere. Own throw-away database (ALTER TABLE ... AUTO_INCREMENT is
#    not transactional/reversible; it would poison the shared session DB
#    forever for every OTHER test in this file).
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_rango_reservado_rechaza_creacion_sin_dejar_filas_nuevas():
    db_name = f"{_DB}_rango{uuid.uuid4().hex[:8]}"
    assert es_base_de_test(db_name)
    setup_conn = await aiomysql.connect(autocommit=True, **_conn_params())
    try:
        async with setup_conn.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{db_name}`")
    finally:
        setup_conn.close()
    try:
        conn = await aiomysql.connect(db=db_name, autocommit=True, cursorclass=aiomysql.DictCursor, **_conn_params())
        pool = None
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
                await cur.execute("INSERT INTO jax_tenants (name,plan,status) VALUES ('t','personal','active')")
                tenant_id = cur.lastrowid
                await cur.execute(
                    "INSERT INTO jax_users (tenant_id,email,password_hash,role,status) VALUES "
                    "(%s,'owner@test.invalid','x','operator','active')", (tenant_id,))
                owner_id = cur.lastrowid
                await cur.execute("ALTER TABLE projects AUTO_INCREMENT=900001")

            pool = await aiomysql.create_pool(
                db=db_name, autocommit=True, minsize=1, maxsize=2, cursorclass=aiomysql.DictCursor, **_conn_params())
            admin_api = _admin(pool)
            scope = ScopeContext(f"user:{owner_id}", "USER", str(owner_id), str(tenant_id), None)
            with pytest.raises(ReservedProjectIdRange):
                await admin_api.create_project(
                    _request(scope, "CREATE_PROJECT"), name="p", description=None, idempotency_key=str(uuid.uuid4()))
            async with pool.acquire() as c2:
                async with c2.cursor() as cur2:
                    for table in ("projects", "jax_project_scope", "jax_project_membership",
                                 "jax_project_membership_event", "jax_project_creation_request"):
                        await cur2.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
                        assert (await cur2.fetchone())["n"] == 0, f"{table} has unexpected rows"
        finally:
            if pool is not None:
                pool.close(); await pool.wait_closed()
            conn.close()
    finally:
        drop_conn = await aiomysql.connect(autocommit=True, **_conn_params())
        async with drop_conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
        drop_conn.close()


# --------------------------------------------------------------------------
# 8. Concurrency: 20 parallel creates, same-key/same-body idempotency,
#    same-key/different-body conflict.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_concurrencia_20_creaciones_paralelas_y_idempotencia():
    pool = await _pool(maxsize=12)
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        creator = await _crear_usuario(tenant_id, role="operator")
        await _crear_usuario(tenant_id, role="admin")
        await _crear_usuario(tenant_id, role="superadmin")
        scope = _scope(creator, tenant_id, None)

        async def _create(key: str, name: str):
            return await admin_api.create_project(
                _request(scope, "CREATE_PROJECT"), name=name, description=None, idempotency_key=key)

        keys = [str(uuid.uuid4()) for _ in range(20)]
        results = await asyncio.gather(*[_create(k, f"p-{i}") for i, k in enumerate(keys)])
        assert len({r.project_id for r in results}) == 20
        for r in results:
            rows = await _sql(
                "SELECT COUNT(*) AS n FROM jax_project_membership WHERE project_id=%s AND status='ACTIVE'",
                (r.project_id,), fetch=True)
            assert rows[0]["n"] == 3  # creator + the two tenant admins

        same_key = str(uuid.uuid4())
        r1, r2 = await asyncio.gather(_create(same_key, "same"), _create(same_key, "same"))
        assert r1.project_id == r2.project_id
        assert {r1.created, r2.created} == {True, False}

        conflict_key = str(uuid.uuid4())
        outcomes = await asyncio.gather(
            _create(conflict_key, "left"), _create(conflict_key, "right"), return_exceptions=True)
        successes = [o for o in outcomes if not isinstance(o, Exception)]
        failures = [o for o in outcomes if isinstance(o, Exception)]
        assert len(successes) == 1 and len(failures) == 1
        assert isinstance(failures[0], IdempotencyKeyConflict)
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 9. Race: admin ascent vs. project creation, 50 iterations -- the §6.5
#    invariant (every active admin OWNs every project of its tenant) holds
#    at the end of each one.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_carrera_ascenso_de_admin_contra_creacion_de_proyecto():
    pool = await _pool(maxsize=8)
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        promotable = await _crear_usuario(tenant_id, role="operator")
        promoter = await _crear_usuario(tenant_id, role="admin")
        creator = await _crear_usuario(tenant_id, role="operator")
        promoter_scope = _scope(promoter, tenant_id, None)
        creator_scope = _scope(creator, tenant_id, None)
        for i in range(50):
            await _sql("UPDATE jax_users SET role='operator' WHERE user_id=%s", (promotable,))
            await asyncio.gather(
                _promote_to_admin(pool, promoter_scope, tenant_id, promotable),
                admin_api.create_project(
                    _request(creator_scope, "CREATE_PROJECT"), name=f"race-{i}", description=None,
                    idempotency_key=str(uuid.uuid4())),
            )
            offenders = await _admins_sin_ownership(tenant_id)
            assert not offenders, f"iteration {i}: {offenders}"
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 10. Last owner / write skew: two non-admin OWNERs revoke each other at
#     once -- exactly one wins, the invariant (at least one active OWNER
#     survives) always holds, and the loser gets a `ProjectAuthorityError`.
#
#     DEVIATION FROM THE PLAN TEXT, verified empirically against this
#     implementation and documented here rather than silently "fixed": the
#     plan says the loser gets `LastOwnerRequired`. Because `jax_tenants`
#     is a per-tenant mutex (decision 1) these two `revoke_member` calls
#     fully serialize -- the SECOND one to acquire the tenant lock finds
#     that ITS OWN actor was just revoked by the first (each actor is the
#     other's target), so it fails on the actor-membership check inside
#     `_resolve_project_actor_cur` with `ProjectNotVisible`, before ever
#     reaching the last-owner count. `LastOwnerRequired` fires when a
#     surviving OWNER is asked to demote/revoke a DIFFERENT, still-active
#     OWNER down to zero, which is not what mutual revocation produces once
#     the tenant mutex removes the classic write-skew race entirely. Both
#     outcomes are `ProjectAuthorityError` and neither leaves zero OWNERs,
#     which is the actual property decision 1 exists to guarantee.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_ultimo_dueno_dos_owners_se_revocan_entre_si_a_la_vez():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        owner_a = await _crear_usuario(tenant_id, role="operator")
        owner_b = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, owner_a, role="OWNER")
        await _crear_membresia(project_id, tenant_id, owner_b, role="OWNER")

        scope_a = _scope(owner_a, tenant_id, project_id)
        scope_b = _scope(owner_b, tenant_id, project_id)
        outcomes = await asyncio.gather(
            admin_api.revoke_member(_request(scope_a, "REVOKE_MEMBER"), project_id, owner_b),
            admin_api.revoke_member(_request(scope_b, "REVOKE_MEMBER"), project_id, owner_a),
            return_exceptions=True)
        successes = [o for o in outcomes if not isinstance(o, Exception)]
        failures = [o for o in outcomes if isinstance(o, Exception)]
        assert len(successes) == 1, outcomes
        assert len(failures) == 1 and isinstance(failures[0], ProjectAuthorityError), outcomes
        rows = await _sql(
            "SELECT status FROM jax_project_membership WHERE project_id=%s AND status='ACTIVE' AND project_role='OWNER'",
            (project_id,), fetch=True)
        assert len(rows) == 1
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 11. REVOKED: change_project_role -> MemberNotFound; grant_member reactivates.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_revoked_change_role_da_member_not_found_y_grant_reactiva():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        owner_user = await _crear_usuario(tenant_id, role="operator")
        target_email = f"revoked-{uuid.uuid4().hex[:8]}@test.invalid"
        target_user = await _crear_usuario(tenant_id, role="operator", email=target_email)
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, owner_user, role="OWNER", origin="CREATOR")
        await _crear_membresia(project_id, tenant_id, target_user, role="VIEWER", status="REVOKED")

        owner_scope = _scope(owner_user, tenant_id, project_id)
        with pytest.raises(MemberNotFound):
            await admin_api.change_project_role(
                _request(owner_scope, "CHANGE_PROJECT_ROLE"), project_id, target_user, ProjectRole.CONTRIBUTOR)

        reactivated_id = await admin_api.grant_member(
            _request(owner_scope, "GRANT_MEMBER"), project_id, email=target_email, role=ProjectRole.CONTRIBUTOR)
        assert reactivated_id == target_user
        rows = await _sql(
            "SELECT status,project_role FROM jax_project_membership WHERE project_id=%s AND user_id=%s",
            (project_id, target_user), fetch=True)
        assert rows[0]["status"] == "ACTIVE" and rows[0]["project_role"] == "CONTRIBUTOR"
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 12. D5: the projects.status mirror never drifts, through every transition.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_d5_espejo_se_mantiene_sincronizado_tras_cada_transicion():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        admin_user = await _crear_usuario(tenant_id, role="admin")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, admin_user, role="OWNER", origin="TENANT_ADMIN")

        scope = _scope(admin_user, tenant_id, project_id)
        for target in (ProjectLifecycle.ARCHIVED, ProjectLifecycle.HIDDEN,
                      ProjectLifecycle.DISABLED, ProjectLifecycle.ACTIVE):
            changed = await admin_api.set_project_lifecycle(
                _request(scope, "SET_PROJECT_LIFECYCLE"), project_id, target)
            assert changed is True
            assert not await _espejo_desincronizado(tenant_id)
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 13. Lock order: the first statement of every project-authority mutation's
#     transaction is `jax_tenants ... FOR UPDATE`.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_orden_de_bloqueo_primera_sentencia_de_cada_transaccion_es_tenant_for_update():
    pool = await _pool()
    try:
        tenant_id = await _crear_tenant()
        creator = await _crear_usuario(tenant_id, role="operator")
        admin_user = await _crear_usuario(tenant_id, role="admin")

        store = _RecordingStore(pool)
        admin_api = ProjectAuthorityAdmin(store, MariaDBScopeAuthorityResolver(pool))
        create_scope = _scope(creator, tenant_id, None)
        created = await admin_api.create_project(
            _request(create_scope, "CREATE_PROJECT"), name="orden", description=None,
            idempotency_key=str(uuid.uuid4()))
        # mutations[0] is the lock-free idempotency pre-read (decision 6);
        # mutations[1] is the actual write transaction.
        assert len(store.mutations) >= 2
        assert "jax_tenants" in store.mutations[1][0] and "FOR UPDATE" in store.mutations[1][0]

        legacy_project_id = await _crear_proyecto_activo(tenant_id, name="legacy-orden")
        store2 = _RecordingStore(pool)
        admin_api2 = ProjectAuthorityAdmin(store2, MariaDBScopeAuthorityResolver(pool))
        bootstrap_scope = _scope(admin_user, tenant_id, legacy_project_id)
        await admin_api2.bootstrap_existing_project(
            _request(bootstrap_scope, "BOOTSTRAP_PROJECT"), legacy_project_id, owner_user_id=creator)
        assert store2.mutations and "jax_tenants" in store2.mutations[0][0] and "FOR UPDATE" in store2.mutations[0][0]

        store3 = _RecordingStore(pool)
        admin_api3 = ProjectAuthorityAdmin(store3, MariaDBScopeAuthorityResolver(pool))
        owner_scope = _scope(creator, tenant_id, created.project_id)
        await admin_api3.set_project_lifecycle(
            _request(owner_scope, "SET_PROJECT_LIFECYCLE"), created.project_id, ProjectLifecycle.ARCHIVED)
        assert store3.mutations and "jax_tenants" in store3.mutations[0][0] and "FOR UPDATE" in store3.mutations[0][0]
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# 14. Descent branch of `sync_tenant_admin_memberships_in_transaction`
#     (Fernando aprobó §2.17 tal cual; ronda 2 de PR-J1, 2026-09-26).
#
# Verificado contra MariaDB real con la credencial de pruebas
# `~/.config/jax/test-db.env` (usuario `jax_test`, ALL solo sobre
# `jax_memory_test`/`jax_memory_test_%`). Los dos primeros (a, b) cayeron
# en rojo en su primera corrida: el escenario dejaba a `x` como ÚNICO OWNER
# del proyecto tras el ascenso, así que el descenso chocaba con el
# invariante de último dueño (`LastOwnerRequired`) en vez de restaurar/
# revocar como el test esperaba -- corregido agregando un segundo OWNER
# explícito a cada proyecto antes de ascender a `x`.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_descenso_sin_membresia_previa_deja_las_filas_tenant_admin_revoked():
    pool = await _pool()
    try:
        tenant_id = await _crear_tenant()
        promoter = await _crear_usuario(tenant_id, role="admin")
        x = await _crear_usuario(tenant_id, role="operator")
        # Otro OWNER explicito en cada proyecto -- si x fuera el UNICO OWNER,
        # el descenso chocaria con el invariante de ultimo dueno (test #17,
        # 2026-09-26: bug descubierto en rojo contra esta misma corrida:
        # LastOwnerRequired en vez del REVOKED esperado, porque x quedaba
        # sola como OWNER tras el ascenso).
        otro_owner = await _crear_usuario(tenant_id, role="operator")
        p1 = await _crear_proyecto_activo(tenant_id, name="p1")
        p2 = await _crear_proyecto_activo(tenant_id, name="p2")
        await _crear_scope(p1, tenant_id)
        await _crear_scope(p2, tenant_id)
        await _crear_membresia(p1, tenant_id, otro_owner, role="OWNER", origin="EXPLICIT")
        await _crear_membresia(p2, tenant_id, otro_owner, role="OWNER", origin="EXPLICIT")
        promoter_scope = _scope(promoter, tenant_id, None)

        await _promote_to_admin(pool, promoter_scope, tenant_id, x)
        for pid in (p1, p2):
            rows = await _sql(
                "SELECT project_role,status,grant_origin FROM jax_project_membership "
                "WHERE project_id=%s AND user_id=%s", (pid, x), fetch=True)
            assert rows[0]["project_role"] == "OWNER" and rows[0]["status"] == "ACTIVE"
            assert rows[0]["grant_origin"] == "TENANT_ADMIN"

        await _demote_from_admin(pool, promoter_scope, tenant_id, x)
        for pid in (p1, p2):
            rows = await _sql(
                "SELECT project_role,status,grant_origin,pre_admin_role,pre_admin_status FROM "
                "jax_project_membership WHERE project_id=%s AND user_id=%s", (pid, x), fetch=True)
            assert rows[0]["status"] == "REVOKED"
            assert rows[0]["grant_origin"] == "TENANT_ADMIN"
            assert rows[0]["pre_admin_role"] is None and rows[0]["pre_admin_status"] is None
            # 1 GRANT_TENANT_ADMIN (ascenso) + 1 REVOKE_TENANT_ADMIN (descenso).
            assert await _contar_eventos_de_sincronizacion(pid, x) == 2
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_descenso_con_membresia_explicita_previa_restaura_pre_admin():
    pool = await _pool()
    try:
        tenant_id = await _crear_tenant()
        promoter = await _crear_usuario(tenant_id, role="admin")
        x = await _crear_usuario(tenant_id, role="operator")
        # Otro OWNER explicito: x pasa a OWNER via TENANT_ADMIN durante el
        # ascenso, y sin otro dueno el descenso chocaria con el invariante de
        # ultimo dueno en vez de restaurar VIEWER (mismo bug que el test
        # anterior, corregido el 2026-09-26 tras verlo en rojo).
        otro_owner = await _crear_usuario(tenant_id, role="operator")
        p1 = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(p1, tenant_id)
        await _crear_membresia(p1, tenant_id, x, role="VIEWER", status="ACTIVE", origin="EXPLICIT")
        await _crear_membresia(p1, tenant_id, otro_owner, role="OWNER", origin="EXPLICIT")
        promoter_scope = _scope(promoter, tenant_id, None)

        await _promote_to_admin(pool, promoter_scope, tenant_id, x)
        rows = await _sql(
            "SELECT project_role,status,grant_origin,pre_admin_role,pre_admin_status FROM "
            "jax_project_membership WHERE project_id=%s AND user_id=%s", (p1, x), fetch=True)
        assert rows[0]["project_role"] == "OWNER" and rows[0]["grant_origin"] == "TENANT_ADMIN"
        assert rows[0]["pre_admin_role"] == "VIEWER" and rows[0]["pre_admin_status"] == "ACTIVE"

        await _demote_from_admin(pool, promoter_scope, tenant_id, x)
        rows = await _sql(
            "SELECT project_role,status,grant_origin,pre_admin_role,pre_admin_status FROM "
            "jax_project_membership WHERE project_id=%s AND user_id=%s", (p1, x), fetch=True)
        assert rows[0]["project_role"] == "VIEWER" and rows[0]["status"] == "ACTIVE"
        assert rows[0]["grant_origin"] == "EXPLICIT"
        assert rows[0]["pre_admin_role"] is None and rows[0]["pre_admin_status"] is None
        assert await _contar_eventos_de_sincronizacion(p1, x) == 2
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_descenso_no_toca_la_fila_creator():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        promoter = await _crear_usuario(tenant_id, role="admin")
        x = await _crear_usuario(tenant_id, role="operator")
        creator_scope = _scope(x, tenant_id, None)
        created = await admin_api.create_project(
            _request(creator_scope, "CREATE_PROJECT"), name="creado-por-x", description=None,
            idempotency_key=str(uuid.uuid4()))
        before = await _sql(
            "SELECT membership_id,project_role,status,grant_origin,version FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (created.project_id, x), fetch=True)
        assert before[0]["grant_origin"] == "CREATOR"

        promoter_scope = _scope(promoter, tenant_id, None)
        await _promote_to_admin(pool, promoter_scope, tenant_id, x)
        await _demote_from_admin(pool, promoter_scope, tenant_id, x)

        after = await _sql(
            "SELECT membership_id,project_role,status,grant_origin,version FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (created.project_id, x), fetch=True)
        assert after[0] == before[0]  # ni una columna tocada, ni siquiera version
        assert await _contar_eventos_de_sincronizacion(created.project_id, x) == 0
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_descenso_bloqueado_por_ultimo_dueno_no_deja_cambios_ni_eventos():
    pool = await _pool()
    try:
        tenant_id = await _crear_tenant()
        promoter = await _crear_usuario(tenant_id, role="admin")
        x = await _crear_usuario(tenant_id, role="operator")
        otro_owner = await _crear_usuario(tenant_id, role="operator")
        p1 = await _crear_proyecto_activo(tenant_id, name="p1")
        p2 = await _crear_proyecto_activo(tenant_id, name="p2")
        await _crear_scope(p1, tenant_id)
        await _crear_scope(p2, tenant_id)
        # p1 ya tiene otro OWNER activo aparte de x -- por si sola, sería
        # segura de bajar. p2 NO: x va a quedar como su unico OWNER.
        await _crear_membresia(p1, tenant_id, otro_owner, role="OWNER", origin="EXPLICIT")
        promoter_scope = _scope(promoter, tenant_id, None)

        await _promote_to_admin(pool, promoter_scope, tenant_id, x)
        owners_p2 = await _sql(
            "SELECT COUNT(*) AS n FROM jax_project_membership WHERE project_id=%s AND status='ACTIVE' "
            "AND project_role='OWNER'", (p2,), fetch=True)
        assert owners_p2[0]["n"] == 1  # x es el unico OWNER de p2

        before_p1 = await _sql(
            "SELECT version,status,project_role,grant_origin FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (p1, x), fetch=True)
        before_p2 = await _sql(
            "SELECT version,status,project_role,grant_origin FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (p2, x), fetch=True)
        eventos_antes = (await _contar_eventos_de_sincronizacion(p1, x)
                        + await _contar_eventos_de_sincronizacion(p2, x))

        with pytest.raises(LastOwnerRequired) as excinfo:
            await _demote_from_admin(pool, promoter_scope, tenant_id, x)
        assert excinfo.value.code == "proyecto_sin_duenio"

        after_p1 = await _sql(
            "SELECT version,status,project_role,grant_origin FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (p1, x), fetch=True)
        after_p2 = await _sql(
            "SELECT version,status,project_role,grant_origin FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (p2, x), fetch=True)
        # Rollback total: ni p1 (que hubiera sido seguro solo) se toca --
        # el chequeo corre sobre TODAS las filas antes de mutar cualquiera.
        assert after_p1 == before_p1
        assert after_p2 == before_p2
        eventos_despues = (await _contar_eventos_de_sincronizacion(p1, x)
                           + await _contar_eventos_de_sincronizacion(p2, x))
        assert eventos_despues == eventos_antes  # 0 eventos nuevos

        # El rol en jax_users tambien vuelve atras: _demote_from_admin lo
        # cambia a 'operator' ANTES de llamar al sync, pero toda esa
        # transaccion se revierte cuando el sync levanta la excepcion.
        rows_role = await _sql("SELECT role FROM jax_users WHERE user_id=%s", (x,), fetch=True)
        assert rows_role[0]["role"] == "admin"
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_descenso_actor_de_otro_tenant_da_papel_insuficiente():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant1 = await _crear_tenant("t1")
        tenant2 = await _crear_tenant("t2")
        actor_de_otro_tenant = await _crear_usuario(tenant2, role="admin")
        x = await _crear_usuario(tenant1, role="operator")
        actor_scope = _scope(actor_de_otro_tenant, tenant2, None)

        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    with pytest.raises(ProjectRoleInsufficient):
                        await admin_api.sync_tenant_admin_memberships_in_transaction(
                            cur, actor_scope=actor_scope, user_id=x, tenant_id=tenant1,
                            was_active_admin=False, is_active_admin=True)
            finally:
                await conn.rollback()
    finally:
        pool.close(); await pool.wait_closed()
