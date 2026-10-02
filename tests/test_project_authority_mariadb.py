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
import re
import uuid
from pathlib import Path

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.b9 import AuthorizationDenied, MutationAuthorizationRequest, ScopeContext, ScopeDenied, Visibility
from jax.memory.b9_mariadb import MariaDBB9Store
from jax.memory.project_authority import (
    AlreadyMember, IdempotencyKeyConflict, InvalidIdempotencyKey, LastOwnerRequired,
    LEGACY_PROJECT_TENANT_ID, MemberNotFound, ProjectAuthorityAdmin, ProjectAuthorityError,
    ProjectNotVisible, ProjectRoleInsufficient, ProjectStateConflict, ReservedProjectIdRange,
    TargetUserNotEligible, TenantAdminMembershipProtected,
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
    conn = await aiomysql.connect(
        db=db_name, autocommit=True, connect_timeout=db_connect_timeout_seconds(),
        **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute("SHOW TABLES LIKE 'projects'")
            if not await cur.fetchone():
                await cur.execute(_projects_ddl())
            await apply_project_authority_migration(cur)
    finally:
        conn.close()


async def _ensure_legacy_tenant(db_name: str) -> None:
    """Ronda 3, MINOR 5: `LEGACY_PROJECT_TENANT_ID` (1) must exist before any
    bootstrap test runs. Claimed EARLY (this fixture runs before any test),
    with an explicit id, so no other test's `_crear_tenant()` (AUTO_INCREMENT)
    can grab id 1 first in a freshly-created session database."""
    conn = await aiomysql.connect(
        db=db_name, autocommit=True, connect_timeout=db_connect_timeout_seconds(),
        **_conn_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT IGNORE INTO jax_tenants (tenant_id,name,plan,status) VALUES (%s,'legacy','personal','active')",
                (LEGACY_PROJECT_TENANT_ID,))
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def _esquema_de_proyectos():
    if not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB):
        return
    asyncio.run(_ensure_schema(_DB))
    asyncio.run(_ensure_legacy_tenant(_DB))


async def _sql(query: str, args: tuple = (), *, db: str | None = None, fetch: bool = False):
    conn = await aiomysql.connect(
        db=db or _DB, autocommit=True, cursorclass=aiomysql.DictCursor,
        connect_timeout=db_connect_timeout_seconds(), **_conn_params())
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
        db=_DB, autocommit=True, minsize=1, maxsize=maxsize,
        cursorclass=aiomysql.DictCursor, connect_timeout=db_connect_timeout_seconds(),
        **_conn_params())


def _scope(user_id: int, tenant_id: int, project_id: int | None = None) -> ScopeContext:
    return ScopeContext(f"user:{user_id}", "USER", str(user_id), str(tenant_id),
                        str(project_id) if project_id is not None else None)


def _request(scope: ScopeContext, operation: str) -> MutationAuthorizationRequest:
    return MutationAuthorizationRequest(scope, operation, Visibility.PROJECT_SHARED)


def _admin(pool) -> ProjectAuthorityAdmin:
    return ProjectAuthorityAdmin(MariaDBB9Store(pool))


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
    """Wraps a real aiomysql cursor and records every SQL statement AND its
    bound args, in order, so test 13 can tell apart two statements that
    share the same SQL text (an actor's own `jax_users` row lock and a
    destino's) by which id they were actually called with -- without a
    resolver/store double (H4's lesson)."""
    def __init__(self, real_cur, calls: list[tuple[str, tuple]]):
        self._real, self._calls = real_cur, calls

    async def execute(self, sql, args=None):
        self._calls.append((sql, args if args is not None else ()))
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
        self.mutations: list[list[tuple[str, tuple]]] = []

    async def mutation(self, operation):
        calls: list[tuple[str, tuple]] = []
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


def _first_index(calls: list[tuple[str, tuple]], *, sql_contains: str, arg=None) -> int:
    """Index of the first recorded statement whose SQL contains
    `sql_contains` and (if `arg` is given) whose args contain it. Raises
    `AssertionError` with the full call list if nothing matches -- a test
    failure here should show what actually ran, not just "not found"."""
    for i, (sql, args) in enumerate(calls):
        if sql_contains in sql and (arg is None or str(arg) in [str(a) for a in args]):
            return i
    raise AssertionError(f"no statement matches sql_contains={sql_contains!r} arg={arg!r} in {calls}")


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
                await cur.execute("UPDATE jax_users SET role='admin' WHERE user_id=%s", (user_id,))
                await admin_api.sync_tenant_admin_memberships_in_transaction(
                    cur, actor_scope=actor_scope, user_id=user_id, tenant_id=tenant_id)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def _demote_from_admin(pool, actor_scope: ScopeContext, tenant_id: int, user_id: int) -> None:
    """The descent mirror of `_promote_to_admin`: same lock order (tenant ->
    admin set -> target row), flips the DB role back, then calls
    `sync_tenant_admin_memberships_in_transaction` inside the SAME
    transaction -- the function derives is_active_admin from the row this
    UPDATE just wrote (ronda 3, MAJOR M2), never from a caller-supplied flag.
    A `LastOwnerRequired` raised by the sync call rolls back EVERYTHING in
    this transaction -- the role flip included -- because the `except` here
    does `conn.rollback()` before re-raising."""
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
                await cur.execute("UPDATE jax_users SET role='operator' WHERE user_id=%s", (user_id,))
                await admin_api.sync_tenant_admin_memberships_in_transaction(
                    cur, actor_scope=actor_scope, user_id=user_id, tenant_id=tenant_id)
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
        drop_conn = await aiomysql.connect(
            autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
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
        # MINOR 5: bootstrap_existing_project solo opera sobre
        # LEGACY_PROJECT_TENANT_ID (los proyectos anteriores a B9 pertenecen
        # al tenant que existia antes) -- no un tenant nuevo cualquiera.
        tenant_id = LEGACY_PROJECT_TENANT_ID
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
    setup_conn = await aiomysql.connect(
        autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
    try:
        async with setup_conn.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{db_name}`")
    finally:
        setup_conn.close()
    try:
        conn = await aiomysql.connect(
            db=db_name, autocommit=True, cursorclass=aiomysql.DictCursor,
            connect_timeout=db_connect_timeout_seconds(), **_conn_params())
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
                db=db_name, autocommit=True, minsize=1, maxsize=2,
                cursorclass=aiomysql.DictCursor, connect_timeout=db_connect_timeout_seconds(),
                **_conn_params())
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
        drop_conn = await aiomysql.connect(
            autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_conn_params())
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
async def test_ultimo_dueno_revoke_y_demote_sobre_uno_mismo_exige_last_owner_required():
    """MAJOR M3: test 10 tiene que exigir `LastOwnerRequired`, tipo exacto.
    Escenario donde la cuenta SI decide: el unico OWNER activo se revoca o
    se degrada a si mismo, y `_count_other_active_owners` da 0 de verdad --
    sin la complicacion de que el chequeo de membresia del actor tape el
    resultado, como pasa en la version "dos OWNERs se revocan mutuamente"
    de abajo (que documenta un hallazgo real y distinto: `ProjectNotVisible`,
    no `LastOwnerRequired`, por el mutex de tenant)."""
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        sole_owner = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, sole_owner, role="OWNER")
        scope = _scope(sole_owner, tenant_id, project_id)

        with pytest.raises(LastOwnerRequired) as excinfo:
            await admin_api.revoke_member(_request(scope, "REVOKE_MEMBER"), project_id, sole_owner)
        assert excinfo.value.code == "ultimo_duenio"

        with pytest.raises(LastOwnerRequired) as excinfo2:
            await admin_api.change_project_role(
                _request(scope, "CHANGE_PROJECT_ROLE"), project_id, sole_owner, ProjectRole.VIEWER)
        assert excinfo2.value.code == "ultimo_duenio"

        rows = await _sql(
            "SELECT status,project_role FROM jax_project_membership WHERE project_id=%s AND user_id=%s",
            (project_id, sole_owner), fetch=True)
        assert rows[0]["status"] == "ACTIVE" and rows[0]["project_role"] == "OWNER"
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_ultimo_dueno_write_skew_dos_owners_se_revocan_entre_si_a_la_vez():
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
        admin_api = ProjectAuthorityAdmin(store)
        create_scope = _scope(creator, tenant_id, None)
        created = await admin_api.create_project(
            _request(create_scope, "CREATE_PROJECT"), name="orden", description=None,
            idempotency_key=str(uuid.uuid4()))
        # mutations[0] is the lock-free idempotency pre-read (decision 6);
        # mutations[1] is the actual write transaction.
        assert len(store.mutations) >= 2
        write_tx = store.mutations[1]
        assert "jax_tenants" in write_tx[0][0] and "FOR UPDATE" in write_tx[0][0]

        store3 = _RecordingStore(pool)
        admin_api3 = ProjectAuthorityAdmin(store3)
        owner_scope = _scope(creator, tenant_id, created.project_id)
        await admin_api3.set_project_lifecycle(
            _request(owner_scope, "SET_PROJECT_LIFECYCLE"), created.project_id, ProjectLifecycle.ARCHIVED)
        assert store3.mutations and "jax_tenants" in store3.mutations[0][0][0] and "FOR UPDATE" in store3.mutations[0][0][0]

        # MAJOR M1: bootstrap locks the DESTINO (owner_user_id) before
        # `projects`/`jax_project_scope` -- the full order, not just the
        # first statement.
        legacy_project_id = await _crear_proyecto_activo(LEGACY_PROJECT_TENANT_ID, name="legacy-orden")
        legacy_admin = await _crear_usuario(LEGACY_PROJECT_TENANT_ID, role="admin")
        legacy_owner = await _crear_usuario(LEGACY_PROJECT_TENANT_ID, role="operator")
        store2 = _RecordingStore(pool)
        admin_api2 = ProjectAuthorityAdmin(store2)
        bootstrap_scope = _scope(legacy_admin, LEGACY_PROJECT_TENANT_ID, legacy_project_id)
        await admin_api2.bootstrap_existing_project(
            _request(bootstrap_scope, "BOOTSTRAP_PROJECT"), legacy_project_id, owner_user_id=legacy_owner)
        bootstrap_calls = store2.mutations[0]
        assert "jax_tenants" in bootstrap_calls[0][0] and "FOR UPDATE" in bootstrap_calls[0][0]
        i_destino = _first_index(bootstrap_calls, sql_contains="FROM jax_users", arg=legacy_owner)
        i_projects = _first_index(bootstrap_calls, sql_contains="FROM projects")
        i_scope = _first_index(bootstrap_calls, sql_contains="FROM jax_project_scope")
        assert i_destino < i_projects < i_scope, bootstrap_calls

        # grant_member/change_project_role/revoke_member: the destino
        # (jax_users row, via `_resolve_project_actor_cur`'s `lock_target`)
        # is locked BEFORE the project scope row -- this is the exact
        # reordering that closes the 1213 deadlock against the chat
        # (test_orden_de_bloqueo_evita_1213_contra_el_chat_real). A FRESH
        # ACTIVE project -- `created.project_id` is ARCHIVED by now.
        segundo_scope = _scope(creator, tenant_id, None)
        segundo = await admin_api.create_project(
            _request(segundo_scope, "CREATE_PROJECT"), name="orden-2", description=None,
            idempotency_key=str(uuid.uuid4()))
        owner_scope_2 = _scope(creator, tenant_id, segundo.project_id)
        target_email = f"orden-target-{uuid.uuid4().hex[:8]}@test.invalid"
        target_id = await _crear_usuario(tenant_id, role="operator", email=target_email)
        store4 = _RecordingStore(pool)
        admin_api4 = ProjectAuthorityAdmin(store4)
        await admin_api4.grant_member(
            _request(owner_scope_2, "GRANT_MEMBER"), segundo.project_id, email=target_email, role=ProjectRole.VIEWER)
        grant_calls = store4.mutations[0]
        i_user_target = _first_index(grant_calls, sql_contains="FROM jax_users WHERE email=")
        i_scope_grant = _first_index(grant_calls, sql_contains="FROM jax_project_scope")
        assert i_user_target < i_scope_grant, grant_calls

        store5 = _RecordingStore(pool)
        admin_api5 = ProjectAuthorityAdmin(store5)
        await admin_api5.change_project_role(
            _request(owner_scope_2, "CHANGE_PROJECT_ROLE"), segundo.project_id, target_id, ProjectRole.CONTRIBUTOR)
        change_calls = store5.mutations[0]
        i_user_target2 = _first_index(change_calls, sql_contains="FROM jax_users", arg=target_id)
        i_scope_change = _first_index(change_calls, sql_contains="FROM jax_project_scope")
        assert i_user_target2 < i_scope_change, change_calls

        store6 = _RecordingStore(pool)
        admin_api6 = ProjectAuthorityAdmin(store6)
        await admin_api6.revoke_member(_request(owner_scope_2, "REVOKE_MEMBER"), segundo.project_id, target_id)
        revoke_calls = store6.mutations[0]
        i_user_target3 = _first_index(revoke_calls, sql_contains="FROM jax_users", arg=target_id)
        i_scope_revoke = _first_index(revoke_calls, sql_contains="FROM jax_project_scope")
        assert i_user_target3 < i_scope_revoke, revoke_calls
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MAJOR M1, ronda 3 (2026-09-26): the SAME deadlock the auditor reproduced,
# against the REAL resolver on both sides (the chat's
# `resolve_mutation_in_transaction` and `grant_member`), with a barrier that
# forces the exact interleaving that used to trigger MariaDB error 1213.
#
# Verified in RED against the pre-fix code (standalone repro, same
# technique, same fixtures): `t1` (chat) died with
# `OperationalError(1213, 'Deadlock found...')`. After reordering
# `_resolve_project_actor_cur` to lock the destino before the scope row,
# the same scenario ends with `t1` succeeding and `t2` hitting the ordinary
# business error (`AlreadyMember`) -- never a deadlock.
# --------------------------------------------------------------------------

class _PausingCursor:
    """Wraps a real cursor; the first time `execute`'s SQL contains
    `trigger`, it signals `ready` and waits for `go` BEFORE running that
    statement -- so the caller can force two transactions to interleave at
    an exact point instead of hoping timing lines up."""
    def __init__(self, real_cur, trigger: str, ready: asyncio.Event, go: asyncio.Event):
        self._real, self._trigger, self._ready, self._go = real_cur, trigger, ready, go
        self._done = False

    async def execute(self, sql, args=None):
        if not self._done and self._trigger in sql:
            self._done = True
            self._ready.set()
            await self._go.wait()
        return await (self._real.execute(sql, args) if args is not None else self._real.execute(sql))

    async def fetchone(self):
        return await self._real.fetchone()

    async def fetchall(self):
        return await self._real.fetchall()


class _PausingStore:
    def __init__(self, pool, trigger: str, ready: asyncio.Event, go: asyncio.Event):
        self._pool, self._trigger, self._ready, self._go = pool, trigger, ready, go

    async def mutation(self, operation):
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as real_cur:
                    wrapped = _PausingCursor(real_cur, self._trigger, self._ready, self._go)
                    value = await operation(wrapped)
                await conn.commit()
                return value
            except Exception:
                await conn.rollback()
                raise


@requiere_db_de_prueba
@asincrono
async def test_orden_de_bloqueo_evita_1213_contra_el_chat_real():
    pool = await _pool(maxsize=4)
    try:
        tenant_id = await _crear_tenant()
        owner = await _crear_usuario(tenant_id, role="operator")
        target_email = f"target-{uuid.uuid4().hex[:8]}@test.invalid"
        target = await _crear_usuario(tenant_id, role="operator", email=target_email)
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, owner, role="OWNER")
        await _crear_membresia(project_id, tenant_id, target, role="CONTRIBUTOR")

        resolver = MariaDBScopeAuthorityResolver(pool)
        e_t1_ready, e_t2_ready = asyncio.Event(), asyncio.Event()
        e_go_t1, e_go_t2 = asyncio.Event(), asyncio.Event()
        outcome: dict[str, Any] = {}

        async def t1_chat_as_target():
            # Same two locks as `resolve_mutation_in_transaction`: jax_users
            # (target) via `_tenant_user_cur`, THEN jax_project_scope via
            # `_project_membership_cur` -- paused right before the second one.
            conn = await aiomysql.connect(
                db=_DB, cursorclass=aiomysql.DictCursor, autocommit=False,
                connect_timeout=db_connect_timeout_seconds(), **_conn_params())
            try:
                async with conn.cursor() as real_cur:
                    wrapped = _PausingCursor(
                        real_cur, "FROM jax_project_scope WHERE project_id=%s FOR UPDATE", e_t1_ready, e_go_t1)
                    scope = _scope(target, tenant_id, project_id)
                    try:
                        await resolver.resolve_mutation_in_transaction(wrapped, scope, "RETRIEVE", Visibility.PROJECT_SHARED)
                        outcome["t1"] = "ok"
                    except Exception as e:  # fail-soft: retained as outcome; the assertion below fails the deadlock test
                        outcome["t1"] = e
                await conn.commit()
            finally:
                conn.close()

        async def t2_grant():
            # Same two locks as `grant_member`: jax_users (owner, the actor)
            # then jax_project_scope, then -- with the fix -- jax_users
            # (target) BEFORE that scope lock. Paused right before the
            # target-row lock so it interleaves with t1's own scope-row wait.
            store = _PausingStore(pool, "FROM jax_users WHERE email=%s AND tenant_id=%s FOR UPDATE",
                                  e_t2_ready, e_go_t2)
            admin_api = ProjectAuthorityAdmin(store)
            owner_scope = _scope(owner, tenant_id, project_id)
            try:
                await admin_api.grant_member(
                    _request(owner_scope, "GRANT_MEMBER"), project_id, email=target_email, role=ProjectRole.VIEWER)
                outcome["t2"] = "ok"
            except Exception as e:  # fail-soft: retained as outcome; the assertion below fails the deadlock test
                outcome["t2"] = e

        t1_task = asyncio.create_task(t1_chat_as_target())
        t2_task = asyncio.create_task(t2_grant())
        await e_t1_ready.wait()
        await e_t2_ready.wait()
        e_go_t1.set()
        e_go_t2.set()
        await asyncio.gather(t1_task, t2_task)

        assert outcome.get("t1") == "ok", outcome
        # t2 hits the ordinary business rule (target already has an ACTIVE
        # membership) -- never a deadlock (pre-fix: one of the two died with
        # `pymysql.err.OperationalError(1213, ...)`, seen in a standalone
        # repro against the code before this reordering).
        assert isinstance(outcome.get("t2"), AlreadyMember), outcome
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MAJOR MJ-1, ronda 4 (2026-09-26): re-audit of b9d13c8 found that
# `_count_other_active_owners` still `JOIN ... FOR UPDATE`-locked the
# `jax_users` row of every OTHER active owner of the project (a third
# party -- never the actor, never the destino) whenever a demotion/
# revocation counted remaining owners. If that third owner was concurrently
# chatting about the SAME project, its transaction (`jax_users` first, then
# `jax_project_scope`) could deadlock with this one (`jax_project_scope`
# first -- via `_resolve_project_actor_cur` -- then the third owner's
# `jax_users` row). Verified in RED against b9d13c8 with two real
# connections and a barrier (same technique as `test_orden_de_bloqueo_...`
# above): `t2` (the chat, as the third owner) died with
# `OperationalError(1213, 'Deadlock found...')`.
#
# The fix (`_count_other_active_owners`) locks ONLY `jax_project_membership`
# rows for the project, then reads `jax_users.status` for the surviving
# owner ids with a PLAIN `SELECT` (no `FOR UPDATE`) -- safe under the
# tenant-row lock invariant documented in the module. Below: the SAME
# barrier reused for `revoke_member`, `change_project_role` (OWNER
# demotion), and the descent branch of
# `sync_tenant_admin_memberships_in_transaction` -- each paused right at
# that plain read (`FROM jax_users WHERE user_id IN (`), which never blocks
# a concurrent holder of that row's `FOR UPDATE` lock, so there is nothing
# left to deadlock on.
# --------------------------------------------------------------------------

async def _tercer_owner_deadlock_scenario(pool, tenant_id, project_id, owner_c):
    """Runs the chat, as `owner_c`, paused right before its `jax_project_scope`
    lock -- the SAME shape `test_orden_de_bloqueo_evita_1213_contra_el_chat_real`
    uses for the destino side. Returns the outcome dict key "t2" once resolved
    (populated by the caller's driver)."""
    resolver = MariaDBScopeAuthorityResolver(pool)

    async def t2_chat_as_owner_c(ready, go, outcome):
        conn = await aiomysql.connect(
            db=_DB, cursorclass=aiomysql.DictCursor, autocommit=False,
            connect_timeout=db_connect_timeout_seconds(), **_conn_params())
        try:
            async with conn.cursor() as real_cur:
                wrapped = _PausingCursor(
                    real_cur, "FROM jax_project_scope WHERE project_id=%s FOR UPDATE", ready, go)
                scope_c = _scope(owner_c, tenant_id, project_id)
                try:
                    await resolver.resolve_mutation_in_transaction(wrapped, scope_c, "RETRIEVE", Visibility.PROJECT_SHARED)
                    outcome["t2"] = "ok"
                except Exception as e:  # fail-soft: retained as outcome; the caller asserts this concurrent path succeeds
                    outcome["t2"] = e
            await conn.commit()
        finally:
            conn.close()

    return t2_chat_as_owner_c


@requiere_db_de_prueba
@asincrono
async def test_interbloqueo_revoke_member_no_bloquea_jax_users_de_terceros():
    pool = await _pool(maxsize=4)
    try:
        tenant_id = await _crear_tenant()
        owner_a = await _crear_usuario(tenant_id, role="operator")
        owner_b = await _crear_usuario(tenant_id, role="operator")
        owner_c = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        for uid in (owner_a, owner_b, owner_c):
            await _crear_membresia(project_id, tenant_id, uid, role="OWNER")

        e_t1_ready, e_t2_ready = asyncio.Event(), asyncio.Event()
        e_go_t1, e_go_t2 = asyncio.Event(), asyncio.Event()
        outcome: dict[str, Any] = {}
        t2_chat_as_owner_c = await _tercer_owner_deadlock_scenario(pool, tenant_id, project_id, owner_c)

        async def t1_revoke():
            store = _PausingStore(pool, "FROM jax_users WHERE user_id IN (", e_t1_ready, e_go_t1)
            api = ProjectAuthorityAdmin(store)
            scope_a = _scope(owner_a, tenant_id, project_id)
            try:
                await api.revoke_member(_request(scope_a, "REVOKE_MEMBER"), project_id, owner_b)
                outcome["t1"] = "ok"
            except Exception as e:  # fail-soft: retained as outcome; the assertion below fails the deadlock test
                outcome["t1"] = e

        t1_task = asyncio.create_task(t1_revoke())
        t2_task = asyncio.create_task(t2_chat_as_owner_c(e_t2_ready, e_go_t2, outcome))
        await asyncio.wait_for(e_t1_ready.wait(), timeout=10)
        await asyncio.wait_for(e_t2_ready.wait(), timeout=10)
        e_go_t1.set()
        e_go_t2.set()
        await asyncio.wait_for(asyncio.gather(t1_task, t2_task), timeout=10)

        assert outcome.get("t1") == "ok", outcome
        assert outcome.get("t2") == "ok", outcome
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_interbloqueo_change_project_role_descenso_de_owner_no_bloquea_jax_users_de_terceros():
    pool = await _pool(maxsize=4)
    try:
        tenant_id = await _crear_tenant()
        owner_a = await _crear_usuario(tenant_id, role="operator")
        owner_b = await _crear_usuario(tenant_id, role="operator")
        owner_c = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        for uid in (owner_a, owner_b, owner_c):
            await _crear_membresia(project_id, tenant_id, uid, role="OWNER")

        e_t1_ready, e_t2_ready = asyncio.Event(), asyncio.Event()
        e_go_t1, e_go_t2 = asyncio.Event(), asyncio.Event()
        outcome: dict[str, Any] = {}
        t2_chat_as_owner_c = await _tercer_owner_deadlock_scenario(pool, tenant_id, project_id, owner_c)

        async def t1_demote():
            store = _PausingStore(pool, "FROM jax_users WHERE user_id IN (", e_t1_ready, e_go_t1)
            api = ProjectAuthorityAdmin(store)
            scope_a = _scope(owner_a, tenant_id, project_id)
            try:
                await api.change_project_role(
                    _request(scope_a, "CHANGE_PROJECT_ROLE"), project_id, owner_b, ProjectRole.VIEWER)
                outcome["t1"] = "ok"
            except Exception as e:  # fail-soft: retained as outcome; the assertion below fails the deadlock test
                outcome["t1"] = e

        t1_task = asyncio.create_task(t1_demote())
        t2_task = asyncio.create_task(t2_chat_as_owner_c(e_t2_ready, e_go_t2, outcome))
        await asyncio.wait_for(e_t1_ready.wait(), timeout=10)
        await asyncio.wait_for(e_t2_ready.wait(), timeout=10)
        e_go_t1.set()
        e_go_t2.set()
        await asyncio.wait_for(asyncio.gather(t1_task, t2_task), timeout=10)

        assert outcome.get("t1") == "ok", outcome
        assert outcome.get("t2") == "ok", outcome
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_interbloqueo_descenso_de_sync_no_bloquea_jax_users_de_terceros():
    """Reproducción B del auditor: el descenso de
    `sync_tenant_admin_memberships_in_transaction` llama a
    `_count_other_active_owners` igual que `revoke_member`/
    `change_project_role` -- mismo defecto, mismo arreglo, mismo tipo de
    prueba. `x` desciende de admin en un proyecto donde `owner_c` es OTRO
    OWNER activo que está chateando sobre el mismo proyecto al mismo tiempo.
    """
    pool = await _pool(maxsize=4)
    try:
        tenant_id = await _crear_tenant()
        promoter = await _crear_usuario(tenant_id, role="admin")
        x = await _crear_usuario(tenant_id, role="operator")
        owner_c = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, owner_c, role="OWNER")
        promoter_scope = _scope(promoter, tenant_id, None)
        await _promote_to_admin(pool, promoter_scope, tenant_id, x)

        e_t1_ready, e_t2_ready = asyncio.Event(), asyncio.Event()
        e_go_t1, e_go_t2 = asyncio.Event(), asyncio.Event()
        outcome: dict[str, Any] = {}
        t2_chat_as_owner_c = await _tercer_owner_deadlock_scenario(pool, tenant_id, project_id, owner_c)

        async def t1_demote_sync():
            admin_api = ProjectAuthorityAdmin(MariaDBB9Store(pool))
            conn = await pool.acquire()
            try:
                await conn.begin()
                async with conn.cursor() as real_cur:
                    wrapped = _PausingCursor(real_cur, "FROM jax_users WHERE user_id IN (", e_t1_ready, e_go_t1)
                    try:
                        await wrapped.execute("SELECT tenant_id FROM jax_tenants WHERE tenant_id=%s FOR UPDATE",
                                             (tenant_id,))
                        await wrapped.fetchone()
                        await wrapped.execute(
                            "SELECT user_id FROM jax_users WHERE tenant_id=%s AND role IN "
                            "('admin','superadmin','super_admin') AND status='active' ORDER BY user_id",
                            (tenant_id,))
                        await wrapped.fetchall()
                        await wrapped.execute("UPDATE jax_users SET role='operator' WHERE user_id=%s", (x,))
                        await admin_api.sync_tenant_admin_memberships_in_transaction(
                            wrapped, actor_scope=promoter_scope, user_id=x, tenant_id=tenant_id)
                        outcome["t1"] = "ok"
                    except Exception as e:  # fail-soft: retained as outcome; the assertion below fails the deadlock test
                        outcome["t1"] = e
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            finally:
                pool.release(conn)

        t1_task = asyncio.create_task(t1_demote_sync())
        t2_task = asyncio.create_task(t2_chat_as_owner_c(e_t2_ready, e_go_t2, outcome))
        await asyncio.wait_for(e_t1_ready.wait(), timeout=10)
        await asyncio.wait_for(e_t2_ready.wait(), timeout=10)
        e_go_t1.set()
        e_go_t2.set()
        await asyncio.wait_for(asyncio.gather(t1_task, t2_task), timeout=10)

        assert outcome.get("t1") == "ok", outcome
        assert outcome.get("t2") == "ok", outcome
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MAJOR MJ-2, ronda 4: `_read_tenant_admins` reads the tenant's admin set
# WITHOUT `FOR UPDATE` (module invariant), via
# `idx_jax_users_tenant_role_status` (migration 005h).
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_explain_lectura_de_admins_usa_indice_por_tenant():
    """MJ-2 pide EXPLAIN con `type=ref` y sin recorrer `PRIMARY`. El plan
    NATURAL de MariaDB, en esta base, ya elige un índice simple por
    `tenant_id` (pre-existente, de la FK) antes que el nuevo compuesto --
    ambos dan `type=ref`, así que el plan natural ya cumple el pedido. Para
    confirmar que el índice NUEVO (`idx_jax_users_tenant_role_status`,
    migración 005h) es en sí mismo válido y usable -- no solo que existe --
    se fuerza con `FORCE INDEX` y se repite la verificación."""
    tenant_id = await _crear_tenant()
    await _crear_usuario(tenant_id, role="admin")
    natural = await _sql(
        "EXPLAIN SELECT user_id FROM jax_users WHERE tenant_id=%s AND role IN ('admin','superadmin','super_admin') "
        "AND status='active' ORDER BY user_id", (tenant_id,), fetch=True)
    assert natural, natural
    assert natural[0]["type"] != "ALL", natural[0]
    assert natural[0]["key"] is not None and natural[0]["key"] != "PRIMARY", natural[0]

    forced = await _sql(
        "EXPLAIN SELECT user_id FROM jax_users FORCE INDEX (idx_jax_users_tenant_role_status) "
        "WHERE tenant_id=%s AND role IN ('admin','superadmin','super_admin') AND status='active' "
        "ORDER BY user_id", (tenant_id,), fetch=True)
    assert forced, forced
    assert forced[0]["key"] == "idx_jax_users_tenant_role_status", forced[0]
    assert forced[0]["type"] in ("ref", "range"), forced[0]


@requiere_db_de_prueba
@asincrono
async def test_create_project_de_otro_tenant_no_bloquea_el_chat_de_este_tenant():
    """MJ-2: leer el conjunto de admins de OTRO tenant durante un
    `create_project` en vuelo no debe bloquear en absoluto al chat de un
    usuario de ESTE tenant -- son índices/tenants distintos, y con MJ-2 esa
    lectura ni siquiera toma `FOR UPDATE`."""
    pool = await _pool(maxsize=4)
    try:
        tenant1 = await _crear_tenant("t1")
        tenant2 = await _crear_tenant("t2")
        creator2 = await _crear_usuario(tenant2, role="operator")
        await _crear_usuario(tenant2, role="admin")
        user1 = await _crear_usuario(tenant1, role="operator")
        project1 = await _crear_proyecto_activo(tenant1)
        await _crear_scope(project1, tenant1)
        await _crear_membresia(project1, tenant1, user1, role="CONTRIBUTOR")

        resolver = MariaDBScopeAuthorityResolver(pool)
        e_t1_ready, e_go_t1 = asyncio.Event(), asyncio.Event()
        outcome: dict[str, Any] = {}

        async def t1_create_en_tenant2():
            store = _PausingStore(pool, "FROM jax_users WHERE tenant_id=%s AND role IN", e_t1_ready, e_go_t1)
            api = ProjectAuthorityAdmin(store)
            scope2 = _scope(creator2, tenant2, None)
            try:
                await api.create_project(
                    _request(scope2, "CREATE_PROJECT"), name="p", description=None,
                    idempotency_key=str(uuid.uuid4()))
                outcome["t1"] = "ok"
            except Exception as e:  # fail-soft: retained as outcome; the assertion below fails the isolation test
                outcome["t1"] = e

        t1_task = asyncio.create_task(t1_create_en_tenant2())
        await asyncio.wait_for(e_t1_ready.wait(), timeout=10)
        # t1 is paused MID transaction, holding tenant2's tenant-row lock and
        # about to read (not lock) tenant2's admin set. The chat, for a
        # DIFFERENT tenant's user, must complete with a SHORT timeout --
        # never wait on t1 at all.
        scope1 = _scope(user1, tenant1, project1)
        async with aiomysql.connect(
                db=_DB, cursorclass=aiomysql.DictCursor, autocommit=True,
                connect_timeout=db_connect_timeout_seconds(), **_conn_params()) as conn:
            async with conn.cursor() as cur:
                await asyncio.wait_for(
                    resolver.resolve_mutation_in_transaction(cur, scope1, "RETRIEVE", Visibility.PROJECT_SHARED),
                    timeout=2)
        e_go_t1.set()
        await asyncio.wait_for(t1_task, timeout=10)
        assert outcome.get("t1") == "ok", outcome
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# Module invariant, ronda 4: no project-authority operation locks a
# `jax_users` row that is not the actor's or the destino's.
# --------------------------------------------------------------------------

def test_invariante_ningun_tercero_se_bloquea_en_jax_users():
    """Escaneo estatico del modulo: ninguna sentencia SQL de
    `project_authority.py` combina `jax_users` con `FOR UPDATE` salvo las
    dos formas permitidas (bloquear la fila del ACTOR o la del DESTINO, por
    `user_id=%s` puntual) -- nunca un `JOIN`/`WHERE ... IN (...)` con
    `FOR UPDATE` que alcance filas de terceros."""
    import inspect

    from jax.memory import project_authority

    source = inspect.getsource(project_authority)
    # Cada SELECT ... FOR UPDATE sobre jax_users tiene que ser un lookup
    # puntual del actor o del destino -- por user_id=%s (el caso comun) o por
    # email=%s (grant_member invita por correo antes de conocer el user_id
    # del destino) -- nunca un JOIN ni un IN (...) que alcance terceros.
    matches = list(re.finditer(r"SELECT[^\"]*?FROM jax_users[^\"]*?FOR UPDATE", source))
    assert matches, "no se encontro ningun SELECT ... FROM jax_users ... FOR UPDATE: el escaneo no prueba nada"
    for match in matches:
        statement = match.group(0)
        assert "JOIN" not in statement, statement
        assert " IN (" not in statement, statement
        assert "user_id=%s" in statement or "email=%s" in statement, statement
# unambiguous USER actor. Verified in RED against the pre-fix code
# (standalone repro): a SERVICE scope created a project outright, and a
# forged `actor_principal` ("user:999999") with a real subject id granted
# membership AND signed the resulting event row with the forged identity.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_b1_actor_service_no_puede_crear_proyectos():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        real_user = await _crear_usuario(tenant_id, role="operator")
        service_scope = ScopeContext("service:memory-extraction", "SERVICE", str(real_user), str(tenant_id), None,
                                     calling_component="memory-extraction")
        with pytest.raises(AuthorizationDenied):
            await admin_api.create_project(
                _request(service_scope, "CREATE_PROJECT"), name="service-created", description=None,
                idempotency_key=str(uuid.uuid4()))
        rows = await _sql("SELECT COUNT(*) AS n FROM projects WHERE name='service-created'", fetch=True)
        assert rows[0]["n"] == 0
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_b1_principal_forjado_no_pasa_ni_firma_el_evento():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        real_user = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, real_user, role="OWNER", origin="EXPLICIT")
        invitee_email = f"invitee-{uuid.uuid4().hex[:8]}@test.invalid"
        await _crear_usuario(tenant_id, role="operator", email=invitee_email)

        # subject_user_id is the REAL owner (so the DB-backed checks would
        # all pass) but actor_principal claims a different identity.
        forged_scope = ScopeContext("user:999999", "USER", str(real_user), str(tenant_id), str(project_id))
        with pytest.raises(ScopeDenied):
            await admin_api.grant_member(
                _request(forged_scope, "GRANT_MEMBER"), project_id, email=invitee_email, role=ProjectRole.VIEWER)
        rows = await _sql(
            "SELECT COUNT(*) AS n FROM jax_project_membership_event WHERE project_id=%s AND actor_principal='user:999999'",
            (project_id,), fetch=True)
        assert rows[0]["n"] == 0
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_b1_delegacion_es_rechazada():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        real_user = await _crear_usuario(tenant_id, role="operator")
        delegated_scope = ScopeContext(f"user:{real_user}", "USER", str(real_user), str(tenant_id), None,
                                       delegation="user:other")
        with pytest.raises(ScopeDenied):
            await admin_api.create_project(
                _request(delegated_scope, "CREATE_PROJECT"), name="delegated", description=None,
                idempotency_key=str(uuid.uuid4()))
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_b1_bootstrap_y_sync_exigen_actor_usuario_canonico():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        legacy_project_id = await _crear_proyecto_activo(LEGACY_PROJECT_TENANT_ID, name="legacy-b1")
        real_admin = await _crear_usuario(LEGACY_PROJECT_TENANT_ID, role="admin")
        owner_user = await _crear_usuario(LEGACY_PROJECT_TENANT_ID, role="operator")

        service_scope = ScopeContext("service:memory-extraction", "SERVICE", str(real_admin),
                                     str(LEGACY_PROJECT_TENANT_ID), str(legacy_project_id),
                                     calling_component="memory-extraction")
        with pytest.raises(AuthorizationDenied):
            await admin_api.bootstrap_existing_project(
                _request(service_scope, "BOOTSTRAP_PROJECT"), legacy_project_id, owner_user_id=owner_user)

        promotable = await _crear_usuario(LEGACY_PROJECT_TENANT_ID, role="operator")
        forged_actor_scope = ScopeContext("user:999999", "USER", str(real_admin), str(LEGACY_PROJECT_TENANT_ID))
        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    with pytest.raises(ScopeDenied):
                        await admin_api.sync_tenant_admin_memberships_in_transaction(
                            cur, actor_scope=forged_actor_scope, user_id=promotable,
                            tenant_id=LEGACY_PROJECT_TENANT_ID)
            finally:
                await conn.rollback()
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
async def test_descenso_restaura_a_owner_sin_bloquear_por_ultimo_dueno():
    """MINOR 1: si `pre_admin_role/status` es OWNER/ACTIVE, restaurar deja al
    usuario siendo dueno -- no es una democion real, y el chequeo de ultimo
    dueno no debe dispararse aunque sea el UNICO OWNER visible en ese
    instante. Visto en rojo contra el codigo previo a esta ronda:
    `LastOwnerRequired` se levantaba igual (repro standalone, restaurado a
    mano contra el commit 6403d5f, no derivado)."""
    pool = await _pool()
    try:
        tenant_id = await _crear_tenant()
        promoter = await _crear_usuario(tenant_id, role="admin")
        x = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, x, role="OWNER", status="ACTIVE", origin="EXPLICIT")
        promoter_scope = _scope(promoter, tenant_id, None)

        await _promote_to_admin(pool, promoter_scope, tenant_id, x)
        rows = await _sql(
            "SELECT pre_admin_role,pre_admin_status FROM jax_project_membership WHERE project_id=%s AND user_id=%s",
            (project_id, x), fetch=True)
        assert rows[0]["pre_admin_role"] == "OWNER" and rows[0]["pre_admin_status"] == "ACTIVE"

        await _demote_from_admin(pool, promoter_scope, tenant_id, x)
        rows = await _sql(
            "SELECT status,project_role,grant_origin FROM jax_project_membership WHERE project_id=%s AND user_id=%s",
            (project_id, x), fetch=True)
        assert rows[0]["status"] == "ACTIVE" and rows[0]["project_role"] == "OWNER"
        assert rows[0]["grant_origin"] == "EXPLICIT"
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
async def test_sync_actor_de_otro_tenant_da_papel_insuficiente():
    """Ronda 3, MAJOR M2: nombre corregido (antes decia "descenso" pero no
    ejercitaba ninguna rama -- el actor fallaba antes de llegar a la
    diferencia ascenso/descenso). El actor tiene que ser un admin ACTIVO del
    MISMO tenant que se le pasa a `tenant_id`; si no, `ProjectRoleInsufficient`,
    sin importar si el llamado hubiera sido un ascenso o un descenso."""
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
                            cur, actor_scope=actor_scope, user_id=x, tenant_id=tenant1)
            finally:
                await conn.rollback()
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_sync_destino_de_otro_tenant_es_rechazado():
    """MAJOR M2(c): un admin ACTIVO legitimo del tenant correcto, pero un
    `user_id` destino que en realidad pertenece a OTRO tenant -> rechazado,
    nunca tratado como "nada que hacer"."""
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant1 = await _crear_tenant("t1")
        tenant2 = await _crear_tenant("t2")
        admin1 = await _crear_usuario(tenant1, role="admin")
        destino_de_otro_tenant = await _crear_usuario(tenant2, role="operator")
        actor_scope = _scope(admin1, tenant1, None)

        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    with pytest.raises(TargetUserNotEligible):
                        await admin_api.sync_tenant_admin_memberships_in_transaction(
                            cur, actor_scope=actor_scope, user_id=destino_de_otro_tenant, tenant_id=tenant1)
            finally:
                await conn.rollback()
        # 0 filas nuevas: el rechazo fue antes de tocar nada.
        rows = await _sql(
            "SELECT COUNT(*) AS n FROM jax_project_membership WHERE user_id=%s", (destino_de_otro_tenant,), fetch=True)
        assert rows[0]["n"] == 0
    finally:
        pool.close(); await pool.wait_closed()


@requiere_db_de_prueba
@asincrono
async def test_sync_no_promueve_a_quien_sigue_siendo_operator_en_la_base():
    """MAJOR M2(a)/(b): antes, un llamador podia mandar `is_active_admin=True`
    para un usuario cuyo `role` real en `jax_users` seguia siendo 'operator',
    y la funcion le otorgaba OWNER en todos los proyectos igual -- confiaba
    en el flag, no en la base. Ahora deriva `is_active_admin` de una lectura
    FOR UPDATE del rol/estado REAL; sin cambiar ese rol primero, la llamada
    es un no-op verificable (0 filas tocadas, 0 eventos)."""
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        real_admin = await _crear_usuario(tenant_id, role="admin")
        sigue_operator = await _crear_usuario(tenant_id, role="operator")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        actor_scope = _scope(real_admin, tenant_id, None)

        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    touched = await admin_api.sync_tenant_admin_memberships_in_transaction(
                        cur, actor_scope=actor_scope, user_id=sigue_operator, tenant_id=tenant_id)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        assert touched == 0
        rows = await _sql(
            "SELECT COUNT(*) AS n FROM jax_project_membership WHERE project_id=%s AND user_id=%s",
            (project_id, sigue_operator), fetch=True)
        assert rows[0]["n"] == 0
        assert await _contar_eventos_de_sincronizacion(project_id, sigue_operator) == 0
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MINOR 2, ronda 3: reactivating a REVOKED row (grant_member) and revoking
# an ACTIVE one (revoke_member) both leave pre_admin_role/pre_admin_status
# NULL -- defensive, regardless of what they were before -- and
# `_wrap_unexpected_db_error` maps any raw pymysql/aiomysql error to a
# `ProjectAuthorityError`, never leaking one past this module.
# --------------------------------------------------------------------------

def test_wrap_unexpected_db_error_mapea_errores_crudos_y_respeta_los_propios():
    import pymysql.err

    from jax.memory.project_authority import ProjectNotVisible, _wrap_unexpected_db_error

    driver_text = "Cannot add or update a child row -- SOME INTERNAL PATH /etc/x"
    raw = pymysql.err.IntegrityError(1452, driver_text)
    wrapped = _wrap_unexpected_db_error(raw)
    assert isinstance(wrapped, ProjectAuthorityError)
    assert wrapped.code == "error_de_base_de_datos"
    assert wrapped.__cause__ is raw
    # Ronda 4, MINOR 2: nunca el texto crudo del driver dentro del mensaje
    # publico -- solo queda en __cause__, para quien lea el traceback.
    assert driver_text not in str(wrapped)

    own = ProjectNotVisible("already ours")
    assert _wrap_unexpected_db_error(own) is own

    plain = ValueError("not a db error")
    assert _wrap_unexpected_db_error(plain) is plain


def test_wrap_unexpected_db_error_1213_y_1205_dan_retryable_sin_texto_del_driver():
    """Ronda 4, MINOR 2: 1213 (deadlock) y 1205 (lock-wait timeout) son
    contencion transitoria, nunca un problema de esquema/autoridad -- se
    mapean a `ProjectAuthorityRetryable` (code "reintentar"), nunca al
    generico `error_de_base_de_datos`, y tampoco incrustan el texto del
    driver en el mensaje."""
    import pymysql.err

    from jax.memory.project_authority import ProjectAuthorityRetryable, _wrap_unexpected_db_error

    for errno in (1213, 1205):
        driver_text = f"Some driver message for {errno} with a path /srv/x"
        raw = pymysql.err.OperationalError(errno, driver_text)
        wrapped = _wrap_unexpected_db_error(raw)
        assert isinstance(wrapped, ProjectAuthorityRetryable), (errno, wrapped)
        assert wrapped.code == "reintentar"
        assert wrapped.__cause__ is raw
        assert driver_text not in str(wrapped)


@requiere_db_de_prueba
@asincrono
async def test_subject_user_id_no_canonico_es_rechazado_antes_de_cualquier_sql():
    """Ronda 4, MINOR 1: `subject_user_id` tiene que matchear
    `^[1-9]\\d{0,9}$` antes de cualquier consulta -- nunca depender de que
    MariaDB rechace un valor no-numerico por su cuenta (modo estricto,
    coercion de tipos), que es un detalle de configuracion, no un contrato.

    Visto en rojo contra el codigo previo a esta ronda (repro standalone,
    restaurado a mano contra b9d13c8): un valor como "<id real> OR 1=1" no
    daba un `ValueError` sin capturar, pero SI daba un
    `ProjectAuthorityError` generico ("unexpected database error: (1265,
    'Data truncated ...')") -- una falla de una consulta real, dependiente
    de que el modo estricto de MariaDB este activado, en vez de un rechazo
    limpio antes de tocar la base. Con el arreglo: `AuthorizationDenied`
    inmediato, mismo mensaje para cualquier valor no canonico.
    """
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        real_user = await _crear_usuario(tenant_id, role="operator")
        await _crear_usuario(tenant_id, role="admin")
        # "" no llega al chequeo de MINOR 1 -- `require_subject` (capa mas
        # basica, ya existente) la rechaza antes por "ausente", con
        # `ScopeDenied`. Las demas SI son el caso que MINOR 1 cierra:
        # valores presentes pero no canonicos.
        for malformed, expected in (
            (f"{real_user} OR 1=1", AuthorizationDenied), ("0" + str(real_user), AuthorizationDenied),
            ("-1", AuthorizationDenied), ("abc", AuthorizationDenied), ("", ScopeDenied),
        ):
            scope = ScopeContext(f"user:{malformed}", "USER", malformed, str(tenant_id), None)
            with pytest.raises(expected):
                await admin_api.create_project(
                    _request(scope, "CREATE_PROJECT"), name="p-minor1", description=None,
                    idempotency_key=str(uuid.uuid4()))
        rows = await _sql("SELECT COUNT(*) AS n FROM projects WHERE name='p-minor1'", fetch=True)
        assert rows[0]["n"] == 0
    finally:
        pool.close(); await pool.wait_closed()


def test_projectauthorityadmin_no_toma_authorization_resolver():
    """Ronda 4, MINOR 3: el parametro `authorization_resolver` -- codigo
    muerto, nunca leido -- se quito del constructor."""
    import inspect

    params = inspect.signature(ProjectAuthorityAdmin.__init__).parameters
    assert list(params) == ["self", "store"], params


def test_exports_invalid_idempotency_key_y_retryable():
    """Ronda 4, MINOR 4."""
    import jax.memory as jm

    assert jm.InvalidIdempotencyKey is InvalidIdempotencyKey
    assert jm.ProjectAuthorityRetryable.__name__ == "ProjectAuthorityRetryable"
    assert "InvalidIdempotencyKey" in jm.__all__
    assert "ProjectAuthorityRetryable" in jm.__all__


@requiere_db_de_prueba
@asincrono
async def test_reactivar_y_revocar_dejan_pre_admin_en_null():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        owner_user = await _crear_usuario(tenant_id, role="operator")
        target_email = f"reactivar-{uuid.uuid4().hex[:8]}@test.invalid"
        target_user = await _crear_usuario(tenant_id, role="operator", email=target_email)
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)
        await _crear_membresia(project_id, tenant_id, owner_user, role="OWNER", origin="CREATOR")
        await _crear_membresia(project_id, tenant_id, target_user, role="VIEWER", status="REVOKED")
        owner_scope = _scope(owner_user, tenant_id, project_id)

        await admin_api.grant_member(
            _request(owner_scope, "GRANT_MEMBER"), project_id, email=target_email, role=ProjectRole.CONTRIBUTOR)
        rows = await _sql(
            "SELECT status,pre_admin_role,pre_admin_status FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (project_id, target_user), fetch=True)
        assert rows[0]["status"] == "ACTIVE"
        assert rows[0]["pre_admin_role"] is None and rows[0]["pre_admin_status"] is None

        await admin_api.revoke_member(_request(owner_scope, "REVOKE_MEMBER"), project_id, target_user)
        rows = await _sql(
            "SELECT status,pre_admin_role,pre_admin_status FROM jax_project_membership "
            "WHERE project_id=%s AND user_id=%s", (project_id, target_user), fetch=True)
        assert rows[0]["status"] == "REVOKED"
        assert rows[0]["pre_admin_role"] is None and rows[0]["pre_admin_status"] is None
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MINOR 3, ronda 3: idempotency_key debe ser un UUID canonico (36
# caracteres) antes de tocar la base.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_idempotency_key_invalida_es_rechazada_antes_de_tocar_la_base():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        creator = await _crear_usuario(tenant_id, role="operator")
        scope = _scope(creator, tenant_id, None)
        for clave_mala in ("no-es-un-uuid", str(uuid.uuid4()) + "0", str(uuid.uuid4())[:-1],
                          "{" + str(uuid.uuid4()) + "}"):
            with pytest.raises(InvalidIdempotencyKey):
                await admin_api.create_project(
                    _request(scope, "CREATE_PROJECT"), name="p", description=None, idempotency_key=clave_mala)
        rows = await _sql("SELECT COUNT(*) AS n FROM jax_project_creation_request", fetch=True)
        # No podemos afirmar 0 filas GLOBALES (otros tests comparten la
        # sesion), pero si que ninguna clave invalida quedo grabada.
        for clave_mala in ("no-es-un-uuid",):
            rows = await _sql(
                "SELECT COUNT(*) AS n FROM jax_project_creation_request WHERE idempotency_key=%s",
                (clave_mala,), fetch=True)
            assert rows[0]["n"] == 0
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MINOR 4, ronda 3: la lectura previa sin bloqueo (idempotencia) no debe
# devolver el proyecto a un actor que ya no esta ACTIVE.
#
# Visto en rojo contra el codigo previo a esta ronda (repro standalone,
# restaurado a mano contra 6403d5f): la segunda llamada, con el actor ya
# `inactive`, devolvia el proyecto igual (`created=False`).
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_idempotencia_no_responde_a_un_actor_ya_inactivo():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        creator = await _crear_usuario(tenant_id, role="operator")
        scope = _scope(creator, tenant_id, None)
        key = str(uuid.uuid4())
        created = await admin_api.create_project(
            _request(scope, "CREATE_PROJECT"), name="p", description=None, idempotency_key=key)
        assert created.created is True

        await _sql("UPDATE jax_users SET status='inactive' WHERE user_id=%s", (creator,))
        with pytest.raises(ProjectNotVisible):
            await admin_api.create_project(
                _request(scope, "CREATE_PROJECT"), name="p", description=None, idempotency_key=key)
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MINOR 5, ronda 3: bootstrap_existing_project solo lo puede hacer un admin
# de LEGACY_PROJECT_TENANT_ID -- otro tenant, aunque sea admin de verdad ahi,
# recibe el mismo proyecto_no_encontrado que un cross-tenant lookup.
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_bootstrap_desde_otro_tenant_da_proyecto_no_encontrado():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        legacy_project_id = await _crear_proyecto_activo(LEGACY_PROJECT_TENANT_ID, name="legacy-minor5")
        otro_tenant = await _crear_tenant("otro")
        admin_de_otro_tenant = await _crear_usuario(otro_tenant, role="admin")
        owner_user = await _crear_usuario(otro_tenant, role="operator")
        scope = _scope(admin_de_otro_tenant, otro_tenant, legacy_project_id)

        with pytest.raises(ProjectNotVisible):
            await admin_api.bootstrap_existing_project(
                _request(scope, "BOOTSTRAP_PROJECT"), legacy_project_id, owner_user_id=owner_user)
        rows = await _sql(
            "SELECT COUNT(*) AS n FROM jax_project_scope WHERE project_id=%s", (legacy_project_id,), fetch=True)
        assert rows[0]["n"] == 0
    finally:
        pool.close(); await pool.wait_closed()


# --------------------------------------------------------------------------
# MINOR 8, ronda 3: bootstrap y sync dejan el invariante de §6.5 en 0 filas
# tambien para scopes PRE-EXISTENTES (el caso de backfill: un admin que ya
# era admin antes de que existiera este codigo, y un proyecto que ya
# existia, sin membresia entre los dos).
# --------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_backfill_sync_repara_scopes_preexistentes_para_un_admin_ya_activo():
    pool = await _pool()
    try:
        admin_api = _admin(pool)
        tenant_id = await _crear_tenant()
        # admin1 YA es admin (simula "admin de antes de este codigo") --
        # nunca paso por _promote_to_admin, así que no tiene OWNER en P.
        admin1 = await _crear_usuario(tenant_id, role="admin")
        project_id = await _crear_proyecto_activo(tenant_id)
        await _crear_scope(project_id, tenant_id)

        offenders_antes = await _admins_sin_ownership(tenant_id)
        assert offenders_antes  # admin1 falta en P

        actor_scope = _scope(admin1, tenant_id, None)
        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    touched = await admin_api.sync_tenant_admin_memberships_in_transaction(
                        cur, actor_scope=actor_scope, user_id=admin1, tenant_id=tenant_id)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        assert touched == 1
        assert not await _admins_sin_ownership(tenant_id)
    finally:
        pool.close(); await pool.wait_closed()


async def _eventos_rename(project_id: int) -> int:
    filas = await _sql(
        "SELECT COUNT(*) AS n FROM jax_project_membership_event WHERE project_id=%s AND operation='RENAME_PROJECT'",
        (project_id,), fetch=True)
    return filas[0]["n"]


@asincrono
async def test_rename_project_cambia_nombre_y_deja_evento():
    t = await _crear_tenant("ren")
    u = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="viejo")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER")
    pool = await _pool()
    try:
        cambio = await _admin(pool).rename_project(
            _request(_scope(u, t, p), "RENAME_PROJECT"), p, name="  Nuevo  ", description="d")
    finally:
        pool.close(); await pool.wait_closed()
    assert cambio is True
    fila = await _sql("SELECT name, description FROM projects WHERE id=%s", (p,), fetch=True)
    assert (fila[0]["name"], fila[0]["description"]) == ("Nuevo", "d")
    assert await _eventos_rename(p) == 1


@asincrono
async def test_rename_project_igual_es_noop_sin_evento():
    t = await _crear_tenant("ren2")
    u = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="mismo")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER")
    await _sql("UPDATE projects SET description=NULL WHERE id=%s", (p,))
    pool = await _pool()
    try:
        cambio = await _admin(pool).rename_project(
            _request(_scope(u, t, p), "RENAME_PROJECT"), p, name="mismo", description=None)
    finally:
        pool.close(); await pool.wait_closed()
    assert cambio is False
    assert await _eventos_rename(p) == 0


@asincrono
async def test_rename_project_exige_owner_y_activo():
    t = await _crear_tenant("ren3")
    owner = await _crear_usuario(t)
    editor = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="x")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, owner, role="OWNER")
    await _crear_membresia(p, t, editor, role="CONTRIBUTOR")
    pool = await _pool()
    try:
        with pytest.raises(ProjectRoleInsufficient):
            await _admin(pool).rename_project(_request(_scope(editor, t, p), "RENAME_PROJECT"), p,
                                              name="y", description=None)
        await _sql("UPDATE jax_project_scope SET status='ARCHIVED' WHERE project_id=%s", (p,))
        with pytest.raises(ProjectStateConflict):
            await _admin(pool).rename_project(_request(_scope(owner, t, p), "RENAME_PROJECT"), p,
                                              name="y", description=None)
    finally:
        pool.close(); await pool.wait_closed()


@asincrono
async def test_rename_project_valida_nombre():
    t = await _crear_tenant("ren4")
    u = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="x")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER")
    pool = await _pool()
    try:
        for malo in ("", "   ", "x" * 256):
            with pytest.raises(AuthorizationDenied):
                await _admin(pool).rename_project(_request(_scope(u, t, p), "RENAME_PROJECT"), p,
                                                  name=malo, description=None)
        with pytest.raises(AuthorizationDenied):
            await _admin(pool).rename_project(_request(_scope(u, t, p), "RENAME_PROJECT"), p,
                                              name="ok", description="d" * 2001)
    finally:
        pool.close(); await pool.wait_closed()


@asincrono
async def test_rename_project_con_scope_de_otro_tenant_niega_sin_tocar_el_candado_de_projects():
    """m-1: el orden es alcance -> projects (igual que set_project_lifecycle). Un actor de OTRO
    tenant debe recibir ProjectNotVisible SIN esperar el candado de `projects(id)`; con el
    candado tomado antes del alcance se queda esperando a quien lo tenga (riesgo de 1213)."""
    t1 = await _crear_tenant("ren5a"); t2 = await _crear_tenant("ren5b")
    owner = await _crear_usuario(t1)
    ajeno = await _crear_usuario(t2)
    p = await _crear_proyecto_activo(t1, name="ajeno")
    await _crear_scope(p, t1)
    await _crear_membresia(p, t1, owner, role="OWNER")
    pool = await _pool()
    retenedor = await aiomysql.connect(db=_DB, autocommit=False, connect_timeout=db_connect_timeout_seconds(),
                                       **_conn_params())
    try:
        async with retenedor.cursor() as cur:
            await cur.execute("SELECT id FROM projects WHERE id=%s FOR UPDATE", (p,))      # candado ajeno abierto
        with pytest.raises(ProjectNotVisible):
            await asyncio.wait_for(
                _admin(pool).rename_project(_request(_scope(ajeno, t2, p), "RENAME_PROJECT"), p,
                                            name="robado", description=None), timeout=4)
    finally:
        await retenedor.rollback()
        retenedor.close()
        pool.close(); await pool.wait_closed()
    fila = await _sql("SELECT name FROM projects WHERE id=%s", (p,), fetch=True)
    assert fila[0]["name"] == "ajeno"
