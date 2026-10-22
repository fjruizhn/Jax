"""Disposable-MariaDB coverage for Office namespace migration and creation."""
from __future__ import annotations

import asyncio
import functools
import os
import re
import math
import time
import uuid
from pathlib import Path

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.project_authority import (IdempotencyKeyConflict,
                                          ProjectAuthorityAdmin,
                                          ProjectAuthorityError)
from jax.memory.project_authority_migrations import (apply_project_authority_migration,
                                                      revert_project_lifecycle_migration)
from jax.memory.scope_authority import (MariaDBScopeAuthorityResolver,
                                        ProjectLifecycle, ProjectNamespace)
from jax.memory.b9_mariadb import MariaDBB9Store
from jax.memory.b9 import MutationAuthorizationRequest, ScopeContext, ScopeDenied, Visibility
from jax.memory.project_queries import (ProjectView, get_project_for_user,
                                        list_project_members, list_projects_for_user)


_DB = os.getenv("JAX_DB_NAME", "")
requires_test_db = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB),
    reason="needs disposable MariaDB and JAX_DB_NAME")
_SCHEMA = Path(__file__).resolve().parents[1] / "jax_memory_schema.sql"


def _async(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapped


def _params():
    return {"host": os.environ["JAX_DB_HOST"], "port": int(os.environ["JAX_DB_PORT"]),
            "user": os.environ["JAX_DB_USER"], "password": os.environ["JAX_DB_PASSWORD"]}


def _projects_ddl() -> str:
    match = re.search(r"CREATE TABLE `projects` \(.*?\n\)[^;\n]*", _SCHEMA.read_text(), re.S)
    assert match
    return match.group(0)


class _Authorizer:
    def __init__(self, *, allow=True):
        self.allow = allow
        self.cursors = []

    async def authorize_in_transaction(self, cursor, **kwargs):
        self.cursors.append((cursor, kwargs))
        if not self.allow:
            raise PermissionError("revoked")


class _FirstHintBarrier:
    """Stops only the first two actor hint reads before either takes a lock."""
    def __init__(self):
        self._barrier = asyncio.Barrier(2)
        self.arrivals = 0

    async def wait(self):
        self.arrivals += 1
        if self.arrivals <= 2:
            await self._barrier.wait()


class _BarrierCursor:
    def __init__(self, cursor, barrier):
        self._cursor = cursor
        self._barrier = barrier

    async def execute(self, sql, args=None):
        result = await self._cursor.execute(sql, args)
        if sql == "SELECT tenant_id FROM jax_users WHERE user_id=%s":
            await self._barrier.wait()
        return result

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _ObservedBarrierStore(MariaDBB9Store):
    """Real mutation boundary plus raw DB-error observation for this test."""
    def __init__(self, pool, barrier, raw_errnos):
        super().__init__(pool)
        self._barrier = barrier
        self._raw_errnos = raw_errnos

    async def mutation(self, operation):
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    value = await operation(_BarrierCursor(cur, self._barrier))
                await conn.commit()
                return value
            except BaseException as exc:
                args = getattr(exc, "args", ()) or ()
                if args and isinstance(args[0], int):
                    self._raw_errnos.append(args[0])
                await conn.rollback()
                raise


async def _setup(name: str):
    root = await aiomysql.connect(autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with root.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{name}`")
    finally:
        root.close()
    conn = await aiomysql.connect(db=name, autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute("CREATE TABLE jax_tenants (tenant_id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) NOT NULL, plan VARCHAR(20), status VARCHAR(20))")
            await cur.execute("CREATE TABLE jax_users (user_id INT AUTO_INCREMENT PRIMARY KEY, tenant_id INT NOT NULL, email VARCHAR(320) UNIQUE NOT NULL, password_hash VARCHAR(255) NOT NULL, role VARCHAR(20), status VARCHAR(20))")
            await cur.execute(_projects_ddl())
            await apply_project_authority_migration(cur)
            await apply_project_authority_migration(cur)
            await cur.execute("INSERT INTO jax_tenants (name,plan,status) VALUES ('office','personal','active')")
            tenant_id = cur.lastrowid
            await cur.execute("INSERT INTO jax_users (tenant_id,email,password_hash,role,status) VALUES (%s,'office@test.invalid','x','operator','active')", (tenant_id,))
            return tenant_id, cur.lastrowid
    finally:
        conn.close()


async def _drop(name: str):
    conn = await aiomysql.connect(autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{name}`")
    finally:
        conn.close()


@requires_test_db
@_async
async def test_office_namespace_migration_creation_replay_rollback_and_index():
    name = f"{_DB}_office_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    tenant_id, actor_id = await _setup(name)
    pool = await aiomysql.create_pool(db=name, autocommit=True, minsize=1, maxsize=4,
                                      cursorclass=aiomysql.DictCursor,
                                      connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        api = ProjectAuthorityAdmin(MariaDBB9Store(pool))
        auth = _Authorizer()
        key = str(uuid.uuid4())
        created = await api.create_namespaced_project(
            actor_user_id=actor_id, target_tenant_id=tenant_id,
            namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="Office", description="private",
            idempotency_key=key, authorizer=auth)
        replay = await api.create_namespaced_project(
            actor_user_id=actor_id, target_tenant_id=tenant_id,
            namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="Office", description="private",
            idempotency_key=key, authorizer=auth)
        assert created.created and not replay.created and replay.project_id == created.project_id
        assert len(auth.cursors) == 2 and auth.cursors[0][0] is not None
        with pytest.raises(IdempotencyKeyConflict):
            await api.create_namespaced_project(
                actor_user_id=actor_id, target_tenant_id=tenant_id,
                namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="changed", description="private",
                idempotency_key=key, authorizer=auth)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO jax_users (tenant_id,email,password_hash,role,status) VALUES (%s,'other@test.invalid','x','operator','active')",
                    (tenant_id,))
                other_actor_id = cur.lastrowid
        with pytest.raises(IdempotencyKeyConflict):
            await api.create_namespaced_project(
                actor_user_id=other_actor_id, target_tenant_id=tenant_id,
                namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="Office", description="private",
                idempotency_key=key, authorizer=auth)
        denied = _Authorizer(allow=False)
        with pytest.raises(PermissionError):
            await api.create_namespaced_project(
                actor_user_id=actor_id, target_tenant_id=tenant_id,
                namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="denied", description=None,
                idempotency_key=str(uuid.uuid4()), authorizer=denied)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT authorization_namespace FROM jax_project_scope WHERE project_id=%s", (created.project_id,))
                assert (await cur.fetchone())["authorization_namespace"] == "OFFICE_CONFIDENTIAL"
                await cur.execute("SELECT COUNT(*) AS n FROM jax_project_membership WHERE project_id=%s", (created.project_id,))
                assert (await cur.fetchone())["n"] == 0
                await cur.execute("SELECT COUNT(*) AS n FROM projects WHERE name='denied'")
                assert (await cur.fetchone())["n"] == 0
                await cur.execute("EXPLAIN SELECT project_id FROM jax_project_scope WHERE authorization_namespace='CLASSIC' AND tenant_id=%s AND status='ACTIVE' ORDER BY project_id", (tenant_id,))
                plan = await cur.fetchone()
                assert plan["key"] == "idx_jax_project_scope_namespace_tenant_status_project"
    finally:
        pool.close(); await pool.wait_closed()
        await _drop(name)


@requires_test_db
@_async
async def test_office_root_list_uses_cross_tenant_namespace_status_cursor_index():
    """The root Office list must not walk scopes from unrelated tenants.

    The exact JXP query joins its authoritative scope to the deployed
    ``projects`` table.  The legacy namespace/tenant/status/project index has
    a tenant gap, so it cannot both apply ``status`` and provide the cursor
    order for a cross-tenant root list.  Seed a deliberately skewed Office
    population, observe that plan without the new index, then re-run the real
    idempotent migration and require its range plan.
    """
    name = f"{_DB}_office_list_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    tenant_id, _actor_id = await _setup(name)
    conn = await aiomysql.connect(
        db=name, autocommit=True, cursorclass=aiomysql.DictCursor,
        connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with conn.cursor() as cur:
            tenant_ids = [tenant_id]
            for number in range(3):
                await cur.execute(
                    "INSERT INTO jax_tenants (name,plan,status) VALUES (%s,'personal','active')",
                    (f"office-list-{number}",))
                tenant_ids.append(cur.lastrowid)

            # 240 older Office rows and 2,000 Classic rows make the post-cursor
            # Office page selective while preserving a realistic mixed schema.
            # ``projects`` is the deployed DDL read by _projects_ddl(), never
            # a hand-made projection of it.
            for number in range(2240):
                namespace = "OFFICE_CONFIDENTIAL" if number < 240 else "CLASSIC"
                await cur.execute(
                    "INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),%s,'active')",
                    (f"list-{number}",))
                project_id = cur.lastrowid
                await cur.execute(
                    "INSERT INTO jax_project_scope "
                    "(project_id,tenant_id,authorization_namespace,status,created_at,created_by,updated_at) "
                    "VALUES (%s,%s,%s,'ACTIVE',NOW(6),'test',NOW(6))",
                    (project_id, tenant_ids[number % len(tenant_ids)], namespace))

            await cur.execute(
                "SELECT project_id FROM jax_project_scope "
                "WHERE authorization_namespace='OFFICE_CONFIDENTIAL' "
                "ORDER BY project_id LIMIT 1 OFFSET 220")
            cursor_id = (await cur.fetchone())["project_id"]
            query = (
                "SELECT p.id,p.name,p.description,s.tenant_id FROM projects p "
                "JOIN jax_project_scope s ON s.project_id=p.id "
                "WHERE s.authorization_namespace='OFFICE_CONFIDENTIAL' AND s.status='ACTIVE' "
                "AND p.id>%s ORDER BY p.id LIMIT %s")

            await cur.execute(
                "DROP INDEX idx_jax_project_scope_namespace_status_project ON jax_project_scope")
            await cur.execute("EXPLAIN " + query, (cursor_id, 25))
            before = await cur.fetchall()
            await apply_project_authority_migration(cur)
            await cur.execute("EXPLAIN " + query, (cursor_id, 25))
            after = await cur.fetchall()

            before_scope = next(row for row in before if row["table"] == "s")
            after_scope = next(row for row in after if row["table"] == "s")
            extra = str(after_scope.get("Extra") or "")
            assert after_scope["key"] == "idx_jax_project_scope_namespace_status_project"
            assert after_scope["type"] == "range"
            assert int(after_scope["rows"]) < int(before_scope["rows"])
            assert "Using filesort" not in extra
            assert "Using temporary" not in extra
            print(
                "office root-list explain: "
                f"before(key={before_scope['key']},type={before_scope['type']},rows={before_scope['rows']}) "
                f"after(key={after_scope['key']},type={after_scope['type']},rows={after_scope['rows']},extra={extra!r})")
    finally:
        conn.close()
        await _drop(name)


@requires_test_db
@_async
async def test_office_same_key_concurrency_creates_one_project_and_stable_replay():
    """Catches duplicate-key handling that leaves concurrent retries failing."""
    name = f"{_DB}_office_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    tenant_id, actor_id = await _setup(name)
    pool = await aiomysql.create_pool(db=name, autocommit=True, minsize=1, maxsize=4,
                                      cursorclass=aiomysql.DictCursor,
                                      connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        api = ProjectAuthorityAdmin(MariaDBB9Store(pool))
        key = str(uuid.uuid4())
        async def create(key, project_name):
            started = time.monotonic()
            result = await api.create_namespaced_project(
                actor_user_id=actor_id, target_tenant_id=tenant_id,
                namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name=project_name, description=None,
                idempotency_key=key, authorizer=_Authorizer())
            return result, time.monotonic() - started

        results = await asyncio.gather(*[create(key, "same") for _ in range(8)])
        projects, durations = zip(*results)
        elapsed = max(durations)
        p95 = sorted(durations)[math.ceil(len(durations) * .95) - 1]
        assert len({result.project_id for result in projects}) == 1
        assert sum(result.created for result in projects) == 1
        # These measurements make a contention regression visible in output.
        assert p95 < 10, f"same-key p95 exceeded bounded test window: {p95:.3f}s (max {elapsed:.3f}s)"
        different = await asyncio.gather(*[
            create(str(uuid.uuid4()), f"different-{index}") for index in range(8)
        ])
        assert len({result.project_id for result, _ in different}) == 8
        different_p95 = sorted(duration for _, duration in different)[math.ceil(len(different) * .95) - 1]
        print(f"office namespace contention: same_key_p95={p95:.6f}s different_key_p95={different_p95:.6f}s")
    finally:
        pool.close(); await pool.wait_closed()
        await _drop(name)


@requires_test_db
@_async
async def test_office_cross_tenant_creates_use_canonical_lock_order_without_deadlock():
    """Opposite directions overlap after the hint yet never hit 1213/1205."""
    name = f"{_DB}_office_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    low_tenant_id, low_actor_id = await _setup(name)
    root = await aiomysql.connect(db=name, autocommit=True, cursorclass=aiomysql.DictCursor,
                                  connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with root.cursor() as cur:
            await cur.execute("INSERT INTO jax_tenants (name,plan,status) VALUES ('other','personal','active')")
            high_tenant_id = cur.lastrowid
            await cur.execute(
                "INSERT INTO jax_users (tenant_id,email,password_hash,role,status) VALUES (%s,'other-actor@test.invalid','x','operator','active')",
                (high_tenant_id,))
            high_actor_id = cur.lastrowid
            await cur.execute("EXPLAIN SELECT tenant_id FROM jax_users WHERE user_id=%s", (low_actor_id,))
            assert (await cur.fetchone())["key"] == "PRIMARY"
    finally:
        root.close()
    pool = await aiomysql.create_pool(db=name, autocommit=True, minsize=1, maxsize=4,
                                      cursorclass=aiomysql.DictCursor,
                                      connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        barrier = _FirstHintBarrier()
        raw_errnos = []
        api = ProjectAuthorityAdmin(_ObservedBarrierStore(pool, barrier, raw_errnos))

        async def create(actor_id, target_id, label, authorizer):
            return await api.create_namespaced_project(
                actor_user_id=actor_id, target_tenant_id=target_id,
                namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name=label, description=None,
                idempotency_key=str(uuid.uuid4()), authorizer=authorizer)

        low_authorizer, high_authorizer = _Authorizer(), _Authorizer()
        first, second = await asyncio.gather(
            create(low_actor_id, high_tenant_id, "low-to-high", low_authorizer),
            create(high_actor_id, low_tenant_id, "high-to-low", high_authorizer),
        )
        assert barrier.arrivals >= 2
        assert not ({1213, 1205} & set(raw_errnos)), raw_errnos
        assert first.created and second.created and first.project_id != second.project_id
        assert len(low_authorizer.cursors) == len(high_authorizer.cursors) == 1
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT tenant_id,authorization_namespace FROM jax_project_scope WHERE project_id IN (%s,%s) ORDER BY project_id",
                    (first.project_id, second.project_id))
                rows = await cur.fetchall()
                assert {row["tenant_id"] for row in rows} == {low_tenant_id, high_tenant_id}
                assert {row["authorization_namespace"] for row in rows} == {"OFFICE_CONFIDENTIAL"}
    finally:
        pool.close(); await pool.wait_closed()
        await _drop(name)


@requires_test_db
@_async
async def test_classic_resolvers_queries_and_admin_mutations_deny_office_scope():
    """Catches an Office scope inheriting a manually present classic OWNER."""
    name = f"{_DB}_office_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    tenant_id, actor_id = await _setup(name)
    pool = await aiomysql.create_pool(db=name, autocommit=True, minsize=1, maxsize=4,
                                      cursorclass=aiomysql.DictCursor,
                                      connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        api = ProjectAuthorityAdmin(MariaDBB9Store(pool))
        created = await api.create_namespaced_project(
            actor_user_id=actor_id, target_tenant_id=tenant_id,
            namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="Office", description=None,
            idempotency_key=str(uuid.uuid4()), authorizer=_Authorizer())
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,status,grant_origin,created_at,created_by,updated_at) "
                    "VALUES (UUID(),%s,%s,%s,'OWNER','ACTIVE','EXPLICIT',NOW(6),'test',NOW(6))",
                    (created.project_id, tenant_id, actor_id))
        scope = ScopeContext(f"user:{actor_id}", "USER", str(actor_id), str(tenant_id), str(created.project_id))
        resolver = MariaDBScopeAuthorityResolver(pool)
        with pytest.raises(ScopeDenied, match="namespace"):
            await resolver.resolve_project_scope(scope)
        with pytest.raises(ScopeDenied, match="namespace"):
            await resolver.resolve_project_read(scope)
        with pytest.raises(ScopeDenied, match="namespace"):
            await resolver.resolve_mutation(scope, "CREATE", Visibility.PROJECT_SHARED)
        assert await list_projects_for_user(pool, tenant_id=tenant_id, user_id=actor_id,
                                            view=ProjectView.ACTIVOS, before_id=None, limit=10) == []
        with pytest.raises(Exception):
            await get_project_for_user(pool, tenant_id=tenant_id, user_id=actor_id, project_id=created.project_id)
        with pytest.raises(Exception):
            await list_project_members(pool, tenant_id=tenant_id, user_id=actor_id, project_id=created.project_id)
        request = MutationAuthorizationRequest(scope, "RENAME_PROJECT", Visibility.PROJECT_SHARED)
        with pytest.raises(Exception, match="namespace"):
            await api.rename_project(request, created.project_id, name="forbidden", description=None)
        request = MutationAuthorizationRequest(scope, "SET_PROJECT_LIFECYCLE", Visibility.PROJECT_SHARED)
        with pytest.raises(Exception, match="namespace"):
            await api.set_project_lifecycle(request, created.project_id, ProjectLifecycle.ARCHIVED)
    finally:
        pool.close(); await pool.wait_closed()
        await _drop(name)


@requires_test_db
@_async
async def test_namespace_rollback_fails_before_ddl_when_office_data_exists():
    """Catches a legacy rollback deleting Office scope/request contract data."""
    name = f"{_DB}_office_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    tenant_id, actor_id = await _setup(name)
    pool = await aiomysql.create_pool(db=name, autocommit=True, minsize=1, maxsize=2,
                                      cursorclass=aiomysql.DictCursor,
                                      connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        await ProjectAuthorityAdmin(MariaDBB9Store(pool)).create_namespaced_project(
            actor_user_id=actor_id, target_tenant_id=tenant_id,
            namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="Office", description=None,
            idempotency_key=str(uuid.uuid4()), authorizer=_Authorizer())
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                with pytest.raises(RuntimeError, match="non-CLASSIC"):
                    await revert_project_lifecycle_migration(cur)
                await cur.execute("SHOW COLUMNS FROM jax_project_scope LIKE 'authorization_namespace'")
                assert await cur.fetchone() is not None
                await cur.execute("SHOW TABLES LIKE 'jax_namespace_project_creation_request'")
                assert await cur.fetchone() is not None
                # The request guard independently protects data if a manual
                # repair mislabeled the Office scope as CLASSIC.
                await cur.execute(
                    "UPDATE jax_project_scope SET authorization_namespace='CLASSIC'")
                with pytest.raises(RuntimeError, match="namespace project creation requests"):
                    await revert_project_lifecycle_migration(cur)
                await cur.execute("SHOW COLUMNS FROM jax_project_scope LIKE 'authorization_namespace'")
                assert await cur.fetchone() is not None
    finally:
        pool.close(); await pool.wait_closed()
        await _drop(name)


@requires_test_db
@_async
async def test_upgrade_existing_classic_scope_backfills_namespace_without_changing_row():
    """Catches an upgrade that rewrites or leaves legacy scope rows unbound."""
    name = f"{_DB}_office_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    root = await aiomysql.connect(autocommit=True, connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with root.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{name}`")
    finally:
        root.close()
    conn = await aiomysql.connect(db=name, autocommit=True, cursorclass=aiomysql.DictCursor,
                                  connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with conn.cursor() as cur:
            await cur.execute("CREATE TABLE jax_tenants (tenant_id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) NOT NULL, plan VARCHAR(20), status VARCHAR(20))")
            await cur.execute("CREATE TABLE jax_users (user_id INT AUTO_INCREMENT PRIMARY KEY, tenant_id INT NOT NULL, email VARCHAR(320) UNIQUE NOT NULL, password_hash VARCHAR(255) NOT NULL, role VARCHAR(20), status VARCHAR(20))")
            await cur.execute(_projects_ddl())
            await cur.execute("CREATE TABLE jax_project_scope (project_id INT NOT NULL PRIMARY KEY, tenant_id INT NOT NULL, status VARCHAR(16) NOT NULL, created_at DATETIME(6) NOT NULL, created_by VARCHAR(128) NOT NULL, updated_at DATETIME(6) NOT NULL, UNIQUE KEY uq_jax_project_scope_project_tenant (project_id,tenant_id))")
            await cur.execute("INSERT INTO jax_tenants (name,plan,status) VALUES ('legacy','personal','active')")
            tenant_id = cur.lastrowid
            await cur.execute("INSERT INTO projects (project_uuid,name,status) VALUES (UUID(),'legacy','active')")
            project_id = cur.lastrowid
            await cur.execute("INSERT INTO jax_project_scope (project_id,tenant_id,status,created_at,created_by,updated_at) VALUES (%s,%s,'ACTIVE',NOW(6),'legacy',NOW(6))", (project_id, tenant_id))
            await apply_project_authority_migration(cur)
            await cur.execute("SELECT tenant_id,status,created_by,authorization_namespace FROM jax_project_scope WHERE project_id=%s", (project_id,))
            row = await cur.fetchone()
            assert row == {"tenant_id": tenant_id, "status": "ACTIVE", "created_by": "legacy", "authorization_namespace": "CLASSIC"}
    finally:
        conn.close()
        await _drop(name)


@requires_test_db
@_async
async def test_real_scope_insert_failure_rolls_back_project_and_namespace_request():
    """Catches a confidential create that publishes a project before scope."""
    name = f"{_DB}_office_{uuid.uuid4().hex[:10]}"
    assert es_base_de_test(name)
    tenant_id, actor_id = await _setup(name)
    pool = await aiomysql.create_pool(db=name, autocommit=True, minsize=1, maxsize=2,
                                      cursorclass=aiomysql.DictCursor,
                                      connect_timeout=db_connect_timeout_seconds(), **_params())
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE TRIGGER fail_office_scope BEFORE INSERT ON jax_project_scope FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='forced scope failure'")
        with pytest.raises(ProjectAuthorityError) as failure:
            await ProjectAuthorityAdmin(MariaDBB9Store(pool)).create_namespaced_project(
                actor_user_id=actor_id, target_tenant_id=tenant_id,
                namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="Office", description=None,
                idempotency_key=str(uuid.uuid4()), authorizer=_Authorizer())
        assert failure.value.code == "error_de_base_de_datos"
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for table in ("projects", "jax_project_scope", "jax_namespace_project_creation_request"):
                    await cur.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
                    assert (await cur.fetchone())["n"] == 0
    finally:
        pool.close(); await pool.wait_closed()
        await _drop(name)
