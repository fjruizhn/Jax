import asyncio
import os
import uuid
from dataclasses import replace

import pytest

from jax.memory.b9 import (
    AuthorizationDenied, EventKind, Lifecycle, MemoryEvent, MemoryRevision,
    MutationAuthorizationContext, MutationAuthorizationRequest, ObjectKind, ScopeContext, ScopeDenied, Visibility, _derive_projection,
)
from jax.memory.b9_mariadb import MariaDBB9Store, MariaDBB9Reader, PersistentMemoryAPI


async def _race_pool(*, dict_cursor=False):
    """A real multi-connection pool for the M1 delete/adoption interleave.

    Unlike `_b9_ci_test_pool`, this uses persistent tables in the disposable
    suffixed CI database because TEMPORARY tables are connection-local and
    cannot prove a race. It is never reachable without the explicit test DB
    guard below.
    """
    import aiomysql
    from base_de_test import exigir_base_de_test
    if not os.getenv("JAX_DB_HOST"):
        pytest.skip("requires the disposable memory-b9 MariaDB job")
    database=exigir_base_de_test()
    return await aiomysql.create_pool(
        host=os.environ["JAX_DB_HOST"], port=int(os.getenv("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", "root"), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=database, minsize=1, maxsize=4,
        cursorclass=aiomysql.DictCursor if dict_cursor else aiomysql.Cursor,
        autocommit=False, connect_timeout=5,
    )


async def _ensure_race_b9_tables(pool):
    """Create only the persistent B9 tables `_write` needs if CI has not
    already applied them. These names live solely in the disposable test DB."""
    ddls=(
        "CREATE TABLE IF NOT EXISTS memory_objects (memory_id CHAR(36) PRIMARY KEY, object_kind VARCHAR(32) NOT NULL, tenant_id VARCHAR(128) NOT NULL, created_at DATETIME(6) NOT NULL, legacy_source_type VARCHAR(64), legacy_source_namespace VARCHAR(255), legacy_source_key VARCHAR(255))",
        "CREATE TABLE IF NOT EXISTS memory_revisions (revision_id CHAR(36) PRIMARY KEY, memory_id CHAR(36) NOT NULL, content_digest CHAR(71) NOT NULL, visibility VARCHAR(32) NOT NULL, user_id VARCHAR(128), project_id VARCHAR(128), lifecycle_state VARCHAR(32) NOT NULL, created_at DATETIME(6) NOT NULL, payload LONGBLOB, provenance_status VARCHAR(64) NOT NULL, prior_revision_id CHAR(36), tenant_id VARCHAR(128))",
        "CREATE TABLE IF NOT EXISTS memory_revision_payloads (revision_id CHAR(36) PRIMARY KEY, payload LONGBLOB, purged_at DATETIME(6), purge_reason VARCHAR(255))",
        "CREATE TABLE IF NOT EXISTS memory_provenance (provenance_id CHAR(36) PRIMARY KEY, revision_id CHAR(36) NOT NULL, source_revisions JSON, transformation_id VARCHAR(128) NOT NULL, transformation_version VARCHAR(64) NOT NULL, actor_principal VARCHAR(255) NOT NULL, actor_type VARCHAR(64) NOT NULL, subject_user_id VARCHAR(128), provider VARCHAR(128), model VARCHAR(255), created_at DATETIME(6) NOT NULL, limitations TEXT)",
        "CREATE TABLE IF NOT EXISTS memory_events (event_id CHAR(36) PRIMARY KEY, memory_id CHAR(36) NOT NULL, revision_id CHAR(36), event_kind VARCHAR(32) NOT NULL, actor_principal VARCHAR(255) NOT NULL, subject_user_id VARCHAR(128), authority_source VARCHAR(255) NOT NULL, occurred_at DATETIME(6) NOT NULL, details JSON, compensates_event_id CHAR(36), actor_type VARCHAR(64), delegation VARCHAR(255), calling_component VARCHAR(255), request_id VARCHAR(255), trace_id VARCHAR(255))",
        "CREATE TABLE IF NOT EXISTS memory_projections (memory_id CHAR(36) PRIMARY KEY, current_revision_id CHAR(36), current_lifecycle_state VARCHAR(32), current_verification_state BOOLEAN NOT NULL, canonical_history_digest CHAR(71) NOT NULL, reconciliation_required BOOLEAN NOT NULL DEFAULT FALSE)",
        "CREATE TABLE IF NOT EXISTS memory_legacy_bindings (tenant_id VARCHAR(128) NOT NULL, legacy_source_type VARCHAR(64) NOT NULL, legacy_source_namespace VARCHAR(255) NOT NULL, legacy_source_key VARCHAR(255) NOT NULL, memory_id CHAR(36) NOT NULL, binding_state VARCHAR(32) NOT NULL, created_at DATETIME(6) NOT NULL, PRIMARY KEY (tenant_id, legacy_source_type, legacy_source_namespace, legacy_source_key))",
    )
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for ddl in ddls:
                await cur.execute(ddl)
        await conn.commit()


async def _ensure_race_legacy_source_tables(pool):
    """Bootstrap only the disposable test DB's legacy source prerequisites."""
    ddls=(
        "CREATE TABLE IF NOT EXISTS jax_tenants (tenant_id INT PRIMARY KEY, name VARCHAR(100) NOT NULL, plan VARCHAR(20), status VARCHAR(20), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        "CREATE TABLE IF NOT EXISTS jax_users (user_id INT PRIMARY KEY, tenant_id INT NOT NULL, email VARCHAR(320) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL, role VARCHAR(20), status VARCHAR(20), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        "CREATE TABLE IF NOT EXISTS facts (id BIGINT AUTO_INCREMENT PRIMARY KEY, fact_uuid CHAR(36) NOT NULL, fact_text TEXT NOT NULL, fact_type VARCHAR(64), confidence DOUBLE, is_verified BOOLEAN, user_id INT NOT NULL, project_id VARCHAR(128), source_message_id BIGINT, source_facet VARCHAR(128), source_fact_ids JSON, superseded_by BIGINT, superseded_at DATETIME, superseded_by_user INT, expires_at DATETIME, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    )
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for ddl in ddls:
                await cur.execute(ddl)
            await cur.execute("INSERT INTO jax_tenants (tenant_id,name,plan,status) VALUES (990330001,'F2-P race tenant','test','active') ON DUPLICATE KEY UPDATE tenant_id=tenant_id")
            await cur.execute("INSERT INTO jax_users (user_id,tenant_id,email,password_hash,role,status) VALUES (990330001,990330001,'f2p-race@example.invalid','test','operator','ACTIVE') ON DUPLICATE KEY UPDATE tenant_id=VALUES(tenant_id),status='ACTIVE'")
        await conn.commit()


class _DeleteLockCursor:
    """Signals only after the real delete cursor has locked its fact row."""
    def __init__(self, cursor, locked, release):
        self._cursor_context, self._cursor = cursor, None
        self._locked, self._release = locked, release
        self._fact_lock_select = False
    async def __aenter__(self):
        self._cursor = await self._cursor_context.__aenter__()
        return self
    async def __aexit__(self, *args):
        return await self._cursor_context.__aexit__(*args)
    async def execute(self, sql, args=()):
        self._fact_lock_select = "FROM facts f JOIN jax_users" in sql and "FOR UPDATE" in sql
        return await self._cursor.execute(sql, args)
    async def fetchone(self):
        row=await self._cursor.fetchone()
        if self._fact_lock_select:
            self._fact_lock_select=False
            self._locked.set()
            await self._release.wait()
        return row
    @property
    def rowcount(self): return self._cursor.rowcount


class _DeleteLockConn:
    def __init__(self, conn, locked, release): self._conn, self._locked, self._release=conn, locked, release
    async def begin(self): return await self._conn.begin()
    async def commit(self): return await self._conn.commit()
    async def rollback(self): return await self._conn.rollback()
    def cursor(self): return _DeleteLockCursor(self._conn.cursor(), self._locked, self._release)


class _DeleteLockAcquire:
    def __init__(self, acquire, locked, release): self._acquire, self._locked, self._release=acquire, locked, release
    async def __aenter__(self): return _DeleteLockConn(await self._acquire.__aenter__(), self._locked, self._release)
    async def __aexit__(self, *args): return await self._acquire.__aexit__(*args)


class _DeleteLockPool:
    def __init__(self, pool, locked, release): self._pool, self._locked, self._release=pool, locked, release
    def acquire(self): return _DeleteLockAcquire(self._pool.acquire(), self._locked, self._release)


class Cursor:
    def __init__(self, fail_at=None): self.fail_at, self.calls = fail_at, []
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass
    async def execute(self, sql, args=()):
        self.calls.append((sql, args))
        if self.fail_at == len(self.calls): raise RuntimeError("write failure")

class Conn:
    def __init__(self, fail_at=None): self.cursor_obj=Cursor(fail_at); self.committed=False; self.rolled=False
    async def begin(self): pass
    async def commit(self): self.committed=True
    async def rollback(self): self.rolled=True
    def cursor(self): return self.cursor_obj

class Acquire:
    def __init__(self, conn): self.conn=conn
    async def __aenter__(self): return self.conn
    async def __aexit__(self, *_): pass

class Pool:
    def __init__(self, conn): self.conn=conn
    def acquire(self): return Acquire(self.conn)

class ScriptCursor(Cursor):
    def __init__(self, ones=(), many=(), fail_at=None):
        super().__init__(fail_at); self.ones=list(ones); self.many=list(many)
    async def fetchone(self): return self.ones.pop(0) if self.ones else None
    async def fetchall(self): return self.many.pop(0) if self.many else []

class ScriptConn(Conn):
    def __init__(self, ones=(), many=(), fail_at=None):
        super().__init__(fail_at); self.cursor_obj=ScriptCursor(ones,many,fail_at)


def current_rows(*, reconciled=True, project_id=None):
    visibility="PROJECT_SHARED" if project_id else "USER_PRIVATE"
    revision={"revision_id":"r1","memory_id":"m1","content_digest":"sha256:old","visibility":visibility,"user_id":"user-1","project_id":project_id,"lifecycle_state":"ACTIVE","created_at":1.0,"payload":"old","provenance_status":"COMPLETE","prior_revision_id":None}
    event={"event_id":"e1","memory_id":"m1","revision_id":"r1","event_kind":"CREATE","actor_principal":"user-1","subject_user_id":"user-1","authority_source":"test-authority","occurred_at":1.0,"details":{},"compensates_event_id":None,"actor_type":"USER","delegation":None,"calling_component":None,"request_id":None,"trace_id":None}
    rev=MemoryRevision("r1","m1","sha256:old",Visibility(visibility),"user-1",project_id,Lifecycle.ACTIVE,1,"old")
    evt=MemoryEvent("e1","m1","r1",EventKind.CREATE,"user-1","user-1","test-authority",1,{},actor_type="USER")
    projection=_derive_projection("m1",[rev],[evt])
    current={"memory_id":"m1","object_kind":"FACT","tenant_id":"tenant-1","object_created_at":1.0,
             **revision}
    current["revision_created_at"] = current.pop("created_at")
    stored={"current_revision_id":projection.current_revision_id,"current_lifecycle_state":projection.current_lifecycle.value,
            "current_verification_state":projection.current_verification,"canonical_history_digest":projection.canonical_history_digest,
            "reconciliation_required":not reconciled}
    return current, revision, event, stored

def lifecycle_api(*, reconciled=True, project_id=None, resolver=None):
    current, revision, event, stored=current_rows(reconciled=reconciled, project_id=project_id)
    # _current.fetchone, then _assert_reconciled.fetchall revisions/events,
    # then its projection fetchone.
    conn=ScriptConn([current, stored], [[revision], [event]])
    return conn, PersistentMemoryAPI(MariaDBB9Store(Pool(conn)), resolver or TxResolver())

class TxResolver:
    def __init__(self, roles=(), caps=(), denied=False, project_role="CONTRIBUTOR"):
        self.roles=frozenset(roles); self.caps=frozenset(caps); self.denied=denied; self.calls=[]
        self.project_role=project_role
    async def resolve_mutation_in_transaction(self, cur, scope, operation, visibility):
        self.calls.append((cur,scope,operation,visibility))
        if self.denied: raise AuthorizationDenied("current authority denied")
        if scope.project_id:
            from jax.memory.scope_authority import ProjectRole, ProjectScopeAuthorization
            role=ProjectRole(self.project_role)
            scope=replace(scope,project_authorization=ProjectScopeAuthorization(
                scope.project_id, scope.tenant_id, scope.subject_user_id or "", "membership-1",
                role, "ACTIVE", "ACTIVE", 1.0))
            caps=set(self.caps)
            if role in {ProjectRole.CONTRIBUTOR,ProjectRole.REVIEWER,ProjectRole.OWNER}: caps.add("memory:project:write")
            if role in {ProjectRole.REVIEWER,ProjectRole.OWNER}: caps.add("memory:project:verify")
            return MutationAuthorizationContext(scope,operation,visibility,self.roles,frozenset(caps),"db-test-authority")
        return MutationAuthorizationContext(scope,operation,visibility,self.roles,self.caps,"db-test-authority")

def auth(operation="CREATE", visibility=Visibility.USER_PRIVATE, *, project_id=None):
    scope=ScopeContext("user-1", "USER", "user-1", "tenant-1", project_id=project_id)
    return MutationAuthorizationRequest(scope, operation, visibility)

def make_api(conn, resolver=None):
    return PersistentMemoryAPI(MariaDBB9Store(Pool(conn)), resolver or TxResolver())


@pytest.mark.asyncio
async def test_persistent_create_builds_complete_atomic_bundle():
    conn=Conn(); api=make_api(conn)
    memory_id=await api.create_memory(auth(), ObjectKind.FACT, "payload", Visibility.USER_PRIVATE, user_id="user-1")
    assert memory_id
    assert conn.committed and not conn.rolled
    text="\n".join(sql for sql, _ in conn.cursor_obj.calls)
    for table in ("memory_objects", "memory_revisions", "memory_revision_payloads", "memory_provenance", "memory_events", "memory_projections"):
        assert table in text

@pytest.mark.asyncio
@pytest.mark.parametrize('actor_type', ['SERVICE','MODEL','AGENT'])
async def test_persistent_nonhuman_actor_cannot_verify(actor_type):
    conn,api=lifecycle_api(resolver=TxResolver(roles={'memory_admin'}))
    request=MutationAuthorizationRequest(ScopeContext(f'{actor_type.lower()}:x',actor_type,'user-1','tenant-1'), 'VERIFY', Visibility.USER_PRIVATE)
    with pytest.raises(AuthorizationDenied,match='human user'):
        await api.verify_memory(request,'m1',method='forged')
    assert conn.rolled and not conn.committed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [2, 3, 4, 5, 6])
async def test_each_canonical_write_failure_rolls_back(failure):
    conn=Conn(failure); api=make_api(conn)
    with pytest.raises(RuntimeError):
        await api.create_memory(auth(), ObjectKind.FACT, "payload", Visibility.USER_PRIVATE, user_id="user-1")
    assert conn.rolled and not conn.committed


@pytest.mark.asyncio
async def test_persistent_api_rejects_raw_authority_input():
    api=make_api(Conn())
    with pytest.raises(AuthorizationDenied):
        await api.create_memory({"admin": True}, ObjectKind.FACT, "x", Visibility.USER_PRIVATE, user_id="user-1")


@pytest.mark.asyncio
async def test_manually_constructed_authorization_context_grants_nothing():
    conn=Conn(); api=make_api(conn)
    forged=MutationAuthorizationContext(ScopeContext("user-1","USER","user-1","tenant-1"),
                                        "CREATE",Visibility.USER_PRIVATE,
                                        frozenset({"memory_admin"}),frozenset({"memory:project:write"}),"forged")
    with pytest.raises(AuthorizationDenied):
        await api.create_memory(forged,ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="user-1")
    assert not conn.committed


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,call", [
    ("CORRECT", lambda api, a: api.correct_memory(a,"m1","new")),
    ("SUPERSEDE", lambda api, a: api.supersede_memory(a,"m1","new")),
    ("EXPIRE", lambda api, a: api.expire_memory(a,"m1",reason="ttl")),
    ("TOMBSTONE", lambda api, a: api.tombstone_memory(a,"m1",reason="withdrawn")),
    ("CONTENT_PURGE", lambda api, a: api.content_purge(a,"m1")),
    ("RE_SCOPE", lambda api, a: api.re_scope_memory(a,"m1",visibility=Visibility.TENANT_SHARED)),
])
async def test_persistent_lifecycle_paths_use_locked_canonical_history(operation, call):
    conn,api=lifecycle_api()
    await call(api,auth(operation, Visibility.TENANT_SHARED if operation == "RE_SCOPE" else Visibility.USER_PRIVATE))
    assert conn.committed and not conn.rolled
    sql="\n".join(x for x,_ in conn.cursor_obj.calls)
    assert "FOR UPDATE" in sql and "memory_events" in sql and "memory_projections" in sql


@pytest.mark.asyncio
async def test_persistent_verify_requires_resolved_reviewer_and_records_worker_subject():
    conn,api=lifecycle_api()
    api._authorization_resolver=TxResolver(roles={"memory_reviewer"})
    await api.verify_memory(auth("VERIFY"),"m1",method="human")
    provenance_args=[args for sql,args in conn.cursor_obj.calls if "INSERT INTO memory_provenance" in sql][0]
    assert provenance_args[5:8] == ("user-1","USER","user-1")
    conn,api=lifecycle_api()
    with pytest.raises(AuthorizationDenied): await api.verify_memory(auth("VERIFY"),"m1",method="forged")
    assert conn.rolled


@pytest.mark.asyncio
@pytest.mark.parametrize("role,allowed", [("VIEWER",False),("CONTRIBUTOR",False),("REVIEWER",True),("OWNER",True)])
async def test_project_role_matrix_is_enforced_at_persistent_boundary(role, allowed):
    resolver=TxResolver(project_role=role)
    conn,api=lifecycle_api(project_id="project-a",resolver=resolver)
    call=api.verify_memory(auth("VERIFY",Visibility.PROJECT_SHARED,project_id="project-a"),"m1",method="human")
    if allowed:
        await call
        assert conn.committed
    else:
        with pytest.raises(AuthorizationDenied): await call
        assert conn.rolled


@pytest.mark.asyncio
async def test_project_a_authority_cannot_mutate_project_b_within_same_tenant():
    conn,api=lifecycle_api(project_id="project-a",resolver=TxResolver())
    with pytest.raises(ScopeDenied):
        await api.correct_memory(auth("CORRECT",Visibility.PROJECT_SHARED,project_id="project-b"),"m1","nope")
    assert conn.rolled


@pytest.mark.asyncio
async def test_rescope_cannot_silently_switch_project_namespace():
    conn,api=lifecycle_api(project_id="project-a",resolver=TxResolver())
    with pytest.raises(ScopeDenied):
        await api.re_scope_memory(auth("RE_SCOPE",Visibility.PROJECT_SHARED,project_id="project-a"),"m1",
                                  visibility=Visibility.PROJECT_SHARED,project_id="project-b")
    assert conn.rolled


@pytest.mark.asyncio
async def test_authority_is_revalidated_inside_mutation_transaction_after_revocation():
    resolver=TxResolver(denied=True)
    conn=Conn(); api=make_api(conn,resolver)
    with pytest.raises(AuthorizationDenied):
        await api.create_memory(auth(),ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="user-1")
    assert resolver.calls and resolver.calls[0][0] is conn.cursor_obj
    assert conn.rolled and not conn.committed


@pytest.mark.asyncio
async def test_projection_mismatch_marks_and_fails_closed():
    conn,api=lifecycle_api(reconciled=False)
    with pytest.raises(Exception): await api.correct_memory(auth("CORRECT"),"m1","new")
    assert conn.rolled
    assert any("reconciliation_required=TRUE" in sql for sql,_ in conn.cursor_obj.calls)


@pytest.mark.asyncio
async def test_persistent_legacy_binding_and_synthesis_are_transactional():
    legacy={'id':1,'fact_text':'original','user_id':'user-1','project_id':None,'superseded_by':None,'expires_at':None}
    owner={'user_id':'user-1','tenant_id':'tenant-1','status':'ACTIVE'}
    conn=ScriptConn([legacy,owner,None]); api=make_api(conn,TxResolver(roles={'memory_admin'},caps={'memory:admin'}))
    mid=await api.import_legacy_memory(auth("IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL),"facts","legacy","1",ObjectKind.FACT,"original")
    assert mid and conn.committed and any("memory_legacy_bindings" in sql for sql,_ in conn.cursor_obj.calls)
    source={"revision_id":"source-r","memory_id":"source-m","object_kind":"FACT","tenant_id":"tenant-1",
            "visibility":"USER_PRIVATE","user_id":"user-1","project_id":None,"lifecycle_state":"ACTIVE",
            "payload":"source","current_revision_id":"source-r","current_verification_state":True}
    conn=ScriptConn([source,dict(source,revision_id="source-r2",current_revision_id="source-r2",memory_id="source-m2")]); api=make_api(conn)
    worker_scope=ScopeContext("worker/extract","SERVICE","user-1","tenant-1",request_id="job-1")
    worker_auth=MutationAuthorizationRequest(worker_scope,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL)
    mid=await api.synthesize_memory(worker_auth,"derived",("source-r","source-r2"),provider="p",model="m",transformation_version="1")
    prov=[args for sql,args in conn.cursor_obj.calls if "INSERT INTO memory_provenance" in sql][0]
    assert mid and prov[5:8] == ("worker/extract","SERVICE","user-1") and prov[8:10] == ("p","m")


@pytest.mark.asyncio
async def test_aud003_authorized_project_read_uses_scoped_query_without_external_guard():
    row={"memory_id":"m1","object_kind":"FACT","tenant_id":"tenant-1","object_created_at":1,
         "revision_id":"r1","content_digest":"sha256:x","visibility":"PROJECT_SHARED","user_id":None,
         "project_id":"project-a","lifecycle_state":"ACTIVE","revision_created_at":1,"payload":"project fact",
         "provenance_status":"COMPLETE","prior_revision_id":None}
    conn=ScriptConn(many=[[row],[]]); resolver=TxResolver(project_role="VIEWER")
    reader=MariaDBB9Reader(Pool(conn),resolver)
    result=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.PROJECT_SHARED,project_id="project-a"))
    assert len(result)==1 and result[0].revision.payload == "project fact"
    assert any("r.project_id=%s" in sql for sql,_ in conn.cursor_obj.calls)
    assert any(args == ("tenant-1","USER_PRIVATE","user-1","project-a",20) for _,args in conn.cursor_obj.calls)
    with pytest.raises(AuthorizationDenied):
        await reader.retrieve(ScopeContext("user-1","USER","user-1","tenant-1",project_id="project-a"))


@pytest.mark.asyncio
async def test_aud001_persistent_project_synthesis_retains_source_scope():
    source={"revision_id":"source-r","memory_id":"source-m","object_kind":"FACT","tenant_id":"tenant-1",
            "visibility":"PROJECT_SHARED","user_id":None,"project_id":"project-a","lifecycle_state":"ACTIVE",
            "payload":"source","current_revision_id":"source-r","current_verification_state":True}
    conn=ScriptConn([source,dict(source,revision_id="source-r2",current_revision_id="source-r2",memory_id="source-m2")]); api=make_api(conn,TxResolver(project_role="CONTRIBUTOR"))
    await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL,project_id="project-a"),
                                "derived",("source-r","source-r2"),provider="p",model="m",transformation_version="1")
    revision_args=next(args for sql,args in conn.cursor_obj.calls if "INSERT INTO memory_revisions" in sql)
    assert revision_args[3] == "PROJECT_SHARED" and revision_args[5] == "project-a"


@pytest.mark.asyncio
async def test_aud001_project_service_synthesis_uses_resolved_service_policy():
    from jax.memory.scope_authority import ProjectScopeAuthorization
    class ServiceResolver:
        async def resolve_mutation_in_transaction(self, cur, scope, operation, visibility):
            resolved=replace(scope,project_authorization=ProjectScopeAuthorization(
                "project-a","tenant-1","user-1",None,None,"SERVICE_POLICY","ACTIVE",1.0))
            return MutationAuthorizationContext(resolved,operation,visibility,frozenset({"memory_service"}),
                                                frozenset({"memory:service:synthesize"}),"service-policy")
    source={"revision_id":"source-r","memory_id":"source-m","object_kind":"FACT","tenant_id":"tenant-1",
            "visibility":"PROJECT_SHARED","user_id":None,"project_id":"project-a","lifecycle_state":"ACTIVE",
            "payload":"source","current_revision_id":"source-r","current_verification_state":True}
    conn=ScriptConn([source,dict(source,revision_id="source-r2",current_revision_id="source-r2",memory_id="source-m2")]); api=make_api(conn,ServiceResolver())
    service=ScopeContext("service:memory-synthesis","SERVICE","user-1","tenant-1",project_id="project-a")
    await api.synthesize_memory(MutationAuthorizationRequest(service,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL),
                                "derived",("source-r","source-r2"),provider="p",model="m",transformation_version="1")
    assert conn.committed


@pytest.mark.asyncio
async def test_aud004_persistent_purge_erases_revision_column_payloads_and_vectors():
    conn,api=lifecycle_api()
    await api.content_purge(auth("CONTENT_PURGE"),"m1")
    sql="\n".join(sql for sql,_ in conn.cursor_obj.calls)
    assert "UPDATE memory_revisions SET payload=NULL WHERE memory_id=%s" in sql
    assert "DELETE p FROM memory_revision_payloads" in sql
    assert "DELETE e FROM embedding_generations" in sql


@pytest.mark.asyncio
async def test_aud005_persistent_legacy_query_and_insert_are_tenant_qualified():
    legacy={'id':1,'fact_text':'original','user_id':'user-1','project_id':None,'superseded_by':None,'expires_at':None}
    owner={'user_id':'user-1','tenant_id':'tenant-1','status':'ACTIVE'}
    conn=ScriptConn([legacy,owner,None]); api=make_api(conn,TxResolver(roles={'memory_admin'},caps={'memory:admin'}))
    await api.import_legacy_memory(auth("IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL),"facts","legacy","1",ObjectKind.FACT,"original")
    statements=[(sql,args) for sql,args in conn.cursor_obj.calls if "memory_legacy_bindings" in sql]
    assert all("tenant_id" in sql for sql,_ in statements)
    assert all("tenant-1" in args for _,args in statements)


def test_aud005_migration_derives_existing_tenant_from_bound_object():
    from pathlib import Path
    migration=(Path(__file__).resolve().parents[1]/"jax/memory/b9_migrations/004_tenant_legacy_binding.sql").read_text()
    assert "JOIN memory_objects" in migration
    assert "o.tenant_id" in migration
    assert "PRIMARY KEY (tenant_id, legacy_source_type, legacy_source_namespace, legacy_source_key)" in migration
    assert "UNIQUE KEY uq_memory_legacy_binding_tenant (tenant_id, legacy_source_type, legacy_source_namespace, legacy_source_key)" in migration


@pytest.mark.asyncio
async def test_aud006_persistent_synthesis_rejects_missing_source_revision():
    conn=ScriptConn([None]); api=make_api(conn)
    with pytest.raises(ScopeDenied):
        await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL),"derived",("missing","missing2"),
                                    provider="p",model="m",transformation_version="1")
    assert conn.rolled and not conn.committed


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"tenant_id":"tenant-2"}, {"current_revision_id":"newer"}, {"lifecycle_state":"EXPIRED"},
    {"payload":None}, {"object_kind":"SYNTHESIS"}, {"object_kind":"CONVERSATION"},
    {"project_id":"project-b"},
])
async def test_aud006_persistent_synthesis_rejects_ineligible_source_revision(change):
    row={"revision_id":"source-r","memory_id":"source-m","object_kind":"FACT","tenant_id":"tenant-1",
         "visibility":"PROJECT_SHARED","user_id":None,"project_id":"project-a","lifecycle_state":"ACTIVE",
         "payload":"source","current_revision_id":"source-r","current_verification_state":True}
    row.update(change)
    conn=ScriptConn([row]); api=make_api(conn,TxResolver(project_role="CONTRIBUTOR"))
    with pytest.raises(ScopeDenied):
        await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL,project_id="project-a"),
                                    "derived",("source-r","source-r2"),provider="p",model="m",transformation_version="1")
    assert conn.rolled and not conn.committed


@pytest.mark.asyncio
async def test_aud007_persistent_verify_after_expire_rejects_transition():
    conn,api=lifecycle_api()
    current,revision,event,stored=current_rows()
    current["lifecycle_state"]="EXPIRED"
    revision["lifecycle_state"]="EXPIRED"
    projection=_derive_projection("m1",[MemoryRevision("r1","m1","sha256:old",Visibility.USER_PRIVATE,
        "user-1",None,Lifecycle.EXPIRED,1,"old")],[MemoryEvent("e1","m1","r1",EventKind.CREATE,
        "user-1","user-1","test-authority",1,{},actor_type="USER")])
    stored.update(current_lifecycle_state="EXPIRED",current_verification_state=False,
                  canonical_history_digest=projection.canonical_history_digest)
    conn=ScriptConn([current,stored],[[revision],[event]])
    api=make_api(conn,TxResolver(roles={"memory_reviewer"}))
    with pytest.raises(ScopeDenied): await api.verify_memory(auth("VERIFY"),"m1",method="human")
    assert conn.rolled and not conn.committed


async def _b9_ci_test_pool(*, authority=False, legacy=False):
    """One-connection temporary schema, restricted to the isolated CI database."""
    import os
    import aiomysql
    from base_de_test import exigir_base_de_test
    if not os.getenv("JAX_DB_HOST"):
        pytest.skip("requires an isolated CI test database")
    database=exigir_base_de_test()
    pool=await aiomysql.create_pool(
        host=os.environ["JAX_DB_HOST"],port=int(os.getenv("JAX_DB_PORT","3306")),
        user=os.getenv("JAX_DB_USER","root"),password=os.getenv("JAX_DB_PASSWORD",""),db=database,
        minsize=1,maxsize=1,cursorclass=aiomysql.DictCursor,connect_timeout=5,
    )
    try:
        tables=(
            "CREATE TEMPORARY TABLE memory_objects (memory_id CHAR(36) PRIMARY KEY, object_kind VARCHAR(32), tenant_id VARCHAR(128), created_at DATETIME(6), legacy_source_type VARCHAR(64), legacy_source_namespace VARCHAR(255), legacy_source_key VARCHAR(255), UNIQUE KEY uq_memory_legacy_binding (legacy_source_type, legacy_source_namespace, legacy_source_key))",
            "CREATE TEMPORARY TABLE memory_revisions (revision_id CHAR(36) PRIMARY KEY, memory_id CHAR(36), content_digest CHAR(71), visibility VARCHAR(32), user_id VARCHAR(128), project_id VARCHAR(128), lifecycle_state VARCHAR(32), created_at DATETIME(6), payload LONGBLOB, provenance_status VARCHAR(64), prior_revision_id CHAR(36),tenant_id VARCHAR(128))",
            "CREATE TEMPORARY TABLE memory_revision_payloads (revision_id CHAR(36) PRIMARY KEY, payload LONGBLOB, purged_at DATETIME(6), purge_reason VARCHAR(255))",
            "CREATE TEMPORARY TABLE memory_provenance (provenance_id CHAR(36) PRIMARY KEY, revision_id CHAR(36), source_revisions JSON, transformation_id VARCHAR(128), transformation_version VARCHAR(64), actor_principal VARCHAR(255), actor_type VARCHAR(64), subject_user_id VARCHAR(128), provider VARCHAR(128), model VARCHAR(255), created_at DATETIME(6), limitations TEXT)",
            "CREATE TEMPORARY TABLE memory_events (event_id CHAR(36) PRIMARY KEY, memory_id CHAR(36), revision_id CHAR(36), event_kind VARCHAR(32), actor_principal VARCHAR(255), subject_user_id VARCHAR(128), authority_source VARCHAR(255), occurred_at DATETIME(6), details JSON, compensates_event_id CHAR(36), actor_type VARCHAR(64), delegation VARCHAR(255), calling_component VARCHAR(255), request_id VARCHAR(255), trace_id VARCHAR(255))",
            "CREATE TEMPORARY TABLE memory_projections (memory_id CHAR(36) PRIMARY KEY, current_revision_id CHAR(36), current_lifecycle_state VARCHAR(32), current_verification_state BOOLEAN, canonical_history_digest CHAR(71), reconciliation_required BOOLEAN)",
            "CREATE TEMPORARY TABLE embedding_generations (generation_id CHAR(36) PRIMARY KEY, revision_id CHAR(36), embedding_space_id CHAR(71), generated_at DATETIME(6), embedding_payload LONGBLOB)",
        )
        if authority:
            tables+=(
                "CREATE TEMPORARY TABLE jax_users (user_id VARCHAR(128), tenant_id VARCHAR(128), status VARCHAR(16), role VARCHAR(32), PRIMARY KEY (user_id,tenant_id))",
                "CREATE TEMPORARY TABLE jax_project_scope (project_id VARCHAR(128) PRIMARY KEY, tenant_id VARCHAR(128), status VARCHAR(16))",
                "CREATE TEMPORARY TABLE jax_project_membership (membership_id CHAR(36) PRIMARY KEY, project_id VARCHAR(128), tenant_id VARCHAR(128), user_id VARCHAR(128), project_role VARCHAR(16), status VARCHAR(16))",
            )
        if legacy:
            tables+=("CREATE TEMPORARY TABLE memory_legacy_bindings (legacy_source_type VARCHAR(64), legacy_source_namespace VARCHAR(255), legacy_source_key VARCHAR(255), memory_id CHAR(36), binding_state VARCHAR(32), created_at DATETIME(6), PRIMARY KEY (legacy_source_type, legacy_source_namespace, legacy_source_key))",)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for ddl in tables: await cur.execute(ddl)
        return pool
    except Exception:
        pool.close()
        await pool.wait_closed()
        raise


async def _seed_project_membership(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("INSERT INTO jax_users VALUES (%s,%s,'ACTIVE','member')",("user-1","tenant-1"))
            for project, role in (("project-a","CONTRIBUTOR"),("project-b","VIEWER")):
                await cur.execute("INSERT INTO jax_project_scope VALUES (%s,%s,'ACTIVE')",(project,"tenant-1"))
                await cur.execute("INSERT INTO jax_project_membership VALUES (%s,%s,%s,%s,%s,'ACTIVE')",
                                  ("membership-"+project,project,"tenant-1","user-1",role))
        await conn.commit()


@pytest.mark.asyncio
async def test_m1_real_mariadb_delete_and_adoption_serialise_on_legacy_fact():
    """M1 race regression against two real transactions, never a mock.

    The two orderings are forced at the same locks used in production:
    `MemoryDB.delete_fact` pauses after its locked source read, while
    `PersistentMemoryAPI.import_legacy_memory` pauses in `_write` after it
    has locked the source. The final state can be delete-without-binding or
    active-binding-with-source, never an orphaned ACTIVE binding.
    """
    from jax.memory.db import MemoryDB

    raw_pool=await _race_pool()
    api_pool=await _race_pool(dict_cursor=True)
    created_memory_ids=[]
    fact_ids=[]
    try:
        await _ensure_race_b9_tables(api_pool)
        await _ensure_race_legacy_source_tables(api_pool)
        async with api_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT user_id,tenant_id FROM jax_users WHERE user_id=990330001 AND status='ACTIVE'")
                owner=await cur.fetchone()
        if not owner:
            pytest.skip("test database has no active authoritative legacy owner")
        user_id, tenant_id=str(owner["user_id"]), str(owner["tenant_id"])

        async def create_fact(label):
            async with api_pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO facts (fact_uuid,fact_text,fact_type,confidence,is_verified,user_id) VALUES (UUID(),%s,'technical',0.1,FALSE,%s)",
                        (label, user_id),
                    )
                    fid=cur.lastrowid
                await conn.commit()
                fact_ids.append(fid)
                return fid

        def request():
            scope=ScopeContext(f"user:{user_id}", "USER", user_id, tenant_id)
            return MutationAuthorizationRequest(scope, "IMPORT_LEGACY", Visibility.SYSTEM_INTERNAL)

        resolver=TxResolver(roles={"memory_admin"}, caps={"memory:admin"})
        api=PersistentMemoryAPI(MariaDBB9Store(api_pool), resolver)
        deleter=MemoryDB()

        # Delete wins: the importer waits on the real source-row lock, then
        # sees no source and cannot create a binding.
        delete_label="M1 race delete first " + uuid.uuid4().hex
        delete_first=await create_fact(delete_label)
        delete_locked, release_delete=asyncio.Event(), asyncio.Event()
        async def after_delete_lock():
            delete_locked.set()
            await release_delete.wait()
        deleter.pool=_DeleteLockPool(raw_pool, delete_locked, release_delete)
        delete_task=asyncio.create_task(deleter.delete_fact(delete_first))
        await asyncio.wait_for(delete_locked.wait(), timeout=5)
        import_task=asyncio.create_task(api.import_legacy_memory(
            request(), "facts", "legacy", str(delete_first), ObjectKind.FACT,
            delete_label,
        ))
        release_delete.set()
        assert await asyncio.wait_for(delete_task, timeout=5) is True
        with pytest.raises(ScopeDenied):
            await asyncio.wait_for(import_task, timeout=5)
        async with api_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) AS n FROM memory_legacy_bindings "
                    "WHERE tenant_id=%s AND legacy_source_type='facts' "
                    "AND legacy_source_namespace='legacy' AND legacy_source_key=%s",
                    (tenant_id, str(delete_first)),
                )
                assert (await cur.fetchone())["n"] == 0

        # Adoption wins: `_write` is reached only while import still owns the
        # locked source row. Delete therefore sees ACTIVE binding after import
        # commits and fails closed.
        adopt_first=await create_fact("M1 race adopt first " + uuid.uuid4().hex)
        deleter.pool=raw_pool
        async with api_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT fact_text FROM facts WHERE id=%s", (adopt_first,))
                source=await cur.fetchone()
        write_started, release_write=asyncio.Event(), asyncio.Event()
        real_write=api._write
        async def paused_write(*args, **kwargs):
            write_started.set()
            await release_write.wait()
            return await real_write(*args, **kwargs)
        api._write=paused_write
        adopt_task=asyncio.create_task(api.import_legacy_memory(
            request(), "facts", "legacy", str(adopt_first), ObjectKind.FACT, source["fact_text"],
        ))
        await asyncio.wait_for(write_started.wait(), timeout=5)
        blocked_delete=asyncio.create_task(deleter.delete_fact(adopt_first))
        release_write.set()
        memory_id=await asyncio.wait_for(adopt_task, timeout=5)
        created_memory_ids.append(memory_id)
        assert await asyncio.wait_for(blocked_delete, timeout=5) is None
        async with api_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) AS n FROM facts WHERE id=%s", (adopt_first,))
                assert (await cur.fetchone())["n"] == 1
                await cur.execute("SELECT binding_state FROM memory_legacy_bindings WHERE tenant_id=%s AND legacy_source_type='facts' AND legacy_source_namespace='legacy' AND legacy_source_key=%s", (tenant_id, str(adopt_first)))
                assert (await cur.fetchone())["binding_state"] == "ACTIVE"
    finally:
        async with api_pool.acquire() as conn:
            async with conn.cursor() as cur:
                for memory_id in created_memory_ids:
                    await cur.execute("DELETE FROM memory_revision_payloads WHERE revision_id IN (SELECT revision_id FROM memory_revisions WHERE memory_id=%s)", (memory_id,))
                    await cur.execute("DELETE FROM memory_provenance WHERE revision_id IN (SELECT revision_id FROM memory_revisions WHERE memory_id=%s)", (memory_id,))
                    await cur.execute("DELETE FROM memory_events WHERE memory_id=%s", (memory_id,))
                    await cur.execute("DELETE FROM memory_projections WHERE memory_id=%s", (memory_id,))
                    await cur.execute("DELETE FROM memory_revisions WHERE memory_id=%s", (memory_id,))
                    await cur.execute("DELETE FROM memory_legacy_bindings WHERE memory_id=%s", (memory_id,))
                    await cur.execute("DELETE FROM memory_objects WHERE memory_id=%s", (memory_id,))
                if fact_ids:
                    await cur.execute("DELETE FROM facts WHERE id IN (" + ",".join(["%s"] * len(fact_ids)) + ")", fact_ids)
                await cur.execute("DELETE FROM jax_users WHERE user_id=990330001")
                await cur.execute("DELETE FROM jax_tenants WHERE tenant_id=990330001")
            await conn.commit()
        raw_pool.close(); api_pool.close()
        await raw_pool.wait_closed(); await api_pool.wait_closed()


@pytest.mark.asyncio
async def test_aud003_real_membership_authorizes_project_read_end_to_end():
    from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
    pool=await _b9_ci_test_pool(authority=True)
    try:
        await _seed_project_membership(pool)
        resolver=MariaDBScopeAuthorityResolver(pool)
        api=PersistentMemoryAPI(MariaDBB9Store(pool),resolver)
        reader=MariaDBB9Reader(pool,resolver)
        source=await api.create_memory(auth("CREATE",Visibility.PROJECT_SHARED,project_id="project-a"),
                                       ObjectKind.FACT,"project A",Visibility.PROJECT_SHARED,project_id="project-a")
        result=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.PROJECT_SHARED,project_id="project-a"))
        assert {item.identity.memory_id for item in result} == {source}
        with pytest.raises(AuthorizationDenied):
            await reader.retrieve(ScopeContext("user-1","USER","user-1","tenant-1",project_id="project-a"))
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_aud001_real_project_synthesis_is_invisible_to_project_b_and_tenant_only():
    from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
    pool=await _b9_ci_test_pool(authority=True)
    try:
        await _seed_project_membership(pool)
        resolver=MariaDBScopeAuthorityResolver(pool)
        api=PersistentMemoryAPI(MariaDBB9Store(pool),resolver)
        reader=MariaDBB9Reader(pool,resolver)
        source=await api.create_memory(auth("CREATE",Visibility.PROJECT_SHARED,project_id="project-a"),
                                       ObjectKind.FACT,"project A",Visibility.PROJECT_SHARED,project_id="project-a")
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE jax_project_membership SET project_role='REVIEWER' WHERE project_id='project-a'")
            await conn.commit()
        source2=await api.create_memory(auth("CREATE",Visibility.PROJECT_SHARED,project_id="project-a"),
                                       ObjectKind.FACT,"project A detail",Visibility.PROJECT_SHARED,project_id="project-a")
        await api.verify_memory(auth("VERIFY",Visibility.PROJECT_SHARED,project_id="project-a"),source,method="human test")
        await api.verify_memory(auth("VERIFY",Visibility.PROJECT_SHARED,project_id="project-a"),source2,method="human test")
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT current_revision_id FROM memory_projections WHERE memory_id IN (%s,%s)",(source,source2))
                revision_ids=tuple(row['current_revision_id'] for row in await cur.fetchall())
        derived=await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL,project_id="project-a"),
                                            "project summary",revision_ids,provider="p",model="m",transformation_version="1")
        project_a=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.PROJECT_SHARED,project_id="project-a"))
        project_b=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.PROJECT_SHARED,project_id="project-b"))
        tenant_only=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.TENANT_SHARED))
        assert derived in {item.identity.memory_id for item in project_a}
        assert derived not in {item.identity.memory_id for item in project_b}
        assert derived not in {item.identity.memory_id for item in tenant_only}
        assert not project_b and not tenant_only
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_persistent_synthesis_is_hidden_after_exact_source_revision_is_corrected():
    """K2: exercise read-time derivation invalidation through the MariaDB
    adapter, not only the reference in-memory implementation."""
    pool=await _b9_ci_test_pool(authority=True)
    try:
        resolver=TxResolver(roles={"memory_reviewer"})
        api=PersistentMemoryAPI(MariaDBB9Store(pool),resolver)
        reader=MariaDBB9Reader(pool)
        source=await api.create_memory(auth("CREATE"),ObjectKind.FACT,"source one",
                                       Visibility.USER_PRIVATE,user_id="user-1")
        source2=await api.create_memory(auth("CREATE"),ObjectKind.FACT,"source two",
                                        Visibility.USER_PRIVATE,user_id="user-1")
        await api.verify_memory(auth("VERIFY"),source,method="human test")
        await api.verify_memory(auth("VERIFY"),source2,method="human test")
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT current_revision_id FROM memory_projections WHERE memory_id IN (%s,%s)",
                    (source,source2),
                )
                revision_ids=tuple(row["current_revision_id"] for row in await cur.fetchall())
        derived=await api.synthesize_memory(
            auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL),"derived summary",revision_ids,
            provider="test",model="independent-test",transformation_version="f2p-k2-v1",
        )
        scope=auth("RETRIEVE").scope
        before=await reader.retrieve(scope)
        assert derived in {item.identity.memory_id for item in before}

        await api.correct_memory(auth("CORRECT"),source,"corrected source one")

        after=await reader.retrieve(scope)
        assert derived not in {item.identity.memory_id for item in after}
    finally:
        pool.close(); await pool.wait_closed()


def _aud005_migration_statements():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"jax/memory/b9_migrations/004_tenant_legacy_binding.sql"
    sql="\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("--"))
    return [statement for statement in sql.split(";") if statement.strip()]


async def _apply_aud005_migration(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for statement in _aud005_migration_statements():
                await cur.execute(statement)


@pytest.mark.asyncio
async def test_aud005_real_migration_qualifies_legacy_binding_by_tenant():
    pool=await _b9_ci_test_pool(legacy=True)
    old_id="00000000-0000-0000-0000-000000000010"
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO memory_objects (memory_id,object_kind,tenant_id,created_at,legacy_source_type,legacy_source_namespace,legacy_source_key) VALUES (%s,'FACT','tenant-1',NOW(6),'facts','legacy','same')",(old_id,))
                await cur.execute("INSERT INTO memory_legacy_bindings VALUES ('facts','legacy','same',%s,'ACTIVE',NOW(6))",(old_id,))
            await conn.commit()
        await _apply_aud005_migration(pool)
        # The migration's namespace uniqueness is independent of adoption's
        # new locked-source authority contract. Exercise the physical keys.
        second_id="00000000-0000-0000-0000-000000000011"
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO memory_objects (memory_id,object_kind,tenant_id,created_at,legacy_source_type,legacy_source_namespace,legacy_source_key) VALUES (%s,'FACT','tenant-2',NOW(6),'facts','legacy','same')",(second_id,))
                await cur.execute("INSERT INTO memory_legacy_bindings (tenant_id,legacy_source_type,legacy_source_namespace,legacy_source_key,memory_id,binding_state,created_at) VALUES ('tenant-2','facts','legacy','same',%s,'ACTIVE',NOW(6))",(second_id,))
                with pytest.raises(Exception):
                    await cur.execute("INSERT INTO memory_legacy_bindings (tenant_id,legacy_source_type,legacy_source_namespace,legacy_source_key,memory_id,binding_state,created_at) VALUES ('tenant-2','facts','legacy','same',%s,'ACTIVE',NOW(6))",(second_id,))
                await cur.execute("SELECT tenant_id,memory_id FROM memory_legacy_bindings WHERE legacy_source_key='same' ORDER BY tenant_id")
                assert [(r['tenant_id'],r['memory_id']) for r in await cur.fetchall()]==[('tenant-1',old_id),('tenant-2',second_id)]
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_aud005_real_migration_fails_closed_on_ambiguous_legacy_row():
    pool=await _b9_ci_test_pool(legacy=True)
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO memory_legacy_bindings VALUES ('facts','legacy','orphan',%s,'ACTIVE',NOW(6))",
                                  ("00000000-0000-0000-0000-000000000011",))
            await conn.commit()
            async with conn.cursor() as cur:
                with pytest.raises(Exception):
                    for statement in _aud005_migration_statements():
                        await cur.execute(statement)
                await cur.execute("SELECT tenant_id FROM memory_legacy_bindings WHERE legacy_source_key='orphan'")
                assert (await cur.fetchone())["tenant_id"] is None
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_aud004_content_purge_scrubs_physical_b9_storage():
    """The CI MariaDB service must leave no recoverable B9 payload or vector."""
    pool=await _b9_ci_test_pool()
    try:
        api=PersistentMemoryAPI(MariaDBB9Store(pool),TxResolver())
        memory_id=await api.create_memory(auth(),ObjectKind.FACT,"aud004 payload",Visibility.USER_PRIVATE,user_id="user-1")
        await api.correct_memory(auth("CORRECT"),memory_id,"aud004 revised payload")
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT revision_id FROM memory_revisions WHERE memory_id=%s ORDER BY created_at LIMIT 1",(memory_id,))
                revision_id=(await cur.fetchone())["revision_id"]
                await cur.execute("INSERT INTO embedding_generations VALUES (%s,%s,%s,NOW(6),%s)",
                                  ("00000000-0000-0000-0000-000000000001",revision_id,"space",b"derived vector"))
            await conn.commit()
        await api.content_purge(auth("CONTENT_PURGE"),memory_id)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT payload FROM memory_revisions WHERE memory_id=%s",(memory_id,))
                assert all(row["payload"] is None for row in await cur.fetchall())
                await cur.execute("SELECT p.payload FROM memory_revision_payloads p JOIN memory_revisions r ON r.revision_id=p.revision_id WHERE r.memory_id=%s",(memory_id,))
                assert all(row["payload"] is None for row in await cur.fetchall())
                await cur.execute("SELECT e.embedding_payload FROM embedding_generations e JOIN memory_revisions r ON r.revision_id=e.revision_id WHERE r.memory_id=%s",(memory_id,))
                assert not await cur.fetchall()
                await cur.execute("SELECT current_lifecycle_state FROM memory_projections WHERE memory_id=%s",(memory_id,))
                assert (await cur.fetchone())["current_lifecycle_state"] == "PURGED"
    finally:
        pool.close()
        await pool.wait_closed()


@pytest.mark.asyncio
async def test_reembed_and_compensation_append_without_rewriting_history():
    from jax.memory.b9 import EmbeddingSpaceIdentity
    conn,api=lifecycle_api()
    identity=EmbeddingSpaceIdentity("1","runtime","model","digest",2,"unit","cosine")
    generation=await api.reembed_memory(auth("RE_EMBED"),"m1",identity,(.1,.2))
    assert generation and any("embedding_generations" in sql for sql,_ in conn.cursor_obj.calls)
    current,revision,event,stored=current_rows()
    conn=ScriptConn([current,stored,{"event_id":"e1"}], [[revision],[event]])
    api=make_api(conn)
    marker=await api.compensating_event(auth("COMPENSATE"),"m1","e1",reason="undo")
    assert marker and any("COMPENSATE" in args for _,args in conn.cursor_obj.calls if args)
