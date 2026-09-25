from dataclasses import replace

import pytest

from jax.memory.b9 import (
    AuthorizationDenied, EventKind, Lifecycle, MemoryEvent, MemoryRevision,
    MutationAuthorizationContext, MutationAuthorizationRequest, ObjectKind, ScopeContext, ScopeDenied, Visibility, _derive_projection,
)
from jax.memory.b9_mariadb import MariaDBB9Store, MariaDBB9Reader, PersistentMemoryAPI


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
    conn=ScriptConn([None]); api=make_api(conn)
    mid=await api.import_legacy_memory(auth("IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL),"facts","legacy","k",ObjectKind.FACT,None)
    assert mid and conn.committed and any("memory_legacy_bindings" in sql for sql,_ in conn.cursor_obj.calls)
    source={"revision_id":"source-r","memory_id":"source-m","object_kind":"FACT","tenant_id":"tenant-1",
            "visibility":"USER_PRIVATE","user_id":"user-1","project_id":None,"lifecycle_state":"ACTIVE",
            "payload":"source","current_revision_id":"source-r"}
    conn=ScriptConn([source]); api=make_api(conn)
    worker_scope=ScopeContext("worker/extract","SERVICE","user-1","tenant-1",request_id="job-1")
    worker_auth=MutationAuthorizationRequest(worker_scope,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL)
    mid=await api.synthesize_memory(worker_auth,"derived",("source-r",),provider="p",model="m",transformation_version="1")
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
    assert any(args == ("tenant-1","user-1","project-a",20) for _,args in conn.cursor_obj.calls)
    with pytest.raises(AuthorizationDenied):
        await reader.retrieve(ScopeContext("user-1","USER","user-1","tenant-1",project_id="project-a"))


@pytest.mark.asyncio
async def test_aud001_persistent_project_synthesis_retains_source_scope():
    source={"revision_id":"source-r","memory_id":"source-m","object_kind":"FACT","tenant_id":"tenant-1",
            "visibility":"PROJECT_SHARED","user_id":None,"project_id":"project-a","lifecycle_state":"ACTIVE",
            "payload":"source","current_revision_id":"source-r"}
    conn=ScriptConn([source]); api=make_api(conn,TxResolver(project_role="CONTRIBUTOR"))
    await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL,project_id="project-a"),
                                "derived",("source-r",),provider="p",model="m",transformation_version="1")
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
            "payload":"source","current_revision_id":"source-r"}
    conn=ScriptConn([source]); api=make_api(conn,ServiceResolver())
    service=ScopeContext("service:memory-synthesis","SERVICE","user-1","tenant-1",project_id="project-a")
    await api.synthesize_memory(MutationAuthorizationRequest(service,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL),
                                "derived",("source-r",),provider="p",model="m",transformation_version="1")
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
    conn=ScriptConn([None]); api=make_api(conn)
    await api.import_legacy_memory(auth("IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL),"facts","legacy","k",ObjectKind.FACT,None)
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
        await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL),"derived",("missing",),
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
         "payload":"source","current_revision_id":"source-r"}
    row.update(change)
    conn=ScriptConn([row]); api=make_api(conn,TxResolver(project_role="CONTRIBUTOR"))
    with pytest.raises(ScopeDenied):
        await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL,project_id="project-a"),
                                    "derived",("source-r",),provider="p",model="m",transformation_version="1")
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
    database=os.getenv("JAX_DB_NAME","")
    if not database.endswith("_test") or not os.getenv("JAX_DB_HOST"):
        pytest.skip("requires an isolated CI test database")
    pool=await aiomysql.create_pool(
        host=os.environ["JAX_DB_HOST"],port=int(os.getenv("JAX_DB_PORT","3306")),
        user=os.getenv("JAX_DB_USER","root"),password=os.getenv("JAX_DB_PASSWORD",""),db=database,
        minsize=1,maxsize=1,cursorclass=aiomysql.DictCursor,
    )
    try:
        tables=(
            "CREATE TEMPORARY TABLE memory_objects (memory_id CHAR(36) PRIMARY KEY, object_kind VARCHAR(32), tenant_id VARCHAR(128), created_at DATETIME(6), legacy_source_type VARCHAR(64), legacy_source_namespace VARCHAR(255), legacy_source_key VARCHAR(255), UNIQUE KEY uq_memory_legacy_binding (legacy_source_type, legacy_source_namespace, legacy_source_key))",
            "CREATE TEMPORARY TABLE memory_revisions (revision_id CHAR(36) PRIMARY KEY, memory_id CHAR(36), content_digest CHAR(71), visibility VARCHAR(32), user_id VARCHAR(128), project_id VARCHAR(128), lifecycle_state VARCHAR(32), created_at DATETIME(6), payload LONGBLOB, provenance_status VARCHAR(64), prior_revision_id CHAR(36))",
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
                await cur.execute("SELECT current_revision_id FROM memory_projections WHERE memory_id=%s",(source,))
                revision_id=(await cur.fetchone())["current_revision_id"]
        derived=await api.synthesize_memory(auth("SYNTHESIZE",Visibility.SYSTEM_INTERNAL,project_id="project-a"),
                                            "project summary",(revision_id,),provider="p",model="m",transformation_version="1")
        project_a=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.PROJECT_SHARED,project_id="project-a"))
        project_b=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.PROJECT_SHARED,project_id="project-b"))
        tenant_only=await reader.retrieve_authorized(auth("RETRIEVE",Visibility.TENANT_SHARED))
        assert derived in {item.identity.memory_id for item in project_a}
        assert derived not in {item.identity.memory_id for item in project_b}
        assert derived not in {item.identity.memory_id for item in tenant_only}
        assert not project_b and not tenant_only
    finally:
        pool.close(); await pool.wait_closed()


async def _apply_aud005_migration(pool):
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"jax/memory/b9_migrations/004_tenant_legacy_binding.sql"
    sql="\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("--"))
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for statement in sql.split(";"):
                if statement.strip(): await cur.execute(statement)


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
        api=PersistentMemoryAPI(MariaDBB9Store(pool),TxResolver())
        first=await api.import_legacy_memory(auth("IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL),
                                             "facts","legacy","same",ObjectKind.FACT,"original")
        other_scope=ScopeContext("user-1","USER","user-1","tenant-2")
        other_auth=MutationAuthorizationRequest(other_scope,"IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL)
        second=await api.import_legacy_memory(other_auth,"facts","legacy","same",ObjectKind.FACT,"other")
        repeat=await api.import_legacy_memory(other_auth,"facts","legacy","same",ObjectKind.FACT,"other")
        assert first == old_id and second != first and repeat == second
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT tenant_id,memory_id FROM memory_legacy_bindings WHERE legacy_source_key='same' ORDER BY tenant_id")
                assert [(r["tenant_id"],r["memory_id"]) for r in await cur.fetchall()] == [
                    ("tenant-1",first),("tenant-2",second)]
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
        with pytest.raises(Exception):
            await _apply_aud005_migration(pool)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
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
