from dataclasses import replace

import pytest

from jax.memory.b9 import (
    AuthorizationDenied, EventKind, Lifecycle, MemoryEvent, MemoryRevision,
    MutationAuthorizationContext, MutationAuthorizationRequest, ObjectKind, ScopeContext, ScopeDenied, Visibility, _derive_projection,
)
from jax.memory.b9_mariadb import MariaDBB9Store, PersistentMemoryAPI


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
    conn=Conn(); api=make_api(conn)
    worker_scope=ScopeContext("worker/extract","SERVICE","user-1","tenant-1",request_id="job-1")
    worker_auth=MutationAuthorizationRequest(worker_scope,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL)
    mid=await api.synthesize_memory(worker_auth,"derived",("source-r",),provider="p",model="m",transformation_version="1")
    prov=[args for sql,args in conn.cursor_obj.calls if "INSERT INTO memory_provenance" in sql][0]
    assert mid and prov[5:8] == ("worker/extract","SERVICE","user-1") and prov[8:10] == ("p","m")


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
