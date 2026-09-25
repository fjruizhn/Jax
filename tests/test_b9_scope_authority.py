import pytest

from jax.memory.b9 import (AuthorizationDenied, MutationAuthorizationContext,
                           MutationAuthorizationRequest, ScopeContext,
                           ScopeDenied, Visibility)
from jax.memory.scope_authority import (MariaDBScopeAuthorityResolver,
                                        ProjectRole, ProjectScopeAuthorization)


class Cursor:
    def __init__(self, row): self.row, self.calls = row, []
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass
    async def execute(self, sql, args): self.calls.append((sql, args))
    async def fetchone(self): return self.row


class Conn:
    def __init__(self, row): self.cursor_obj = Cursor(row)
    def cursor(self): return self.cursor_obj


class Acquire:
    def __init__(self, conn): self.conn = conn
    async def __aenter__(self): return self.conn
    async def __aexit__(self, *_): pass


class Pool:
    def __init__(self, row): self.conn = Conn(row)
    def acquire(self): return Acquire(self.conn)


class SequentialCursor:
    def __init__(self, rows): self.rows=list(rows); self.calls=[]
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass
    async def execute(self, sql, args): self.calls.append((sql,args))
    async def fetchone(self): return self.rows.pop(0)
class SequentialConn:
    def __init__(self, rows): self.cursor_obj=SequentialCursor(rows)
    def cursor(self): return self.cursor_obj
class SequentialPool:
    def __init__(self, rows): self.conn=SequentialConn(rows)
    def acquire(self): return Acquire(self.conn)


def scope(*, actor="u1", actor_type="USER", subject="u1", tenant="t1", project=None, delegation=None):
    return ScopeContext(actor, actor_type, subject, tenant, project, delegation)


@pytest.mark.asyncio
async def test_active_jax_user_resolves_tenant_scope_and_authorization_context():
    pool = Pool({"tenant_id": "t1", "status": "active", "role": "admin"})
    resolver = MariaDBScopeAuthorityResolver(pool)
    actual = await resolver.resolve_scope(scope())
    auth = await resolver.resolve_mutation(actual, "VERIFY", Visibility.USER_PRIVATE)
    assert actual.tenant_id == "t1"
    assert {"tenant_member", "memory_admin", "memory_reviewer"} <= auth.resolved_roles
    assert auth.authority_source == "jax_users.active_tenant_role"
    sql, args = pool.conn.cursor_obj.calls[0]
    assert "FROM jax_users" in sql and args == ("u1", "t1")


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [None, {"tenant_id": "t1", "status": "locked", "role": "admin"}, {"tenant_id": "t2", "status": "active", "role": "admin"}])
async def test_missing_inactive_or_wrong_tenant_membership_fails_closed(row):
    with pytest.raises(ScopeDenied):
        await MariaDBScopeAuthorityResolver(Pool(row)).resolve_scope(scope())


@pytest.mark.asyncio
async def test_project_claim_and_delegation_fail_closed_without_authoritative_sources():
    resolver = MariaDBScopeAuthorityResolver(Pool({"tenant_id": "t1", "status": "active", "role": "admin"}))
    with pytest.raises(ScopeDenied, match="project membership"):
        await resolver.resolve_scope(scope(project="p1"))
    with pytest.raises(ScopeDenied, match="delegation"):
        await resolver.resolve_scope(scope(actor="service:worker", actor_type="SERVICE", delegation="on-behalf-of"))


@pytest.mark.asyncio
async def test_user_cannot_select_a_different_subject():
    resolver = MariaDBScopeAuthorityResolver(Pool({"tenant_id": "t1", "status": "active", "role": "admin"}))
    with pytest.raises(ScopeDenied, match="actor and subject"):
        await resolver.resolve_scope(scope(subject="u2"))


@pytest.mark.asyncio
async def test_reviewer_authority_comes_from_persisted_role_not_caller_claim():
    resolver = MariaDBScopeAuthorityResolver(Pool({"tenant_id": "t1", "status": "active", "role": "member"}))
    with pytest.raises(AuthorizationDenied):
        await resolver.resolve_mutation(scope(), "VERIFY", Visibility.USER_PRIVATE)


@pytest.mark.asyncio
async def test_service_actor_preserves_subject_without_inheriting_subject_role():
    resolver = MariaDBScopeAuthorityResolver(Pool(("t1", "active", "reviewer")))
    worker_scope = scope(actor="service:memory-extraction", actor_type="SERVICE")
    worker_scope = ScopeContext(**{**worker_scope.__dict__, "calling_component": "memory-extraction"})
    auth = await resolver.resolve_service_mutation(worker_scope, "CREATE", Visibility.SYSTEM_INTERNAL)
    assert auth.scope.actor_principal == "service:memory-extraction"
    assert auth.scope.subject_user_id == "u1"
    assert "memory_reviewer" not in auth.resolved_roles
    assert "memory:verify" not in auth.resolved_capabilities


@pytest.mark.asyncio
async def test_service_cannot_copy_subject_membership_or_global_reviewer_authority():
    # The third row represents an OWNER membership.  Service resolution must
    # not query it and therefore cannot inherit it by copying subject_user_id.
    resolver = MariaDBScopeAuthorityResolver(SequentialPool([
        {"tenant_id": "t1", "status": "ACTIVE", "role": "admin"},
        {"tenant_id": "t1", "status": "ACTIVE"},
        {"membership_id": "m1", "tenant_id": "t1", "user_id": "u1", "project_role": "OWNER", "status": "ACTIVE"},
    ]))
    service = ScopeContext("service:memory-extraction", "SERVICE", "u1", "t1", "p1", calling_component="memory-extraction")
    auth = await resolver.resolve_service_mutation(service, "CREATE", Visibility.PROJECT_SHARED)
    assert auth.resolved_roles == frozenset({"memory_service"})
    assert "project:membership:admin" not in auth.resolved_capabilities
    assert "memory:project:verify" not in auth.resolved_capabilities
    with pytest.raises(ScopeDenied, match="must use"):
        await resolver.resolve_mutation(service, "VERIFY", Visibility.PROJECT_SHARED)


@pytest.mark.asyncio
async def test_fabricated_project_authorization_is_not_authority():
    # A constructor call cannot replace the membership lookup; it is ignored
    # and current DB state wins.
    fake = ProjectScopeAuthorization("p1", "t1", "u1", "forged", ProjectRole.OWNER,
                                     "ACTIVE", "ACTIVE", 0)
    requested = ScopeContext("u1", "USER", "u1", "t1", "p1", project_authorization=fake)
    resolver = MariaDBScopeAuthorityResolver(SequentialPool([
        {"tenant_id": "t1", "status": "ACTIVE", "role": "member"},
        {"tenant_id": "t1", "status": "ACTIVE"},
        None,
    ]))
    with pytest.raises(ScopeDenied, match="membership is missing"):
        await resolver.resolve_mutation(requested, "CREATE", Visibility.PROJECT_SHARED)


@pytest.mark.asyncio
async def test_transaction_resolver_rejects_constructed_authorization_context():
    resolver = MariaDBScopeAuthorityResolver(Pool(None))
    fabricated = MutationAuthorizationContext(scope(), "VERIFY", Visibility.USER_PRIVATE,
                                              frozenset({"memory_admin"}), frozenset({"memory:admin"}), "forged")
    class Cur:
        async def execute(self, *_): pass
        async def fetchone(self): return None
    with pytest.raises(AuthorizationDenied, match="untrusted mutation request"):
        await resolver.resolve_mutation_in_transaction(Cur(), fabricated)


@pytest.mark.asyncio
async def test_transaction_resolver_rechecks_current_membership_with_locks():
    class Cur:
        def __init__(self): self.rows = [
            {"tenant_id": "t1", "status": "ACTIVE", "role": "member"},
            {"tenant_id": "t1", "status": "ACTIVE"},
            {"membership_id": "m", "tenant_id": "t1", "user_id": "u1", "project_role": "REVIEWER", "status": "ACTIVE"},
        ]; self.calls=[]
        async def execute(self, sql, args): self.calls.append(sql)
        async def fetchone(self): return self.rows.pop(0)
    cur=Cur(); request=MutationAuthorizationRequest(scope(project="p1"), "VERIFY", Visibility.PROJECT_SHARED)
    auth=await MariaDBScopeAuthorityResolver(Pool(None)).resolve_mutation_in_transaction(cur, request)
    assert "memory:project:verify" in auth.resolved_capabilities
    assert all("FOR UPDATE" in sql for sql in cur.calls)
