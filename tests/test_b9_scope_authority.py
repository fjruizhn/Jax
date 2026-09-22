import pytest

from jax.memory.b9 import AuthorizationDenied, ScopeContext, ScopeDenied, Visibility
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver


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
async def test_service_actor_preserves_distinct_subject_after_membership_check():
    resolver = MariaDBScopeAuthorityResolver(Pool(("t1", "active", "reviewer")))
    auth = await resolver.resolve_mutation(
        scope(actor="service:memory-worker", actor_type="SERVICE"), "CREATE", Visibility.SYSTEM_INTERNAL
    )
    assert auth.scope.actor_principal == "service:memory-worker"
    assert auth.scope.subject_user_id == "u1"
    assert "memory_reviewer" in auth.resolved_roles
