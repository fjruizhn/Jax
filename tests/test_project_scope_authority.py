import pytest

from jax.memory.b9 import (AuthorizationDenied, MutationAuthorizationContext,
                           MutationAuthorizationRequest, ScopeContext, ScopeDenied, Visibility)
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
from jax.memory.scope_authority import ProjectRole
from jax.memory.project_authority import ProjectAuthorityAdmin


class _TrustedAdminResolver:
    """Test double for the designated transaction resolver, not caller input."""
    async def resolve_mutation_in_transaction(self, _cur, request, operation, visibility):
        return MutationAuthorizationContext(request.scope, operation, visibility,
                                            frozenset({"memory_admin"}),
                                            frozenset({"memory:admin"}), "db-test")


def _admin_request(operation="GRANT_MEMBER"):
    return MutationAuthorizationRequest(
        ScopeContext("user:7", "USER", "7", "1", "9"), operation, Visibility.PROJECT_SHARED
    )


class Cursor:
    def __init__(self, rows): self.rows=list(rows); self.calls=[]
    async def __aenter__(self): return self
    async def __aexit__(self,*_): pass
    async def execute(self,sql,args): self.calls.append((sql,args))
    async def fetchone(self): return self.rows.pop(0)
class Conn:
    def __init__(self, rows): self.cursor_obj=Cursor(rows)
    def cursor(self): return self.cursor_obj
class Acquire:
    def __init__(self,conn): self.conn=conn
    async def __aenter__(self): return self.conn
    async def __aexit__(self,*_): pass
class Pool:
    def __init__(self,rows): self.conn=Conn(rows)
    def acquire(self): return Acquire(self.conn)

def requested(tenant="1",project="9",user="7"):
    return ScopeContext(f"user:{user}","USER",user,tenant,project)
def rows(*, project_status="ACTIVE", membership_status="ACTIVE", membership_tenant="1", role="CONTRIBUTOR"):
    return [{"tenant_id":1,"status":"ACTIVE","role":"admin"}, {"tenant_id":1,"status":project_status}, {"membership_id":"m","tenant_id":membership_tenant,"user_id":7,"project_role":role,"status":membership_status}]

@pytest.mark.asyncio
async def test_active_matching_project_membership_resolves_and_allows_write():
    resolver=MariaDBScopeAuthorityResolver(Pool(rows()))
    scope=await resolver.resolve_scope(requested())
    assert scope.project_authorization.project_role.value=="CONTRIBUTOR"
    auth=await MariaDBScopeAuthorityResolver(Pool(rows()+[{"tenant_id":1,"status":"ACTIVE","role":"admin"}])).resolve_mutation(requested(),"CREATE",Visibility.PROJECT_SHARED)
    assert "memory:project:write" in auth.resolved_capabilities

@pytest.mark.asyncio
@pytest.mark.parametrize("data,match", [([{"tenant_id":1,"status":"ACTIVE","role":"admin"},None],"UNBOUND"), (rows(membership_status="REVOKED"),"revoked"), (rows(project_status="DISABLED"),"disabled"), (rows(membership_tenant="2"),"tenant")])
async def test_project_scope_mismatches_fail_closed(data,match):
    with pytest.raises(ScopeDenied,match=match): await MariaDBScopeAuthorityResolver(Pool(data)).resolve_scope(requested())

@pytest.mark.asyncio
async def test_viewer_cannot_write_or_verify_even_when_global_role_is_admin():
    with pytest.raises(AuthorizationDenied,match="CONTRIBUTOR"):
        await MariaDBScopeAuthorityResolver(Pool(rows(role="VIEWER")+[{"tenant_id":1,"status":"ACTIVE","role":"admin"}])).resolve_mutation(requested(),"CREATE",Visibility.PROJECT_SHARED)
    with pytest.raises(AuthorizationDenied,match="REVIEWER"):
        await MariaDBScopeAuthorityResolver(Pool(rows(role="VIEWER")+[{"tenant_id":1,"status":"ACTIVE","role":"admin"}])).resolve_mutation(requested(),"VERIFY",Visibility.PROJECT_SHARED)

@pytest.mark.asyncio
async def test_requested_project_id_does_not_bypass_membership_lookup():
    pool=Pool([{"tenant_id":1,"status":"ACTIVE","role":"admin"},{"tenant_id":1,"status":"ACTIVE"},None])
    with pytest.raises(ScopeDenied,match="missing"): await MariaDBScopeAuthorityResolver(pool).resolve_scope(requested())


@pytest.mark.asyncio
async def test_grant_rejects_target_user_from_another_tenant_before_insert():
    class Store:
        async def mutation(self, operation):
            class Cur:
                def __init__(self): self.rows=[{"status":"ACTIVE","role":"admin"},{"tenant_id":1,"status":"ACTIVE"},{"tenant_id":2,"status":"ACTIVE"}]; self.calls=[]
                async def execute(self, sql, args): self.calls.append((sql,args))
                async def fetchone(self): return self.rows.pop(0)
            self.cur=Cur()
            return await operation(self.cur)
    with pytest.raises(ScopeDenied,match="target user tenant"):
        await ProjectAuthorityAdmin(Store(), _TrustedAdminResolver()).grant_member(
            _admin_request(),9,1,8,ProjectRole.VIEWER)


def test_bind_legacy_scope_is_global_admin_only_idempotent_and_audited():
    class Store:
        def __init__(self, rows): self.rows = rows; self.calls = []
        async def mutation(self, operation):
            store = self
            class Cur:
                async def execute(self, sql, args): store.calls.append((sql, args))
                async def fetchone(self): return store.rows.pop(0)
            return await operation(Cur())

    async def run():
        # The pre-existing global role is DB data; the caller does not pass an
        # is_admin flag.  A project OWNER cannot satisfy bootstrap=True.
        request = _admin_request("BIND_LEGACY_PROJECT_SCOPE")
        created = Store([
            {"status":"ACTIVE", "role":"admin"}, {"id": 9}, None, {"tenant_id": 1},
        ])
        assert await ProjectAuthorityAdmin(created, _TrustedAdminResolver()).bind_legacy_project_scope(request, 9, 1) is True
        sql = "\n".join(query for query, _ in created.calls)
        assert "INSERT INTO jax_project_scope" in sql
        assert "BIND_LEGACY_PROJECT_SCOPE" in str(created.calls)

        repeat = Store([
            {"status":"ACTIVE", "role":"admin"}, {"id": 9}, {"tenant_id": 1, "status":"ACTIVE"},
        ])
        assert await ProjectAuthorityAdmin(repeat, _TrustedAdminResolver()).bind_legacy_project_scope(request, 9, 1) is False
        assert "BIND_LEGACY_PROJECT_SCOPE_NOOP" in str(repeat.calls)

        conflict = Store([
            {"status":"ACTIVE", "role":"admin"}, {"id": 9}, {"tenant_id": 2, "status":"ACTIVE"},
        ])
        with pytest.raises(ScopeDenied, match="conflicting"):
            await ProjectAuthorityAdmin(conflict, _TrustedAdminResolver()).bind_legacy_project_scope(request, 9, 1)
    import asyncio
    asyncio.run(run())
