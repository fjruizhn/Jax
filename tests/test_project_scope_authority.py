import pathlib

import pytest

from jax.memory.b9 import (AuthorizationDenied, ScopeContext, ScopeDenied, Visibility)
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
from jax.memory.scope_authority import ProjectRole


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


def test_no_trusted_admin_resolver_double_reappears():
    """Section 3-bis (2026-09-25 plan): project-authority tests exercise the
    REAL transactional resolver against MariaDB (see
    `test_project_authority_mariadb.py`), never a resolver double that always
    answers "admin, allowed" the way the retired test double for
    `resolve_mutation_in_transaction` used to. A grep across the test tree,
    not an import check, so it still catches a reintroduction under a
    different call site.

    The forbidden name is assembled at runtime (never written out whole in
    this file) so this very check does not flag itself as an offender.
    """
    forbidden = "_Trusted" + "AdminResolver"
    here = pathlib.Path(__file__).resolve().parent
    offenders = [
        str(path) for path in here.glob("test_project_*.py")
        if forbidden in path.read_text(encoding="utf-8")
    ]
    assert not offenders, "forbidden resolver double reappeared in: " + repr(offenders)
