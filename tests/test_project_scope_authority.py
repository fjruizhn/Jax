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
    `resolve_mutation_in_transaction` used to.

    Ronda 3, MINOR 7 (auditor de escalón 3, 2026-09-26): esto buscaba un
    nombre EXACTO en un subconjunto de archivos (`test_project_*.py`). Un
    double con otro nombre, o pasado a `ProjectAuthorityAdmin` desde
    cualquier otro archivo de `tests/`, no lo hubiera detectado. Ahora es
    un escaneo AST de TODO `tests/`, por patrón: cualquier clase que define
    un método `resolve_*` Y que además aparece como argumento en una
    llamada a `ProjectAuthorityAdmin(...)` en el MISMO archivo -- sin
    importar cómo se llame la clase. Esto no marca dobles legítimos de
    OTROS subsistemas (p.ej. `TxResolver` en `test_b9_persistent_api.py`,
    que implementa `resolve_mutation_in_transaction` pero nunca se pasa a
    `ProjectAuthorityAdmin`): el patrón es la combinación de las dos cosas,
    no el nombre del método por sí solo.
    """
    import ast

    here = pathlib.Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in here.glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError:
            continue
        resolver_like_classes = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("resolve_")
                   for item in node.body)
        }
        if not resolver_like_classes:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "ProjectAuthorityAdmin"):
                continue
            for arg in (*node.args, *(kw.value for kw in node.keywords)):
                if isinstance(arg, ast.Name) and arg.id in resolver_like_classes:
                    offenders.append(f"{path.name}: {arg.id} passed to ProjectAuthorityAdmin(...)")
    assert not offenders, "resolver double passed to ProjectAuthorityAdmin: " + repr(offenders)
