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


def test_ningun_resolver_falso_se_usa_como_si_fuera_el_real():
    """Section 3-bis (2026-09-25 plan): project-authority tests exercise the
    REAL transactional resolver against MariaDB (see
    `test_project_authority_mariadb.py`), never a resolver double that
    always answers "admin, allowed" the way the retired test double for
    `resolve_mutation_in_transaction` used to.

    Ronda 4, MINOR 3 (auditor de escalón 3, 2026-09-26): el meta-test
    anterior (ronda 3, MINOR 7) buscaba una clase pasada como SEGUNDO
    ARGUMENTO de `ProjectAuthorityAdmin(...)` -- ese constructor ya no toma
    ningún resolver (ronda 4, MINOR 3 de este mismo cambio: parámetro
    muerto retirado), así que ese patrón dejó de proteger nada. Ahora
    detecta algo real: cualquier clase, en un archivo que MENCIONE
    `project_authority` (acota el barrido: no marca dobles legítimos de
    OTROS subsistemas, p.ej. `TxResolver` en `test_b9_persistent_api.py`,
    que no toca `project_authority` en absoluto), que define
    `resolve_project_read` o `resolve_mutation_in_transaction` -- y que
    ADEMÁS se usa (por `Name`, instanciada o el nombre de la clase mismo) en
    cualquier otro `Call` del mismo archivo. Detección por `Name` y por
    `Call`, no por la firma de un constructor puntual que puede cambiar.
    """
    import ast

    here = pathlib.Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in here.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "project_authority" not in text:
            continue
        try:
            tree = ast.parse(text, str(path))
        except SyntaxError:
            continue
        fake_resolver_classes = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and item.name in ("resolve_project_read", "resolve_mutation_in_transaction")
                   for item in node.body)
        }
        if not fake_resolver_classes:
            continue
        instance_names = {
            target.id
            for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            and node.value.func.id in fake_resolver_classes
            for target in node.targets if isinstance(target, ast.Name)
        }
        watched = fake_resolver_classes | instance_names
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            for arg in (*node.args, *(kw.value for kw in node.keywords)):
                if isinstance(arg, ast.Name) and arg.id in watched and arg.id != func_name:
                    offenders.append(f"{path.name}: {arg.id} used in {func_name or '<call>'}(...)")
    assert not offenders, "fake resolver used as if real: " + repr(offenders)
