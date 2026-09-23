"""Current DB-backed tenant and explicit project authority for B9."""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .b9 import (AuthorizationDenied, MutationAuthorizationContext,
                 MutationAuthorizationRequest, ScopeContext, ScopeDenied, Visibility)


class ProjectRole(str, Enum):
    VIEWER="VIEWER"; CONTRIBUTOR="CONTRIBUTOR"; REVIEWER="REVIEWER"; OWNER="OWNER"


@dataclass(frozen=True)
class ProjectScopeAuthorization:
    project_id: str; tenant_id: str; subject_user_id: str; membership_id: str | None
    project_role: ProjectRole | None; membership_status: str; project_status: str; resolved_at: float
    source_identity_version: str | None = None


def _issue_project_authorization(project_id: str, tenant_id: str, subject_user_id: str, membership_id: str | None,
                                 project_role: ProjectRole | None, membership_status: str="ACTIVE", project_status: str="ACTIVE") -> ProjectScopeAuthorization:
    # This is provenance for a resolver decision, not a cryptographic or
    # object-identity capability.  Every execution boundary re-resolves from
    # current DB state; construction of this value grants nothing.
    return ProjectScopeAuthorization(project_id,tenant_id,subject_user_id,membership_id,project_role,membership_status,project_status,time.time())


def bind_worker_project_scope(*_: Any, **__: Any) -> ScopeContext:
    """Removed unsafe worker convenience API.

    A worker may not turn a subject's project-membership result into its own
    authority.  Use ``resolve_service_mutation`` with a fixed service policy.
    """
    raise ScopeDenied("workers must resolve a fixed service operation policy")


class MariaDBScopeAuthorityResolver:
    """Each resolution reads current `jax_users`, scope and membership state."""
    authority_source = "jax_users+jax_project_scope+jax_project_membership"
    _TENANT_AUTHORITY_SOURCE = "jax_users.active_tenant_role"
    _ADMIN_ROLES=frozenset({"admin","superadmin","super_admin"})
    _REVIEWER_ROLES=frozenset({"reviewer","memory_reviewer"})
    # Deliberately narrow, code-owned worker contract.  Subject identity is
    # provenance only and cannot add a user/project role to a service actor.
    DEFAULT_SERVICE_OPERATION_POLICY: Mapping[str, frozenset[str]] = {
        "service:memory-extraction": frozenset({"CREATE"}),
        "service:memory-synthesis": frozenset({"SYNTHESIZE"}),
        "service:embedding": frozenset({"RE_EMBED"}),
    }
    _SERVICE_COMPONENTS: Mapping[str, str] = {
        "service:memory-extraction": "memory-extraction",
        "service:memory-synthesis": "memory-synthesis",
        "service:embedding": "embedding",
    }

    def __init__(self, pool: Any, *, service_operation_policy: Mapping[str, frozenset[str]] | None=None):
        self._pool=pool
        self._service_operation_policy = dict(service_operation_policy or self.DEFAULT_SERVICE_OPERATION_POLICY)
    @staticmethod
    def _value(row: Any, name: str, pos: int) -> Any: return row.get(name) if isinstance(row,Mapping) else row[pos]
    @staticmethod
    def _active(value: Any) -> bool: return str(value or "").upper() == "ACTIVE"
    @staticmethod
    def _require_subject(scope: ScopeContext) -> None:
        if not scope.tenant_id or not scope.actor_principal or not scope.actor_type or not scope.subject_user_id:
            raise ScopeDenied("authenticated tenant subject is required")
        if scope.delegation: raise ScopeDenied("delegation authority is unavailable")
        if scope.actor_type == "USER" and scope.actor_principal not in {scope.subject_user_id,f"user:{scope.subject_user_id}"}:
            raise ScopeDenied("user actor and subject do not match")
    async def _one(self, sql: str, args: tuple[Any,...]) -> Any:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql,args); return await cur.fetchone()
    async def _tenant_user(self, scope: ScopeContext) -> tuple[str,str]:
        row=await self._one("SELECT tenant_id,status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s LIMIT 1",(scope.subject_user_id,scope.tenant_id))
        if not row: raise ScopeDenied("subject is not a tenant member")
        tenant,status,role=self._value(row,"tenant_id",0),self._value(row,"status",1),self._value(row,"role",2)
        if str(tenant)!=str(scope.tenant_id) or not self._active(status): raise ScopeDenied("tenant membership is inactive or mismatched")
        return str(status),str(role or "").lower()

    async def _tenant_user_cur(self, cur: Any, scope: ScopeContext) -> tuple[str, str]:
        await cur.execute("SELECT tenant_id,status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE", (scope.subject_user_id,scope.tenant_id))
        row=await cur.fetchone()
        if not row: raise ScopeDenied("subject is not a tenant member")
        tenant,status,role=self._value(row,"tenant_id",0),self._value(row,"status",1),self._value(row,"role",2)
        if str(tenant)!=str(scope.tenant_id) or not self._active(status):
            raise ScopeDenied("tenant membership is inactive or mismatched")
        return str(status),str(role or "").lower()

    async def _project_membership_cur(self, cur: Any, requested: ScopeContext) -> tuple[ScopeContext, ProjectRole]:
        await cur.execute("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s FOR UPDATE",(requested.project_id,))
        project=await cur.fetchone()
        if not project: raise ScopeDenied("PROJECT_SCOPE_UNBOUND")
        tenant,status=self._value(project,"tenant_id",0),str(self._value(project,"status",1) or "").upper()
        if str(tenant)!=str(requested.tenant_id): raise ScopeDenied("project tenant mismatch")
        if status!="ACTIVE": raise ScopeDenied("project scope is disabled")
        await cur.execute("SELECT membership_id,tenant_id,user_id,project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",(requested.project_id,requested.subject_user_id))
        member=await cur.fetchone()
        if not member: raise ScopeDenied("project membership is missing")
        mid,mtenant,muser,role,mstatus=(self._value(member,"membership_id",0),self._value(member,"tenant_id",1),self._value(member,"user_id",2),str(self._value(member,"project_role",3) or "").upper(),str(self._value(member,"status",4) or "").upper())
        if str(mtenant)!=str(tenant) or str(muser)!=str(requested.subject_user_id): raise ScopeDenied("project membership tenant or subject mismatch")
        if mstatus!="ACTIVE": raise ScopeDenied("project membership is revoked")
        try: project_role=ProjectRole(role)
        except ValueError as e: raise ScopeDenied("project membership role is invalid") from e
        auth=_issue_project_authorization(str(requested.project_id),str(tenant),str(requested.subject_user_id),str(mid),project_role,mstatus,status)
        return replace(requested,tenant_id=str(tenant),project_authorization=auth),project_role

    async def resolve_project_scope(self, requested: ScopeContext) -> ScopeContext:
        """Resolve active scope; no row means legacy PROJECT_SCOPE_UNBOUND."""
        self._require_subject(requested); await self._tenant_user(requested)
        if not requested.project_id: raise ScopeDenied("project id is required for project scope resolution")
        project=await self._one("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s LIMIT 1",(requested.project_id,))
        if not project: raise ScopeDenied("PROJECT_SCOPE_UNBOUND")
        tenant,status=self._value(project,"tenant_id",0),str(self._value(project,"status",1) or "").upper()
        if str(tenant)!=str(requested.tenant_id): raise ScopeDenied("project tenant mismatch")
        if status!="ACTIVE": raise ScopeDenied("project scope is disabled")
        member=await self._one("SELECT membership_id,tenant_id,user_id,project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s LIMIT 1",(requested.project_id,requested.subject_user_id))
        if not member: raise ScopeDenied("project membership is missing")
        mid,mtenant,muser,role,mstatus=(self._value(member,"membership_id",0),self._value(member,"tenant_id",1),self._value(member,"user_id",2),str(self._value(member,"project_role",3) or "").upper(),str(self._value(member,"status",4) or "").upper())
        if str(mtenant)!=str(tenant) or str(muser)!=str(requested.subject_user_id): raise ScopeDenied("project membership tenant or subject mismatch")
        if mstatus!="ACTIVE": raise ScopeDenied("project membership is revoked")
        try: project_role=ProjectRole(role)
        except ValueError as e: raise ScopeDenied("project membership role is invalid") from e
        auth=_issue_project_authorization(str(requested.project_id),str(tenant),str(requested.subject_user_id),str(mid),project_role,mstatus,status)
        return replace(requested,tenant_id=str(tenant),project_authorization=auth)

    async def resolve_scope(self, requested: ScopeContext) -> ScopeContext:
        self._require_subject(requested)
        if requested.actor_type == "SERVICE":
            raise ScopeDenied("service actors require an explicit service operation policy")
        if requested.project_id: return await self.resolve_project_scope(requested)
        await self._tenant_user(requested); return requested

    async def resolve_service_mutation(self, requested: ScopeContext, operation: str,
                                       target_visibility: Visibility) -> MutationAuthorizationContext:
        """Resolve a worker's narrowly defined operation without user-role inheritance.

        This is intentionally separate from user membership resolution.  It
        validates the originating subject and exact tenant/project scope as
        current DB state, but never reads the subject's membership or global
        role as authority for the service principal.
        """
        self._require_subject(requested)
        if requested.actor_type != "SERVICE":
            raise ScopeDenied("service operation policy requires a service actor")
        allowed = self._service_operation_policy.get(requested.actor_principal, frozenset())
        if operation not in allowed:
            raise AuthorizationDenied("service operation is not permitted by fixed policy")
        if requested.calling_component != self._SERVICE_COMPONENTS.get(requested.actor_principal):
            raise AuthorizationDenied("service calling component does not match fixed policy")
        await self._tenant_user(requested)
        resolved = requested
        if requested.project_id:
            project = await self._one("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s LIMIT 1",(requested.project_id,))
            if not project:
                raise ScopeDenied("PROJECT_SCOPE_UNBOUND")
            tenant,status=self._value(project,"tenant_id",0),str(self._value(project,"status",1) or "").upper()
            if str(tenant) != str(requested.tenant_id) or status != "ACTIVE":
                raise ScopeDenied("project scope is inactive or tenant-mismatched")
            # This has no membership/role.  It records the narrow service
            # decision so scope shape can be carried to persistence, where it
            # is revalidated in the mutation transaction.
            provenance=ProjectScopeAuthorization(str(requested.project_id),str(tenant),str(requested.subject_user_id),None,None,"SERVICE_POLICY",status,time.time())
            resolved=replace(requested,tenant_id=str(tenant),project_authorization=provenance)
        elif target_visibility is Visibility.PROJECT_SHARED:
            raise ScopeDenied("project shared service operation requires a project scope")
        caps=frozenset({f"memory:service:{operation.lower()}"})
        return MutationAuthorizationContext(resolved,operation,target_visibility,frozenset({"memory_service"}),caps,"jax-service-operation-policy:v1")

    async def resolve_mutation_in_transaction(self, cur: Any,
                                              request: MutationAuthorizationRequest | ScopeContext,
                                              operation: str | None=None,
                                              target_visibility: Visibility | None=None) -> MutationAuthorizationContext:
        """JIT authority decision under the caller's mutation transaction.

        ``MutationAuthorizationContext`` is intentionally not accepted.  The
        lock order is user -> project scope -> membership, matching state
        mutations, so a committed revoke/disable always wins if it serialized
        first.
        """
        if isinstance(request, MutationAuthorizationRequest):
            scope, operation, target_visibility = request.scope, request.operation, request.target_visibility
        elif isinstance(request, ScopeContext) and operation is not None and target_visibility is not None:
            scope = request
        else:
            raise AuthorizationDenied("untrusted mutation request is required")
        assert operation is not None and target_visibility is not None
        self._require_subject(scope)
        _, global_role = await self._tenant_user_cur(cur, scope)
        if scope.actor_type == "SERVICE":
            allowed=self._service_operation_policy.get(scope.actor_principal, frozenset())
            if operation not in allowed or scope.calling_component != self._SERVICE_COMPONENTS.get(scope.actor_principal):
                raise AuthorizationDenied("service operation is not permitted by fixed policy")
            resolved=scope
            if scope.project_id:
                await cur.execute("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s FOR UPDATE",(scope.project_id,))
                project=await cur.fetchone()
                if not project: raise ScopeDenied("PROJECT_SCOPE_UNBOUND")
                tenant,status=self._value(project,"tenant_id",0),str(self._value(project,"status",1) or "").upper()
                if str(tenant)!=str(scope.tenant_id) or status!="ACTIVE": raise ScopeDenied("project scope is inactive or tenant-mismatched")
                resolved=replace(scope,project_authorization=ProjectScopeAuthorization(str(scope.project_id),str(tenant),str(scope.subject_user_id),None,None,"SERVICE_POLICY",status,time.time()))
            elif target_visibility is Visibility.PROJECT_SHARED:
                raise ScopeDenied("project shared service operation requires a project scope")
            return MutationAuthorizationContext(resolved,operation,target_visibility,frozenset({"memory_service"}),frozenset({f"memory:service:{operation.lower()}"}),"jax-service-operation-policy:v1")
        if scope.project_id:
            resolved, project_role = await self._project_membership_cur(cur, scope)
            roles,caps=self._tenant_roles(global_role)
            proles,pcaps=self._project_roles(project_role); roles|=proles; caps|=pcaps
            if target_visibility is Visibility.PROJECT_SHARED:
                if operation=="VERIFY" and "memory:project:verify" not in caps: raise AuthorizationDenied("project verification requires REVIEWER")
                if operation in {"CREATE","CORRECT","RE_SCOPE","SYNTHESIZE"} and "memory:project:write" not in caps: raise AuthorizationDenied("project write requires CONTRIBUTOR")
            return MutationAuthorizationContext(resolved,operation,target_visibility,roles,caps,self.authority_source)
        roles,caps=self._tenant_roles(global_role)
        if target_visibility is Visibility.PROJECT_SHARED: raise ScopeDenied("project shared memory requires resolved project authorization")
        if operation=="VERIFY" and not roles.intersection({"memory_reviewer","memory_admin"}): raise AuthorizationDenied("verification requires resolved reviewer authority")
        return MutationAuthorizationContext(scope,operation,target_visibility,roles,caps,self._TENANT_AUTHORITY_SOURCE)

    @classmethod
    def _tenant_roles(cls, role: str) -> tuple[frozenset[str],frozenset[str]]:
        roles={"tenant_member"}; caps={"memory:tenant-scope"}
        if role in cls._ADMIN_ROLES: roles.update({"memory_admin","memory_reviewer"}); caps.update({"memory:admin","memory:verify"})
        elif role in cls._REVIEWER_ROLES: roles.add("memory_reviewer"); caps.add("memory:verify")
        return frozenset(roles),frozenset(caps)
    @staticmethod
    def _project_roles(role: ProjectRole) -> tuple[frozenset[str],frozenset[str]]:
        roles={"project_member",f"project_role:{role.value}"}; caps={"memory:project:read"}
        if role in {ProjectRole.CONTRIBUTOR,ProjectRole.REVIEWER,ProjectRole.OWNER}: caps.add("memory:project:write")
        if role in {ProjectRole.REVIEWER,ProjectRole.OWNER}: caps.add("memory:project:verify")
        if role is ProjectRole.OWNER: caps.add("project:membership:admin")
        return frozenset(roles),frozenset(caps)
    async def resolve_mutation(self, scope: ScopeContext, operation: str, target_visibility: Visibility) -> MutationAuthorizationContext:
        if scope.actor_type == "SERVICE":
            # Do not accidentally give a worker the subject's membership by
            # routing it through the ordinary user resolver.
            raise ScopeDenied("service actors must use resolve_service_mutation")
        resolved=await self.resolve_scope(scope); _,global_role=await self._tenant_user(resolved)
        roles,caps=self._tenant_roles(global_role); source=self._TENANT_AUTHORITY_SOURCE
        if resolved.project_id:
            auth=resolved.project_authorization
            assert isinstance(auth,ProjectScopeAuthorization)
            if auth.project_role is None:
                raise ScopeDenied("ordinary project authorization requires an active membership role")
            proles,pcaps=self._project_roles(auth.project_role); roles|=proles; caps|=pcaps; source=self.authority_source
            if target_visibility is Visibility.PROJECT_SHARED:
                if operation=="VERIFY" and "memory:project:verify" not in caps: raise AuthorizationDenied("project verification requires REVIEWER")
                if operation in {"CREATE","CORRECT","RE_SCOPE","SYNTHESIZE"} and "memory:project:write" not in caps: raise AuthorizationDenied("project write requires CONTRIBUTOR")
        elif target_visibility is Visibility.PROJECT_SHARED: raise ScopeDenied("project shared memory requires resolved project authorization")
        if operation=="VERIFY" and target_visibility is not Visibility.PROJECT_SHARED and not roles.intersection({"memory_reviewer","memory_admin"}): raise AuthorizationDenied("verification requires resolved reviewer authority")
        return MutationAuthorizationContext(resolved,operation,target_visibility,roles,caps,source)


ProjectScopeAuthorityResolver = MariaDBScopeAuthorityResolver
