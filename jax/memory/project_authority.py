"""Transactional mutation boundary for project scope/membership authority."""
from __future__ import annotations

import time
from typing import Any

from .b9 import (AuthorizationDenied, MutationAuthorizationContext,
                 MutationAuthorizationRequest, ScopeDenied, Visibility, _uuid7)
from .scope_authority import ProjectRole


class ProjectAuthorityAdmin:
    """Only supported API for scope/member state changes and audit events.

    Its caller supplies a *resolved* mutation context, never an is_owner flag
    or caller role.  Event rows are inserted in the same transaction as state.
    """
    def __init__(self, store: Any, authorization_resolver: Any):
        self._store, self._authorization_resolver = store, authorization_resolver
    @staticmethod
    def _admin_shape(auth: MutationAuthorizationContext, project_id: int, *, bootstrap: bool=False) -> None:
        """Check a freshly resolver-issued decision, never caller input."""
        caps=auth.resolved_capabilities
        if "memory:admin" in caps: return
        if not bootstrap and str(auth.scope.project_id)==str(project_id) and "project:membership:admin" in caps: return
        raise AuthorizationDenied("project membership administration requires OWNER or global admin")
    async def _resolve_admin(self, cur: Any, request: MutationAuthorizationRequest,
                             project_id: int, *, expected_operation: str,
                             bootstrap: bool=False) -> MutationAuthorizationContext:
        """Recheck the administrative actor under the same transaction locks.

        A stale resolved context cannot grant a member mutation after an OWNER
        has been revoked.  The preliminary capability shape check is retained
        only to reject malformed contexts before database work.
        """
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("membership mutation requires authorization request")
        if (request.operation != expected_operation
                or request.target_visibility is not Visibility.PROJECT_SHARED):
            raise AuthorizationDenied("membership mutation operation request is invalid")
        resolve = getattr(self._authorization_resolver, "resolve_mutation_in_transaction", None)
        if resolve is None:
            raise AuthorizationDenied("membership mutation requires transaction authority resolver")
        auth = await resolve(cur, request, request.operation, request.target_visibility)
        if not isinstance(auth, MutationAuthorizationContext):
            raise AuthorizationDenied("authority resolver returned invalid decision")
        ProjectAuthorityAdmin._admin_shape(auth,project_id,bootstrap=bootstrap)
        scope=auth.scope
        await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",(scope.subject_user_id,scope.tenant_id))
        user=await cur.fetchone()
        if not user: raise AuthorizationDenied("administrative subject is not an active tenant user")
        status,role=(user[0],user[1]) if not isinstance(user,dict) else (user["status"],user["role"])
        if str(status).upper()!="ACTIVE": raise AuthorizationDenied("administrative subject is inactive")
        if str(role or "").lower() in {"admin","superadmin","super_admin"}: return auth
        if bootstrap: raise AuthorizationDenied("project scope administration requires global admin")
        await cur.execute("SELECT project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",(project_id,scope.subject_user_id))
        member=await cur.fetchone()
        if not member: raise AuthorizationDenied("administrative owner membership is missing")
        mrole,mstatus=(member[0],member[1]) if not isinstance(member,dict) else (member["project_role"],member["status"])
        if str(mstatus).upper()!="ACTIVE" or str(mrole).upper()!="OWNER": raise AuthorizationDenied("membership administration requires active OWNER")
        return auth
    @staticmethod
    async def _event(cur: Any, auth: MutationAuthorizationContext, operation: str, project_id: int, target_user_id: int | None, tenant_id: int, old_role: str | None, new_role: str | None, old_status: str | None, new_status: str | None) -> None:
        s=auth.scope
        await cur.execute("INSERT INTO jax_project_membership_event (event_id,operation,actor_principal,actor_type,project_id,target_user_id,tenant_id,old_project_role,new_project_role,old_status,new_status,occurred_at,request_id,trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s)",(_uuid7(),operation,s.actor_principal,s.actor_type,project_id,target_user_id,tenant_id,old_role,new_role,old_status,new_status,time.time(),s.request_id,s.trace_id))
    async def grant_member(self, request: MutationAuthorizationRequest, project_id: int, tenant_id: int, user_id: int, role: ProjectRole) -> None:
        async def op(cur: Any) -> None:
            auth=await self._resolve_admin(cur,request,project_id,expected_operation="GRANT_MEMBER")
            await cur.execute("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s FOR UPDATE",(project_id,)); scope=await cur.fetchone()
            if not scope: raise ScopeDenied("PROJECT_SCOPE_UNBOUND")
            stenant=scope[0] if not isinstance(scope,dict) else scope["tenant_id"]
            if int(stenant)!=int(tenant_id): raise ScopeDenied("project tenant mismatch")
            # Do not create an inconsistent membership and rely on the
            # resolver to reject it later.  The target identity is current
            # DB-backed state and is locked with the new membership write.
            await cur.execute("SELECT tenant_id,status FROM jax_users WHERE user_id=%s FOR UPDATE",(user_id,))
            target=await cur.fetchone()
            if not target: raise ScopeDenied("target user is missing")
            target_tenant,target_status=(target[0],target[1]) if not isinstance(target,dict) else (target["tenant_id"],target["status"])
            if int(target_tenant)!=int(tenant_id) or str(target_status).upper()!="ACTIVE":
                raise ScopeDenied("target user tenant is inactive or mismatched")
            await cur.execute("SELECT membership_id,project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",(project_id,user_id)); old=await cur.fetchone()
            if old: raise ScopeDenied("membership already exists; use change or grant after revoke")
            await cur.execute("INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,status,created_at,created_by,updated_at) VALUES (%s,%s,%s,%s,%s,'ACTIVE',NOW(6),%s,NOW(6))",(_uuid7(),project_id,tenant_id,user_id,role.value,auth.scope.actor_principal))
            await self._event(cur,auth,"GRANT_MEMBER",project_id,user_id,tenant_id,None,role.value,None,"ACTIVE")
        await self._store.mutation(op)
    async def change_project_role(self, request: MutationAuthorizationRequest, project_id: int, user_id: int, role: ProjectRole) -> None:
        async def op(cur: Any) -> None:
            auth=await self._resolve_admin(cur,request,project_id,expected_operation="CHANGE_PROJECT_ROLE")
            await cur.execute("SELECT membership_id,tenant_id,project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",(project_id,user_id)); old=await cur.fetchone()
            if not old: raise ScopeDenied("project membership is missing")
            vals=(old[1],old[2],old[3]) if not isinstance(old,dict) else (old["tenant_id"],old["project_role"],old["status"])
            await cur.execute("UPDATE jax_project_membership SET project_role=%s,version=version+1,updated_at=NOW(6) WHERE project_id=%s AND user_id=%s",(role.value,project_id,user_id))
            await self._event(cur,auth,"CHANGE_PROJECT_ROLE",project_id,user_id,int(vals[0]),vals[1],role.value,vals[2],vals[2])
        await self._store.mutation(op)
    async def revoke_member(self, request: MutationAuthorizationRequest, project_id: int, user_id: int) -> None:
        async def op(cur: Any) -> None:
            auth=await self._resolve_admin(cur,request,project_id,expected_operation="REVOKE_MEMBER")
            await cur.execute("SELECT tenant_id,project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",(project_id,user_id)); old=await cur.fetchone()
            if not old: raise ScopeDenied("project membership is missing")
            tenant,role,status=(old[0],old[1],old[2]) if not isinstance(old,dict) else (old["tenant_id"],old["project_role"],old["status"])
            await cur.execute("UPDATE jax_project_membership SET status='REVOKED',version=version+1,updated_at=NOW(6) WHERE project_id=%s AND user_id=%s",(project_id,user_id))
            await self._event(cur,auth,"REVOKE_MEMBER",project_id,user_id,int(tenant),role,role,status,"REVOKED")
        await self._store.mutation(op)

    async def set_project_scope_status(self, request: MutationAuthorizationRequest, project_id: int, *, enabled: bool) -> None:
        """Enable/disable requires DB-backed global admin; an OWNER cannot bootstrap scope state."""
        operation,status=("ENABLE_PROJECT_SCOPE","ACTIVE") if enabled else ("DISABLE_PROJECT_SCOPE","DISABLED")
        async def op(cur: Any) -> None:
            auth=await self._resolve_admin(cur,request,project_id,expected_operation=operation,bootstrap=True)
            await cur.execute("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s FOR UPDATE",(project_id,)); old=await cur.fetchone()
            if not old: raise ScopeDenied("PROJECT_SCOPE_UNBOUND")
            tenant,old_status=(old[0],old[1]) if not isinstance(old,dict) else (old["tenant_id"],old["status"])
            await cur.execute("UPDATE jax_project_scope SET status=%s,version=version+1,updated_at=NOW(6) WHERE project_id=%s",(status,project_id))
            await self._event(cur,auth,operation,project_id,None,int(tenant),None,None,old_status,status)
        await self._store.mutation(op)

    async def bind_legacy_project_scope(
        self, request: MutationAuthorizationRequest, project_id: int, tenant_id: int
    ) -> bool:
        """Explicitly bind one previously-unbound legacy project.

        There is intentionally no inference path here: a global DB-backed
        administrator supplies the target tenant, the deployed project row is
        locked and checked, and the binding plus its authorization event are
        committed atomically.  ``True`` means a new binding was created;
        ``False`` is the safe idempotent same-tenant result.
        """
        async def op(cur: Any) -> bool:
            auth = await self._resolve_admin(cur, request, project_id,
                                             expected_operation="BIND_LEGACY_PROJECT_SCOPE", bootstrap=True)
            await cur.execute("SELECT id FROM projects WHERE id=%s FOR UPDATE", (project_id,))
            if not await cur.fetchone():
                raise ScopeDenied("legacy project does not exist")
            await cur.execute(
                "SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s FOR UPDATE",
                (project_id,),
            )
            existing = await cur.fetchone()
            if existing:
                existing_tenant = existing[0] if not isinstance(existing, dict) else existing["tenant_id"]
                if int(existing_tenant) != int(tenant_id):
                    raise ScopeDenied("conflicting legacy project tenant binding")
                # Same binding is a deliberate, observable no-op.  It is an
                # audit event rather than a silent success, while preserving
                # the original active/disabled state.
                status = existing[1] if not isinstance(existing, dict) else existing["status"]
                await self._event(cur, auth, "BIND_LEGACY_PROJECT_SCOPE_NOOP", project_id,
                                  None, int(tenant_id), None, None, status, status)
                return False
            await cur.execute(
                "SELECT tenant_id FROM jax_tenants WHERE tenant_id=%s FOR UPDATE", (tenant_id,)
            )
            if not await cur.fetchone():
                raise ScopeDenied("target tenant does not exist")
            await cur.execute(
                "INSERT INTO jax_project_scope "
                "(project_id,tenant_id,status,created_at,created_by,updated_at) "
                "VALUES (%s,%s,'ACTIVE',NOW(6),%s,NOW(6))",
                (project_id, tenant_id, auth.scope.actor_principal),
            )
            await self._event(cur, auth, "BIND_LEGACY_PROJECT_SCOPE", project_id,
                              None, tenant_id, None, None, None, "ACTIVE")
            return True

        return await self._store.mutation(op)
