"""Transactional mutation boundary for project scope/membership authority.

Rewritten for D1-D5 / section 3-bis (plan `2026-09-25-proyectos-d1d5-plan.md`).
Project administration no longer routes through the memory resolver's
`resolve_mutation_in_transaction` (that path is ACTIVE-only and exists for
memory reads/writes, H3/H4). It has its own transactional resolver,
`_resolve_project_actor_cur`, which:

  - resolves against the TARGET project (an argument, never inferred from a
    caller-supplied role) and requires the caller's own already-resolved
    scope to point at that same project (closes the H4 "admin_shape" laundering,
    where `memory:admin` on any `project_id` satisfied any target);
  - takes the tenant from the actor's own current DB row, not the request body
    (D4);
  - requires an ACTIVE membership in the target project, with no
    `memory:admin` shortcut (D2: no implicit access -- an admin's authority
    over a project is an explicit OWNER/TENANT_ADMIN membership row, never a
    bypass).

Every mutation here takes `jax_tenants(tenant_id) FOR UPDATE` as its first
statement. That single-row lock is a per-tenant mutex: two project-authority
transactions on the same tenant fully serialize on it, so there is no need to
replicate the theoretical inter-transaction lock order beyond it (decision 1
of the plan). `sync_tenant_admin_memberships_in_transaction` is the one
exception -- its caller (the platform's user-admin transaction) already holds
that lock before calling in.
"""
from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from .b9 import (AuthorizationDenied, MutationAuthorizationRequest, ScopeContext,
                 ScopeDenied, Visibility, _uuid7)
from .scope_authority import ProjectLifecycle, ProjectRole, TENANT_ADMIN_ROLES


# --------------------------------------------------------------------------
# Errors. Every one carries a stable `code` a caller (the platform's HTTP
# layer) can map to a status/i18n key without string-matching a message.
# --------------------------------------------------------------------------

class ProjectAuthorityError(Exception):
    """Base error. `code` is a class attribute, overridable per-raise for the
    handful of errors that carry more than one stable code (e.g. a state
    conflict is either `proyecto_no_activo` or `transicion_invalida`)."""
    code: str = "project_authority_error"

    def __init__(self, message: str = "", *, code: str | None = None):
        super().__init__(message or self.code)
        if code is not None:
            self.code = code


class ProjectNotVisible(ProjectAuthorityError, ScopeDenied):
    code = "proyecto_no_encontrado"


class ProjectRoleInsufficient(ProjectAuthorityError, AuthorizationDenied):
    code = "papel_insuficiente"


class ProjectStateConflict(ProjectAuthorityError):
    code = "proyecto_no_activo"


class LastOwnerRequired(ProjectAuthorityError):
    code = "ultimo_duenio"


class TenantAdminMembershipProtected(ProjectAuthorityError):
    code = "membresia_de_administracion"


class AlreadyMember(ProjectAuthorityError):
    code = "ya_es_miembro"


class MemberNotFound(ProjectAuthorityError):
    code = "miembro_no_encontrado"


class TargetUserNotEligible(ProjectAuthorityError):
    code = "usuario_no_elegible"


class IdempotencyKeyConflict(ProjectAuthorityError):
    code = "clave_reutilizada"


class ReservedProjectIdRange(ProjectAuthorityError):
    code = "id_reservado"


#: See spec Sec9.3: legacy orphaned ids fall in this INT range. A newly
#: created project landing here is a data-corruption signal, not a normal
#: collision -- the whole transaction is rolled back and nothing is written.
RESERVED_PROJECT_ID_RANGE = (900001, 1400055)

_ADMIN_ROLE_PLACEHOLDERS = ",".join(["%s"] * len(TENANT_ADMIN_ROLES))
_ADMIN_ROLE_PARAMS = tuple(sorted(TENANT_ADMIN_ROLES))

#: `to_lifecycle -> jax_project_scope.status` mirror written to `projects.status`
#: in the same transaction (D5).
_LIFECYCLE_MIRROR = {
    ProjectLifecycle.ACTIVE: "active",
    ProjectLifecycle.ARCHIVED: "archived",
    ProjectLifecycle.HIDDEN: "hidden",
    ProjectLifecycle.DISABLED: "disabled",
}

#: (from, to) -> (event name, requires tenant-admin role in addition to OWNER).
_LIFECYCLE_TRANSITIONS: dict[tuple[ProjectLifecycle, ProjectLifecycle], tuple[str, bool]] = {
    (ProjectLifecycle.ACTIVE, ProjectLifecycle.ARCHIVED): ("ARCHIVE_PROJECT", False),
    (ProjectLifecycle.ARCHIVED, ProjectLifecycle.ACTIVE): ("RESTORE_PROJECT", False),
    (ProjectLifecycle.ARCHIVED, ProjectLifecycle.HIDDEN): ("HIDE_PROJECT", True),
    (ProjectLifecycle.HIDDEN, ProjectLifecycle.ARCHIVED): ("UNHIDE_PROJECT", True),
    (ProjectLifecycle.ACTIVE, ProjectLifecycle.DISABLED): ("DISABLE_PROJECT", True),
    (ProjectLifecycle.ARCHIVED, ProjectLifecycle.DISABLED): ("DISABLE_PROJECT", True),
    (ProjectLifecycle.HIDDEN, ProjectLifecycle.DISABLED): ("DISABLE_PROJECT", True),
    (ProjectLifecycle.DISABLED, ProjectLifecycle.ACTIVE): ("ENABLE_PROJECT", True),
}

#: OWNER outranks every other membership role; VIEWER/CONTRIBUTOR/REVIEWER are
#: all below it. `min_role=OWNER` is the only threshold this PR ever checks,
#: but the ranking is kept generic rather than a single `== OWNER` test.
_PROJECT_ROLE_RANK = {ProjectRole.VIEWER: 0, ProjectRole.CONTRIBUTOR: 1,
                      ProjectRole.REVIEWER: 1, ProjectRole.OWNER: 2}


def _is_duplicate_key_error(exc: Exception) -> bool:
    """True for a MySQL/MariaDB 1062 (duplicate key) error, however the
    driver wraps it -- both pymysql/aiomysql expose it as `args[0]`."""
    args = getattr(exc, "args", None)
    return bool(args) and args[0] == 1062


@dataclass(frozen=True)
class CreatedProject:
    project_id: int
    project_uuid: str
    created: bool


@dataclass(frozen=True)
class _Actor:
    """A resolved, lock-held, DB-current view of the acting user on the
    target project. Never constructed from caller input."""
    tenant_id: int
    user_id: int
    is_tenant_admin: bool
    project_role: ProjectRole
    lifecycle: ProjectLifecycle


class ProjectAuthorityAdmin:
    """The only supported API for project scope/membership state changes and
    their audit events. Every public method opens exactly one transaction
    (`self._store.mutation`) and re-resolves the actor inside it -- a stale
    resolved context from before the transaction cannot grant a mutation
    after a concurrent revoke committed first.
    """

    def __init__(self, store: Any, authorization_resolver: Any):
        self._store, self._authorization_resolver = store, authorization_resolver

    # -- shared row helpers -------------------------------------------------

    @staticmethod
    def _value(row: Any, name: str, pos: int) -> Any:
        return row.get(name) if isinstance(row, dict) else row[pos]

    @staticmethod
    async def _lock_tenant(cur: Any, tenant_id: Any) -> None:
        await cur.execute("SELECT tenant_id FROM jax_tenants WHERE tenant_id=%s FOR UPDATE", (tenant_id,))
        if not await cur.fetchone():
            raise ProjectNotVisible("tenant does not exist")

    @staticmethod
    async def _lock_tenant_admins(cur: Any, tenant_id: Any) -> list[int]:
        # jax_users' collation is case-insensitive (fixed by a test): this
        # still matches a role stored as e.g. 'Admin'.
        await cur.execute(
            f"SELECT user_id FROM jax_users WHERE tenant_id=%s AND role IN ({_ADMIN_ROLE_PLACEHOLDERS}) "
            "AND status='active' ORDER BY user_id FOR UPDATE",
            (tenant_id, *_ADMIN_ROLE_PARAMS))
        rows = await cur.fetchall()
        return [int(ProjectAuthorityAdmin._value(r, "user_id", 0)) for r in rows]

    @staticmethod
    async def _mirror_status(cur: Any, project_id: Any, lifecycle: ProjectLifecycle) -> None:
        await cur.execute("UPDATE projects SET status=%s WHERE id=%s",
                          (_LIFECYCLE_MIRROR[lifecycle], project_id))

    @staticmethod
    async def _count_other_active_owners(cur: Any, project_id: Any, excluded_user_id: Any) -> int:
        # A locking read (current read, not a snapshot): it must see any
        # concurrent revoke/demote that already committed, to avoid write skew
        # between two OWNERs revoking each other at the same time.
        await cur.execute(
            "SELECT COUNT(*) AS n FROM jax_project_membership m "
            "JOIN jax_users u ON u.user_id=m.user_id "
            "WHERE m.project_id=%s AND m.user_id<>%s AND m.status='ACTIVE' "
            "AND m.project_role='OWNER' AND u.status='active' FOR UPDATE",
            (project_id, excluded_user_id))
        row = await cur.fetchone()
        return int(ProjectAuthorityAdmin._value(row, "n", 0))

    @staticmethod
    async def _event(cur: Any, scope: ScopeContext, operation: str, project_id: Any,
                     target_user_id: int | None, tenant_id: Any, old_role: str | None,
                     new_role: str | None, old_status: str | None, new_status: str | None) -> None:
        await cur.execute(
            "INSERT INTO jax_project_membership_event (event_id,operation,actor_principal,actor_type,"
            "project_id,target_user_id,tenant_id,old_project_role,new_project_role,old_status,new_status,"
            "occurred_at,request_id,trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s)",
            (_uuid7(), operation, scope.actor_principal, scope.actor_type, project_id, target_user_id,
             tenant_id, old_role, new_role, old_status, new_status, time.time(), scope.request_id, scope.trace_id))

    # -- the transactional actor resolver ------------------------------------

    async def _resolve_project_actor_cur(
        self, cur: Any, request: MutationAuthorizationRequest, project_id: Any, *,
        expected_operation: str, min_role: ProjectRole | None, allowed_states: frozenset[ProjectLifecycle],
        require_tenant_admin: bool = False, lock_admins: bool = False,
    ) -> _Actor:
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("project mutation requires an authorization request")
        if request.operation != expected_operation or request.target_visibility is not Visibility.PROJECT_SHARED:
            raise AuthorizationDenied("project mutation operation request is invalid")
        scope = request.scope
        # Anti-laundering (H4): the caller's own already-resolved scope has to
        # point at the exact project being administered. `memory:admin`/any
        # other capability on a *different* project id grants nothing here.
        if str(scope.project_id) != str(project_id):
            raise ProjectNotVisible("scope project id does not match the target project")
        await self._lock_tenant(cur, scope.tenant_id)
        await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                          (scope.subject_user_id, scope.tenant_id))
        actor_row = await cur.fetchone()
        if not actor_row or str(self._value(actor_row, "status", 0)).upper() != "ACTIVE":
            raise ProjectNotVisible("actor is not an active tenant member")
        is_tenant_admin = str(self._value(actor_row, "role", 1) or "").lower() in TENANT_ADMIN_ROLES
        if lock_admins:
            await self._lock_tenant_admins(cur, scope.tenant_id)
        await cur.execute("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s FOR UPDATE", (project_id,))
        scope_row = await cur.fetchone()
        if not scope_row:
            raise ProjectNotVisible("project scope does not exist")
        scope_tenant = self._value(scope_row, "tenant_id", 0)
        # D4: the target project's tenant has to match the ACTOR's own
        # current tenant, never a tenant asserted by the caller.
        if int(scope_tenant) != int(scope.tenant_id):
            raise ProjectNotVisible("project belongs to a different tenant")
        await cur.execute(
            "SELECT project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",
            (project_id, scope.subject_user_id))
        member_row = await cur.fetchone()
        if not member_row or str(self._value(member_row, "status", 1)).upper() != "ACTIVE":
            raise ProjectNotVisible("actor project membership is missing or revoked")
        try:
            lifecycle = ProjectLifecycle(str(self._value(scope_row, "status", 1) or "").upper())
        except ValueError as e:
            raise ProjectNotVisible("project lifecycle is invalid") from e
        if lifecycle in (ProjectLifecycle.HIDDEN, ProjectLifecycle.DISABLED) and not is_tenant_admin:
            raise ProjectNotVisible("project scope is not visible to a non-admin actor")
        if lifecycle not in allowed_states:
            raise ProjectStateConflict(f"project lifecycle {lifecycle.value} does not allow {expected_operation}")
        try:
            project_role = ProjectRole(str(self._value(member_row, "project_role", 0) or "").upper())
        except ValueError as e:
            raise ProjectNotVisible("actor membership role is invalid") from e
        if require_tenant_admin and not is_tenant_admin:
            raise ProjectRoleInsufficient("operation requires a tenant administrator")
        if min_role is not None and _PROJECT_ROLE_RANK[project_role] < _PROJECT_ROLE_RANK[min_role]:
            raise ProjectRoleInsufficient(f"operation requires role >= {min_role.value}")
        return _Actor(int(scope.tenant_id), int(scope.subject_user_id), is_tenant_admin, project_role, lifecycle)

    # -- public operations ----------------------------------------------------

    async def create_project(self, request: MutationAuthorizationRequest, *, name: str,
                             description: str | None, idempotency_key: str) -> CreatedProject:
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("project mutation requires an authorization request")
        if request.operation != "CREATE_PROJECT" or request.target_visibility is not Visibility.PROJECT_SHARED:
            raise AuthorizationDenied("project mutation operation request is invalid")
        scope = request.scope
        if scope.project_id is not None:
            raise AuthorizationDenied("create_project must not target an existing project scope")
        normalized_name = unicodedata.normalize("NFC", name or "").strip()
        if not (1 <= len(normalized_name) <= 255):
            raise AuthorizationDenied("project name must be 1-255 characters")
        if description is not None and len(description) > 2000:
            raise AuthorizationDenied("project description must be <= 2000 characters")
        digest_payload = {"t": str(scope.tenant_id), "u": str(scope.subject_user_id),
                          "n": normalized_name, "d": description}
        request_digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        async def _read_existing(cur: Any) -> Any:
            await cur.execute(
                "SELECT project_id,request_digest FROM jax_project_creation_request WHERE idempotency_key=%s",
                (idempotency_key,))
            return await cur.fetchone()

        async def _read_uuid(cur: Any, existing_project_id: Any) -> Any:
            await cur.execute("SELECT project_uuid FROM projects WHERE id=%s", (existing_project_id,))
            return await cur.fetchone()

        async def _resolve_existing(row: Any) -> CreatedProject:
            existing_project_id = self._value(row, "project_id", 0)
            existing_digest = self._value(row, "request_digest", 1)
            if str(existing_digest) != request_digest:
                raise IdempotencyKeyConflict("idempotency key already used with a different request")
            uuid_row = await self._store.mutation(lambda cur: _read_uuid(cur, existing_project_id))
            project_uuid = self._value(uuid_row, "project_uuid", 0) if uuid_row else None
            return CreatedProject(int(existing_project_id), str(project_uuid), False)

        existing = await self._store.mutation(_read_existing)
        if existing is not None:
            return await _resolve_existing(existing)

        async def op(cur: Any) -> CreatedProject:
            await self._lock_tenant(cur, scope.tenant_id)
            admin_ids = await self._lock_tenant_admins(cur, scope.tenant_id)
            await cur.execute("SELECT status FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                              (scope.subject_user_id, scope.tenant_id))
            actor = await cur.fetchone()
            if not actor or str(self._value(actor, "status", 0)).upper() != "ACTIVE":
                raise ProjectNotVisible("actor is not an active tenant member")
            project_uuid = _uuid7()
            await cur.execute(
                "INSERT INTO projects (project_uuid,name,description,status) VALUES (%s,%s,%s,'active')",
                (project_uuid, normalized_name, description))
            await cur.execute("SELECT LAST_INSERT_ID() AS id")
            new_id = int(self._value(await cur.fetchone(), "id", 0))
            if RESERVED_PROJECT_ID_RANGE[0] <= new_id <= RESERVED_PROJECT_ID_RANGE[1]:
                raise ReservedProjectIdRange(f"project id {new_id} falls in the reserved legacy range")
            await cur.execute(
                "INSERT INTO jax_project_scope (project_id,tenant_id,status,created_at,created_by,updated_at) "
                "VALUES (%s,%s,'ACTIVE',NOW(6),%s,NOW(6))",
                (new_id, scope.tenant_id, scope.actor_principal))
            await cur.execute(
                "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,"
                "status,grant_origin,created_at,created_by,updated_at) VALUES "
                "(%s,%s,%s,%s,'OWNER','ACTIVE','CREATOR',NOW(6),%s,NOW(6))",
                (_uuid7(), new_id, scope.tenant_id, scope.subject_user_id, scope.actor_principal))
            await self._event(cur, scope, "CREATE_PROJECT", new_id, int(scope.subject_user_id),
                              scope.tenant_id, None, "OWNER", None, "ACTIVE")
            for admin_id in admin_ids:
                if admin_id == int(scope.subject_user_id):
                    continue
                await cur.execute(
                    "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,"
                    "status,grant_origin,created_at,created_by,updated_at) VALUES "
                    "(%s,%s,%s,%s,'OWNER','ACTIVE','TENANT_ADMIN',NOW(6),%s,NOW(6))",
                    (_uuid7(), new_id, scope.tenant_id, admin_id, scope.actor_principal))
                await self._event(cur, scope, "GRANT_TENANT_ADMIN", new_id, admin_id,
                                  scope.tenant_id, None, "OWNER", None, "ACTIVE")
            # Last statement of the transaction, as designed: a concurrent
            # winner with the same key commits its own row first and this one
            # dies with 1062, rolling back the whole project it just created.
            await cur.execute(
                "INSERT INTO jax_project_creation_request (idempotency_key,tenant_id,user_id,request_digest,"
                "project_id,created_at) VALUES (%s,%s,%s,%s,%s,NOW(6))",
                (idempotency_key, scope.tenant_id, scope.subject_user_id, request_digest, new_id))
            return CreatedProject(new_id, project_uuid, True)

        try:
            return await self._store.mutation(op)
        except Exception as exc:
            if not _is_duplicate_key_error(exc):
                raise
            reread = await self._store.mutation(_read_existing)
            if reread is None:
                raise
            return await _resolve_existing(reread)

    async def bootstrap_existing_project(self, request: MutationAuthorizationRequest, project_id: int, *,
                                         owner_user_id: int) -> bool:
        """Explicitly bind one previously-unbound legacy project (replaces
        `bind_legacy_project_scope`). The tenant is always the ACTOR's own
        tenant (D4) -- `bind_legacy_project_scope`'s caller-supplied
        `tenant_id` parameter is gone on purpose (H5)."""
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("project mutation requires an authorization request")
        if request.operation != "BOOTSTRAP_PROJECT" or request.target_visibility is not Visibility.PROJECT_SHARED:
            raise AuthorizationDenied("project mutation operation request is invalid")
        scope = request.scope
        if str(scope.project_id) != str(project_id):
            raise ProjectNotVisible("scope project id does not match the target project")
        if RESERVED_PROJECT_ID_RANGE[0] <= int(project_id) <= RESERVED_PROJECT_ID_RANGE[1]:
            raise ReservedProjectIdRange(f"project id {project_id} falls in the reserved legacy range")

        async def op(cur: Any) -> bool:
            await self._lock_tenant(cur, scope.tenant_id)
            await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                              (scope.subject_user_id, scope.tenant_id))
            actor = await cur.fetchone()
            if not actor:
                raise ProjectNotVisible("actor is not an active tenant member")
            actor_status = str(self._value(actor, "status", 0)).upper()
            actor_role = str(self._value(actor, "role", 1) or "").lower()
            if actor_status != "ACTIVE" or actor_role not in TENANT_ADMIN_ROLES:
                raise ProjectRoleInsufficient("bootstrap requires an active tenant administrator")
            admin_ids = await self._lock_tenant_admins(cur, scope.tenant_id)
            await cur.execute("SELECT id FROM projects WHERE id=%s FOR UPDATE", (project_id,))
            if not await cur.fetchone():
                raise ProjectNotVisible("legacy project does not exist")
            await cur.execute("SELECT tenant_id,status FROM jax_project_scope WHERE project_id=%s FOR UPDATE", (project_id,))
            existing_scope = await cur.fetchone()
            if existing_scope:
                existing_tenant = self._value(existing_scope, "tenant_id", 0)
                existing_status = self._value(existing_scope, "status", 1)
                if int(existing_tenant) != int(scope.tenant_id):
                    raise ProjectStateConflict("legacy project is bound to a different tenant")
                await self._event(cur, scope, "BOOTSTRAP_PROJECT_NOOP", project_id, None, scope.tenant_id,
                                  None, None, existing_status, existing_status)
                return False
            await cur.execute("SELECT status FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                              (owner_user_id, scope.tenant_id))
            owner = await cur.fetchone()
            if not owner or str(self._value(owner, "status", 0)).upper() != "ACTIVE":
                raise TargetUserNotEligible("owner is not an active tenant member")
            await cur.execute(
                "INSERT INTO jax_project_scope (project_id,tenant_id,status,created_at,created_by,updated_at) "
                "VALUES (%s,%s,'ACTIVE',NOW(6),%s,NOW(6))",
                (project_id, scope.tenant_id, scope.actor_principal))
            await cur.execute(
                "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,"
                "status,grant_origin,created_at,created_by,updated_at) VALUES "
                "(%s,%s,%s,%s,'OWNER','ACTIVE','EXPLICIT',NOW(6),%s,NOW(6))",
                (_uuid7(), project_id, scope.tenant_id, owner_user_id, scope.actor_principal))
            await self._event(cur, scope, "BOOTSTRAP_PROJECT", project_id, int(owner_user_id),
                              scope.tenant_id, None, "OWNER", None, "ACTIVE")
            for admin_id in admin_ids:
                if admin_id == int(owner_user_id):
                    continue
                await cur.execute(
                    "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,"
                    "status,grant_origin,created_at,created_by,updated_at) VALUES "
                    "(%s,%s,%s,%s,'OWNER','ACTIVE','TENANT_ADMIN',NOW(6),%s,NOW(6))",
                    (_uuid7(), project_id, scope.tenant_id, admin_id, scope.actor_principal))
                await self._event(cur, scope, "GRANT_TENANT_ADMIN", project_id, admin_id,
                                  scope.tenant_id, None, "OWNER", None, "ACTIVE")
            await self._mirror_status(cur, project_id, ProjectLifecycle.ACTIVE)
            return True

        return await self._store.mutation(op)

    async def grant_member(self, request: MutationAuthorizationRequest, project_id: int, *,
                           email: str, role: ProjectRole) -> int:
        if role not in (ProjectRole.VIEWER, ProjectRole.CONTRIBUTOR, ProjectRole.OWNER):
            raise AuthorizationDenied("role must be VIEWER, CONTRIBUTOR or OWNER")

        async def op(cur: Any) -> int:
            actor = await self._resolve_project_actor_cur(
                cur, request, project_id, expected_operation="GRANT_MEMBER",
                min_role=ProjectRole.OWNER, allowed_states=frozenset({ProjectLifecycle.ACTIVE}))
            scope = request.scope
            await cur.execute("SELECT user_id,status,role FROM jax_users WHERE email=%s AND tenant_id=%s FOR UPDATE",
                              (email, actor.tenant_id))
            target = await cur.fetchone()
            if not target or str(self._value(target, "status", 1)).upper() != "ACTIVE":
                raise TargetUserNotEligible("target user is not eligible")
            target_id = int(self._value(target, "user_id", 0))
            if str(self._value(target, "role", 2) or "").lower() in TENANT_ADMIN_ROLES:
                raise AlreadyMember("target user is already an administrator")
            await cur.execute(
                "SELECT status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",
                (project_id, target_id))
            existing = await cur.fetchone()
            if existing:
                if str(self._value(existing, "status", 0)).upper() == "ACTIVE":
                    raise AlreadyMember("membership already active")
                await cur.execute(
                    "UPDATE jax_project_membership SET status='ACTIVE',project_role=%s,grant_origin='EXPLICIT',"
                    "version=version+1,updated_at=NOW(6) WHERE project_id=%s AND user_id=%s",
                    (role.value, project_id, target_id))
            else:
                await cur.execute(
                    "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,project_role,"
                    "status,grant_origin,created_at,created_by,updated_at) VALUES "
                    "(%s,%s,%s,%s,%s,'ACTIVE','EXPLICIT',NOW(6),%s,NOW(6))",
                    (_uuid7(), project_id, actor.tenant_id, target_id, role.value, scope.actor_principal))
            await self._event(cur, scope, "GRANT_MEMBER", project_id, target_id, actor.tenant_id,
                              None, role.value, None, "ACTIVE")
            return target_id

        return await self._store.mutation(op)

    async def change_project_role(self, request: MutationAuthorizationRequest, project_id: int,
                                  user_id: int, role: ProjectRole) -> None:
        async def op(cur: Any) -> None:
            actor = await self._resolve_project_actor_cur(
                cur, request, project_id, expected_operation="CHANGE_PROJECT_ROLE",
                min_role=ProjectRole.OWNER, allowed_states=frozenset({ProjectLifecycle.ACTIVE}))
            scope = request.scope
            await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                              (user_id, actor.tenant_id))
            target_user = await cur.fetchone()
            if (target_user and str(self._value(target_user, "status", 0)).upper() == "ACTIVE"
                    and str(self._value(target_user, "role", 1) or "").lower() in TENANT_ADMIN_ROLES):
                raise TenantAdminMembershipProtected("membership of an active tenant administrator cannot be changed")
            await cur.execute(
                "SELECT project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",
                (project_id, user_id))
            member = await cur.fetchone()
            if not member or str(self._value(member, "status", 1)).upper() != "ACTIVE":
                raise MemberNotFound("project membership is missing")
            old_role = str(self._value(member, "project_role", 0))
            if old_role.upper() == "OWNER" and role is not ProjectRole.OWNER:
                if await self._count_other_active_owners(cur, project_id, user_id) == 0:
                    raise LastOwnerRequired("cannot demote the last active owner")
            await cur.execute(
                "UPDATE jax_project_membership SET project_role=%s,version=version+1,updated_at=NOW(6) "
                "WHERE project_id=%s AND user_id=%s", (role.value, project_id, user_id))
            await self._event(cur, scope, "CHANGE_PROJECT_ROLE", project_id, int(user_id), actor.tenant_id,
                              old_role, role.value, "ACTIVE", "ACTIVE")

        await self._store.mutation(op)

    async def revoke_member(self, request: MutationAuthorizationRequest, project_id: int, user_id: int) -> None:
        async def op(cur: Any) -> None:
            actor = await self._resolve_project_actor_cur(
                cur, request, project_id, expected_operation="REVOKE_MEMBER",
                min_role=ProjectRole.OWNER,
                allowed_states=frozenset({ProjectLifecycle.ACTIVE, ProjectLifecycle.ARCHIVED}))
            scope = request.scope
            await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                              (user_id, actor.tenant_id))
            target_user = await cur.fetchone()
            if (target_user and str(self._value(target_user, "status", 0)).upper() == "ACTIVE"
                    and str(self._value(target_user, "role", 1) or "").lower() in TENANT_ADMIN_ROLES):
                raise TenantAdminMembershipProtected("membership of an active tenant administrator cannot be revoked")
            await cur.execute(
                "SELECT project_role,status FROM jax_project_membership WHERE project_id=%s AND user_id=%s FOR UPDATE",
                (project_id, user_id))
            member = await cur.fetchone()
            if not member or str(self._value(member, "status", 1)).upper() != "ACTIVE":
                raise MemberNotFound("project membership is missing")
            old_role = str(self._value(member, "project_role", 0))
            if old_role.upper() == "OWNER" and await self._count_other_active_owners(cur, project_id, user_id) == 0:
                raise LastOwnerRequired("cannot revoke the last active owner")
            await cur.execute(
                "UPDATE jax_project_membership SET status='REVOKED',version=version+1,updated_at=NOW(6) "
                "WHERE project_id=%s AND user_id=%s", (project_id, user_id))
            await self._event(cur, scope, "REVOKE_MEMBER", project_id, int(user_id), actor.tenant_id,
                              old_role, old_role, "ACTIVE", "REVOKED")

        await self._store.mutation(op)

    async def set_project_lifecycle(self, request: MutationAuthorizationRequest, project_id: int,
                                    target: ProjectLifecycle) -> bool:
        async def op(cur: Any) -> bool:
            actor = await self._resolve_project_actor_cur(
                cur, request, project_id, expected_operation="SET_PROJECT_LIFECYCLE",
                min_role=ProjectRole.OWNER, allowed_states=frozenset(ProjectLifecycle))
            scope = request.scope
            current = actor.lifecycle
            if current is target:
                # Deliberate no-op: a retry after a DESCONOCIDO outcome must
                # not fail or double-write an event.
                return False
            key = (current, target)
            if key not in _LIFECYCLE_TRANSITIONS:
                raise ProjectStateConflict(f"cannot transition from {current.value} to {target.value}",
                                           code="transicion_invalida")
            event_name, needs_admin = _LIFECYCLE_TRANSITIONS[key]
            if needs_admin and not actor.is_tenant_admin:
                raise ProjectRoleInsufficient("this transition requires a tenant administrator")
            await cur.execute(
                "UPDATE jax_project_scope SET status=%s,version=version+1,updated_at=NOW(6) WHERE project_id=%s",
                (target.value, project_id))
            await self._mirror_status(cur, project_id, target)
            await self._event(cur, scope, event_name, project_id, None, actor.tenant_id,
                              None, None, current.value, target.value)
            return True

        return await self._store.mutation(op)

    async def sync_tenant_admin_memberships_in_transaction(
        self, cur: Any, *, actor_scope: ScopeContext, user_id: int, tenant_id: int,
        was_active_admin: bool, is_active_admin: bool,
    ) -> int:
        """Called by the platform's own `jax_users` admin-role transaction,
        which has already locked `jax_tenants(tenant_id)` -> admin set ->
        target user row (decision 1's lock order) before calling in. This
        function does not re-lock the tenant.
        """
        await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                          (actor_scope.subject_user_id, tenant_id))
        actor = await cur.fetchone()
        if (not actor or str(self._value(actor, "status", 0)).upper() != "ACTIVE"
                or str(self._value(actor, "role", 1) or "").lower() not in TENANT_ADMIN_ROLES):
            raise ProjectRoleInsufficient("sync requires an active tenant administrator actor")
        touched = 0
        if is_active_admin and not was_active_admin:
            await cur.execute(
                "SELECT project_id FROM jax_project_scope WHERE tenant_id=%s ORDER BY project_id FOR UPDATE",
                (tenant_id,))
            project_ids = [self._value(r, "project_id", 0) for r in await cur.fetchall()]
            for pid in project_ids:
                await cur.execute(
                    "SELECT project_role,status,grant_origin FROM jax_project_membership "
                    "WHERE project_id=%s AND user_id=%s FOR UPDATE", (pid, user_id))
                existing = await cur.fetchone()
                if not existing:
                    await cur.execute(
                        "INSERT INTO jax_project_membership (membership_id,project_id,tenant_id,user_id,"
                        "project_role,status,grant_origin,created_at,created_by,updated_at) VALUES "
                        "(%s,%s,%s,%s,'OWNER','ACTIVE','TENANT_ADMIN',NOW(6),%s,NOW(6))",
                        (_uuid7(), pid, tenant_id, user_id, actor_scope.actor_principal))
                    await self._event(cur, actor_scope, "GRANT_TENANT_ADMIN", pid, user_id, tenant_id,
                                      None, "OWNER", None, "ACTIVE")
                    touched += 1
                    continue
                role = str(self._value(existing, "project_role", 0))
                status = str(self._value(existing, "status", 1))
                origin = self._value(existing, "grant_origin", 2)
                if str(origin) == "CREATOR" and role.upper() == "OWNER" and status.upper() == "ACTIVE":
                    continue
                await cur.execute(
                    "UPDATE jax_project_membership SET pre_admin_role=%s,pre_admin_status=%s,"
                    "project_role='OWNER',status='ACTIVE',grant_origin='TENANT_ADMIN',version=version+1,"
                    "updated_at=NOW(6) WHERE project_id=%s AND user_id=%s", (role, status, pid, user_id))
                await self._event(cur, actor_scope, "GRANT_TENANT_ADMIN", pid, user_id, tenant_id,
                                  role, "OWNER", status, "ACTIVE")
                touched += 1
        elif was_active_admin and not is_active_admin:
            await cur.execute(
                "SELECT project_id,pre_admin_role,pre_admin_status FROM jax_project_membership "
                "WHERE user_id=%s AND tenant_id=%s AND grant_origin='TENANT_ADMIN' AND status='ACTIVE' "
                "ORDER BY project_id FOR UPDATE", (user_id, tenant_id))
            rows = await cur.fetchall()
            # Check every row BEFORE mutating any of them: a partial descent
            # left half-applied by a raised LastOwnerRequired would be worse
            # than refusing the whole operation up front.
            for row in rows:
                pid = self._value(row, "project_id", 0)
                if await self._count_other_active_owners(cur, pid, user_id) == 0:
                    raise LastOwnerRequired("cannot demote the last active owner of a project",
                                            code="proyecto_sin_duenio")
            for row in rows:
                pid = self._value(row, "project_id", 0)
                pre_role = self._value(row, "pre_admin_role", 1)
                pre_status = self._value(row, "pre_admin_status", 2)
                if pre_role is not None and pre_status is not None:
                    await cur.execute(
                        "UPDATE jax_project_membership SET project_role=%s,status=%s,grant_origin='EXPLICIT',"
                        "pre_admin_role=NULL,pre_admin_status=NULL,version=version+1,updated_at=NOW(6) "
                        "WHERE project_id=%s AND user_id=%s", (pre_role, pre_status, pid, user_id))
                    new_status = pre_status
                else:
                    await cur.execute(
                        "UPDATE jax_project_membership SET status='REVOKED',version=version+1,updated_at=NOW(6) "
                        "WHERE project_id=%s AND user_id=%s", (pid, user_id))
                    new_status = "REVOKED"
                await self._event(cur, actor_scope, "REVOKE_TENANT_ADMIN", pid, user_id, tenant_id,
                                  "OWNER", pre_role, "ACTIVE", new_status)
                touched += 1
        return touched
