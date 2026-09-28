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
statement (a per-tenant mutex), then locks the ACTOR's own row, then (ronda 3,
MAJOR M1) any DESTINO row this operation touches, all BEFORE the project
scope/membership rows: `tenant -> actor -> destino -> projects -> scope ->
membership -> events` (plan section 2.1). The chat's own resolver
(`scope_authority.py`'s `_tenant_user_cur` -> `_project_membership_cur`)
locks a user row before the scope row for the SAME reason; the two now agree,
so they cannot deadlock (MariaDB error 1213) over the same pair of rows --
reproduced and fixed ronda 3, 2026-09-26, see
`tests/test_project_authority_mariadb.py::test_orden_de_bloqueo_...`.

Ronda 3, BLOCK B1 (2026-09-26): every public entry point (and
`sync_tenant_admin_memberships_in_transaction`) requires a human, unambiguous
USER actor -- `_require_project_actor_subject` builds on
`scope_authority.require_subject` (never duplicates its logic) and adds the
project-specific narrowing: no SERVICE actor, and the canonical `user:<id>`
principal form, not a bare id. Before this fix a SERVICE-typed scope or a
USER scope whose `actor_principal` did not match its `subject_user_id` sailed
straight through every operation, misattributing (or fully forging) the
audit trail in `jax_project_membership_event`.

INVARIANTE (ronda 4, 2026-09-26, decisión de la sesión principal): ninguna
operación de este módulo bloquea (`FOR UPDATE`) una fila de `jax_users` que
no sea la del ACTOR o la del DESTINO -- nunca la de un tercero (otro OWNER
que se está contando, un admin del conjunto que se está leyendo). La
serialización de cambios a `jax_users.role`/`jax_users.status` la da,
enteramente, el candado de `jax_tenants(tenant_id) FOR UPDATE` que esta
transacción ya tomó como primera sentencia: TODO escritor de
`jax_users.role`/`status` (este módulo y la transacción propia de la
plataforma en `jax_users`, plan sección 4) tiene que tomar ese MISMO candado
antes de escribir. Bajo esa garantía, leer `jax_users.status`/`role` de un
tercero sin `FOR UPDATE` es una lectura consistente, no una carrera: nadie
más puede estar cambiándola mientras este candado siga en pie. Violar este
invariante fue exactamente el defecto de la ronda 3
(`_count_other_active_owners` y `_lock_tenant_admins` bloqueaban filas de
terceros vía `JOIN ... FOR UPDATE`), reproducido en rojo con dos conexiones
reales y una barrera contra `revoke_member`, `change_project_role` y el
descenso de `sync_tenant_admin_memberships_in_transaction` -- ver
`test_interbloqueo_...` en `tests/test_project_authority_mariadb.py` y
`test_invariante_ningun_tercero_se_bloquea_en_jax_users`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
import uuid as uuid_module
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .b9 import (AuthorizationDenied, MutationAuthorizationRequest, ScopeContext,
                 ScopeDenied, Visibility, _uuid7)
from .scope_authority import ProjectLifecycle, ProjectRole, TENANT_ADMIN_ROLES, require_subject

_logger = logging.getLogger(__name__)


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


class InvalidIdempotencyKey(ProjectAuthorityError):
    code = "clave_invalida"


class ReservedProjectIdRange(ProjectAuthorityError):
    code = "id_reservado"


class ProjectAuthorityRetryable(ProjectAuthorityError):
    """Ronda 4, MINOR 2: a transient MariaDB contention error (1213 deadlock,
    1205 lock-wait timeout) that a caller can retry as-is -- never a schema
    or authority problem. Kept distinct from the generic DB-error wrapper on
    purpose: a caller that sees `code == "reintentar"` knows to retry the
    same request; anything else means stop and look."""
    code = "reintentar"


#: See spec Sec9.3: legacy orphaned ids fall in this INT range. A newly
#: created project landing here is a data-corruption signal, not a normal
#: collision -- the whole transaction is rolled back and nothing is written.
RESERVED_PROJECT_ID_RANGE = (900001, 1400055)

#: Ronda 3, MINOR 5 (decision de la sesion principal, 2026-09-26): los
#: proyectos heredados (anteriores a B9) pertenecen al tenant que existia
#: antes de B9. Hoy `jax_tenants` tiene una sola fila en produccion
#: (verificado); si algun dia hay mas de un tenant legado, esto deja de ser
#: una constante y bootstrap necesita un parametro explicito -- no antes.
LEGACY_PROJECT_TENANT_ID = 1

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


def _is_raw_db_error(exc: Exception) -> bool:
    """True for a pymysql/aiomysql exception, however it is subclassed."""
    module = type(exc).__module__ or ""
    return module.startswith("pymysql") or module.startswith("aiomysql")


#: MariaDB 1213 (ER_LOCK_DEADLOCK) and 1205 (ER_LOCK_WAIT_TIMEOUT): transient
#: contention, never a schema/authority problem -- mapped to
#: `ProjectAuthorityRetryable`, never to the generic DB-error wrapper.
_RETRYABLE_MYSQL_ERRNOS = frozenset({1213, 1205})


def _wrap_unexpected_db_error(exc: Exception) -> Exception:
    """Ronda 3, MINOR 2 / ronda 4, MINOR 2: no raw DB error may leak past
    this module's public API -- a CHECK/FK violation this code should have
    prevented, a lock-wait timeout, anything unmapped. `ProjectAuthorityError`
    subclasses and already-expected control-flow exceptions pass through
    untouched.

    Ronda 4: the driver's own message text (`str(exc)`) is never embedded in
    the exception this raises -- a caller-facing message built from raw
    driver text is one string-match change away from leaking implementation
    detail (or breaking every caller that matches on it). The original is
    logged here, once, and kept as `__cause__` for anyone reading a traceback
    -- never as text inside the message a caller might display or match on.
    """
    if isinstance(exc, ProjectAuthorityError):
        return exc
    if _is_raw_db_error(exc):
        args = getattr(exc, "args", None)
        errno = args[0] if args else None
        _logger.warning("unexpected database error in project_authority (errno=%s): %s", errno, exc)
        if errno in _RETRYABLE_MYSQL_ERRNOS:
            wrapped: ProjectAuthorityError = ProjectAuthorityRetryable("transient database contention, retry")
        else:
            wrapped = ProjectAuthorityError("unexpected database error", code="error_de_base_de_datos")
        wrapped.__cause__ = exc
        return wrapped
    return exc


def _validate_idempotency_key(idempotency_key: str) -> None:
    """Ronda 3, MINOR 3: a canonical UUID (36 chars, RFC 4122 form) before
    anything else touches the database with it -- `jax_project_creation_request
    .idempotency_key` is `CHAR(36)`, and a non-canonical value (a URN, braces,
    no dashes, wrong length) is a caller bug, not a legitimate retry key."""
    if not isinstance(idempotency_key, str) or len(idempotency_key) != 36:
        raise InvalidIdempotencyKey("idempotency_key must be a canonical 36-character UUID")
    try:
        parsed = uuid_module.UUID(idempotency_key)
    except ValueError as e:
        raise InvalidIdempotencyKey("idempotency_key is not a valid UUID") from e
    if str(parsed) != idempotency_key.lower():
        raise InvalidIdempotencyKey("idempotency_key must be the canonical UUID form")


#: A `jax_users.user_id` is an INT AUTO_INCREMENT: positive, no leading zero,
#: at most 10 digits (fits a signed 32-bit int with room to spare). Ronda 4,
#: MINOR 1: `subject_user_id` has to match this BEFORE any SQL or any bare
#: `int(...)` call downstream -- a non-numeric or malformed value must never
#: reach a query as a raw parameter, and must never raise an uncaught
#: `ValueError` out of this module either.
_SUBJECT_USER_ID_RE = re.compile(r"^[1-9]\d{0,9}$")


def _require_project_actor_subject(scope: ScopeContext) -> None:
    """Ronda 3, BLOCK B1. Reuses `scope_authority.require_subject` for the
    baseline checks (tenant/actor/subject present, no delegation, and for a
    USER actor, principal/subject consistency) and adds the constraints
    specific to project administration, which `require_subject` deliberately
    does not enforce on its own (it also serves SERVICE actors elsewhere):
    the actor must be a USER, the principal must be the exact canonical
    `user:<subject_user_id>` form -- never a bare id, never anyone else's
    identity -- and (ronda 4, MINOR 1) `subject_user_id` itself has to be a
    canonical positive integer string. Every `_event()` row's
    `actor_principal` comes straight from `scope.actor_principal`; this is
    what keeps that column truthful.
    """
    require_subject(scope)
    if scope.actor_type != "USER":
        raise AuthorizationDenied("project administration requires a USER actor")
    if not _SUBJECT_USER_ID_RE.match(str(scope.subject_user_id or "")):
        raise AuthorizationDenied("subject_user_id must be a canonical positive integer")
    if scope.actor_principal != f"user:{scope.subject_user_id}":
        raise AuthorizationDenied("actor principal must be the canonical user:<id> form")


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

    def __init__(self, store: Any):
        # Ronda 4, MINOR 3: dropped the `authorization_resolver` parameter --
        # it was never read (H6: no production caller ever existed, and this
        # class resolves its own authority in-transaction via
        # `_resolve_project_actor_cur`, never through an injected resolver).
        # Dead code left in a constructor is a trap for the next reader who
        # assumes it does something.
        self._store = store

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
    async def _read_tenant_admins(cur: Any, tenant_id: Any) -> list[int]:
        """Ronda 4, MAJOR MJ-2: reads the tenant's admin set WITHOUT `FOR
        UPDATE` -- these are third-party `jax_users` rows (not the actor's,
        not a destino's), and the module invariant is that only the
        actor's/destino's rows ever get locked here. Safe as a plain read
        under the tenant-row lock this transaction already holds (see the
        module docstring): no concurrent writer of `jax_users.role`/`status`
        in this tenant can be running without that same lock. Uses
        `idx_jax_users_tenant_role_status` (migration 005h) so this is an
        index range scan, not a table scan that would also grab
        next-key locks on unrelated tenants' rows under REPEATABLE READ.

        jax_users' collation is case-insensitive (fixed by a test): this
        still matches a role stored as e.g. 'Admin'.
        """
        await cur.execute(
            f"SELECT user_id FROM jax_users WHERE tenant_id=%s AND role IN ({_ADMIN_ROLE_PLACEHOLDERS}) "
            "AND status='active' ORDER BY user_id",
            (tenant_id, *_ADMIN_ROLE_PARAMS))
        rows = await cur.fetchall()
        return [int(ProjectAuthorityAdmin._value(r, "user_id", 0)) for r in rows]

    @staticmethod
    async def _mirror_status(cur: Any, project_id: Any, lifecycle: ProjectLifecycle) -> None:
        await cur.execute("UPDATE projects SET status=%s WHERE id=%s",
                          (_LIFECYCLE_MIRROR[lifecycle], project_id))

    @staticmethod
    async def _count_other_active_owners(cur: Any, project_id: Any, excluded_user_id: Any) -> int:
        """Ronda 4, MAJOR MJ-1: locks ONLY `jax_project_membership` rows for
        this project (a locking read -- it must see any concurrent
        revoke/demote that already committed, to avoid write skew between
        two OWNERs revoking each other at the same time). It does NOT lock
        the matching `jax_users` rows anymore: those belong to OTHER
        owners, never the actor or the destino, and locking them was
        exactly the deadlock this round fixed (`test_interbloqueo_...`).
        `jax_users.status` is read separately, WITHOUT `FOR UPDATE` -- safe
        under the tenant-row lock this transaction already holds (module
        invariant).
        """
        await cur.execute(
            "SELECT user_id FROM jax_project_membership WHERE project_id=%s AND user_id<>%s "
            "AND status='ACTIVE' AND project_role='OWNER' FOR UPDATE",
            (project_id, excluded_user_id))
        owner_ids = [int(ProjectAuthorityAdmin._value(r, "user_id", 0)) for r in await cur.fetchall()]
        if not owner_ids:
            return 0
        placeholders = ",".join(["%s"] * len(owner_ids))
        await cur.execute(
            f"SELECT COUNT(*) AS n FROM jax_users WHERE user_id IN ({placeholders}) AND status='active'",
            tuple(owner_ids))
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
        lock_target: Callable[[Any], Awaitable[Any]] | None = None,
    ) -> tuple[_Actor, Any]:
        """Returns `(actor, target_result)`. `target_result` is whatever
        `lock_target(cur)` returned, or `None` if no `lock_target` was given.
        `lock_target` runs right after the actor's own row is locked and
        BEFORE the project scope row (ronda 3, MAJOR M1) -- callers that need
        to lock a destino row (an invitee, a role-change target) pass it in
        instead of querying that row themselves afterward.
        """
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("project mutation requires an authorization request")
        if request.operation != expected_operation or request.target_visibility is not Visibility.PROJECT_SHARED:
            raise AuthorizationDenied("project mutation operation request is invalid")
        scope = request.scope
        _require_project_actor_subject(scope)
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
            await self._read_tenant_admins(cur, scope.tenant_id)
        # Destino BEFORE the project scope row (plan 2.1 / MAJOR M1): the
        # chat's own resolver locks a user row before the scope row for the
        # same reason, and the two now agree on which comes first.
        target_result = await lock_target(cur) if lock_target is not None else None
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
        return (_Actor(int(scope.tenant_id), int(scope.subject_user_id), is_tenant_admin, project_role, lifecycle),
               target_result)

    # -- public operations ----------------------------------------------------

    async def create_project(self, request: MutationAuthorizationRequest, *, name: str,
                             description: str | None, idempotency_key: str) -> CreatedProject:
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("project mutation requires an authorization request")
        if request.operation != "CREATE_PROJECT" or request.target_visibility is not Visibility.PROJECT_SHARED:
            raise AuthorizationDenied("project mutation operation request is invalid")
        scope = request.scope
        _require_project_actor_subject(scope)
        if scope.project_id is not None:
            raise AuthorizationDenied("create_project must not target an existing project scope")
        _validate_idempotency_key(idempotency_key)
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

        async def _actor_still_active(cur: Any) -> bool:
            # Ronda 3, MINOR 4: the lock-free idempotency pre-read must not
            # hand back a project to a subject who is no longer an active
            # tenant member -- that identity has no standing to ask at all,
            # idempotent replay or not.
            await cur.execute("SELECT status FROM jax_users WHERE user_id=%s AND tenant_id=%s",
                              (scope.subject_user_id, scope.tenant_id))
            row = await cur.fetchone()
            return bool(row) and str(self._value(row, "status", 0)).upper() == "ACTIVE"

        async def _resolve_existing(row: Any) -> CreatedProject:
            existing_project_id = self._value(row, "project_id", 0)
            existing_digest = self._value(row, "request_digest", 1)
            if str(existing_digest) != request_digest:
                raise IdempotencyKeyConflict("idempotency key already used with a different request")
            if not await self._store.mutation(_actor_still_active):
                raise ProjectNotVisible("actor is not an active tenant member")
            uuid_row = await self._store.mutation(lambda cur: _read_uuid(cur, existing_project_id))
            project_uuid = self._value(uuid_row, "project_uuid", 0) if uuid_row else None
            return CreatedProject(int(existing_project_id), str(project_uuid), False)

        existing = await self._store.mutation(_read_existing)
        if existing is not None:
            return await _resolve_existing(existing)

        async def op(cur: Any) -> CreatedProject:
            await self._lock_tenant(cur, scope.tenant_id)
            admin_ids = await self._read_tenant_admins(cur, scope.tenant_id)
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
                raise _wrap_unexpected_db_error(exc) from exc
            reread = await self._store.mutation(_read_existing)
            if reread is None:
                raise _wrap_unexpected_db_error(exc) from exc
            return await _resolve_existing(reread)

    async def bootstrap_existing_project(self, request: MutationAuthorizationRequest, project_id: int, *,
                                         owner_user_id: int) -> bool:
        """Explicitly bind one previously-unbound legacy project (replaces
        `bind_legacy_project_scope`). The tenant is always the ACTOR's own
        tenant (D4) -- `bind_legacy_project_scope`'s caller-supplied
        `tenant_id` parameter is gone on purpose (H5). Ronda 3, MINOR 5:
        legacy (pre-B9) projects belong to `LEGACY_PROJECT_TENANT_ID`; an
        admin of any other tenant gets the same `proyecto_no_encontrado` a
        cross-tenant lookup would (never a hint that the project exists
        elsewhere)."""
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("project mutation requires an authorization request")
        if request.operation != "BOOTSTRAP_PROJECT" or request.target_visibility is not Visibility.PROJECT_SHARED:
            raise AuthorizationDenied("project mutation operation request is invalid")
        scope = request.scope
        _require_project_actor_subject(scope)
        if str(scope.project_id) != str(project_id):
            raise ProjectNotVisible("scope project id does not match the target project")
        if RESERVED_PROJECT_ID_RANGE[0] <= int(project_id) <= RESERVED_PROJECT_ID_RANGE[1]:
            raise ReservedProjectIdRange(f"project id {project_id} falls in the reserved legacy range")
        if str(scope.tenant_id) != str(LEGACY_PROJECT_TENANT_ID):
            raise ProjectNotVisible("legacy project does not exist")

        async def op(cur: Any) -> bool:
            try:
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
                admin_ids = await self._read_tenant_admins(cur, scope.tenant_id)
                # Destino (the intended owner) BEFORE `projects`/scope
                # (MAJOR M1): same reordering as grant/change/revoke.
                await cur.execute("SELECT status FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                                  (owner_user_id, scope.tenant_id))
                owner = await cur.fetchone()
                if not owner or str(self._value(owner, "status", 0)).upper() != "ACTIVE":
                    raise TargetUserNotEligible("owner is not an active tenant member")
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
            except ProjectAuthorityError:
                raise
            except Exception as exc:
                raise _wrap_unexpected_db_error(exc) from exc

        return await self._store.mutation(op)

    async def grant_member(self, request: MutationAuthorizationRequest, project_id: int, *,
                           email: str, role: ProjectRole) -> int:
        if role not in (ProjectRole.VIEWER, ProjectRole.CONTRIBUTOR, ProjectRole.OWNER):
            raise AuthorizationDenied("role must be VIEWER, CONTRIBUTOR or OWNER")

        async def op(cur: Any) -> int:
            try:
                scope = request.scope

                async def _lock_target(cur: Any) -> Any:
                    await cur.execute(
                        "SELECT user_id,status,role FROM jax_users WHERE email=%s AND tenant_id=%s FOR UPDATE",
                        (email, scope.tenant_id))
                    return await cur.fetchone()

                actor, target = await self._resolve_project_actor_cur(
                    cur, request, project_id, expected_operation="GRANT_MEMBER",
                    min_role=ProjectRole.OWNER, allowed_states=frozenset({ProjectLifecycle.ACTIVE}),
                    lock_target=_lock_target)
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
                    # Reactivating a REVOKED row (MINOR 2): explicitly clear
                    # pre_admin_* -- a defensive NULL, not an assumption that
                    # it was already null, so this can never trip
                    # chk_jax_project_membership_pre_admin regardless of how
                    # the row got here.
                    await cur.execute(
                        "UPDATE jax_project_membership SET status='ACTIVE',project_role=%s,grant_origin='EXPLICIT',"
                        "pre_admin_role=NULL,pre_admin_status=NULL,"
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
            except ProjectAuthorityError:
                raise
            except Exception as exc:
                raise _wrap_unexpected_db_error(exc) from exc

        return await self._store.mutation(op)

    async def change_project_role(self, request: MutationAuthorizationRequest, project_id: int,
                                  user_id: int, role: ProjectRole) -> None:
        # REVIEWER is an approval/audit function, not an assignable project
        # membership role.  Reject it before opening a transaction so an
        # invalid request cannot acquire locks or issue any SQL.
        if role is ProjectRole.REVIEWER:
            raise AuthorizationDenied("REVIEWER cannot be assigned through change_project_role")

        async def op(cur: Any) -> None:
            try:
                scope = request.scope

                async def _lock_target(cur: Any) -> Any:
                    await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                                      (user_id, scope.tenant_id))
                    return await cur.fetchone()

                actor, target_user = await self._resolve_project_actor_cur(
                    cur, request, project_id, expected_operation="CHANGE_PROJECT_ROLE",
                    min_role=ProjectRole.OWNER, allowed_states=frozenset({ProjectLifecycle.ACTIVE}),
                    lock_target=_lock_target)
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
            except ProjectAuthorityError:
                raise
            except Exception as exc:
                raise _wrap_unexpected_db_error(exc) from exc

        await self._store.mutation(op)

    async def revoke_member(self, request: MutationAuthorizationRequest, project_id: int, user_id: int) -> None:
        async def op(cur: Any) -> None:
            try:
                scope = request.scope

                async def _lock_target(cur: Any) -> Any:
                    await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                                      (user_id, scope.tenant_id))
                    return await cur.fetchone()

                actor, target_user = await self._resolve_project_actor_cur(
                    cur, request, project_id, expected_operation="REVOKE_MEMBER",
                    min_role=ProjectRole.OWNER,
                    allowed_states=frozenset({ProjectLifecycle.ACTIVE, ProjectLifecycle.ARCHIVED}),
                    lock_target=_lock_target)
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
                # MINOR 2: clear pre_admin_* defensively -- a revoked,
                # non-TENANT_ADMIN-origin row must never carry them (the
                # CHECK constraint requires it), regardless of how it got here.
                await cur.execute(
                    "UPDATE jax_project_membership SET status='REVOKED',pre_admin_role=NULL,pre_admin_status=NULL,"
                    "version=version+1,updated_at=NOW(6) WHERE project_id=%s AND user_id=%s", (project_id, user_id))
                await self._event(cur, scope, "REVOKE_MEMBER", project_id, int(user_id), actor.tenant_id,
                                  old_role, old_role, "ACTIVE", "REVOKED")
            except ProjectAuthorityError:
                raise
            except Exception as exc:
                raise _wrap_unexpected_db_error(exc) from exc

        await self._store.mutation(op)

    async def set_project_lifecycle(self, request: MutationAuthorizationRequest, project_id: int,
                                    target: ProjectLifecycle) -> bool:
        async def op(cur: Any) -> bool:
            try:
                actor, _ = await self._resolve_project_actor_cur(
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
            except ProjectAuthorityError:
                raise
            except Exception as exc:
                raise _wrap_unexpected_db_error(exc) from exc

        return await self._store.mutation(op)

    async def sync_tenant_admin_memberships_in_transaction(
        self, cur: Any, *, actor_scope: ScopeContext, user_id: int, tenant_id: int,
    ) -> int:
        """Called by the platform's own `jax_users` admin-role transaction,
        AFTER it has already updated the target's `role`/`status` -- this
        function reads that CURRENT state itself (ronda 3, MAJOR M2) rather
        than trusting caller-supplied `was_active_admin`/`is_active_admin`
        flags, which is what let a forged pair of booleans grant OWNER on
        every project of a tenant, or skip the descent entirely. There is
        nothing left to trust: both branches below are independently
        idempotent no-ops when they do not apply (the ascent loop's
        "existing row" branch, the descent query's own
        `grant_origin='TENANT_ADMIN' AND status='ACTIVE'` filter), so a
        single DB-derived `is_active_admin` is enough -- no separate "was"
        needed.

        Takes `jax_tenants(tenant_id) FOR UPDATE` itself; re-acquiring a row
        lock a caller's own transaction already holds is a no-op in InnoDB,
        so this is safe whether or not the platform's own transaction did it
        first.
        """
        _require_project_actor_subject(actor_scope)
        await self._lock_tenant(cur, tenant_id)
        await cur.execute("SELECT status,role FROM jax_users WHERE user_id=%s AND tenant_id=%s FOR UPDATE",
                          (actor_scope.subject_user_id, tenant_id))
        actor = await cur.fetchone()
        if (not actor or str(self._value(actor, "status", 0)).upper() != "ACTIVE"
                or str(self._value(actor, "role", 1) or "").lower() not in TENANT_ADMIN_ROLES):
            raise ProjectRoleInsufficient("sync requires an active tenant administrator actor")
        # Destino: read tenant/role/status FOR UPDATE, not filtered by
        # tenant_id, so a cross-tenant target gets a clear rejection (MAJOR
        # M2c) instead of a silent "not found" that could be confused with
        # "already fine, nothing to do".
        await cur.execute("SELECT tenant_id,status,role FROM jax_users WHERE user_id=%s FOR UPDATE", (user_id,))
        target = await cur.fetchone()
        if not target:
            raise TargetUserNotEligible("target user does not exist")
        target_tenant = self._value(target, "tenant_id", 0)
        if int(target_tenant) != int(tenant_id):
            raise TargetUserNotEligible("target user belongs to a different tenant")
        is_active_admin = (str(self._value(target, "role", 2) or "").lower() in TENANT_ADMIN_ROLES
                           and str(self._value(target, "status", 1) or "").lower() == "active")
        touched = 0
        if is_active_admin:
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
                if str(origin) == "TENANT_ADMIN" and role.upper() == "OWNER" and status.upper() == "ACTIVE":
                    continue  # already synced -- true idempotence, no re-touch
                await cur.execute(
                    "UPDATE jax_project_membership SET pre_admin_role=%s,pre_admin_status=%s,"
                    "project_role='OWNER',status='ACTIVE',grant_origin='TENANT_ADMIN',version=version+1,"
                    "updated_at=NOW(6) WHERE project_id=%s AND user_id=%s", (role, status, pid, user_id))
                await self._event(cur, actor_scope, "GRANT_TENANT_ADMIN", pid, user_id, tenant_id,
                                  role, "OWNER", status, "ACTIVE")
                touched += 1
        else:
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
                pre_role = self._value(row, "pre_admin_role", 1)
                pre_status = self._value(row, "pre_admin_status", 2)
                # MINOR 1: restoring to OWNER/ACTIVE keeps the user an
                # owner -- it is not a demotion at all, so it must never
                # trip the last-owner guard.
                remains_owner = (pre_role is not None and str(pre_role).upper() == "OWNER"
                                and pre_status is not None and str(pre_status).upper() == "ACTIVE")
                if not remains_owner and await self._count_other_active_owners(cur, pid, user_id) == 0:
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
                        "UPDATE jax_project_membership SET status='REVOKED',pre_admin_role=NULL,pre_admin_status=NULL,"
                        "version=version+1,updated_at=NOW(6) WHERE project_id=%s AND user_id=%s", (pid, user_id))
                    new_status = "REVOKED"
                await self._event(cur, actor_scope, "REVOKE_TENANT_ADMIN", pid, user_id, tenant_id,
                                  "OWNER", pre_role, "ACTIVE", new_status)
                touched += 1
        return touched
