"""Authoritative tenant-only scope resolution for the B9 boundary.

``jax_users`` is the identity source for the currently deployed tenant
relationship.  This resolver deliberately does *not* turn JWT claims or a
request's role/project fields into authority.  Project membership and
delegation have no designated persisted source yet, therefore those scopes
fail closed rather than acquiring a parallel B9 authority store.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .b9 import AuthorizationDenied, MutationAuthorizationContext, ScopeContext, ScopeDenied, Visibility


class MariaDBScopeAuthorityResolver:
    """Resolve tenant membership and mutation roles from active ``jax_users``.

    The supplied pool follows the aiomysql pool contract.  Resolution is
    intentionally async because membership is checked at the authoritative
    database for every boundary call; it is not a cache of JWT assertions.
    """

    authority_source = "jax_users.active_tenant_role"
    _ADMIN_ROLES = frozenset({"admin", "superadmin", "super_admin"})
    _REVIEWER_ROLES = frozenset({"reviewer", "memory_reviewer"})

    def __init__(self, pool: Any):
        self._pool = pool

    @staticmethod
    def _row_value(row: Any, key: str, position: int) -> Any:
        if isinstance(row, Mapping):
            return row.get(key)
        return row[position]

    @staticmethod
    def _require_non_delegated_tenant_scope(scope: ScopeContext) -> None:
        scope.validate()
        if scope.project_id:
            raise ScopeDenied("project membership authority is unavailable")
        if scope.delegation:
            raise ScopeDenied("delegation authority is unavailable")
        if not scope.subject_user_id:
            raise ScopeDenied("tenant membership requires a subject user")
        # A user is not allowed to select a different subject.  Service actors
        # remain distinct from their subject and are retained in provenance.
        if scope.actor_type == "USER" and scope.actor_principal not in {
            scope.subject_user_id, f"user:{scope.subject_user_id}"
        }:
            raise ScopeDenied("user actor and subject do not match")

    async def _active_membership(self, scope: ScopeContext) -> tuple[str, str]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT tenant_id,status,role FROM jax_users "
                    "WHERE user_id=%s AND tenant_id=%s LIMIT 1",
                    (scope.subject_user_id, scope.tenant_id),
                )
                row = await cur.fetchone()
        if not row:
            raise ScopeDenied("subject is not a tenant member")
        tenant_id = self._row_value(row, "tenant_id", 0)
        status = self._row_value(row, "status", 1)
        role = self._row_value(row, "role", 2)
        if str(tenant_id) != scope.tenant_id or str(status).lower() != "active":
            raise ScopeDenied("tenant membership is inactive or mismatched")
        return str(status), str(role or "").lower()

    @classmethod
    def _roles_and_capabilities(cls, role: str) -> tuple[frozenset[str], frozenset[str]]:
        roles = {"tenant_member"}
        capabilities = {"memory:tenant-scope"}
        if role in cls._ADMIN_ROLES:
            roles.update({"memory_admin", "memory_reviewer"})
            capabilities.update({"memory:admin", "memory:verify"})
        elif role in cls._REVIEWER_ROLES:
            roles.add("memory_reviewer")
            capabilities.add("memory:verify")
        return frozenset(roles), frozenset(capabilities)

    async def resolve_scope(self, scope: ScopeContext) -> ScopeContext:
        """Return only a scope proven by current tenant identity state."""
        self._require_non_delegated_tenant_scope(scope)
        await self._active_membership(scope)
        return scope

    async def resolve_mutation(self, scope: ScopeContext, operation: str,
                               target_visibility: Visibility) -> MutationAuthorizationContext:
        """Produce the only accepted input to persistent B9 mutations."""
        self._require_non_delegated_tenant_scope(scope)
        _, role = await self._active_membership(scope)
        roles, capabilities = self._roles_and_capabilities(role)
        if target_visibility is Visibility.PROJECT_SHARED:
            # This is redundant with the input check, but makes the denial
            # invariant explicit if a caller changes ScopeContext validation.
            raise ScopeDenied("project membership authority is unavailable")
        if operation == "VERIFY" and not roles.intersection({"memory_reviewer", "memory_admin"}):
            raise AuthorizationDenied("verification requires resolved reviewer authority")
        return MutationAuthorizationContext(
            scope, operation, target_visibility, roles, capabilities,
            self.authority_source,
        )
