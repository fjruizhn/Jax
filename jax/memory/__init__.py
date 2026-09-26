"""JAX-owned memory modules.

Legacy ``MemoryDB`` remains available during B9 adoption. New model-facing
memory code must use the B9 boundary in :mod:`jax.memory.b9`.
"""

from .b9 import MemoryAPI, MemoryEnvelope, PromptMemoryContext, ScopeContext
from .b9_resolvers import DesignatedSourceResolver
from .scope_authority import (MariaDBScopeAuthorityResolver, ProjectLifecycle, ProjectReadAuthorization,
                              ProjectRole, ProjectScopeAuthorization, ProjectScopeAuthorityResolver,
                              TENANT_ADMIN_ROLES)
from .project_authority_migrations import (PROJECT_AUTHORITY_DDL, apply_project_authority_migration,
                                           revert_project_lifecycle_migration)
from .project_authority import (AlreadyMember, CreatedProject, IdempotencyKeyConflict, InvalidIdempotencyKey,
                                LastOwnerRequired, MemberNotFound, ProjectAuthorityAdmin, ProjectAuthorityError,
                                ProjectAuthorityRetryable, ProjectNotVisible, ProjectRoleInsufficient,
                                ProjectStateConflict, ReservedProjectIdRange, TargetUserNotEligible,
                                TenantAdminMembershipProtected)

__all__ = (
    "MemoryAPI", "MemoryEnvelope", "PromptMemoryContext", "ScopeContext", "DesignatedSourceResolver",
    "MariaDBScopeAuthorityResolver", "ProjectScopeAuthorityResolver", "ProjectScopeAuthorization",
    "ProjectReadAuthorization", "ProjectRole", "ProjectLifecycle", "TENANT_ADMIN_ROLES",
    "PROJECT_AUTHORITY_DDL", "apply_project_authority_migration", "revert_project_lifecycle_migration",
    "ProjectAuthorityAdmin", "ProjectAuthorityError", "ProjectAuthorityRetryable", "ProjectNotVisible",
    "ProjectRoleInsufficient", "ProjectStateConflict", "LastOwnerRequired", "TenantAdminMembershipProtected",
    "AlreadyMember", "MemberNotFound", "TargetUserNotEligible", "IdempotencyKeyConflict",
    "InvalidIdempotencyKey", "ReservedProjectIdRange", "CreatedProject",
)
