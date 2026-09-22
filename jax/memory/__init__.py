"""JAX-owned memory modules.

Legacy ``MemoryDB`` remains available during B9 adoption. New model-facing
memory code must use the B9 boundary in :mod:`jax.memory.b9`.
"""

from .b9 import MemoryAPI, MemoryEnvelope, PromptMemoryContext, ScopeContext
from .b9_resolvers import DesignatedSourceResolver
from .scope_authority import MariaDBScopeAuthorityResolver

__all__ = ("MemoryAPI", "MemoryEnvelope", "PromptMemoryContext", "ScopeContext", "DesignatedSourceResolver", "MariaDBScopeAuthorityResolver")
