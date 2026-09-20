"""Typed failures at the loader/adapter/resolver boundary."""

class ResolverInputError(ValueError):
    """Base class for invalid resolver contracts."""

class InvalidEvaluationContextError(ResolverInputError):
    """The evaluation context is malformed."""

class InvalidValidatedCorpusError(ResolverInputError):
    """The loader did not produce a valid atomic corpus object."""

class ResolverContractError(ResolverInputError):
    """A caller supplied a raw or otherwise incompatible object."""

class ResolverInvariantError(RuntimeError):
    """An internal invariant that should be unreachable was violated."""
