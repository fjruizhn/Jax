"""Errores cerrados del canonicalizer shadow."""


class CanonicalizationError(ValueError):
    """Base para entradas que no pueden canonicalizarse sin ambigüedad."""


class StrictYAMLError(CanonicalizationError):
    """La fuente viola el subconjunto YAML admitido."""


class BootstrapIntegrityError(CanonicalizationError):
    """El bootstrap no coincide con la raíz pinneada en la implementación."""


class SchemaValidationError(CanonicalizationError):
    """Un documento no satisface el schema bootstrap verificado."""


class DuplicateRuleIdError(CanonicalizationError):
    """Dos miembros del corpus declaran el mismo id normativo."""
