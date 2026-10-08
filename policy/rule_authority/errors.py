"""Errores cerrados del Rule Authority Kernel (F1.1 paso 3).

Fallo cerrado: toda falla de schema o de snapshot es un error tipado; ningun
camino convierte un error en «regla aceptada».
"""
from __future__ import annotations


class RuleAuthorityError(RuntimeError):
    """Base de los errores de este paquete."""


class RuleSchemaError(RuleAuthorityError):
    """Una regla no cumple el shape cerrado rule-v1 (§7 + R-4)."""


class RuleSnapshotError(RuleAuthorityError):
    """El snapshot de policy/faro no se puede cargar con garantias (§6)."""
