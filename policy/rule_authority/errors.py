"""Errores cerrados del Rule Authority Kernel (F1.1 paso 3).

Fallo cerrado: toda falla de schema o de snapshot es un error tipado; ningun
camino convierte un error en «regla aceptada».
"""
from __future__ import annotations


class RuleAuthorityError(RuntimeError):
    """Base de los errores de este paquete."""


class RuleAuthorityStorageError(RuleAuthorityError):
    """Persistencia durable no disponible o inconsistente; nunca equivale a PERMIT."""


class RuleSchemaError(RuleAuthorityError):
    """Una regla no cumple el shape cerrado rule-v1 (§7 + R-4)."""


class RuleSnapshotError(RuleAuthorityError):
    """El snapshot de policy/faro no se puede cargar con garantias (§6)."""


# ------------------------------------------------------------- paso 7 (§9, §11)

class ProveedorInvalido(RuleAuthorityError):
    """Un proveedor confiable falta o no satisface el contrato (p.ej. solo
    relee valores, sin leases): NO se emite ni se consume nada."""


class VersionRetrocedio(RuleAuthorityError):
    """Una version monotonica no repite ni retrocede (sin ABA, §9)."""


class StopDesconocido(RuleAuthorityError):
    """STOP ilegible o sin configurar: el kernel traduce esto en DENY (fail-closed)."""


class CheckpointInvalido(RuleAuthorityError):
    """El checkpoint externo rechaza retrocesos y heads conflictivos (§11)."""


class RelojInvalido(RuleAuthorityError):
    """El reloj entrego una hora sin timezone o invalida."""


class RelojRetrocedio(RuleAuthorityError):
    """El reloj retrocedio respecto de la emision: DENY (§13)."""


class ClasificacionDesconocida(RuleAuthorityError):
    """Capability sin clasificacion confiable: DENY, nunca rebaja (§7)."""
