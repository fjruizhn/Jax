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


# -------------------------------------------------- paso 7 r2 (§9, §10, §11)

class ProveedorInvalido(RuleAuthorityError):
    """Un proveedor confiable falta o no satisface el contrato (isinstance,
    leases con vista, tipos devueltos): NO se emite ni se consume nada."""


class StopDesconocido(RuleAuthorityError):
    """STOP ilegible o sin configurar: el kernel lo traduce en DENY (fail-closed)."""


class CheckpointInvalido(RuleAuthorityError):
    """El checkpoint externo rechaza retrocesos, heads conflictivos e
    idempotencias inexactas (§11)."""


class RelojInvalido(RuleAuthorityError):
    """El reloj entrego una hora sin timezone o invalida."""


class RelojRetrocedio(RuleAuthorityError):
    """El reloj retrocedio (contra la ultima lectura o contra la emision): DENY."""


class ClasificacionDesconocida(RuleAuthorityError):
    """Capability sin clasificacion confiable: el kernel OBLIGA, nunca rebaja."""
