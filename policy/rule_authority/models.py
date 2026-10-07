"""Closed immutable request/evaluation/decision models for Faro F1.1.

Una sola fuente de verdad para los topes: el catálogo SELLADO del pin
(``jax.faro.catalogo_topes.CatalogoTopes``, #370) y el modelo de regla de #370
(``ReglaValidada`` / ``LimitesObligatorios``). Este módulo no define taxonomía
de unidades ni campos de límites propios: los límites se derivan con
``limites_de`` y el catálogo se exige por tipo exacto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
import unicodedata
from types import MappingProxyType
from typing import Any, Mapping

from jax.faro.catalogo_topes import CatalogoTopes, es_de_catalogo
from policy.authority_ledger.canonical import domain_hash
from policy.authority_ledger.errors import AuthorityEventValidationError
from policy.authority_ledger.ids import uuid7_text
from .schema import (MAX_CANTIDAD, Cantidad, Frecuencia, LimitesObligatorios, Monto,
                     ReglaValidada, Tope)

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9._:-]{0,127}\Z")
_ISO_MONEDA = re.compile(r"[A-Z]{3}\Z", re.ASCII)
_REQUEST_HASH_DOMAIN = "JAX-FARO-RULE-REQUEST"
_REQUEST_HASH_VERSION = "2"        # v2: la identidad del catálogo (OID del pin) entra al hash
_DENY_REASONS = frozenset({
    "AUTHORITY_INVALID",
    "CAPABILITY_MISMATCH",
    "GRANT_INVALID",
    "OUT_OF_SCOPE",
    "RULE_CHANGED",
    "RULE_EXPIRED",
    "RULE_NOT_FOUND",
    "STOP_ACTIVE",
})


def _catalogo(value: object, field_name: str) -> CatalogoTopes:
    # Tipo EXACTO: ni un Mapping armado a mano ni una subclase con __slots__ que
    # salte el testigo del constructor.
    if type(value) is not CatalogoTopes:
        raise AuthorityEventValidationError(
            f"{field_name} requiere el catálogo sellado del snapshot (CatalogoTopes)")
    return value


def _entero(value: object, field_name: str) -> int:
    if type(value) is not int or not 0 < value <= MAX_CANTIDAD:
        raise AuthorityEventValidationError(
            f"{field_name} debe ser entero positivo <= {MAX_CANTIDAD}")
    return value


def _unidad_de(catalogo: CatalogoTopes, clase: str, value: object, field_name: str) -> str:
    # `type(...) is str`: una subclase con __eq__ == True pasaría el `in` del catálogo.
    if type(value) is not str or value not in catalogo.get(clase, ()):
        raise AuthorityEventValidationError(f"{field_name} fuera del catálogo {clase}")
    return value


def _identifier(value: object, field: str) -> str:
    if (type(value) is not str or not value or
            unicodedata.normalize("NFC", value) != value or
            not _IDENTIFIER.fullmatch(value)):
        raise AuthorityEventValidationError(f"{field} inválido")
    return value


def _freeze_json(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            value = unicodedata.normalize("NFC", value)
        elif isinstance(value, int) and not isinstance(value, bool) and abs(value) > MAX_CANTIDAD:
            raise AuthorityEventValidationError(
                f"{field} contiene un entero fuera de rango (|n| <= {MAX_CANTIDAD})")
        return value
    if isinstance(value, float):
        raise AuthorityEventValidationError(f"{field} no admite valores float")
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise AuthorityEventValidationError(f"{field} contiene clave inválida")
            key = unicodedata.normalize("NFC", key)
            if key in frozen:
                raise AuthorityEventValidationError(f"{field} contiene claves duplicadas tras NFC")
            frozen[key] = _freeze_json(item, field)
        return MappingProxyType(frozen)
    raise AuthorityEventValidationError(f"{field} debe ser JSON cerrado")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AuthorityEventValidationError(f"{field} debe incluir zona horaria")
    return value.astimezone(timezone.utc)


_LIMITES_DE = object()      # procedencia: solo `limites_de` lo tiene


@dataclass(frozen=True)
class RuleLimits:
    """Envoltorio inmutable de los límites de UNA regla validada (#370).

    No tiene campos de límites propios: ``limites`` es el ``LimitesObligatorios``
    de la ``ReglaValidada`` y ``tope`` su ``Tope`` (donde viven frecuencia,
    duración y tokens/costo, siempre ``<clase>.<subid>`` del catálogo). Solo se
    obtiene con ``limites_de``; ningún llamador arma límites a mano."""

    catalogo: CatalogoTopes
    limites: LimitesObligatorios
    tope: Tope | None
    _origen: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._origen is not _LIMITES_DE:
            raise AuthorityEventValidationError("RuleLimits solo se obtiene con limites_de(regla, catalogo)")
        catalogo = _catalogo(self.catalogo, "RuleLimits")
        limites = self.limites
        if type(limites) is not LimitesObligatorios:
            raise AuthorityEventValidationError("limites debe ser LimitesObligatorios de la regla")
        if self.tope is not None and type(self.tope) is not Tope:
            raise AuthorityEventValidationError("tope debe ser el Tope de la regla")
        if limites.quantity is None and limites.amount is None and limites.frequency is None \
                and self.tope is None:
            raise AuthorityEventValidationError("RuleLimits vacío")
        if limites.quantity is not None:
            if type(limites.quantity) is not Cantidad:
                raise AuthorityEventValidationError("quantity debe ser Cantidad")
            _unidad_de(catalogo, "actos_externos", limites.quantity.unit, "quantity.unit")
            _entero(limites.quantity.max, "quantity.max")
        if limites.amount is not None:
            monto = limites.amount
            if type(monto) is not Monto:
                raise AuthorityEventValidationError("amount debe ser Monto")
            # Monto usa la forma ISO de la regla (USD); el catálogo, su subid (usd).
            if type(monto.currency) is not str or not _ISO_MONEDA.fullmatch(monto.currency):
                raise AuthorityEventValidationError("amount.currency fuera del catálogo monto_dinero")
            _unidad_de(catalogo, "monto_dinero", monto.currency.lower(), "amount.currency")
            _entero(monto.max, "amount.max")
        if limites.frequency is not None:
            if type(limites.frequency) is not Frecuencia:
                raise AuthorityEventValidationError("frequency debe ser Frecuencia")
            _entero(limites.frequency.max_occurrences, "frequency.max_occurrences")
            _entero(limites.frequency.window_seconds, "frequency.window_seconds")
        if self.tope is not None:
            tope = self.tope
            if (type(tope.resource_class) is not str or type(tope.resource) is not str
                    or type(tope.period) is not str or not tope.period
                    or not es_de_catalogo(tope.resource, catalogo)
                    or tope.resource.partition(".")[0] != tope.resource_class):
                raise AuthorityEventValidationError("tope fuera del catálogo cerrado (<clase>.<subid>)")
            _entero(tope.maximum, "tope.maximum")


def limites_de(regla: ReglaValidada, catalogo: CatalogoTopes) -> RuleLimits:
    """Los límites de la regla validada, comprobados contra el catálogo sellado."""
    if type(regla) is not ReglaValidada:
        raise AuthorityEventValidationError("limites_de requiere una ReglaValidada")
    return RuleLimits(catalogo, regla.obligation_limits, regla.tope, _origen=_LIMITES_DE)


@dataclass(frozen=True)
class RuleEvaluationRequest:
    request_id: str
    rule_id: str
    subject: str
    capability: str
    objective: str
    resource_id: str
    arguments: Mapping[str, Any]
    catalogo: CatalogoTopes
    quantity: int | None = None
    quantity_unit: str | None = None
    amount: int | None = None
    currency: str | None = None
    request_hash: str | None = None

    def __post_init__(self) -> None:
        uuid7_text(self.request_id, "request_id")
        _catalogo(self.catalogo, "RuleEvaluationRequest")
        for field_name in ("rule_id", "subject", "capability", "objective", "resource_id"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        if not isinstance(self.arguments, Mapping):
            raise AuthorityEventValidationError("arguments debe ser un objeto JSON")
        frozen = _freeze_json(self.arguments, "arguments")
        object.__setattr__(self, "arguments", frozen)
        if self.quantity is not None:
            _entero(self.quantity, "quantity")
        if (self.quantity is None) != (self.quantity_unit is None):
            raise AuthorityEventValidationError("quantity y quantity_unit deben ir juntos")
        if self.quantity_unit is not None:
            _unidad_de(self.catalogo, "actos_externos", self.quantity_unit, "quantity_unit")
        if self.amount is not None:
            _entero(self.amount, "amount")
        if self.quantity is not None and self.amount is not None:
            raise AuthorityEventValidationError("solicitud no puede limitarse por quantity y amount a la vez")
        if (self.amount is None) != (self.currency is None):
            raise AuthorityEventValidationError("amount y currency deben ir juntos")
        if self.currency is not None:
            _unidad_de(self.catalogo, "monto_dinero", self.currency, "currency")
        computed = domain_hash(_REQUEST_HASH_DOMAIN, _REQUEST_HASH_VERSION, self.canonical_projection())
        if self.request_hash is not None and self.request_hash != computed:
            raise AuthorityEventValidationError("request_hash no coincide con los campos")
        object.__setattr__(self, "request_hash", computed)

    def canonical_projection(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "rule_id": self.rule_id,
            "subject": self.subject,
            "capability": self.capability,
            "objective": self.objective,
            "resource_id": self.resource_id,
            "arguments": _thaw_json(self.arguments),
            "quantity": self.quantity,
            "quantity_unit": self.quantity_unit,
            "amount": self.amount,
            "currency": self.currency,
            "catalogo_oid": self.catalogo.oid_pin,
        }


class RuleDecisionStatus(str, Enum):
    MISSING_RULE = "MISSING_RULE"
    DENY = "DENY"
    PERMIT = "PERMIT"


@dataclass(frozen=True)
class RuleEvaluation:
    request: RuleEvaluationRequest
    limits: RuleLimits

    def __post_init__(self) -> None:
        if type(self.request) is not RuleEvaluationRequest or type(self.limits) is not RuleLimits:
            raise AuthorityEventValidationError("evaluación requiere request y RuleLimits")
        # CatalogoTopes.__eq__ compara clases Y OID del pin (tipo exacto ya comprobado).
        if self.request.catalogo != self.limits.catalogo:
            raise AuthorityEventValidationError(
                "solicitud y límites deben usar el mismo catálogo (misma identidad del pin)")


@dataclass(frozen=True)
class RuleDecision:
    request_id: str
    request_hash: str
    status: RuleDecisionStatus
    required_rule_id: str
    reason_code: str | None
    decided_at_utc: datetime

    def __post_init__(self) -> None:
        uuid7_text(self.request_id, "request_id")
        _identifier(self.required_rule_id, "required_rule_id")
        if type(self.request_hash) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.request_hash):
            raise AuthorityEventValidationError("request_hash inválido")
        if not isinstance(self.status, RuleDecisionStatus):
            raise AuthorityEventValidationError("status inválido")
        if self.status is RuleDecisionStatus.PERMIT:
            if self.reason_code is not None:
                raise AuthorityEventValidationError("PERMIT no lleva reason_code")
        elif self.reason_code not in _DENY_REASONS:
            raise AuthorityEventValidationError("negativa requiere reason_code estable")
        elif self.status is RuleDecisionStatus.MISSING_RULE and self.reason_code != "RULE_NOT_FOUND":
            raise AuthorityEventValidationError("MISSING_RULE requiere RULE_NOT_FOUND")
        elif self.status is RuleDecisionStatus.DENY and self.reason_code == "RULE_NOT_FOUND":
            raise AuthorityEventValidationError("RULE_NOT_FOUND requiere MISSING_RULE")
        object.__setattr__(self, "decided_at_utc", _utc(self.decided_at_utc, "decided_at_utc"))
