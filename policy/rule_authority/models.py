"""Closed immutable request/evaluation/decision models for Faro F1.1."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
import unicodedata
from types import MappingProxyType
from typing import Any, Mapping

from policy.authority_ledger.canonical import domain_hash
from policy.authority_ledger.errors import AuthorityEventValidationError
from policy.authority_ledger.ids import uuid7_text

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9._:-]{0,127}\Z")
# This is only the ISO 4217 alphabetic-code shape. Membership is checked against
# the trusted rule/capability contract before an evaluation can be authoritative.
_CURRENCY = re.compile(r"[A-Z]{3}\Z")
_REQUEST_HASH_DOMAIN = "JAX-FARO-RULE-REQUEST"
_REQUEST_HASH_VERSION = "1"
_DENY_REASONS = frozenset({
    "AUTHORITY_INVALID",
    "CAPABILITY_MISMATCH",
    "GRANT_INVALID",
    "OUT_OF_SCOPE",
    "RULE_CHANGED",
    "RULE_EXPIRED",
    "RULE_NOT_FOUND",
    "STOP_ACTIVE",
    "STORAGE_FAILURE",
})


def _identifier(value: object, field: str) -> str:
    if (not isinstance(value, str) or not value or
            unicodedata.normalize("NFC", value) != value or
            not _IDENTIFIER.fullmatch(value)):
        raise AuthorityEventValidationError(f"{field} inválido")
    return value


def _freeze_json(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and unicodedata.normalize("NFC", value) != value:
            raise AuthorityEventValidationError(f"{field} debe estar en NFC")
        return value
    if isinstance(value, float):
        raise AuthorityEventValidationError(f"{field} no admite valores float")
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or unicodedata.normalize("NFC", key) != key:
                raise AuthorityEventValidationError(f"{field} contiene clave inválida")
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


@dataclass(frozen=True)
class RuleLimits:
    """Only effect and cost limits are representable; infrastructure limits have no fields."""

    quantity: int | None = None
    quantity_unit: str | None = None
    amount: int | None = None
    currency: str | None = None
    frequency_count: int | None = None
    frequency_window_seconds: int | None = None
    duration_seconds: int | None = None
    tokens: int | None = None
    cost_minor_units: int | None = None
    cost_currency: str | None = None

    def __post_init__(self) -> None:
        for name in ("quantity", "amount", "frequency_count",
                     "frequency_window_seconds", "duration_seconds", "tokens",
                     "cost_minor_units"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise AuthorityEventValidationError(f"{name} debe ser entero positivo")
        for name in ("currency", "cost_currency"):
            currency = getattr(self, name)
            if currency is not None and (not isinstance(currency, str) or not _CURRENCY.fullmatch(currency)):
                raise AuthorityEventValidationError(f"{name} debe tener formato alpha-3")
        if (self.quantity is None) != (self.quantity_unit is None):
            raise AuthorityEventValidationError("quantity y quantity_unit deben ir juntos")
        if self.quantity_unit is not None:
            object.__setattr__(self, "quantity_unit", _identifier(self.quantity_unit, "quantity_unit"))
        if (self.amount is None) != (self.currency is None):
            raise AuthorityEventValidationError("amount y currency deben ir juntos")
        if (self.cost_minor_units is None) != (self.cost_currency is None):
            raise AuthorityEventValidationError("cost_minor_units y cost_currency deben ir juntos")
        if (self.frequency_count is None) != (self.frequency_window_seconds is None):
            raise AuthorityEventValidationError("frecuencia requiere count y window")
        if self.quantity is not None and self.amount is not None:
            raise AuthorityEventValidationError("límites no pueden combinar quantity y amount")


@dataclass(frozen=True)
class RuleEvaluationRequest:
    request_id: str
    rule_id: str
    subject: str
    capability: str
    objective: str
    resource_id: str
    arguments: Mapping[str, Any]
    quantity: int | None = None
    quantity_unit: str | None = None
    amount: int | None = None
    currency: str | None = None
    request_hash: str | None = None

    def __post_init__(self) -> None:
        uuid7_text(self.request_id, "request_id")
        for field in ("rule_id", "subject", "capability", "objective", "resource_id"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field))
        if not isinstance(self.arguments, Mapping):
            raise AuthorityEventValidationError("arguments debe ser un objeto JSON")
        frozen = _freeze_json(self.arguments, "arguments")
        object.__setattr__(self, "arguments", frozen)
        if self.quantity is not None and (type(self.quantity) is not int or self.quantity <= 0):
            raise AuthorityEventValidationError("quantity debe ser entero positivo")
        if (self.quantity is None) != (self.quantity_unit is None):
            raise AuthorityEventValidationError("quantity y quantity_unit deben ir juntos")
        if self.quantity_unit is not None:
            object.__setattr__(self, "quantity_unit", _identifier(self.quantity_unit, "quantity_unit"))
        if self.amount is not None and (type(self.amount) is not int or self.amount <= 0):
            raise AuthorityEventValidationError("amount debe ser entero positivo en unidades menores")
        if (self.quantity is not None and self.amount is not None):
            raise AuthorityEventValidationError("solicitud no puede limitarse por quantity y amount a la vez")
        if (self.amount is None) != (self.currency is None):
            raise AuthorityEventValidationError("amount y currency deben ir juntos")
        if self.currency is not None and (not isinstance(self.currency, str) or not _CURRENCY.fullmatch(self.currency)):
            raise AuthorityEventValidationError("currency debe tener formato alpha-3")
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
        if not isinstance(self.request, RuleEvaluationRequest) or not isinstance(self.limits, RuleLimits):
            raise AuthorityEventValidationError("evaluación requiere request y RuleLimits")


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
        if not isinstance(self.request_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.request_hash):
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
