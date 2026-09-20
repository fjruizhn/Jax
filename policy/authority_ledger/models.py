"""Closed, immutable authority-ledger value objects.

The ledger owns lifecycle/effective-state facts.  It deliberately never
models a permission or an execution decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
import unicodedata
from typing import Any

from .canonical import canonical_bytes, domain_hash
from .errors import AuthorityEventValidationError, AuthorityStateError
from .ids import canonical_evidence_refs, sha256_id, uuid7_text

_TOKEN = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_DOC = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_ACTOR = re.compile(r"(?:human|actor):[a-z][a-z0-9-]{0,63}\Z")


def _nfc_token(value: object, field: str, pattern=_TOKEN) -> str:
    if not isinstance(value, str) or not value or unicodedata.normalize("NFC", value) != value or not pattern.fullmatch(value):
        raise AuthorityEventValidationError(f"{field} inválido")
    return value


def _doc_id(value: object, field: str) -> str:
    return _nfc_token(value, field, _DOC)


def _actor(value: object, field: str) -> str:
    return _nfc_token(value, field, _ACTOR)


def _time(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AuthorityEventValidationError(f"{field} debe ser datetime UTC")
    return value.astimezone(timezone.utc)


def _set_ids(values: object, field: str, parser=_doc_id) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise AuthorityEventValidationError(f"{field} debe ser array")
    result = tuple(parser(item, field) for item in values)
    if len(result) != len(set(result)):
        raise AuthorityEventValidationError(f"{field} duplicado")
    return tuple(sorted(result))


class AuthorityEventType(str, Enum):
    RATIFICATION_GRANTED = "RATIFICATION_GRANTED"
    RATIFICATION_REVOKED = "RATIFICATION_REVOKED"
    ACTIVATION_GRANTED = "ACTIVATION_GRANTED"
    ACTIVATION_DEACTIVATED = "ACTIVATION_DEACTIVATED"
    OVERLAY_ISSUED = "OVERLAY_ISSUED"
    OVERLAY_REVOKED = "OVERLAY_REVOKED"


class OverlayType(str, Enum):
    EXCEPTION = "EXCEPTION"
    DELEGATION = "DELEGATION"
    SUSPENSION = "SUSPENSION"
    BINDING_PARTICULAR_INTERPRETATION = "BINDING_PARTICULAR_INTERPRETATION"


class OverlayApplicability(str, Enum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class OverlayScope:
    subjects: tuple[str, ...]
    actions: tuple[str, ...]
    conditions_all: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subjects", _set_ids(self.subjects, "scope.subjects", _nfc_token))
        object.__setattr__(self, "actions", _set_ids(self.actions, "scope.actions", _nfc_token))
        object.__setattr__(self, "conditions_all", _set_ids(self.conditions_all, "scope.conditions_all", _nfc_token))
        if not self.subjects or not self.actions:
            raise AuthorityEventValidationError("scope requiere subjects/actions")


@dataclass(frozen=True)
class OverlayPayload:
    overlay_id: str
    overlay_type: OverlayType
    policy_corpus_hash: str
    scope: OverlayScope
    valid_from_utc: datetime
    valid_until_utc: datetime | None
    target_rule_ids: tuple[str, ...] = ()
    delegate_actor_id: str | None = None
    delegated_action_ids: tuple[str, ...] = ()
    interpretation_code: str | None = None
    target_overlay_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "overlay_id", _nfc_token(self.overlay_id, "overlay_id", _DOC))
        sha256_id(self.policy_corpus_hash, "policy_corpus_hash")
        object.__setattr__(self, "valid_from_utc", _time(self.valid_from_utc, "valid_from_utc"))
        if self.valid_until_utc is not None:
            object.__setattr__(self, "valid_until_utc", _time(self.valid_until_utc, "valid_until_utc"))
            if self.valid_until_utc <= self.valid_from_utc:
                raise AuthorityEventValidationError("valid_until debe ser posterior")
        object.__setattr__(self, "target_rule_ids", _set_ids(self.target_rule_ids, "target_rule_ids"))
        object.__setattr__(self, "delegated_action_ids", _set_ids(self.delegated_action_ids, "delegated_action_ids", _nfc_token))
        if self.delegate_actor_id is not None:
            object.__setattr__(self, "delegate_actor_id", _actor(self.delegate_actor_id, "delegate_actor_id"))
        if self.target_overlay_id is not None:
            object.__setattr__(self, "target_overlay_id", _nfc_token(self.target_overlay_id, "target_overlay_id", _DOC))
        if self.interpretation_code is not None:
            object.__setattr__(self, "interpretation_code", _nfc_token(self.interpretation_code, "interpretation_code"))
        if self.overlay_type is OverlayType.DELEGATION:
            if self.delegate_actor_id is None or not self.delegated_action_ids:
                raise AuthorityEventValidationError("delegación requiere delegate y actions")
        elif self.overlay_type is OverlayType.SUSPENSION:
            if self.target_overlay_id is None or self.target_rule_ids or self.delegate_actor_id or self.interpretation_code:
                raise AuthorityEventValidationError("suspensión sólo puede targetear overlay")
        elif self.overlay_type is OverlayType.BINDING_PARTICULAR_INTERPRETATION:
            if not self.target_rule_ids or self.interpretation_code is None:
                raise AuthorityEventValidationError("interpretación requiere target/code")
        elif self.overlay_type is OverlayType.EXCEPTION:
            if not self.target_rule_ids:
                raise AuthorityEventValidationError("exception requiere target_rule_ids")

    def semantic_projection(self) -> dict[str, Any]:
        """Identity projection; excludes forensic IDs/evidence/signatures."""
        return {
            "overlay_type": self.overlay_type.value,
            "policy_corpus_hash": self.policy_corpus_hash,
            "scope": {"subjects": list(self.scope.subjects), "actions": list(self.scope.actions), "conditions_all": list(self.scope.conditions_all)},
            "valid_from_utc": self.valid_from_utc.isoformat(),
            "valid_until_utc": self.valid_until_utc.isoformat() if self.valid_until_utc else None,
            "target_rule_ids": list(self.target_rule_ids),
            "delegate_actor_id": self.delegate_actor_id,
            "delegated_action_ids": list(self.delegated_action_ids),
            "interpretation_code": self.interpretation_code,
            "target_overlay_id": self.target_overlay_id,
        }


@dataclass(frozen=True)
class AuthorityEventIntent:
    event_type: AuthorityEventType
    actor_id: str
    evidence_refs: tuple[str, ...] = ()
    policy_corpus_hash: str | None = None
    static_policy_view_projection: dict[str, Any] | None = None
    ratification_event_id: str | None = None
    overlay: OverlayPayload | None = None
    overlay_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _actor(self.actor_id, "actor_id"))
        object.__setattr__(self, "evidence_refs", canonical_evidence_refs(self.evidence_refs))
        if self.policy_corpus_hash is not None:
            sha256_id(self.policy_corpus_hash, "policy_corpus_hash")
        if self.ratification_event_id is not None:
            uuid7_text(self.ratification_event_id, "ratification_event_id")
        if self.overlay_id is not None:
            _nfc_token(self.overlay_id, "overlay_id", _DOC)
        required = {
            AuthorityEventType.RATIFICATION_GRANTED: ("policy_corpus_hash",),
            AuthorityEventType.RATIFICATION_REVOKED: ("ratification_event_id",),
            AuthorityEventType.ACTIVATION_GRANTED: ("ratification_event_id",),
            AuthorityEventType.ACTIVATION_DEACTIVATED: (),
            AuthorityEventType.OVERLAY_ISSUED: ("overlay",),
            AuthorityEventType.OVERLAY_REVOKED: ("overlay_id",),
        }[self.event_type]
        for name in required:
            if getattr(self, name) is None:
                raise AuthorityEventValidationError(f"{self.event_type}: {name} requerido")
        if self.event_type is AuthorityEventType.RATIFICATION_GRANTED:
            if not isinstance(self.static_policy_view_projection, dict) or self.static_policy_view_projection.get("policy_corpus_hash") != self.policy_corpus_hash:
                raise AuthorityEventValidationError("ratificación requiere static policy view ligado al hash")

    def canonical_projection(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value, "actor_id": self.actor_id,
            "evidence_refs": list(self.evidence_refs), "policy_corpus_hash": self.policy_corpus_hash,
            "static_policy_view_projection": self.static_policy_view_projection,
            "ratification_event_id": self.ratification_event_id,
            "overlay": self.overlay.semantic_projection() | {"overlay_id": self.overlay.overlay_id} if self.overlay else None,
            "overlay_id": self.overlay_id,
        }


@dataclass(frozen=True)
class AuthorityEvent:
    event_id: str
    sequence: int
    previous_event_hash: str | None
    intent: AuthorityEventIntent
    recorded_at_utc: datetime
    signature: str
    event_hash: str

    def __post_init__(self) -> None:
        uuid7_text(self.event_id)
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1:
            raise AuthorityEventValidationError("sequence inválido")
        if self.previous_event_hash is not None:
            sha256_id(self.previous_event_hash, "previous_event_hash")
        object.__setattr__(self, "recorded_at_utc", _time(self.recorded_at_utc, "recorded_at_utc"))
        sha256_id(self.event_hash, "event_hash")

    def unsigned_projection(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "sequence": self.sequence, "previous_event_hash": self.previous_event_hash, "intent": self.intent.canonical_projection(), "recorded_at_utc": self.recorded_at_utc.isoformat()}


@dataclass(frozen=True)
class AuthorityLedgerGenesis:
    schema_version: str
    kind: str
    ledger_identity: str
    constitutional_actor_id: str
    constitutional_key_id: str
    constitutional_public_key: str

    def __post_init__(self) -> None:
        if (self.schema_version, self.kind, self.ledger_identity) != ("1.0", "JAX_AUTHORITY_LEDGER_GENESIS", "JAX-AUTHORITY-LEDGER/1"):
            raise AuthorityEventValidationError("genesis inválido")
        _actor(self.constitutional_actor_id, "constitutional_actor_id")
        if self.constitutional_actor_id != "human:fernando":
            raise AuthorityEventValidationError("sólo Fernando es ratificador constitucional")

    def projection(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "kind": self.kind, "ledger_identity": self.ledger_identity, "constitutional_actor_id": self.constitutional_actor_id, "constitutional_key_id": self.constitutional_key_id, "constitutional_public_key": self.constitutional_public_key}


@dataclass(frozen=True)
class AuthorityLedgerCheckpoint:
    schema_version: str
    kind: str
    ledger_identity: str
    sequence: int
    head_event_id: str | None
    head_event_hash: str | None

    def projection(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "kind": self.kind, "ledger_identity": self.ledger_identity, "sequence": self.sequence, "head_event_id": self.head_event_id, "head_event_hash": self.head_event_hash}

    @property
    def authority_ledger_checkpoint_hash(self) -> str:
        return domain_hash("JAX-AUTHORITY-LEDGER-CHECKPOINT", "1.0", self.projection())


@dataclass(frozen=True)
class EffectiveAuthorityContext:
    schema_version: str
    kind: str
    active_policy_corpus_hash: str | None
    resolver_identity: str
    resolver_version: str
    effective_overlays: tuple[OverlayPayload, ...]
    verified_external_ceilings: tuple[dict[str, Any], ...] = ()

    def projection(self) -> dict[str, Any]:
        # Evaluation facts and forensic event data are deliberately absent.
        return {"schema_version": self.schema_version, "kind": self.kind, "active_policy_corpus_hash": self.active_policy_corpus_hash, "resolver_identity": self.resolver_identity, "resolver_version": self.resolver_version, "effective_overlays": [x.semantic_projection() for x in self.effective_overlays], "verified_external_ceilings": list(self.verified_external_ceilings)}

    @property
    def effective_authority_context_hash(self) -> str:
        return domain_hash("JAX-EFFECTIVE-AUTHORITY-CONTEXT", "1.0", self.projection())


@dataclass(frozen=True)
class EffectiveAuthorityEnvelope:
    schema_version: str
    kind: str
    static_resolution: Any
    active_policy_corpus_hash: str
    effective_authority_context_hash: str
    authority_ledger_checkpoint_hash: str
    relevant_overlay_ids: tuple[str, ...]
    lifecycle_status: str

    # No execution eligibility or authorization field exists by contract.
