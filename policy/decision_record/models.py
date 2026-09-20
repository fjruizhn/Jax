"""Deeply immutable value objects for Block 5 decision identity and records."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
import unicodedata
from typing import Protocol

from policy.authority_ledger.models import AuthorityLedgerCheckpoint, EffectiveAuthorityEnvelope
from policy.authority_resolution.models import EvaluationContext, StaticAuthorityResolution

from .canonical import decision_input_hash, decision_record_hash, utc_text
from .errors import (DecisionContractError, DecisionRecordIntegrityError,
                     InvalidDecisionFactError, InvalidDecisionInputError,
                     UnverifiedAuthorityEvaluationError)
from .ids import canonical_evidence_refs, decision_id, sha256_id

_FACT_ID = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_INPUT_SEAL = object()
_EVALUATION_SEAL = object()
_RECORD_SEAL = object()


def _nfc(value: object, label: str) -> str:
    if not isinstance(value, str) or unicodedata.normalize("NFC", value) != value:
        raise InvalidDecisionFactError(f"{label} debe ser string NFC")
    return value


def _time(value: object, label: str, error=InvalidDecisionInputError) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise error(f"{label} debe ser datetime timezone-aware")
    return value.astimezone(timezone.utc)


class DecisionFactValueType(str, Enum):
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    STRING = "STRING"
    STRING_SET = "STRING_SET"


@dataclass(frozen=True)
class DecisionFact:
    id: str
    value_type: DecisionFactValueType
    value: bool | int | str | tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        value_id = _nfc(self.id, "fact.id")
        if not _FACT_ID.fullmatch(value_id):
            raise InvalidDecisionFactError("fact.id inválido")
        object.__setattr__(self, "id", value_id)
        if not isinstance(self.value_type, DecisionFactValueType):
            raise InvalidDecisionFactError("fact.value_type inválido")
        value = self.value
        if self.value_type is DecisionFactValueType.BOOLEAN:
            if type(value) is not bool:
                raise InvalidDecisionFactError("BOOLEAN requiere bool exacto")
        elif self.value_type is DecisionFactValueType.INTEGER:
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise InvalidDecisionFactError("INTEGER requiere int64 exacto")
        elif self.value_type is DecisionFactValueType.STRING:
            object.__setattr__(self, "value", _nfc(value, "fact.value"))
        else:
            if not isinstance(value, tuple):
                raise InvalidDecisionFactError("STRING_SET requiere tuple")
            values = tuple(_nfc(item, "fact.value") for item in value)
            if len(values) != len(set(values)):
                raise InvalidDecisionFactError("STRING_SET duplicado")
            object.__setattr__(self, "value", tuple(sorted(values)))
        try:
            object.__setattr__(self, "evidence_refs", canonical_evidence_refs(self.evidence_refs))
        except Exception as exc:
            raise InvalidDecisionFactError("evidence_refs inválidas") from exc

    def semantic_projection(self) -> dict:
        return {"id": self.id, "value_type": self.value_type.value, "value": list(self.value) if self.value_type is DecisionFactValueType.STRING_SET else self.value}

    def record_projection(self) -> dict:
        return self.semantic_projection() | {"evidence_refs": list(self.evidence_refs)}


@dataclass(frozen=True)
class DecisionInput:
    schema_version: str
    kind: str
    evaluation_context: EvaluationContext
    evaluation_time_utc: datetime
    facts: tuple[DecisionFact, ...]
    _seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (self.schema_version, self.kind) != ("1.0", "JAX_DECISION_INPUT"):
            raise InvalidDecisionInputError("DecisionInput schema/kind inválido")
        if not isinstance(self.evaluation_context, EvaluationContext):
            raise InvalidDecisionInputError("EvaluationContext requerido")
        object.__setattr__(self, "evaluation_time_utc", _time(self.evaluation_time_utc, "evaluation_time_utc"))
        if not isinstance(self.facts, tuple) or any(not isinstance(x, DecisionFact) for x in self.facts):
            raise InvalidDecisionInputError("facts debe ser tuple de DecisionFact")
        ids = tuple(item.id for item in self.facts)
        if len(ids) != len(set(ids)):
            raise InvalidDecisionInputError("facts duplicados")
        object.__setattr__(self, "facts", tuple(sorted(self.facts, key=lambda x: x.id)))
        if self._seal is not _INPUT_SEAL:
            raise InvalidDecisionInputError("DecisionInput sólo puede originarse en build_decision_input")

    def _is_sealed(self) -> bool:
        return self._seal is _INPUT_SEAL

    def semantic_projection(self) -> dict:
        context = self.evaluation_context
        return {
            "schema_version": self.schema_version, "kind": self.kind,
            "evaluation_context": {
                "schema_version": context.schema_version, "kind": context.kind,
                "jurisdiction": context.jurisdiction, "subject": context.subject,
                "action": context.action,
                "conditions": [{"id": item.id, "value": item.value} for item in context.conditions],
            },
            "evaluation_time_utc": utc_text(self.evaluation_time_utc),
            "facts": [item.semantic_projection() for item in self.facts],
        }

    def record_projection(self) -> dict:
        return self.semantic_projection() | {"facts": [item.record_projection() for item in self.facts]}

    @property
    def decision_input_hash(self) -> str:
        return decision_input_hash(self.semantic_projection())


@dataclass(frozen=True)
class DecisionAuthorityBinding:
    active_policy_corpus_hash: str
    effective_authority_context_hash: str
    authority_ledger_checkpoint: AuthorityLedgerCheckpoint
    authority_ledger_checkpoint_hash: str
    resolver_identity: str
    resolver_version: str

    def __post_init__(self) -> None:
        for name in ("active_policy_corpus_hash", "effective_authority_context_hash", "authority_ledger_checkpoint_hash"):
            try: sha256_id(getattr(self, name), name)
            except Exception as exc: raise DecisionRecordIntegrityError(f"{name} inválido") from exc
        if not isinstance(self.authority_ledger_checkpoint, AuthorityLedgerCheckpoint):
            raise DecisionRecordIntegrityError("checkpoint inválido")
        if self.authority_ledger_checkpoint.authority_ledger_checkpoint_hash != self.authority_ledger_checkpoint_hash:
            raise DecisionRecordIntegrityError("checkpoint/hash no coinciden")
        if (self.resolver_identity, self.resolver_version) != ("JAX-AUTHORITY-RESOLVER/1", "1.0"):
            raise DecisionRecordIntegrityError("resolver no soportado")

    def projection(self) -> dict:
        return {"active_policy_corpus_hash": self.active_policy_corpus_hash,
                "effective_authority_context_hash": self.effective_authority_context_hash,
                "authority_ledger_checkpoint": self.authority_ledger_checkpoint.projection(),
                "authority_ledger_checkpoint_hash": self.authority_ledger_checkpoint_hash,
                "resolver_identity": self.resolver_identity, "resolver_version": self.resolver_version}


@dataclass(frozen=True)
class EffectiveAuthorityEnvelopeSnapshot:
    schema_version: str
    kind: str
    static_resolution: StaticAuthorityResolution
    active_policy_corpus_hash: str
    effective_authority_context_hash: str
    authority_ledger_checkpoint_hash: str
    relevant_overlay_ids: tuple[str, ...]
    lifecycle_status: str

    @classmethod
    def from_envelope(cls, value: EffectiveAuthorityEnvelope) -> "EffectiveAuthorityEnvelopeSnapshot":
        if not isinstance(value, EffectiveAuthorityEnvelope) or not isinstance(value.static_resolution, StaticAuthorityResolution):
            raise UnverifiedAuthorityEvaluationError("EffectiveAuthorityEnvelope inválido")
        return cls(value.schema_version, value.kind, value.static_resolution, value.active_policy_corpus_hash,
                   value.effective_authority_context_hash, value.authority_ledger_checkpoint_hash,
                   tuple(sorted(set(value.relevant_overlay_ids))), value.lifecycle_status)

    def __post_init__(self) -> None:
        if (self.schema_version, self.kind, self.lifecycle_status) != ("1.0", "JAX_EFFECTIVE_AUTHORITY_ENVELOPE", "EFFECTIVE_STATIC_AUTHORITY_ONLY"):
            raise DecisionRecordIntegrityError("envelope snapshot inválido")
        if not isinstance(self.static_resolution, StaticAuthorityResolution):
            raise DecisionRecordIntegrityError("static resolution inválida")
        if self.static_resolution.policy_corpus_hash != self.active_policy_corpus_hash:
            raise DecisionRecordIntegrityError("resultado/corpus no coinciden")
        if tuple(self.relevant_overlay_ids) != tuple(sorted(self.relevant_overlay_ids)) or len(self.relevant_overlay_ids) != len(set(self.relevant_overlay_ids)):
            raise DecisionRecordIntegrityError("overlay IDs no canónicos")


@dataclass(frozen=True)
class DecisionResult:
    schema_version: str
    kind: str
    result_semantics: str
    effective_authority_envelope: EffectiveAuthorityEnvelopeSnapshot

    def __post_init__(self) -> None:
        if (self.schema_version, self.kind, self.result_semantics) != ("1.0", "JAX_DECISION_RESULT", "EFFECTIVE_AUTHORITY_ANALYSIS_ONLY"):
            raise DecisionRecordIntegrityError("DecisionResult inválido")
        if not isinstance(self.effective_authority_envelope, EffectiveAuthorityEnvelopeSnapshot):
            raise DecisionRecordIntegrityError("snapshot de envelope requerido")


@dataclass(frozen=True)
class VerifiedDecisionEvaluation:
    decision_input: DecisionInput
    authority_binding: DecisionAuthorityBinding
    result: DecisionResult
    _seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.decision_input._is_sealed() or self._seal is not _EVALUATION_SEAL:
            raise UnverifiedAuthorityEvaluationError("evaluación no verificada")
        envelope = self.result.effective_authority_envelope
        if (envelope.active_policy_corpus_hash != self.authority_binding.active_policy_corpus_hash or
                envelope.effective_authority_context_hash != self.authority_binding.effective_authority_context_hash or
                envelope.authority_ledger_checkpoint_hash != self.authority_binding.authority_ledger_checkpoint_hash or
                envelope.static_resolution.context != self.decision_input.evaluation_context):
            raise UnverifiedAuthorityEvaluationError("binding/evaluación no coinciden")

    def _is_verified(self) -> bool:
        return self._seal is _EVALUATION_SEAL


@dataclass(frozen=True)
class DecisionRecord:
    schema_version: str
    kind: str
    decision_id: str
    decision_input: DecisionInput
    decision_input_hash: str
    authority_binding: DecisionAuthorityBinding
    result: DecisionResult
    evidence_refs: tuple[str, ...]
    recorded_at_utc: datetime
    decision_record_hash: str
    _seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (self.schema_version, self.kind) != ("1.0", "JAX_DECISION_RECORD"):
            raise DecisionRecordIntegrityError("DecisionRecord schema/kind inválido")
        decision_id(self.decision_id)
        if not self.decision_input._is_sealed():
            raise DecisionRecordIntegrityError("DecisionInput no sellado")
        if self.decision_input_hash != self.decision_input.decision_input_hash:
            raise DecisionRecordIntegrityError("decision_input_hash no coincide")
        object.__setattr__(self, "recorded_at_utc", _time(self.recorded_at_utc, "recorded_at_utc", DecisionRecordIntegrityError))
        try: object.__setattr__(self, "evidence_refs", canonical_evidence_refs(self.evidence_refs))
        except Exception as exc: raise DecisionRecordIntegrityError("evidence_refs inválidas") from exc
        if self._seal is not _RECORD_SEAL:
            raise DecisionRecordIntegrityError("DecisionRecord sólo puede originarse en factory/verify")
        if self.decision_record_hash != decision_record_hash(self.projection_without_hash()):
            raise DecisionRecordIntegrityError("decision_record_hash no coincide")

    def _is_sealed(self) -> bool:
        return self._seal is _RECORD_SEAL

    def projection_without_hash(self) -> dict:
        from .serialization import result_projection
        return {"schema_version": self.schema_version, "kind": self.kind, "decision_id": self.decision_id,
                "decision_input": self.decision_input.record_projection(), "decision_input_hash": self.decision_input_hash,
                "authority_binding": self.authority_binding.projection(), "result": result_projection(self.result),
                "evidence_refs": list(self.evidence_refs), "recorded_at_utc": utc_text(self.recorded_at_utc)}


class EvidenceProvider(Protocol):
    def read(self, evidence_ref: str) -> bytes: ...


class DecisionReplayStatus(str, Enum):
    REPLAY_MATCH = "REPLAY_MATCH"
    REPLAY_DIVERGENCE = "REPLAY_DIVERGENCE"


class DecisionReplayDifference(str, Enum):
    ACTIVE_POLICY_CORPUS_HASH_MISMATCH = "ACTIVE_POLICY_CORPUS_HASH_MISMATCH"
    EFFECTIVE_AUTHORITY_CONTEXT_HASH_MISMATCH = "EFFECTIVE_AUTHORITY_CONTEXT_HASH_MISMATCH"
    STATIC_RESOLUTION_MISMATCH = "STATIC_RESOLUTION_MISMATCH"
    RELEVANT_OVERLAY_SET_MISMATCH = "RELEVANT_OVERLAY_SET_MISMATCH"
    LIFECYCLE_STATUS_MISMATCH = "LIFECYCLE_STATUS_MISMATCH"
    EFFECTIVE_ENVELOPE_MISMATCH = "EFFECTIVE_ENVELOPE_MISMATCH"


@dataclass(frozen=True)
class DecisionReplayResult:
    schema_version: str
    kind: str
    decision_id: str
    decision_record_hash: str
    status: DecisionReplayStatus
    differences: tuple[DecisionReplayDifference, ...]
    replayed_result: DecisionResult
