"""Pure creation and evidence validation plus the record writer boundary."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from policy.authority_resolution.models import EvaluationContext

from .errors import DecisionEvidenceUnavailableError, DecisionRecordIntegrityError, InvalidDecisionInputError, UnverifiedAuthorityEvaluationError
from .ids import canonical_evidence_refs, decision_id
from .models import (DecisionFact, DecisionInput, DecisionRecord, EvidenceProvider,
    VerifiedDecisionEvaluation, _register_decision_input, _register_decision_record)
from .canonical import decision_record_hash, utc_text
from .serialization import result_projection


def build_decision_input(evaluation_context: EvaluationContext, evaluation_time_utc: datetime, *, facts: tuple[DecisionFact, ...] = ()) -> DecisionInput:
    return _register_decision_input(DecisionInput("1.0", "JAX_DECISION_INPUT", evaluation_context, evaluation_time_utc, facts))


def compute_decision_input_hash(decision_input: DecisionInput) -> str:
    if not isinstance(decision_input, DecisionInput) or not decision_input._is_sealed():
        raise InvalidDecisionInputError("DecisionInput no sellado")
    return decision_input.decision_input_hash


def _verify_evidence(refs: tuple[str, ...], provider: EvidenceProvider | None) -> None:
    if not refs:
        return
    if provider is None:
        raise DecisionEvidenceUnavailableError("evidence provider requerido")
    for ref in refs:
        try:
            data = provider.read(ref)
        except OSError as exc:
            raise DecisionEvidenceUnavailableError(f"evidence ausente: {ref}") from exc
        if not isinstance(data, bytes) or "sha256:" + hashlib.sha256(data).hexdigest() != ref:
            raise DecisionEvidenceUnavailableError(f"evidence inválida: {ref}")


def build_decision_record(evaluation: VerifiedDecisionEvaluation, *, decision_id: str,
                          recorded_at_utc: datetime, evidence_refs: tuple[str, ...] = (),
                          evidence_provider: EvidenceProvider | None = None) -> DecisionRecord:
    if not isinstance(evaluation, VerifiedDecisionEvaluation) or not evaluation._is_verified():
        raise UnverifiedAuthorityEvaluationError("evaluación no verificada")
    if not isinstance(recorded_at_utc, datetime) or recorded_at_utc.tzinfo is None:
        raise DecisionRecordIntegrityError("recorded_at_utc debe ser timezone-aware")
    recorded_at_utc = recorded_at_utc.astimezone(timezone.utc)
    value_id = decision_id
    globals_refs = canonical_evidence_refs(evidence_refs)
    _verify_evidence(globals_refs, evidence_provider)
    for fact in evaluation.decision_input.facts:
        _verify_evidence(fact.evidence_refs, evidence_provider)
    # Hash the exact record projection before adding the self-hash field.
    record_hash = decision_record_hash({
        "schema_version": "1.0", "kind": "JAX_DECISION_RECORD", "decision_id": value_id,
        "decision_input": evaluation.decision_input.record_projection(),
        "decision_input_hash": evaluation.decision_input.decision_input_hash,
        "authority_binding": evaluation.authority_binding.projection(),
        "result": result_projection(evaluation.result), "evidence_refs": list(globals_refs),
        "recorded_at_utc": utc_text(recorded_at_utc),
    })
    return _register_decision_record(DecisionRecord("1.0", "JAX_DECISION_RECORD", value_id, evaluation.decision_input,
        evaluation.decision_input.decision_input_hash, evaluation.authority_binding, evaluation.result,
        globals_refs, recorded_at_utc, record_hash))


def record_decision(store, evaluation: VerifiedDecisionEvaluation, *, decision_id: str,
                    recorded_at_utc: datetime, evidence_refs: tuple[str, ...] = (),
                    evidence_provider: EvidenceProvider | None = None) -> DecisionRecord:
    return store.insert(build_decision_record(evaluation, decision_id=decision_id, recorded_at_utc=recorded_at_utc,
                                               evidence_refs=evidence_refs, evidence_provider=evidence_provider))


def load_decision(store, decision_id: str) -> DecisionRecord:
    from .serialization import decision_record_from_bytes
    return decision_record_from_bytes(store.load_canonical_bytes(decision_id))


def verify_decision_record(canonical_record_bytes: bytes) -> DecisionRecord:
    from .serialization import decision_record_from_bytes
    return decision_record_from_bytes(canonical_record_bytes)


def is_verified_decision_record(record: DecisionRecord) -> bool:
    """Public, capability-free provenance query for governed execution."""
    return isinstance(record, DecisionRecord) and record._is_sealed()
