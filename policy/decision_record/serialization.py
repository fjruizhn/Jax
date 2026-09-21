"""Strict codecs for immutable decision-record canonical bytes."""
from __future__ import annotations

import json
from datetime import datetime

from policy.authority_ledger.models import AuthorityLedgerCheckpoint
from policy.authority_resolution.models import (ApplicabilityState, ConflictRecord,
    ConditionResult, EvaluationContext, ProtectedConstraint, RuleResolutionRecord,
    SelectionState, StaticAuthorityResolution)

from .canonical import canonical_bytes
from .errors import DecisionRecordIntegrityError, UnsupportedDecisionSchemaError
from .models import (DecisionAuthorityBinding, DecisionFact, DecisionFactValueType,
    DecisionInput, DecisionRecord, DecisionResult, EffectiveAuthorityEnvelopeSnapshot,
    _register_decision_input)


def _load(value):
    if isinstance(value, bytes): value = value.decode("utf-8")
    if isinstance(value, str): value = json.loads(value)
    if not isinstance(value, dict): raise UnsupportedDecisionSchemaError("record canonical inválido")
    return value


def resolution_projection(value: StaticAuthorityResolution) -> dict:
    return {
        "schema_version": value.schema_version, "kind": value.kind,
        "resolver_identity": value.resolver_identity, "resolver_version": value.resolver_version,
        "policy_corpus_hash": value.policy_corpus_hash,
        "context": {"schema_version": value.context.schema_version, "kind": value.context.kind,
                    "jurisdiction": value.context.jurisdiction, "subject": value.context.subject,
                    "action": value.context.action,
                    "conditions": [{"id": x.id, "value": x.value} for x in value.context.conditions]},
        "rule_results": [{"rule_id": x.rule_id, "document_class": x.document_class,
                          "normative_layer": x.normative_layer, "applicability_state": x.applicability_state.value,
                          "selection_state": x.selection_state.value, "superseded_by": list(x.superseded_by),
                          "suppression_sources": list(x.suppression_sources), "indeterminacy_relevant": x.indeterminacy_relevant}
                         for x in value.rule_results],
        "controlling_rule_id": value.controlling_rule_id,
        "winning_precedence_layer": value.winning_precedence_layer,
        "conflicts": [{"kind": x.kind, "precedence_layer": x.precedence_layer, "rule_ids": list(x.rule_ids)} for x in value.conflicts],
        "relevant_indeterminate_rule_ids": list(value.relevant_indeterminate_rule_ids),
        "protected_constraints": [{"id": x.id, "statement": x.statement, "non_waivable": x.non_waivable} for x in value.protected_constraints],
        "external_constraint_mode": value.external_constraint_mode, "resolution_status": value.resolution_status,
    }


def resolution_from_projection(data: dict) -> StaticAuthorityResolution:
    try:
        context_data = data["context"]
        context = EvaluationContext.from_mapping(context_data)
        rules = tuple(RuleResolutionRecord(x["rule_id"], x["document_class"], x["normative_layer"],
            ApplicabilityState(x["applicability_state"]), SelectionState(x["selection_state"]),
            tuple(x["superseded_by"]), tuple(x["suppression_sources"]), x["indeterminacy_relevant"])
            for x in data["rule_results"])
        conflicts = tuple(ConflictRecord(x["kind"], x["precedence_layer"], tuple(x["rule_ids"])) for x in data["conflicts"])
        protected = tuple(ProtectedConstraint(x["id"], x["statement"], x["non_waivable"]) for x in data["protected_constraints"])
        return StaticAuthorityResolution(data["schema_version"], data["kind"], data["resolver_identity"], data["resolver_version"],
            data["policy_corpus_hash"], context, rules, data["controlling_rule_id"], data["winning_precedence_layer"], conflicts,
            tuple(data["relevant_indeterminate_rule_ids"]), protected, data["external_constraint_mode"], data["resolution_status"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionRecordIntegrityError("static resolution serializada inválida") from exc


def result_projection(value: DecisionResult) -> dict:
    env = value.effective_authority_envelope
    return {"schema_version": value.schema_version, "kind": value.kind, "result_semantics": value.result_semantics,
            "effective_authority_envelope": {"schema_version": env.schema_version, "kind": env.kind,
                "static_resolution": resolution_projection(env.static_resolution),
                "active_policy_corpus_hash": env.active_policy_corpus_hash,
                "effective_authority_context_hash": env.effective_authority_context_hash,
                "authority_ledger_checkpoint_hash": env.authority_ledger_checkpoint_hash,
                "relevant_overlay_ids": list(env.relevant_overlay_ids), "lifecycle_status": env.lifecycle_status}}


def result_from_projection(data: dict) -> DecisionResult:
    try:
        env = data["effective_authority_envelope"]
        snapshot = EffectiveAuthorityEnvelopeSnapshot(env["schema_version"], env["kind"], resolution_from_projection(env["static_resolution"]),
            env["active_policy_corpus_hash"], env["effective_authority_context_hash"], env["authority_ledger_checkpoint_hash"],
            tuple(env["relevant_overlay_ids"]), env["lifecycle_status"])
        return DecisionResult(data["schema_version"], data["kind"], data["result_semantics"], snapshot)
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionRecordIntegrityError("DecisionResult serializado inválido") from exc


def canonical_decision_record_bytes(record: DecisionRecord) -> bytes:
    if not isinstance(record, DecisionRecord):
        raise DecisionRecordIntegrityError("DecisionRecord requerido")
    return canonical_bytes(record.projection_without_hash() | {"decision_record_hash": record.decision_record_hash})


def _canonical_record_content_bytes(record: DecisionRecord) -> bytes:
    """Persistence-only content encoding before provenance is established."""
    if not isinstance(record, DecisionRecord):
        raise DecisionRecordIntegrityError("DecisionRecord requerido")
    return canonical_bytes(record.projection_without_hash() | {"decision_record_hash": record.decision_record_hash})


def canonical_decision_input_bytes(value: DecisionInput) -> bytes:
    if not isinstance(value, DecisionInput) or not value._is_sealed():
        raise DecisionRecordIntegrityError("DecisionInput no verificado")
    return canonical_bytes(value.semantic_projection())


def decision_record_from_bytes(value: bytes | str) -> DecisionRecord:
    data = _load(value)
    required = {"schema_version", "kind", "decision_id", "decision_input", "decision_input_hash", "authority_binding", "result", "evidence_refs", "recorded_at_utc", "decision_record_hash"}
    if set(data) != required:
        raise UnsupportedDecisionSchemaError("DecisionRecord cerrado inválido")
    try:
        input_data = data["decision_input"]
        c = input_data["evaluation_context"]
        context = EvaluationContext.from_mapping(c)
        facts = tuple(DecisionFact(x["id"], DecisionFactValueType(x["value_type"]), tuple(x["value"]) if x["value_type"] == "STRING_SET" else x["value"], tuple(x.get("evidence_refs", ()))) for x in input_data["facts"])
        decision_input = _register_decision_input(DecisionInput(input_data["schema_version"], input_data["kind"], context,
            datetime.fromisoformat(input_data["evaluation_time_utc"].replace("Z", "+00:00")), facts))
        b = data["authority_binding"]
        checkpoint = AuthorityLedgerCheckpoint(**b["authority_ledger_checkpoint"])
        binding = DecisionAuthorityBinding(b["active_policy_corpus_hash"], b["effective_authority_context_hash"], checkpoint,
            b["authority_ledger_checkpoint_hash"], b["resolver_identity"], b["resolver_version"])
        result = result_from_projection(data["result"])
        # Parsing canonical public bytes establishes content integrity only.
        # It must never turn the resulting object into a trusted record.
        return DecisionRecord(data["schema_version"], data["kind"], data["decision_id"], decision_input, data["decision_input_hash"],
            binding, result, tuple(data["evidence_refs"]), datetime.fromisoformat(data["recorded_at_utc"].replace("Z", "+00:00")),
            data["decision_record_hash"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionRecordIntegrityError("DecisionRecord serializado inválido") from exc
