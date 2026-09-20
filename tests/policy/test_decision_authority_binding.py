from datetime import timezone

import pytest

from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.decision_record.authority_binding import evaluate_decision_input
from policy.decision_record.errors import UnverifiedAuthorityEvaluationError
from policy.decision_record.errors import DecisionRecordIntegrityError
from policy.decision_record.models import (DecisionAuthorityBinding, DecisionResult,
    EffectiveAuthorityEnvelopeSnapshot, VerifiedDecisionEvaluation)
from policy.decision_record.service import build_decision_input, build_decision_record
from policy.decision_record.ids import new_decision_id
from tests.policy.test_authority_ledger_events import setup_ledger, ratification_intent
from tests.policy.test_decision_input import context, instant


def active_state():
    store, root, key = setup_ledger()
    rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    return store, root, verify_authority_ledger(store.get_genesis(), store.events(), root)


def test_binding_is_derived_from_verified_block4_state():
    _, _, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    assert evaluation.authority_binding.active_policy_corpus_hash == state.active_policy_corpus_hash
    assert evaluation.result.effective_authority_envelope.authority_ledger_checkpoint_hash == state.checkpoint.authority_ledger_checkpoint_hash


def test_unsealed_input_and_state_are_rejected():
    _, _, state = active_state()
    with pytest.raises(UnverifiedAuthorityEvaluationError):
        evaluate_decision_input(state, object())


def test_authority_hash_cannot_be_mixed_with_verified_envelope():
    _, _, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    forged = DecisionAuthorityBinding(
            evaluation.authority_binding.active_policy_corpus_hash,
            "sha256:" + "f" * 64,
            evaluation.authority_binding.authority_ledger_checkpoint,
            evaluation.authority_binding.authority_ledger_checkpoint_hash,
            evaluation.authority_binding.resolver_identity,
            evaluation.authority_binding.resolver_version,
        )
    with pytest.raises(UnverifiedAuthorityEvaluationError):
        VerifiedDecisionEvaluation(evaluation.decision_input, forged, evaluation.result)


def test_public_fields_cannot_mint_verified_evaluation_with_consistent_forgery():
    _, _, state = active_state()
    genuine = evaluate_decision_input(state, build_decision_input(context(), instant()))
    env = genuine.result.effective_authority_envelope
    forged_hash = "sha256:" + "f" * 64
    forged_snapshot = EffectiveAuthorityEnvelopeSnapshot(
        env.schema_version, env.kind, env.static_resolution,
        env.active_policy_corpus_hash, forged_hash,
        env.authority_ledger_checkpoint_hash, env.relevant_overlay_ids,
        env.lifecycle_status,
    )
    forged_binding = DecisionAuthorityBinding(
        env.active_policy_corpus_hash, forged_hash,
        genuine.authority_binding.authority_ledger_checkpoint,
        env.authority_ledger_checkpoint_hash,
        genuine.authority_binding.resolver_identity,
        genuine.authority_binding.resolver_version,
    )
    lookalike = VerifiedDecisionEvaluation(
        genuine.decision_input,
        forged_binding,
        DecisionResult("1.0", "JAX_DECISION_RESULT", "EFFECTIVE_AUTHORITY_ANALYSIS_ONLY", forged_snapshot),
    )
    with pytest.raises(UnverifiedAuthorityEvaluationError):
        build_decision_record(lookalike, decision_id=new_decision_id(), recorded_at_utc=instant(2))


def test_genuine_factory_evaluation_is_accepted():
    _, _, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    record = build_decision_record(evaluation, decision_id=new_decision_id(), recorded_at_utc=instant(2))
    assert record.authority_binding == evaluation.authority_binding
