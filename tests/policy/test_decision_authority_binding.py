from datetime import timezone

import pytest

from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.decision_record.authority_binding import evaluate_decision_input
from policy.decision_record.errors import UnverifiedAuthorityEvaluationError
from policy.decision_record.errors import DecisionRecordIntegrityError
from policy.decision_record.models import DecisionAuthorityBinding, VerifiedDecisionEvaluation
from policy.decision_record.service import build_decision_input
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
        VerifiedDecisionEvaluation(evaluation.decision_input, forged, evaluation.result, evaluation._seal)
