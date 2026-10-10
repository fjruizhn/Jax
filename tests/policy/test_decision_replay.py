from policy.decision_record.replay import replay_decision
from policy.decision_record.service import build_decision_record, build_decision_input
from policy.decision_record.ids import new_decision_id
from policy.decision_record.authority_binding import evaluate_decision_input
from policy.decision_record.models import DecisionReplayStatus
from tests.policy.test_decision_authority_binding import active_state
from tests.policy.test_decision_input import context, instant


def test_replay_matches_historical_checkpoint():
    store, root, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    record = build_decision_record(evaluation, decision_id=new_decision_id(), recorded_at_utc=instant(2))
    # Anchor the current head for the complete-chain verification requirement.
    # Use the matched external checkpoint and bootstrap receipt created by the
    # ledger fixture; decision replay must not manufacture a second anchor.
    checkpoints = store._checkpoint_store
    replay = replay_decision(record, store, root, checkpoints)
    assert replay.status is DecisionReplayStatus.REPLAY_MATCH


def test_valid_replay_artifact_mismatch_is_divergence(monkeypatch):
    store, root, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    record = build_decision_record(evaluation, decision_id=new_decision_id(), recorded_at_utc=instant(2))
    checkpoints = store._checkpoint_store
    import policy.decision_record.replay as subject
    original = subject._build_historical_authority_envelope
    def changed(*args):
        value = original(*args)
        from dataclasses import replace
        return replace(value, relevant_overlay_ids=("different-overlay",))
    monkeypatch.setattr(subject, "_build_historical_authority_envelope", changed)
    assert replay_decision(record, store, root, checkpoints).status is DecisionReplayStatus.REPLAY_DIVERGENCE
