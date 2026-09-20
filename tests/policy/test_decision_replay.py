from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore
from policy.decision_record.replay import replay_decision
from policy.decision_record.service import build_decision_record, build_decision_input
from policy.decision_record.ids import new_decision_id
from policy.decision_record.authority_binding import evaluate_decision_input
from policy.decision_record.models import DecisionReplayStatus
from tests.policy.test_decision_authority_binding import active_state
from tests.policy.test_decision_input import context, instant


def test_replay_matches_historical_checkpoint(tmp_path):
    store, root, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    record = build_decision_record(evaluation, decision_id=new_decision_id(), recorded_at_utc=instant(2))
    # Anchor the current head for the complete-chain verification requirement.
    checkpoints = TrustedCheckpointStore(tmp_path / "checkpoints")
    checkpoints.append(state.checkpoint)
    replay = replay_decision(record, store, root, checkpoints)
    assert replay.status is DecisionReplayStatus.REPLAY_MATCH


def test_valid_replay_artifact_mismatch_is_divergence(tmp_path, monkeypatch):
    store, root, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    record = build_decision_record(evaluation, decision_id=new_decision_id(), recorded_at_utc=instant(2))
    checkpoints = TrustedCheckpointStore(tmp_path / "checkpoints")
    checkpoints.append(state.checkpoint)
    import policy.decision_record.replay as subject
    original = subject.evaluate_decision_input
    def changed(*args):
        value = original(*args)
        snap = value.result.effective_authority_envelope
        from policy.decision_record.models import EffectiveAuthorityEnvelopeSnapshot, DecisionResult, VerifiedDecisionEvaluation
        altered = EffectiveAuthorityEnvelopeSnapshot(snap.schema_version, snap.kind, snap.static_resolution,
            snap.active_policy_corpus_hash, snap.effective_authority_context_hash,
            snap.authority_ledger_checkpoint_hash, ("different-overlay",), snap.lifecycle_status)
        return type(value)(value.decision_input, value.authority_binding,
            DecisionResult("1.0", "JAX_DECISION_RESULT", "EFFECTIVE_AUTHORITY_ANALYSIS_ONLY", altered))
    monkeypatch.setattr(subject, "evaluate_decision_input", changed)
    assert replay_decision(record, store, root, checkpoints).status is DecisionReplayStatus.REPLAY_DIVERGENCE
