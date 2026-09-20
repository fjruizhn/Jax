"""Only trusted Block 4 state may become a Block 5 authority binding."""
from __future__ import annotations

from policy.authority_ledger.effective_context import build_effective_authority_envelope
from policy.authority_ledger.replay import ReconstructedAuthorityState

from .errors import UnverifiedAuthorityEvaluationError
from .models import (DecisionAuthorityBinding, DecisionInput, DecisionResult,
    EffectiveAuthorityEnvelopeSnapshot, VerifiedDecisionEvaluation, _EVALUATION_SEAL)


def evaluate_decision_input(authority_state: ReconstructedAuthorityState, decision_input: DecisionInput) -> VerifiedDecisionEvaluation:
    if not isinstance(decision_input, DecisionInput) or not decision_input._is_sealed():
        raise UnverifiedAuthorityEvaluationError("DecisionInput no sellado")
    if not isinstance(authority_state, ReconstructedAuthorityState) or not authority_state._is_verified():
        raise UnverifiedAuthorityEvaluationError("authority state no verificado")
    envelope = build_effective_authority_envelope(authority_state, decision_input.evaluation_context, decision_input.evaluation_time_utc)
    snapshot = EffectiveAuthorityEnvelopeSnapshot.from_envelope(envelope)
    resolution = snapshot.static_resolution
    binding = DecisionAuthorityBinding(snapshot.active_policy_corpus_hash, snapshot.effective_authority_context_hash,
        authority_state.checkpoint, snapshot.authority_ledger_checkpoint_hash, resolution.resolver_identity, resolution.resolver_version)
    return VerifiedDecisionEvaluation(decision_input, binding,
        DecisionResult("1.0", "JAX_DECISION_RESULT", "EFFECTIVE_AUTHORITY_ANALYSIS_ONLY", snapshot), _EVALUATION_SEAL)
