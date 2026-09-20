"""Replay immutable decision records against their recorded ledger checkpoint."""
from __future__ import annotations

from policy.authority_ledger.replay import verify_authority_ledger

from .authority_binding import evaluate_decision_input
from .errors import (DecisionEvidenceUnavailableError, HistoricalAuthorityUnavailableError,
    HistoricalCheckpointMismatchError, UnsupportedResolverError)
from .models import (DecisionReplayDifference, DecisionReplayResult, DecisionReplayStatus,
    DecisionRecord, DecisionResult)
from .serialization import result_projection
from .service import _verify_evidence


def replay_decision(record: DecisionRecord, authority_store, trusted_root, checkpoint_store, *, evidence_provider=None) -> DecisionReplayResult:
    if not isinstance(record, DecisionRecord) or not record._is_sealed():
        from .errors import DecisionRecordIntegrityError
        raise DecisionRecordIntegrityError("DecisionRecord no verificado")
    _verify_evidence(record.evidence_refs, evidence_provider)
    for fact in record.decision_input.facts:
        _verify_evidence(fact.evidence_refs, evidence_provider)
    binding = record.authority_binding
    if (binding.resolver_identity, binding.resolver_version) != ("JAX-AUTHORITY-RESOLVER/1", "1.0"):
        raise UnsupportedResolverError("resolver no soportado")
    genesis = authority_store.get_genesis()
    events = authority_store.events()
    # Verify the complete, currently anchored history first; then replay only
    # the authenticated historical prefix without asking its old head to equal latest.
    verify_authority_ledger(genesis, events, trusted_root, checkpoint_store)
    checkpoint = binding.authority_ledger_checkpoint
    if checkpoint.sequence > len(events):
        raise HistoricalAuthorityUnavailableError("ledger no alcanza checkpoint histórico")
    prefix = events[:checkpoint.sequence]
    if checkpoint.sequence:
        head = prefix[-1]
        if (head.event_id, head.event_hash) != (checkpoint.head_event_id, checkpoint.head_event_hash):
            raise HistoricalCheckpointMismatchError("checkpoint no pertenece a cadena anclada")
    elif checkpoint.head_event_id is not None or checkpoint.head_event_hash is not None:
        raise HistoricalCheckpointMismatchError("checkpoint genesis inválido")
    historical = verify_authority_ledger(genesis, prefix, trusted_root)
    if historical.checkpoint.projection() != checkpoint.projection() or historical.checkpoint.authority_ledger_checkpoint_hash != binding.authority_ledger_checkpoint_hash:
        raise HistoricalCheckpointMismatchError("replay no reconstruyó checkpoint grabado")
    replayed = evaluate_decision_input(historical, record.decision_input).result
    old = record.result.effective_authority_envelope
    new = replayed.effective_authority_envelope
    differences = []
    if old.active_policy_corpus_hash != new.active_policy_corpus_hash: differences.append(DecisionReplayDifference.ACTIVE_POLICY_CORPUS_HASH_MISMATCH)
    if old.effective_authority_context_hash != new.effective_authority_context_hash: differences.append(DecisionReplayDifference.EFFECTIVE_AUTHORITY_CONTEXT_HASH_MISMATCH)
    if result_projection(DecisionResult("1.0", "JAX_DECISION_RESULT", "EFFECTIVE_AUTHORITY_ANALYSIS_ONLY", old)) != result_projection(DecisionResult("1.0", "JAX_DECISION_RESULT", "EFFECTIVE_AUTHORITY_ANALYSIS_ONLY", new)):
        differences.append(DecisionReplayDifference.EFFECTIVE_ENVELOPE_MISMATCH)
    status = DecisionReplayStatus.REPLAY_MATCH if not differences else DecisionReplayStatus.REPLAY_DIVERGENCE
    return DecisionReplayResult("1.0", "JAX_DECISION_REPLAY_RESULT", record.decision_id, record.decision_record_hash,
                                status, tuple(differences), replayed)
