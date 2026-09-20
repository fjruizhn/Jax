"""Direct regressions for B4-AUDIT-001..006."""
from datetime import timedelta

import pytest

from policy.authority_ledger.errors import (AuthorityEventValidationError,
                                            AuthorityStateError,
                                            LedgerRollbackError,
                                            UnanchoredLedgerHeadError)
from policy.authority_ledger.models import (AuthorityEventIntent, AuthorityEventType,
                                            OverlayPayload, OverlayScope, OverlayType)
from policy.authority_ledger.replay import effective_overlays, verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_ledger.storage import MariaDBAuthorityLedgerStore
from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore
from policy.authority_resolution.models import EvaluationContext
from tests.policy.test_authority_ledger_events import base_time, overlay, ratification_intent, setup_ledger


def _context():
    return EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "ALICE", "READ", ())


def _active():
    store, root, key = setup_ledger()
    rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=rat.event_id))
    return store, root, key, rat


def test_audit_001_unsealed_hash_snapshot_pair_rejected():
    with pytest.raises(AuthorityEventValidationError):
        AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", (), "sha256:" + "f" * 64, {"policy_corpus_hash": "sha256:" + "f" * 64})


def test_audit_002_replay_state_is_deeply_sealed():
    store, root, key, _ = _active()
    state = verify_authority_ledger(store.get_genesis(), store.events(), root)
    with pytest.raises(TypeError):
        state.overlays["unsigned"] = overlay()  # type: ignore[index]
    with pytest.raises(AuthorityStateError):
        effective_overlays(type(state)({}, frozenset(), None, {}, frozenset(), state.checkpoint), _context(), base_time())


def test_audit_003_cross_corpus_overlay_never_effective():
    store, root, key, rat = _active()
    bad = overlay("other")
    object.__setattr__(bad, "policy_corpus_hash", "sha256:" + "b" * 64)
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=bad))
    state = verify_authority_ledger(store.get_genesis(), store.events(), root)
    assert effective_overlays(state, _context(), base_time()) == ()


def test_audit_004_external_checkpoint_rejects_old_prefix(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    rat = append_authority_event(store, root, key, ratification_intent(), checkpoint_store=anchor)
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=rat.event_id), checkpoint_store=anchor)
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), checkpoint_store=anchor)
    with pytest.raises(LedgerRollbackError):
        verify_authority_ledger(store.get_genesis(), store.events()[:-1], root, anchor)


def test_audit_005_mariadb_adapter_is_production_store():
    assert MariaDBAuthorityLedgerStore.__name__ == "MariaDBAuthorityLedgerStore"


def test_audit_006_hybrid_and_unbounded_payloads_rejected():
    with pytest.raises(AuthorityEventValidationError):
        OverlayPayload("hybrid", OverlayType.EXCEPTION, "sha256:" + "a" * 64, OverlayScope(("ALICE",), ("READ",)), base_time(), base_time() + timedelta(days=1), ("rule-a",), exception_code="X", delegate_actor_id="actor:x")
    with pytest.raises(AuthorityEventValidationError):
        OverlayPayload("wide", OverlayType.DELEGATION, "sha256:" + "a" * 64, OverlayScope(("ALICE",), ("READ",), ("MUST",)), base_time(), base_time() + timedelta(days=1), (), delegate_actor_id="actor:x", delegate_actor_kind="HUMAN_LOGICAL", delegated_scope=OverlayScope(("ALICE",), ("READ",)), may_subdelegate=False)
