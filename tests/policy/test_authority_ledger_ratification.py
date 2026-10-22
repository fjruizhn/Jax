from pathlib import Path

import pytest

from policy.authority_ledger.errors import AuthorityEventValidationError, AuthorityStateError
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from tests.policy.test_authority_ledger_events import (append_authority_event,
                                                        append_ratification,
                                                        setup_ledger,
                                                        verify_authority_ledger)
from policy.authority_ledger.service import (append_ratification_from_candidate,
                                             ratification_intent_from_candidate,
                                             _append_authority_event)
from policy.authority_ledger import service as authority_ledger_service
from policy.authority_resolution.candidate_loader import load_validated_candidate


def test_ratification_is_distinct_from_activation():
    store, root, key = setup_ledger()
    event = append_ratification(store, root, key, checkpoint_store=store._checkpoint_store)
    state = verify_authority_ledger(store.get_genesis(), store.events(), root)
    assert event.event_id in state.ratifications
    assert state.active_policy_corpus_hash is None


def test_real_candidate_snapshot_is_bound_to_its_hash():
    corpus = load_validated_candidate(Path(__file__).resolve().parents[2])
    intent = ratification_intent_from_candidate(corpus)
    assert intent.policy_corpus_hash == corpus.policy_corpus_hash
    assert intent.static_policy_view_projection["policy_corpus_hash"] == corpus.policy_corpus_hash


class AlwaysEqual:
    def __eq__(self, other):
        return True


def test_ratification_seal_is_not_constructor_reachable():
    corpus_hash = "sha256:" + "f" * 64

    # El sello es init=False: el constructor no lo acepta (ni un AlwaysEqual
    # que engañaría a un __eq__), y dataclasses.replace no lo transporta.
    # Sin sello el intent se construye, pero la frontera que firma lo rechaza
    # (test_append_rejects_equal_but_unsealed_ratification_intent y los
    # ataques de test_authority_ledger_seal_attacks.py).
    with pytest.raises(TypeError):
        AuthorityEventIntent(
            AuthorityEventType.RATIFICATION_GRANTED,
            "human:fernando",
            (),
            corpus_hash,
            {"policy_corpus_hash": corpus_hash},
            _ratification_snapshot_seal=AlwaysEqual(),
        )


def test_generic_append_rejects_all_caller_created_ratification_intents():
    corpus = load_validated_candidate(Path(__file__).resolve().parents[2])
    intent = ratification_intent_from_candidate(corpus)
    object.__setattr__(intent, "_ratification_snapshot_seal", AlwaysEqual())
    store, root, key = setup_ledger()

    with pytest.raises(AuthorityStateError, match="append_ratification_from_candidate"):
        append_authority_event(store, root, key, intent)

    assert store.events() == ()


def test_internal_non_ratification_writer_rejects_caller_ratification_intent():
    corpus = load_validated_candidate(Path(__file__).resolve().parents[2])
    intent = ratification_intent_from_candidate(corpus)
    store, root, key = setup_ledger()

    with pytest.raises(AuthorityStateError, match="append_ratification_from_candidate"):
        _append_authority_event(store, root, key, intent, checkpoint_store=store._checkpoint_store)

    assert store.events() == ()


def test_no_unlocked_writer_accepts_caller_constructed_ratification_intent():
    # Signing must stay inside the validated-candidate API and its checkpoint
    # lock. A private helper is importable Python and is not an authority
    # boundary.
    assert not hasattr(authority_ledger_service,
                       "_append_ratification_from_candidate_unlocked")


def test_atomic_append_captures_candidate_view_and_signs_it():
    corpus = load_validated_candidate(Path(__file__).resolve().parents[2])
    store, root, key = setup_ledger()
    event = append_ratification_from_candidate(
        store, root, key, corpus, checkpoint_store=store._checkpoint_store
    )
    assert event.intent.policy_corpus_hash == corpus.policy_corpus_hash
    assert event.intent.static_policy_view_projection["policy_corpus_hash"] == corpus.policy_corpus_hash
