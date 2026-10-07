from pathlib import Path

import pytest

from policy.authority_ledger.errors import AuthorityEventValidationError, AuthorityStateError
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event, ratification_intent_from_candidate
from policy.authority_resolution.candidate_loader import load_validated_candidate
from tests.policy.test_authority_ledger_events import setup_ledger, ratification_intent


def test_ratification_is_distinct_from_activation():
    store, root, key = setup_ledger()
    event = append_authority_event(store, root, key, ratification_intent())
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


def test_ratification_model_rejects_equal_but_unsealed_snapshot():
    corpus_hash = "sha256:" + "f" * 64

    with pytest.raises(AuthorityEventValidationError, match="snapshot sellado"):
        AuthorityEventIntent(
            AuthorityEventType.RATIFICATION_GRANTED,
            "human:fernando",
            (),
            corpus_hash,
            {"policy_corpus_hash": corpus_hash},
            _ratification_snapshot_seal=AlwaysEqual(),
        )


def test_append_rejects_equal_but_unsealed_ratification_intent():
    corpus = load_validated_candidate(Path(__file__).resolve().parents[2])
    intent = ratification_intent_from_candidate(corpus)
    object.__setattr__(intent, "_ratification_snapshot_seal", AlwaysEqual())
    store, root, key = setup_ledger()

    with pytest.raises(AuthorityStateError, match="snapshot sellado"):
        append_authority_event(store, root, key, intent)

    assert store.events() == ()
