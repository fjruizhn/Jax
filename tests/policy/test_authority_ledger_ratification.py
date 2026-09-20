from pathlib import Path

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
