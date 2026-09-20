from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from tests.policy.test_authority_ledger_events import setup_ledger, ratification_intent


def test_activation_requires_ratification_and_deactivation_has_no_fallback():
    store, root, key = setup_ledger()
    rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    assert verify_authority_ledger(store.get_genesis(), store.events(), root).active_policy_corpus_hash == "sha256:" + "a" * 64
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    assert verify_authority_ledger(store.get_genesis(), store.events(), root).active_policy_corpus_hash is None
