from tests.policy.test_authority_ledger_events import verify_authority_ledger
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from tests.policy.test_authority_ledger_events import (append_authority_event,
                                                        append_ratification,
                                                        setup_ledger)


def test_activation_requires_ratification_and_deactivation_has_no_fallback():
    store, root, key = setup_ledger()
    rat = append_ratification(store, root, key, checkpoint_store=store._checkpoint_store)
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    assert verify_authority_ledger(store.get_genesis(), store.events(), root).active_policy_corpus_hash == rat.intent.policy_corpus_hash
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    assert verify_authority_ledger(store.get_genesis(), store.events(), root).active_policy_corpus_hash is None
