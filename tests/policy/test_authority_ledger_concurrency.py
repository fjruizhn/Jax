from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.service import append_authority_event
from tests.policy.test_authority_ledger_events import setup_ledger


def test_append_assigns_linear_monotonic_sequence():
    store, root, key = setup_ledger()
    first = append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    second = append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    assert (first.sequence, second.sequence, second.previous_event_hash) == (1, 2, first.event_hash)
