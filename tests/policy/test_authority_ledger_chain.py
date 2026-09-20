import pytest

from policy.authority_ledger.errors import TrustedRootMismatchError
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from tests.policy.test_authority_ledger_events import setup_ledger


def test_chain_is_verified_and_db_root_replacement_fails():
    store, root, key = setup_ledger()
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    assert verify_authority_ledger(store.get_genesis(), store.events(), root).checkpoint.sequence == 1
    attacker_store, _, _ = setup_ledger()
    with pytest.raises(TrustedRootMismatchError):
        verify_authority_ledger(attacker_store.get_genesis(), (), root)
