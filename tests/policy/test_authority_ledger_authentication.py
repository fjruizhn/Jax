import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.errors import AuthorityStateError, TrustedRootMismatchError
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
from tests.policy.test_authority_ledger_events import setup_ledger


def test_wrong_signer_and_trusted_fingerprint_fail():
    store, root, key = setup_ledger()
    with pytest.raises(AuthorityStateError):
        append_authority_event(store, root, Ed25519PrivateKey.generate(), AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    bad = TrustedAuthorityRoot("1.0", "JAX_TRUSTED_AUTHORITY_ROOT", root.ledger_identity, root.genesis_hash, root.constitutional_key_id, "sha256:" + "f" * 64)
    with pytest.raises(TrustedRootMismatchError):
        verify_authority_ledger(store.get_genesis(), (), bad)
