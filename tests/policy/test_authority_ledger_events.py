from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis, OverlayPayload, OverlayScope, OverlayType
from policy.authority_ledger.signatures import encode_public_key, public_key_bytes, public_key_fingerprint
from policy.authority_ledger.storage import InMemoryAuthorityLedgerStore
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot


def base_time():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def setup_ledger():
    key = Ed25519PrivateKey.generate()
    genesis = AuthorityLedgerGenesis("1.0", "JAX_AUTHORITY_LEDGER_GENESIS", "JAX-AUTHORITY-LEDGER/1", "human:fernando", "fernando-1", encode_public_key(key.public_key()))
    from policy.authority_ledger.replay import genesis_hash
    root = TrustedAuthorityRoot("1.0", "JAX_TRUSTED_AUTHORITY_ROOT", "JAX-AUTHORITY-LEDGER/1", genesis_hash(genesis), "fernando-1", public_key_fingerprint(public_key_bytes(key.public_key())))
    return InMemoryAuthorityLedgerStore(genesis), root, key


def overlay(overlay_id="exception-a", *, conditions=(), target=("rule-a",), kind=OverlayType.EXCEPTION, code=None, delegate=None, target_overlay=None):
    return OverlayPayload(overlay_id, kind, "sha256:" + "a" * 64, OverlayScope(("ALICE",), ("READ",), conditions), base_time(), base_time() + timedelta(days=1), target, delegate, ("READ",) if delegate else (), code, target_overlay)


def ratification_intent(policy_hash="sha256:" + "a" * 64):
    return AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", (), policy_hash, {"policy_corpus_hash": policy_hash})


def test_evidence_refs_are_set_and_committed():
    one = AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", ("sha256:" + "b" * 64, "sha256:" + "a" * 64), "sha256:" + "c" * 64, {"policy_corpus_hash": "sha256:" + "c" * 64})
    two = AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", ("sha256:" + "a" * 64, "sha256:" + "b" * 64), "sha256:" + "c" * 64, {"policy_corpus_hash": "sha256:" + "c" * 64})
    changed = AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", ("sha256:" + "d" * 64,), "sha256:" + "c" * 64, {"policy_corpus_hash": "sha256:" + "c" * 64})
    assert canonical_bytes(one.canonical_projection()) == canonical_bytes(two.canonical_projection())
    assert canonical_bytes(one.canonical_projection()) != canonical_bytes(changed.canonical_projection())


def test_duplicate_evidence_rejected():
    with pytest.raises(Exception):
        AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando", ("sha256:" + "a" * 64,) * 2)
