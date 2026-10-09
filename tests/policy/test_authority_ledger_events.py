from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.canonical import canonical_bytes, plain
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


def overlay(overlay_id="exception-a", *, conditions=(), target=("rule-a",), kind=OverlayType.EXCEPTION, code="EXCEPTION", delegate=None, target_overlay=None):
    corpus_hash = ratification_intent().policy_corpus_hash
    kwargs = dict(overlay_id=overlay_id, overlay_type=kind, policy_corpus_hash=corpus_hash,
                  scope=OverlayScope(("ALICE",), ("READ",), conditions), valid_from_utc=base_time(), valid_until_utc=base_time() + timedelta(days=1), target_rule_ids=target)
    if kind is OverlayType.EXCEPTION:
        kwargs["exception_code"] = code
    elif kind is OverlayType.SUSPENSION:
        kwargs.update(target_rule_ids=(), target_overlay_id=target_overlay)
    elif kind is OverlayType.DELEGATION:
        kwargs.update(delegate_actor_id=delegate or "actor:delegate", delegate_actor_kind="HUMAN_LOGICAL", delegated_scope=OverlayScope(("ALICE",), ("READ",), conditions), may_subdelegate=False, target_rule_ids=())
    elif kind is OverlayType.BINDING_PARTICULAR_INTERPRETATION:
        kwargs["interpretation_code"] = code or "INTERPRET"
    return OverlayPayload(**kwargs)


def ratification_intent(policy_hash=None):
    from policy.authority_resolution.candidate_loader import load_validated_candidate
    from policy.authority_resolution.adapter import to_static_policy_view
    corpus = load_validated_candidate(Path(__file__).resolve().parents[2])
    projection = plain(to_static_policy_view(corpus))
    return AuthorityEventIntent(
        AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", (),
        projection["policy_corpus_hash"], projection,
    )


def append_ratification(store, root, key, *, evidence_refs=(), **kwargs):
    """Test helper for the public atomic corpus-ratification write path."""
    from policy.authority_ledger.service import append_ratification_from_candidate
    from policy.authority_resolution.candidate_loader import load_validated_candidate
    return append_ratification_from_candidate(
        store, root, key, load_validated_candidate(Path(__file__).resolve().parents[2]),
        evidence_refs, **kwargs,
    )


def test_evidence_refs_are_set_and_committed():
    base = ratification_intent()
    one = AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", ("sha256:" + "b" * 64, "sha256:" + "a" * 64), base.policy_corpus_hash, base.static_policy_view_projection)
    two = AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", ("sha256:" + "a" * 64, "sha256:" + "b" * 64), base.policy_corpus_hash, base.static_policy_view_projection)
    changed = AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", ("sha256:" + "d" * 64,), base.policy_corpus_hash, base.static_policy_view_projection)
    assert canonical_bytes(one.canonical_projection()) == canonical_bytes(two.canonical_projection())
    assert canonical_bytes(one.canonical_projection()) != canonical_bytes(changed.canonical_projection())


def test_duplicate_evidence_rejected():
    with pytest.raises(Exception):
        AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando", ("sha256:" + "a" * 64,) * 2)
