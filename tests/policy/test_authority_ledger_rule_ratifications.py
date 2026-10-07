from datetime import datetime, timezone

import pytest

from policy.authority_ledger.errors import AuthorityEventValidationError, AuthorityStateError
from policy.authority_ledger.models import (
    AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    RuleRatificationGrantPayload, _RULE_RATIFICATION_SNAPSHOT_SEAL,
)
from policy.authority_ledger.replay import genesis_hash, verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_ledger.serialization import intent_from_projection, intent_projection
from policy.authority_ledger.signatures import (
    encode_public_key, public_key_bytes, public_key_fingerprint,
)
from policy.authority_ledger.storage import InMemoryAuthorityLedgerStore
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot


def sample_grant():
    return RuleRatificationGrantPayload(
        rule_id="send-receipt",
        rule_path="policy/faro/send-receipt.yaml",
        rule_blob_oid="a" * 40,
        rule_content_hash="sha256:" + "b" * 64,
        ratified_policy_revision="c" * 40,
        ratified_policy_tree_oid="d" * 40,
        ratified_policy_snapshot_hash="sha256:" + "e" * 64,
        valid_from_utc=datetime(2026, 10, 6, tzinfo=timezone.utc),
        valid_until_utc=None,
    )


def _ledger():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    public_key = key.public_key()
    genesis = AuthorityLedgerGenesis(
        "1.0", "JAX_AUTHORITY_LEDGER_GENESIS", "JAX-AUTHORITY-LEDGER/1",
        "human:fernando", "test-key", encode_public_key(public_key),
    )
    root = TrustedAuthorityRoot(
        "1.0", "JAX_TRUSTED_AUTHORITY_ROOT", genesis.ledger_identity,
        genesis_hash(genesis), "test-key",
        public_key_fingerprint(public_key_bytes(public_key)),
    )
    return InMemoryAuthorityLedgerStore(genesis), root, key


def _grant_intent(grant=None):
    value = grant or sample_grant()
    return intent_from_projection({
        "event_type": "RULE_RATIFICATION_GRANTED",
        "actor_id": "human:fernando",
        "evidence_refs": [],
        "rule_ratification": value.projection(),
    })


def _test_signable_grant_intent(grant=None):
    return AuthorityEventIntent(
        AuthorityEventType.RULE_RATIFICATION_GRANTED,
        "human:fernando",
        rule_ratification=grant or sample_grant(),
        _rule_ratification_snapshot_seal=_RULE_RATIFICATION_SNAPSHOT_SEAL,
    )


def test_rule_ratification_grant_has_a_closed_payload_and_roundtrips():
    grant = sample_grant()
    expected = {
        "rule_id", "rule_path", "rule_blob_oid", "rule_content_hash",
        "ratified_policy_revision", "ratified_policy_tree_oid",
        "ratified_policy_snapshot_hash", "valid_from_utc", "valid_until_utc",
    }

    assert set(grant.projection()) == expected
    projection = {
        "event_type": "RULE_RATIFICATION_GRANTED",
        "actor_id": "human:fernando",
        "evidence_refs": [],
        "rule_ratification": grant.projection(),
    }
    restored = intent_from_projection(projection)

    assert restored.event_type is AuthorityEventType.RULE_RATIFICATION_GRANTED
    assert restored.rule_ratification == grant
    assert intent_projection(restored) == projection


def test_rule_ratification_grant_cannot_be_built_from_caller_supplied_fields():
    with pytest.raises(AuthorityEventValidationError):
        AuthorityEventIntent(
            AuthorityEventType.RULE_RATIFICATION_GRANTED,
            "human:fernando",
            rule_ratification=sample_grant(),
        )


def test_storage_decoding_does_not_authorize_a_new_human_rule_grant():
    store, root, key = _ledger()

    with pytest.raises(AuthorityStateError, match="sealed Faro snapshot"):
        append_authority_event(
            store, root, key, _grant_intent(),
            event_id="018cc251-f400-7000-8000-000000000010",
        )


def test_rule_ratification_grant_rejects_bad_path_oid_hash_and_empty_validity():
    with pytest.raises(AuthorityEventValidationError):
        RuleRatificationGrantPayload(**(sample_grant().__dict__ | {"rule_path": "policy/faro/../rule.yaml"}))
    with pytest.raises(AuthorityEventValidationError):
        RuleRatificationGrantPayload(**(sample_grant().__dict__ | {"rule_blob_oid": "not-an-oid"}))
    with pytest.raises(AuthorityEventValidationError):
        RuleRatificationGrantPayload(**(sample_grant().__dict__ | {"rule_content_hash": "sha256:" + "A" * 64}))
    with pytest.raises(AuthorityEventValidationError):
        RuleRatificationGrantPayload(**(sample_grant().__dict__ | {
            "valid_until_utc": datetime(2026, 10, 6, tzinfo=timezone.utc),
        }))


def test_rule_ratification_codec_rejects_unexpected_fields_and_non_fernando_actor():
    projection = intent_projection(_grant_intent())
    projection["overlay_id"] = "not-part-of-rule-grant"
    with pytest.raises(AuthorityEventValidationError):
        intent_from_projection(projection)

    projection = intent_projection(_grant_intent())
    projection["actor_id"] = "actor:service"
    with pytest.raises(AuthorityEventValidationError):
        intent_from_projection(projection)


def test_rule_ratification_revoke_points_to_one_exact_grant_event():
    event_id = "018cc251-f400-7000-8000-000000000001"
    intent = AuthorityEventIntent(
        AuthorityEventType.RULE_RATIFICATION_REVOKED,
        "human:fernando",
        rule_ratification_event_id=event_id,
    )

    assert intent_projection(intent) == {
        "event_type": "RULE_RATIFICATION_REVOKED",
        "actor_id": "human:fernando",
        "evidence_refs": [],
        "rule_ratification_event_id": event_id,
    }


def test_replay_selects_latest_grant_and_revoking_it_does_not_restore_previous_grant():
    store, root, key = _ledger()
    first_id = "018cc251-f400-7000-8000-000000000001"
    second_id = "018cc251-f400-7000-8000-000000000002"
    append_authority_event(store, root, key, _test_signable_grant_intent(), event_id=first_id)
    append_authority_event(store, root, key, _test_signable_grant_intent(), event_id=second_id)
    append_authority_event(
        store, root, key,
        AuthorityEventIntent(AuthorityEventType.RULE_RATIFICATION_REVOKED,
                             "human:fernando", rule_ratification_event_id=second_id),
        event_id="018cc251-f400-7000-8000-000000000003",
    )

    state = verify_authority_ledger(store.get_genesis(), store.events(), root)

    assert state.latest_rule_ratifications["send-receipt"].event_id == second_id
    assert second_id in state.revoked_rule_ratifications


def test_replay_rejects_unknown_and_duplicate_rule_ratification_revocations():
    store, root, key = _ledger()
    missing = "018cc251-f400-7000-8000-000000000009"
    append_authority_event(
        store, root, key,
        AuthorityEventIntent(AuthorityEventType.RULE_RATIFICATION_REVOKED,
                             "human:fernando", rule_ratification_event_id=missing),
        event_id="018cc251-f400-7000-8000-000000000001",
    )
    with pytest.raises(AuthorityStateError):
        verify_authority_ledger(store.get_genesis(), store.events(), root)

    store, root, key = _ledger()
    grant_id = "018cc251-f400-7000-8000-000000000002"
    revoke_id = "018cc251-f400-7000-8000-000000000003"
    append_authority_event(store, root, key, _test_signable_grant_intent(), event_id=grant_id)
    revoke = AuthorityEventIntent(
        AuthorityEventType.RULE_RATIFICATION_REVOKED,
        "human:fernando", rule_ratification_event_id=grant_id,
    )
    append_authority_event(store, root, key, revoke, event_id=revoke_id)
    append_authority_event(
        store, root, key, revoke, event_id="018cc251-f400-7000-8000-000000000004",
    )
    with pytest.raises(AuthorityStateError):
        verify_authority_ledger(store.get_genesis(), store.events(), root)
