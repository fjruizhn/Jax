from datetime import datetime, timezone

import pytest

from policy.authority_ledger.errors import AuthorityEventValidationError, AuthorityStateError, LedgerIntegrityError
from policy.authority_ledger.models import (
    AuthorityEvent, AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    RuleRatificationGrantPayload,
)
from policy.authority_ledger.replay import event_hash, event_unsigned_bytes, genesis_hash, verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_ledger.serialization import intent_from_projection, intent_projection
from policy.authority_ledger.signatures import (
    encode_public_key, public_key_bytes, public_key_fingerprint,
)
from policy.authority_ledger.storage import InMemoryAuthorityLedgerStore
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
from tests.policy._sellos_de_prueba import rule_grant_intent
from policy.authority_ledger.signatures import sign


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
    return rule_grant_intent(grant or sample_grant())


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


def test_rule_grant_seal_is_not_constructor_reachable():
    # El sello es init=False: ni el constructor ni dataclasses.replace pueden
    # portarlo. Sin sello el intent se construye, pero la frontera que firma
    # (append) lo rechaza -- ver test_storage_decoding_does_not_authorize_a_new_human_rule_grant
    # y los ataques de test_authority_ledger_seal_attacks.py.
    with pytest.raises(TypeError):
        AuthorityEventIntent(
            AuthorityEventType.RULE_RATIFICATION_GRANTED,
            "human:fernando",
            rule_ratification=sample_grant(),
            _rule_ratification_snapshot_seal=object(),
        )
    unsealed = AuthorityEventIntent(
        AuthorityEventType.RULE_RATIFICATION_GRANTED,
        "human:fernando",
        rule_ratification=sample_grant(),
    )
    store, root, key = _ledger()
    with pytest.raises(AuthorityStateError, match="sealed Faro snapshot"):
        append_authority_event(
            store, root, key, unsealed,
            event_id="018cc251-f400-7000-8000-000000000011",
        )
    assert store.events() == ()


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
    with pytest.raises(AuthorityEventValidationError, match="OID"):
        RuleRatificationGrantPayload(**(sample_grant().__dict__ | {"ratified_policy_tree_oid": "d" * 64}))

    sha256_oids = RuleRatificationGrantPayload(**(sample_grant().__dict__ | {
        "rule_blob_oid": "a" * 64,
        "ratified_policy_revision": "c" * 64,
        "ratified_policy_tree_oid": "d" * 64,
    }))
    assert len(sha256_oids.rule_blob_oid) == 64


def test_rule_ratification_codec_rejects_unexpected_fields_and_non_fernando_actor():
    projection = intent_projection(_grant_intent())
    projection["overlay_id"] = "not-part-of-rule-grant"
    with pytest.raises(AuthorityEventValidationError):
        intent_from_projection(projection)

    projection = intent_projection(_grant_intent())
    projection["policy_corpus_hash"] = "sha256:" + "a" * 64
    with pytest.raises(AuthorityEventValidationError):
        intent_from_projection(projection)

    projection = intent_projection(_grant_intent())
    projection["actor_id"] = "actor:service"
    with pytest.raises(AuthorityEventValidationError):
        intent_from_projection(projection)

    with pytest.raises(AuthorityEventValidationError, match="payload Block 4 heredado"):
        AuthorityEventIntent(
            AuthorityEventType.RULE_RATIFICATION_GRANTED,
            "human:fernando",
            policy_corpus_hash="sha256:" + "f" * 64,
            rule_ratification=sample_grant(),
        )


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

    assert state.latest_unrevoked_rule_ratification("send-receipt") is None
    assert state.latest_unrevoked_rule_ratification("missing-rule") is None


def test_replay_exposes_only_the_unrevoked_latest_rule_grant():
    store, root, key = _ledger()
    second_id = "018cc251-f400-7000-8000-000000000002"
    append_authority_event(store, root, key, _test_signable_grant_intent(), event_id="018cc251-f400-7000-8000-000000000001")
    append_authority_event(store, root, key, _test_signable_grant_intent(), event_id=second_id)

    state = verify_authority_ledger(store.get_genesis(), store.events(), root)

    assert state.latest_unrevoked_rule_ratification("send-receipt").event_id == second_id

    # E4 (#369): una ACTIVATION_GRANTED hacia un id que no es ratificación de
    # corpus se rechaza ANTES de escribir -- con triggers append-only, escribirla
    # volvería el ledger inverificable para siempre.
    with pytest.raises(AuthorityStateError, match="activación requiere ratificación"):
        append_authority_event(
            store, root, key,
            AuthorityEventIntent(
                AuthorityEventType.ACTIVATION_GRANTED,
                "human:fernando",
                ratification_event_id=second_id,
            ),
            event_id="018cc251-f400-7000-8000-000000000003",
        )
    verify_authority_ledger(store.get_genesis(), store.events(), root)
    assert len(store.events()) == 2


def test_replay_rejects_unknown_and_duplicate_rule_ratification_revocations():
    store, root, key = _ledger()
    missing = "018cc251-f400-7000-8000-000000000009"
    with pytest.raises(AuthorityStateError, match="rule ratification desconocida"):
        append_authority_event(
            store, root, key,
            AuthorityEventIntent(AuthorityEventType.RULE_RATIFICATION_REVOKED,
                                 "human:fernando", rule_ratification_event_id=missing),
            event_id="018cc251-f400-7000-8000-000000000001",
        )
    verify_authority_ledger(store.get_genesis(), store.events(), root)
    assert store.events() == ()


@pytest.mark.parametrize("direction", ["corpus_to_rule", "rule_to_corpus"])
def test_append_rejects_cross_type_revocations_of_real_grant_ids(direction):
    from tests.policy.test_authority_ledger_events import ratification_intent

    store, root, key = _ledger()
    corpus_id = "018cc251-f400-7000-8000-000000000020"
    rule_id = "018cc251-f400-7000-8000-000000000021"
    if direction == "corpus_to_rule":
        append_authority_event(store, root, key, ratification_intent(), event_id=corpus_id)
        with pytest.raises(AuthorityStateError, match="rule ratification desconocida"):
            append_authority_event(
                store, root, key,
                AuthorityEventIntent(AuthorityEventType.RULE_RATIFICATION_REVOKED,
                                     "human:fernando", rule_ratification_event_id=corpus_id),
                event_id=rule_id,
            )
    else:
        append_authority_event(store, root, key, _test_signable_grant_intent(), event_id=rule_id)
        with pytest.raises(AuthorityStateError, match="ratificación desconocida"):
            append_authority_event(
                store, root, key,
                AuthorityEventIntent(AuthorityEventType.RATIFICATION_REVOKED,
                                     "human:fernando", ratification_event_id=rule_id),
                event_id=corpus_id,
            )

    verify_authority_ledger(store.get_genesis(), store.events(), root)
    assert len(store.events()) == 1


def test_replay_rejects_duplicate_event_id_even_with_valid_signature_and_chain_hash():
    store, root, key = _ledger()
    first = append_authority_event(
        store, root, key,
        AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"),
        event_id="018cc251-f400-7000-8000-000000000030",
    )
    provisional = AuthorityEvent(first.event_id, 2, first.event_hash, first.intent,
                                 first.recorded_at_utc, "", "sha256:" + "0" * 64)
    signature = sign(key, event_unsigned_bytes(provisional))
    signed = AuthorityEvent(first.event_id, 2, first.event_hash, first.intent,
                            first.recorded_at_utc, signature, "sha256:" + "0" * 64)
    duplicate = AuthorityEvent(first.event_id, 2, first.event_hash, first.intent,
                               first.recorded_at_utc, signature, event_hash(signed))

    with pytest.raises(LedgerIntegrityError, match="duplicado"):
        verify_authority_ledger(store.get_genesis(), (first, duplicate), root)

    # La doble revocación del mismo grant ya no se escribe: append la rechaza
    # contra el estado antes de firmar hacia el ledger.
    store, root, key = _ledger()
    grant_id = "018cc251-f400-7000-8000-000000000002"
    revoke_id = "018cc251-f400-7000-8000-000000000003"
    append_authority_event(store, root, key, _test_signable_grant_intent(), event_id=grant_id)
    revoke = AuthorityEventIntent(
        AuthorityEventType.RULE_RATIFICATION_REVOKED,
        "human:fernando", rule_ratification_event_id=grant_id,
    )
    append_authority_event(store, root, key, revoke, event_id=revoke_id)
    with pytest.raises(AuthorityStateError, match="revocada más de una vez"):
        append_authority_event(
            store, root, key, revoke, event_id="018cc251-f400-7000-8000-000000000004",
        )
    verify_authority_ledger(store.get_genesis(), store.events(), root)
    assert len(store.events()) == 2
