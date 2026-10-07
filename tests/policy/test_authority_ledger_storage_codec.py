from datetime import datetime, timezone

import pytest

from policy.authority_ledger.errors import AuthorityEventValidationError
from policy.authority_ledger.models import (
    AuthorityEvent, AuthorityEventIntent, AuthorityEventType,
    OverlayPayload, OverlayScope, OverlayType,
)
from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.serialization import event_from_storage_row, event_projection, intent_projection
from policy.authority_ledger.storage import MariaDBAuthorityLedgerStore


def sample_event(intent=None, number=1):
    intent = intent or AuthorityEventIntent(
        AuthorityEventType.ACTIVATION_DEACTIVATED,
        "human:fernando",
        ("sha256:" + "a" * 64,),
    )
    return AuthorityEvent(
        f"018cc251-f400-7000-8000-{number:012d}",
        1,
        None,
        intent,
        datetime(2026, 10, 6, tzinfo=timezone.utc),
        "test-signature",
        "sha256:" + "b" * 64,
    )


def sample_intents():
    corpus_hash = "sha256:" + "c" * 64
    return (
        AuthorityEventIntent._from_validated_snapshot(corpus_hash, {"policy_corpus_hash": corpus_hash}),
        AuthorityEventIntent(AuthorityEventType.RATIFICATION_REVOKED, "human:fernando", ratification_event_id="018cc251-f400-7000-8000-000000000002"),
        AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id="018cc251-f400-7000-8000-000000000002"),
        AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"),
        AuthorityEventIntent(
            AuthorityEventType.OVERLAY_ISSUED,
            "human:fernando",
            overlay=OverlayPayload(
                "exception-a", OverlayType.EXCEPTION, corpus_hash,
                OverlayScope(("ALICE",), ("READ",)),
                datetime(2026, 10, 6, tzinfo=timezone.utc), None,
                target_rule_ids=("rule-a",), exception_code="EXCEPTION",
            ),
        ),
        AuthorityEventIntent(AuthorityEventType.OVERLAY_REVOKED, "human:fernando", overlay_id="exception-a"),
    )


def test_event_storage_projection_is_closed_and_roundtrips_public_payload():
    event = sample_event()

    projection = event_projection(event)

    assert set(projection) == {
        "event_id", "sequence", "previous_event_hash", "intent",
        "recorded_at_utc", "signature", "event_hash",
    }
    assert projection["intent"] == {
        "event_type": "ACTIVATION_DEACTIVATED",
        "actor_id": "human:fernando",
        "evidence_refs": ["sha256:" + "a" * 64],
        "policy_corpus_hash": None,
        "static_policy_view_projection": None,
        "ratification_event_id": None,
        "overlay": None,
        "overlay_id": None,
    }
    assert "_ratification_snapshot_seal" not in repr(projection)
    restored = event_from_storage_row((1, event.event_id, projection))
    assert restored == event
    assert restored.unsigned_projection() == event.unsigned_projection()


def test_all_six_existing_event_variants_roundtrip_without_changing_signed_projection():
    for number, intent in enumerate(sample_intents(), 1):
        event = sample_event(intent, number)

        restored = event_from_storage_row((
            event.sequence,
            event.event_id,
            canonical_bytes(intent_projection(event.intent)),
            canonical_bytes(event_projection(event)),
            canonical_bytes(event.intent.evidence_refs),
        ))

        assert restored == event
        assert restored.unsigned_projection() == event.unsigned_projection()

    event = sample_event()
    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row((
            event.sequence, event.event_id,
            canonical_bytes(intent_projection(event.intent)),
            canonical_bytes(event_projection(event)),
            canonical_bytes(("sha256:" + "d" * 64,)),
        ))


def test_event_storage_decoder_rejects_unknown_event_fields():
    projection = event_projection(sample_event())
    projection["unexpected"] = "ignored fields can hide unsigned data"

    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row((1, projection["event_id"], projection))

    projection = event_projection(sample_event())
    projection["intent"]["overlay_id"] = "unexpected-payload"
    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row((1, projection["event_id"], projection))


def test_event_storage_decoder_accepts_only_the_exact_legacy_dataclass_shape():
    event = sample_event()
    projection = event_projection(event)
    projection["intent"]["_ratification_snapshot_seal"] = None

    assert event_from_storage_row((1, event.event_id, projection)) == event

    projection["intent"]["_ratification_snapshot_seal"] = "forged"
    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row((1, event.event_id, projection))


def test_mariadb_writer_persists_all_three_public_projections():
    event = sample_event()
    statements = []

    class Cursor:
        def execute(self, sql, params=None):
            statements.append((sql, params))

        def fetchone(self):
            return (0, None)

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    MariaDBAuthorityLedgerStore(Connection).append(event)

    insert_sql, params = next((sql, values) for sql, values in statements if sql.startswith("INSERT"))
    assert "canonical_intent,canonical_event,evidence_refs" in insert_sql
    assert params[4] == canonical_bytes(intent_projection(event.intent))
    assert params[5] == canonical_bytes(event_projection(event))
    assert params[6] == canonical_bytes(event.intent.evidence_refs)
