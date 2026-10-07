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

_UNSET = object()


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


def storage_row(event, *, sequence=None, event_id=None, event_type=None, actor_id=None,
                canonical_intent=_UNSET, canonical_event=None, evidence_refs=None,
                previous_event_hash=_UNSET, event_hash_value=None, signature=None,
                recorded_at_utc=None):
    return (
        event.sequence if sequence is None else sequence,
        event.event_id if event_id is None else event_id,
        event.intent.event_type.value if event_type is None else event_type,
        event.intent.actor_id if actor_id is None else actor_id,
        canonical_bytes(intent_projection(event.intent)) if canonical_intent is _UNSET else canonical_intent,
        canonical_bytes(event_projection(event)) if canonical_event is None else canonical_event,
        canonical_bytes(event.intent.evidence_refs) if evidence_refs is None else evidence_refs,
        event.previous_event_hash if previous_event_hash is _UNSET else previous_event_hash,
        event.event_hash if event_hash_value is None else event_hash_value,
        event.signature if signature is None else signature,
        event.recorded_at_utc.replace(tzinfo=None) if recorded_at_utc is None else recorded_at_utc,
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
    restored = event_from_storage_row(storage_row(event))
    assert restored == event
    assert restored.unsigned_projection() == event.unsigned_projection()


def test_all_six_existing_event_variants_roundtrip_without_changing_signed_projection():
    for number, intent in enumerate(sample_intents(), 1):
        event = sample_event(intent, number)

        restored = event_from_storage_row(storage_row(event))

        assert restored == event
        assert restored.unsigned_projection() == event.unsigned_projection()

    event = sample_event()
    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row(storage_row(event, evidence_refs=canonical_bytes(("sha256:" + "d" * 64,))))


def test_event_storage_decoder_rejects_unknown_event_fields():
    projection = event_projection(sample_event())
    projection["unexpected"] = "ignored fields can hide unsigned data"

    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row(storage_row(sample_event(), canonical_event=canonical_bytes(projection)))

    projection = event_projection(sample_event())
    projection["intent"]["policy_corpus_hash"] = "sha256:" + "f" * 64
    with pytest.raises(AuthorityEventValidationError, match="payload inválido para ACTIVATION_DEACTIVATED"):
        event_from_storage_row(storage_row(
            sample_event(),
            canonical_event=canonical_bytes(projection),
            canonical_intent=canonical_bytes(projection["intent"]),
        ))

    projection = event_projection(sample_event())
    projection["intent"]["overlay_id"] = "unexpected-payload"
    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row(storage_row(sample_event(), canonical_event=canonical_bytes(projection)))


def test_decoded_ratification_cannot_be_resigned_and_appended_as_validated_candidate():
    from policy.authority_ledger.service import append_authority_event
    from policy.authority_ledger.errors import AuthorityStateError
    from tests.policy.test_authority_ledger_events import setup_ledger

    store, root, key = setup_ledger()
    corpus_hash = "sha256:" + "f" * 64
    projection = {
        "event_type": "RATIFICATION_GRANTED",
        "actor_id": "human:fernando",
        "evidence_refs": [],
        "policy_corpus_hash": corpus_hash,
        "static_policy_view_projection": {"policy_corpus_hash": corpus_hash},
        "ratification_event_id": None,
        "overlay": None,
        "overlay_id": None,
    }
    forged = __import__("policy.authority_ledger.serialization", fromlist=["intent_from_projection"]).intent_from_projection(projection)

    with pytest.raises(AuthorityStateError, match="storage"):
        append_authority_event(store, root, key, forged)

    assert store.events() == ()

def test_event_storage_decoder_rejects_legacy_dataclass_shape():
    event = sample_event()
    projection = event_projection(event)
    projection["intent"]["_ratification_snapshot_seal"] = None

    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row(storage_row(event, canonical_event=canonical_bytes(projection)))


def test_event_storage_decoder_rejects_three_column_rows():
    event = sample_event()

    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row((event.sequence, event.event_id, canonical_bytes(event_projection(event))))


@pytest.mark.parametrize("changes", [
    {"event_id": "018cc251-f400-7000-8000-000000000002"},
    {"sequence": 2},
    {"event_type": "RATIFICATION_GRANTED"},
    {"actor_id": "actor:other"},
    {"event_hash_value": "sha256:" + "c" * 64},
    {"signature": "different-signature"},
    {"recorded_at_utc": datetime(2026, 10, 7, tzinfo=timezone.utc)},
    {"canonical_intent": canonical_bytes({"forged": True})},
    {"canonical_intent": None},
])
def test_event_storage_decoder_rejects_table_columns_that_disagree_with_canonical_event(changes):
    event = sample_event()

    with pytest.raises(AuthorityEventValidationError):
        event_from_storage_row(storage_row(event, **changes))


def test_event_storage_reader_selects_all_columns_it_validates():
    event = sample_event()
    with pytest.raises(AuthorityEventValidationError, match="columnas authority_event"):
        event_from_storage_row(
            storage_row(event, previous_event_hash="sha256:" + "c" * 64)
        )
    queries = []

    class Cursor:
        def execute(self, sql):
            queries.append(sql)

        def fetchall(self):
            return [storage_row(event)]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def cursor(self):
            return Cursor()

    assert MariaDBAuthorityLedgerStore(Connection).events() == (event,)
    assert queries == [
        "SELECT sequence,event_id,event_type,actor_id,canonical_intent,canonical_event,evidence_refs,previous_event_hash,event_hash,signature,recorded_at_utc "
        "FROM jax_authority.authority_events ORDER BY sequence ASC"
    ]


def test_mariadb_writer_persists_all_three_public_projections():
    event = sample_event()
    statements = []

    class Cursor:
        rowcount = 1

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
    assert next(sql for sql, _ in statements if sql.startswith("SELECT")) == (
        "SELECT sequence,head_event_hash FROM jax_authority.authority_ledger_head "
        "WHERE singleton=1 FOR UPDATE"
    )


@pytest.mark.parametrize("head_row,update_rowcount", [(None, 1), ((0, None), 0)])
def test_mariadb_writer_fails_closed_if_head_is_missing_or_update_did_not_touch_one_row(head_row, update_rowcount):
    from policy.authority_ledger.errors import AuthorityStateError

    statements = []
    commits = []

    class Cursor:
        rowcount = 1

        def execute(self, sql, params=None):
            statements.append(sql)
            if sql.startswith("UPDATE"):
                self.rowcount = update_rowcount

        def fetchone(self):
            return head_row

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            commits.append(True)

        def rollback(self):
            pass

        def close(self):
            pass

    with pytest.raises(AuthorityStateError):
        MariaDBAuthorityLedgerStore(Connection).append(sample_event())
    assert commits == []

    assert any(statement.startswith("INSERT") for statement in statements) is (head_row is not None)
