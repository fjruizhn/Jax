import pytest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from policy.decision_record.errors import DecisionIdConflictError
from policy.decision_record.service import build_decision_record
from policy.decision_record.storage import InMemoryDecisionRecordStore
from tests.policy.test_decision_record import make_evaluation, make_record


def test_duplicate_id_identical_is_idempotent_and_different_rejected():
    store = InMemoryDecisionRecordStore()
    record = make_record()
    assert store.insert(record).decision_record_hash == record.decision_record_hash
    assert store.insert(record).decision_record_hash == record.decision_record_hash
    changed = build_decision_record(make_evaluation(), decision_id=record.decision_id,
                                    recorded_at_utc=record.recorded_at_utc.replace(hour=3))
    with pytest.raises(DecisionIdConflictError): store.insert(changed)


def test_concurrent_same_id_creates_one_immutable_record():
    store, record = InMemoryDecisionRecordStore(), make_record()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.insert(record).decision_record_hash, range(4)))
    assert results == [record.decision_record_hash] * 4


def test_migration_contains_real_immutability_triggers():
    sql = (Path(__file__).resolve().parents[2] / "policy/decision_record/migrations/001_decision_records.sql").read_text()
    assert "BEFORE UPDATE" in sql and "BEFORE DELETE" in sql
