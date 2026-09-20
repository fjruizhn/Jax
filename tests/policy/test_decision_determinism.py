import pytest

from policy.decision_record.errors import DecisionRecordIntegrityError
from policy.decision_record.models import DecisionRecord
from policy.decision_record.serialization import canonical_decision_record_bytes, decision_record_from_bytes
from tests.policy.test_decision_record import make_record


def test_canonical_round_trip_is_byte_stable():
    record = make_record()
    raw = canonical_decision_record_bytes(record)
    assert canonical_decision_record_bytes(decision_record_from_bytes(raw)) == raw


def test_publicly_reconstructed_record_is_not_registered_as_verified():
    record = make_record()
    lookalike = DecisionRecord(record.schema_version, record.kind, record.decision_id,
                               record.decision_input, record.decision_input_hash,
                               record.authority_binding, record.result, record.evidence_refs,
                               record.recorded_at_utc, record.decision_record_hash)
    with pytest.raises(DecisionRecordIntegrityError):
        canonical_decision_record_bytes(lookalike)
