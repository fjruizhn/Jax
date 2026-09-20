from policy.decision_record.serialization import canonical_decision_record_bytes, decision_record_from_bytes
from tests.policy.test_decision_record import make_record


def test_canonical_round_trip_is_byte_stable():
    record = make_record()
    raw = canonical_decision_record_bytes(record)
    assert canonical_decision_record_bytes(decision_record_from_bytes(raw)) == raw
