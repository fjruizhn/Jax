from datetime import timedelta

import pytest

from policy.decision_record.errors import DecisionRecordIntegrityError
from policy.decision_record.ids import new_decision_id
from policy.decision_record.service import build_decision_record, build_decision_input
from tests.policy.test_decision_authority_binding import active_state
from policy.decision_record.authority_binding import evaluate_decision_input
from tests.policy.test_decision_input import context, instant


def make_evaluation():
    _, _, state = active_state()
    return evaluate_decision_input(state, build_decision_input(context(), instant()))


def make_record():
    return build_decision_record(make_evaluation(), decision_id=new_decision_id(), recorded_at_utc=instant(2))


def test_record_is_integrity_bound_and_immutable():
    record = make_record()
    with pytest.raises(Exception): record.evidence_refs += ("sha256:" + "a" * 64,)
    with pytest.raises(DecisionRecordIntegrityError):
        type(record)(record.schema_version, record.kind, record.decision_id, record.decision_input,
                     "sha256:" + "f" * 64, record.authority_binding, record.result, record.evidence_refs,
                     record.recorded_at_utc, record.decision_record_hash)


def test_same_case_different_record_instances_keep_input_identity():
    _, _, state = active_state()
    evaluation = evaluate_decision_input(state, build_decision_input(context(), instant()))
    a = build_decision_record(evaluation, decision_id=new_decision_id(), recorded_at_utc=instant(2))
    b = build_decision_record(evaluation, decision_id=new_decision_id(), recorded_at_utc=instant(3))
    assert a.decision_input_hash == b.decision_input_hash
    assert a.decision_record_hash != b.decision_record_hash
