from policy.decision_record.ids import decision_id
from policy.decision_record.service import build_decision_input
from policy.decision_record.models import DecisionFact, DecisionFactValueType
from tests.policy.test_decision_input import context, instant


def test_authority_is_excluded_from_case_identity():
    first = build_decision_input(context(), instant(), facts=(DecisionFact("X", DecisionFactValueType.STRING, "same"),))
    second = build_decision_input(context(), instant(), facts=(DecisionFact("X", DecisionFactValueType.STRING, "same"),))
    assert first.decision_input_hash == second.decision_input_hash


def test_uuid7_validator_rejects_non_v7():
    import pytest
    from policy.decision_record.errors import DecisionContractError
    with pytest.raises(DecisionContractError): decision_id("00000000-0000-4000-8000-000000000000")
