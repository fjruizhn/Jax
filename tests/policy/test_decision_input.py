from datetime import datetime, timedelta, timezone

import pytest

from policy.authority_resolution.models import ConditionResult, EvaluationContext
from policy.decision_record.errors import InvalidDecisionFactError, InvalidDecisionInputError
from policy.decision_record.models import DecisionFact, DecisionFactValueType, DecisionInput
from policy.decision_record.service import build_decision_input, compute_decision_input_hash


def context(subject="ALICE", action="READ", conditions=()):
    return EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", subject, action, tuple(conditions))


def instant(hour=0):
    return datetime(2026, 1, 1, hour, tzinfo=timezone.utc)


def test_equivalent_semantic_input_has_same_hash():
    facts = (DecisionFact("B", DecisionFactValueType.INTEGER, 2), DecisionFact("A", DecisionFactValueType.STRING_SET, ("Y", "X")))
    left = build_decision_input(context(conditions=(ConditionResult("READY", True),)), instant(), facts=facts)
    right = build_decision_input(context(conditions=(ConditionResult("READY", True),)), instant(), facts=tuple(reversed(facts)))
    assert compute_decision_input_hash(left) == compute_decision_input_hash(right)


@pytest.mark.parametrize("changed", [
    lambda: build_decision_input(context("BOB"), instant()),
    lambda: build_decision_input(context(action="WRITE"), instant()),
    lambda: build_decision_input(context(conditions=(ConditionResult("READY", False),)), instant()),
    lambda: build_decision_input(context(), instant(), facts=(DecisionFact("COUNT", DecisionFactValueType.INTEGER, 2),)),
    lambda: build_decision_input(context(), instant(1)),
])
def test_semantic_change_changes_hash(changed):
    base = build_decision_input(context(), instant(), facts=(DecisionFact("COUNT", DecisionFactValueType.INTEGER, 1),))
    assert base.decision_input_hash != changed().decision_input_hash


def test_evidence_is_provenance_not_case_identity():
    a = build_decision_input(context(), instant(), facts=(DecisionFact("FLAG", DecisionFactValueType.BOOLEAN, True, ("sha256:" + "a" * 64,)),))
    b = build_decision_input(context(), instant(), facts=(DecisionFact("FLAG", DecisionFactValueType.BOOLEAN, True, ("sha256:" + "b" * 64,)),))
    assert a.decision_input_hash == b.decision_input_hash


def test_closed_input_rejects_float_bool_int_and_naive_time():
    with pytest.raises(InvalidDecisionFactError): DecisionFact("X", DecisionFactValueType.BOOLEAN, 1)
    with pytest.raises(InvalidDecisionFactError): DecisionFact("X", DecisionFactValueType.INTEGER, True)
    with pytest.raises(InvalidDecisionFactError): DecisionFact("X", DecisionFactValueType.STRING_SET, ("X", "X"))
    with pytest.raises(InvalidDecisionInputError): build_decision_input(context(), datetime(2026, 1, 1))


def test_publicly_reconstructed_input_is_not_registered_as_verified():
    genuine = build_decision_input(context(), instant())
    lookalike = DecisionInput(genuine.schema_version, genuine.kind, genuine.evaluation_context,
                              genuine.evaluation_time_utc, genuine.facts)
    with pytest.raises(InvalidDecisionInputError):
        compute_decision_input_hash(lookalike)
