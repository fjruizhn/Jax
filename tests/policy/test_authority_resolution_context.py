from pathlib import Path
import pytest

from policy.authority_resolution import (
    ConditionResult, EvaluationContext, load_validated_candidate,
    to_static_policy_view, resolve_static_authority,
)
from policy.authority_resolution.errors import InvalidEvaluationContextError, ResolverContractError

ROOT = Path(__file__).resolve().parents[2]

def empty_view():
    return to_static_policy_view(load_validated_candidate(ROOT))

def context(**changes):
    value = {
        "schema_version": "1.0", "kind": "JAX_AUTHORITY_EVALUATION_CONTEXT",
        "jurisdiction": "JAX", "subject": "ALICE", "action": "READ", "conditions": [],
    }
    value.update(changes)
    return EvaluationContext.from_mapping(value)

def test_scalar_subject_action_and_empty_candidate():
    result = resolve_static_authority(empty_view(), context())
    assert result.resolution_status == "NO_STATIC_RULE_SELECTED"

def test_boolean_is_exact_and_missing_is_distinct():
    assert ConditionResult("HAS_APPROVAL", True).value is True
    with pytest.raises(InvalidEvaluationContextError):
        ConditionResult("HAS_APPROVAL", 1)
    with pytest.raises(InvalidEvaluationContextError):
        ConditionResult("HAS_APPROVAL", "TRUE")

def test_raw_dict_and_bad_context_are_rejected():
    with pytest.raises(ResolverContractError):
        resolve_static_authority(empty_view(), {"subject": "ALICE"})
    with pytest.raises(InvalidEvaluationContextError):
        context(subject="alice")
