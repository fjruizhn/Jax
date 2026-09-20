from policy.authority_resolution.models import ConflictRecord
from policy.authority_resolution import resolve_static_authority
from policy.authority_resolution.models import FrozenNormativeDocument, FrozenRelationships, FrozenScope
from test_authority_resolution_context import context
from test_authority_resolution_supersession import _view

def test_conflict_record_shape_is_closed():
    record = ConflictRecord("UNRESOLVED_SAME_RANK_MULTIPLE_RULES", "PRODUCT_POLICY", ("a", "b"))
    assert len(record.rule_ids) == 2

def test_conflict_record_is_stably_sorted_by_caller_contract():
    record = ConflictRecord("UNRESOLVED_SAME_RANK_MULTIPLE_RULES", "PRODUCT_POLICY", ("a", "b"))
    assert record.kind == "UNRESOLVED_SAME_RANK_MULTIPLE_RULES"
    assert record.precedence_layer == "PRODUCT_POLICY"

def _doc(identifier, conditions=()):
    return FrozenNormativeDocument(identifier, "PRODUCT_POLICY", "PRODUCT_POLICY", "ACTIVE_WHEN_CORPUS_ACTIVE", FrozenScope("JAX", ("ALICE",), ("READ",), tuple(conditions)), FrozenRelationships((), ()))

def test_top_conflict_plus_relevant_unknown_is_indeterminate():
    result = resolve_static_authority(_view((_doc("a"), _doc("b"), _doc("c", ("UNKNOWN",)))), context())
    assert result.resolution_status == "STATIC_INDETERMINATE"
    assert result.conflicts == ()

def test_two_top_deterministic_rules_are_one_conflict_record():
    result = resolve_static_authority(_view((_doc("a"), _doc("b"))), context())
    assert result.resolution_status == "STATIC_CONFLICT"
    assert len(result.conflicts) == 1
    assert result.conflicts[0].rule_ids == ("a", "b")
