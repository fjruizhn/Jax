from test_authority_resolution_context import context
from policy.authority_resolution import resolve_static_authority
from policy.authority_resolution.models import FrozenNormativeDocument, FrozenRelationships, FrozenScope
from test_authority_resolution_supersession import _view

def _doc(identifier, layer, conditions=(), supersedes=()):
    return FrozenNormativeDocument(identifier, layer, layer, "ACTIVE_WHEN_CORPUS_ACTIVE", FrozenScope("JAX", ("ALICE",), ("READ",), tuple(conditions)), FrozenRelationships(tuple(supersedes), ()))

def _result(docs):
    return resolve_static_authority(_view(docs), context())

def test_missing_condition_is_representable_without_lookup():
    assert context().conditions == ()

def test_case_a_lower_unknown_does_not_block_constitutional_applicable():
    result = _result((_doc("c1", "CONSTITUTIONAL_CORE"), _doc("s1", "SUBORDINATE_POLICY", ("UNKNOWN",))))
    assert result.resolution_status == "STATIC_CONTROLLING_RULE_SELECTED"
    states = {r.rule_id: r for r in result.rule_results}
    assert states["c1"].selection_state.value == "CONTROLLING"
    assert states["s1"].selection_state.value == "IRRELEVANT"

def test_case_b_constitutional_unknown_blocks_product_rule():
    result = _result((_doc("c1", "CONSTITUTIONAL_CORE", ("UNKNOWN",)), _doc("p1", "PRODUCT_POLICY")))
    assert result.resolution_status == "STATIC_INDETERMINATE"
    states = {r.rule_id: r for r in result.rule_results}
    assert states["c1"].selection_state.value == "ELIGIBLE"
    assert states["p1"].selection_state.value == "LOWER_PRECEDENCE"

def test_case_c_same_rank_unknown_blocks_selection():
    result = _result((_doc("c1", "CONSTITUTIONAL_CORE"), _doc("c2", "CONSTITUTIONAL_CORE", ("UNKNOWN",))))
    assert result.resolution_status == "STATIC_INDETERMINATE"
    assert all(r.selection_state.value == "ELIGIBLE" for r in result.rule_results)

def test_case_d_unknown_superseder_is_relevant():
    result = _result((_doc("a", "CONSTITUTIONAL_CORE", ("UNKNOWN",), ("b",)), _doc("b", "CONSTITUTIONAL_CORE")))
    assert result.resolution_status == "STATIC_INDETERMINATE"

def test_case_e_lower_unknown_is_irrelevant_below_constitutional_winner():
    result = _result((_doc("a", "PRODUCT_POLICY", ("UNKNOWN",)), _doc("b", "PRODUCT_POLICY"), _doc("c", "CONSTITUTIONAL_CORE")))
    assert result.resolution_status == "STATIC_CONTROLLING_RULE_SELECTED"
