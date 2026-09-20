from dataclasses import replace
from policy.authority_resolution.models import (
    FrozenAuthorityMetaContract, FrozenExternalConstraints, FrozenFailClosed,
    FrozenNormativeDocument, FrozenNormativeSources, FrozenPrecedence,
    FrozenRelationships, FrozenScope, ProtectedConstraint, ValidatedStaticPolicyView,
)
from policy.authority_resolution import resolve_static_authority
from test_authority_resolution_context import context

def _view(docs):
    authority = FrozenAuthorityMetaContract("JAX", FrozenPrecedence("ROOT_META_LEVEL", ("PROTECTED_METANORM", "CONSTITUTIONAL_CORE", "PRODUCT_POLICY", "SUBORDINATE_POLICY"), "POST_CORPUS_WITHIN_VALID_DELEGATED_SCOPE", "EXTERNAL_CEILING", "REJECT_UNTIL_RATIFIED_RESOLUTION", "REJECT"), FrozenNormativeSources(("CONSTITUTIONAL_CORE", "PRODUCT_POLICY", "SUBORDINATE_POLICY"), True, False, False, False), (ProtectedConstraint("FAIL_CLOSED_BY_DEFAULT", "x"),), FrozenExternalConstraints(False, "CEILING_ONLY", False, True), FrozenFailClosed("REJECT", "REJECT", "REJECT", "REJECT", "REJECT", "REJECT", "REJECT"))
    return ValidatedStaticPolicyView("1.0", "JAX_VALIDATED_STATIC_POLICY_VIEW", "sha256:" + "0"*64, "JAX-POLICY-C14N/3", "sha256:" + "1"*64, authority, tuple(sorted(docs, key=lambda d:d.id)))

def _doc(identifier, supersedes=()):
    return FrozenNormativeDocument(identifier, "CONSTITUTIONAL_CORE", "CONSTITUTIONAL_CORE", "ACTIVE_WHEN_CORPUS_ACTIVE", FrozenScope("JAX", ("ALICE",), ("READ",), ()), FrozenRelationships(tuple(supersedes), ()))

def test_empty_graph_is_deterministic():
    assert resolve_static_authority(_view((_doc("a"),)), context()).controlling_rule_id == "a"

def test_applicable_source_suppresses_transitive_target_through_non_applicable_node():
    a = _doc("a", ("b",)); b = _doc("b", ("c",)); c = _doc("c")
    b = replace(b, scope=FrozenScope("JAX", ("BOB",), ("READ",), ()))
    # The frozen fixture expresses the structural graph; the resolver traverses
    # it even when an intermediate applicability state is not applicable.
    view = _view((a, b, c))
    result = resolve_static_authority(view, context())
    assert result.resolution_status == "STATIC_CONTROLLING_RULE_SELECTED"

def test_applicable_source_suppresses_indeterminate_target():
    a = _doc("a", ("b",)); b = replace(_doc("b"), scope=FrozenScope("JAX", ("ALICE",), ("READ",), ("UNKNOWN",)))
    result = resolve_static_authority(_view((a, b)), context())
    states = {r.rule_id: r for r in result.rule_results}
    assert states["b"].selection_state.value == "SUPPRESSED"
    assert not states["b"].indeterminacy_relevant
