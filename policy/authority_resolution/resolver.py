"""Pure static rule selection.  This module deliberately performs no I/O."""
from __future__ import annotations

from dataclasses import replace

from .errors import ResolverContractError, ResolverInvariantError
from .models import (
    ApplicabilityState, ConflictRecord, EvaluationContext, RuleResolutionRecord,
    SelectionState, StaticAuthorityResolution, ValidatedStaticPolicyView,
)

IDENTITY = "JAX-AUTHORITY-RESOLVER/1"
VERSION = "1.0"
RANK = {"CONSTITUTIONAL_CORE": 3, "PRODUCT_POLICY": 2, "SUBORDINATE_POLICY": 1}

def _initial(view, context):
    conditions = {c.id: c.value for c in context.conditions}
    states = {}
    for doc in view.ordinary_documents:
        if context.subject not in doc.scope.subjects or context.action not in doc.scope.actions:
            state = ApplicabilityState.NOT_APPLICABLE
        else:
            missing = any(c not in conditions for c in doc.scope.conditions_all)
            false = any(not conditions[c] for c in doc.scope.conditions_all if c in conditions)
            state = ApplicabilityState.INDETERMINATE if missing else (ApplicabilityState.NOT_APPLICABLE if false else ApplicabilityState.APPLICABLE)
        states[doc.id] = [doc, state, SelectionState.IRRELEVANT, False, set(), set()]
    return states

def _closure(view):
    return {d.id: tuple(d.relationships.supersedes) for d in view.ordinary_documents}

def _descendants(edges, source):
    out, stack = set(), list(edges.get(source, ()))
    while stack:
        item = stack.pop()
        if item in out:
            continue
        out.add(item)
        stack.extend(edges.get(item, ()))
    return out

def resolve_static_authority(policy_view: ValidatedStaticPolicyView, context: EvaluationContext) -> StaticAuthorityResolution:
    if not isinstance(policy_view, ValidatedStaticPolicyView) or not isinstance(context, EvaluationContext):
        raise ResolverContractError("resolver requiere objetos validados e inmutables")
    states = _initial(policy_view, context)
    edges = _closure(policy_view)
    for source, (doc, applicability, *_rest) in states.items():
        if applicability is not ApplicabilityState.APPLICABLE:
            continue
        for target in _descendants(edges, source):
            if target not in states:
                raise ResolverInvariantError("grafo apunta a documento inexistente")
            target_state = states[target]
            if target_state[1] in (ApplicabilityState.APPLICABLE, ApplicabilityState.INDETERMINATE):
                target_state[2] = SelectionState.SUPPRESSED
                target_state[3] = False
                target_state[4].add(source)

    deterministic = []
    unknown = []
    for item in states.values():
        if item[2] is SelectionState.SUPPRESSED:
            continue
        if item[1] is ApplicabilityState.INDETERMINATE:
            unknown.append(item)
        elif item[1] is ApplicabilityState.APPLICABLE:
            deterministic.append(item)

    top_deterministic_rank = max((RANK[x[0].normative_layer] for x in deterministic), default=None)
    relevant_unknown = [x for x in unknown if top_deterministic_rank is None or RANK[x[0].normative_layer] >= top_deterministic_rank]
    for item in relevant_unknown:
        item[2] = SelectionState.ELIGIBLE
        item[3] = True
    for item in unknown:
        if item not in relevant_unknown and item[2] is not SelectionState.SUPPRESSED:
            item[2] = SelectionState.IRRELEVANT
    if relevant_unknown:
        status = "STATIC_INDETERMINATE"
        controlling = None
        winning = None
        conflicts = ()
        unknown_rank = max(RANK[x[0].normative_layer] for x in relevant_unknown)
        for item in deterministic:
            item[2] = (SelectionState.LOWER_PRECEDENCE
                       if RANK[item[0].normative_layer] < unknown_rank
                       else SelectionState.ELIGIBLE)
    else:
        top = max((RANK[x[0].normative_layer] for x in deterministic), default=None)
        top_items = [x for x in deterministic if RANK[x[0].normative_layer] == top] if top is not None else []
        if not top_items:
            status, controlling, winning, conflicts = "NO_STATIC_RULE_SELECTED", None, None, ()
        elif len(top_items) == 1:
            top_items[0][2] = SelectionState.CONTROLLING
            status, controlling, winning, conflicts = "STATIC_CONTROLLING_RULE_SELECTED", top_items[0][0].id, top_items[0][0].normative_layer, ()
        else:
            for item in top_items:
                item[2] = SelectionState.CONFLICTING
            ids = tuple(sorted(item[0].id for item in top_items))
            record = ConflictRecord("UNRESOLVED_SAME_RANK_MULTIPLE_RULES", top_items[0][0].normative_layer, ids)
            status, controlling, winning, conflicts = "STATIC_CONFLICT", None, top_items[0][0].normative_layer, (record,)
    if winning is not None:
        for item in deterministic:
            if item[2] is SelectionState.ELIGIBLE and RANK[item[0].normative_layer] < RANK[winning]:
                item[2] = SelectionState.LOWER_PRECEDENCE
    results = []
    for doc, app, sel, relevant, sources, _ in sorted(states.values(), key=lambda x: x[0].id):
        if app is ApplicabilityState.NOT_APPLICABLE:
            sel = SelectionState.IRRELEVANT
        results.append(RuleResolutionRecord(doc.id, doc.document_class, doc.normative_layer, app, sel, tuple(sorted(doc.relationships.superseded_by)), tuple(sorted(sources)), relevant))
    return StaticAuthorityResolution("1.0", "JAX_STATIC_AUTHORITY_RESOLUTION", IDENTITY, VERSION, policy_view.policy_corpus_hash, context, tuple(results), controlling, winning, conflicts, tuple(sorted(x[0].id for x in relevant_unknown)), policy_view.authority_meta_contract.protected_metanorms, "STATIC_DECLARATION_ONLY", status)
