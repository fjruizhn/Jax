from datetime import timedelta

from policy.authority_ledger.effective_context import build_effective_authority_context
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_resolution.models import EvaluationContext
from tests.policy.test_authority_ledger_events import base_time, setup_ledger, ratification_intent, overlay


def _state():
    store, root, key = setup_ledger()
    rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    return verify_authority_ledger(store.get_genesis(), store.events(), root)


def test_context_facts_and_time_do_not_enter_effective_authority_hash():
    state = _state()
    a = EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "ALICE", "READ", ())
    b = EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "BOB", "WRITE", ())
    assert build_effective_authority_context(state, a, base_time()).effective_authority_context_hash == build_effective_authority_context(state, b, base_time() + timedelta(hours=2)).effective_authority_context_hash


def test_effective_overlay_changes_context_hash():
    store, root, key = setup_ledger()
    rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    context = EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "ALICE", "READ", ())
    before = build_effective_authority_context(verify_authority_ledger(store.get_genesis(), store.events(), root), context, base_time()).effective_authority_context_hash
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=overlay()))
    after = build_effective_authority_context(verify_authority_ledger(store.get_genesis(), store.events(), root), context, base_time()).effective_authority_context_hash
    assert before != after
