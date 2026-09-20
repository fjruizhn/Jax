from datetime import datetime, timezone

from policy.authority_ledger.effective_context import build_effective_authority_context
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_resolution.models import EvaluationContext
from tests.policy.test_authority_ledger_events import setup_ledger, ratification_intent


def test_effective_context_is_not_execution_authorization():
    store, root, key = setup_ledger()
    rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    context = EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "ALICE", "READ", ())
    effective = build_effective_authority_context(verify_authority_ledger(store.get_genesis(), store.events(), root), context, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert not hasattr(effective, "decision_input_hash") and not hasattr(effective, "execution_eligible")
