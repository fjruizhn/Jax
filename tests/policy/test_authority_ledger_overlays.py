from datetime import timedelta
import pytest

from policy.authority_ledger.errors import OverlayApplicabilityIndeterminateError, OverlayConflictError
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType, OverlayType
from policy.authority_ledger.replay import effective_overlays, verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_resolution.models import ConditionResult, EvaluationContext
from tests.policy.test_authority_ledger_events import base_time, overlay, setup_ledger
from policy.authority_ledger.models import OverlayPayload, OverlayScope


def context(*conditions):
    return EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "ALICE", "READ", tuple(conditions))


def state_with(*items):
    store, root, key = setup_ledger()
    from tests.policy.test_authority_ledger_events import ratification_intent
    rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=rat.event_id))
    for item in items:
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=item))
    return verify_authority_ledger(store.get_genesis(), store.events(), root)


def test_false_not_applicable_missing_fails_closed_and_identical_deduplicates():
    required = overlay(conditions=("APPROVED",))
    assert effective_overlays(state_with(required), context(ConditionResult("APPROVED", False)), base_time()) == ()
    with pytest.raises(OverlayApplicabilityIndeterminateError):
        effective_overlays(state_with(required), context(), base_time())
    duplicate = overlay("exception-b")
    assert len(effective_overlays(state_with(overlay(), duplicate), context(), base_time())) == 1


def test_overlapping_different_overlay_fails_and_suspension_removes_target():
    different = OverlayPayload(overlay_id="exception-b", overlay_type=OverlayType.EXCEPTION, policy_corpus_hash=overlay().policy_corpus_hash, scope=OverlayScope(("ALICE",), ("READ",)), valid_from_utc=base_time(), valid_until_utc=base_time() + timedelta(hours=2), target_rule_ids=("rule-a",), exception_code="EXCEPTION")
    with pytest.raises(OverlayConflictError):
        effective_overlays(state_with(overlay(), different), context(), base_time())
    source = overlay()
    suspension = overlay("suspension-a", kind=OverlayType.SUSPENSION, target=(), target_overlay=source.overlay_id)
    assert effective_overlays(state_with(source, suspension), context(), base_time()) == ()
