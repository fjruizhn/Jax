from datetime import timedelta
import pytest

from policy.authority_ledger.errors import (AuthorityStateError, OverlayApplicabilityIndeterminateError,
                                            OverlayConflictError)
from policy.authority_ledger.models import AuthorityEvent, AuthorityEventIntent, AuthorityEventType, OverlayType
from policy.authority_ledger.replay import (ReconstructedAuthorityState, effective_overlays,
                                            event_hash, event_unsigned_bytes)
from policy.authority_ledger.replay import replay_authority_history, _effective_overlays_for_history
from tests.policy.test_authority_ledger_events import verify_authority_ledger
from tests.policy.test_authority_ledger_events import append_authority_event
from policy.authority_ledger.signatures import sign
from policy.authority_resolution.models import ConditionResult, EvaluationContext
from tests.policy.test_authority_ledger_events import base_time, overlay, setup_ledger
from policy.authority_ledger.models import OverlayPayload, OverlayScope


def context(*conditions):
    return EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "ALICE", "READ", tuple(conditions))


def state_with(*items):
    store, root, key = setup_ledger()
    from tests.policy.test_authority_ledger_events import append_ratification
    rat = append_ratification(store, root, key)
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=rat.event_id))
    for item in items:
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=item))
    return verify_authority_ledger(store.get_genesis(), store.events(), root)


def test_reconstructed_state_preserves_the_prior_positional_constructor_layout():
    """Quarantine is additive and cannot rebind legacy private replay fields."""
    checkpoint = state_with().checkpoint
    grants, latest, revoked, seal = {"grant": "event"}, {"rule": "event"}, frozenset({"grant"}), object()
    state = ReconstructedAuthorityState({}, frozenset(), None, {}, frozenset(), checkpoint,
                                        grants, latest, revoked, seal)
    assert state._rule_ratification_grants == grants
    assert state._latest_rule_ratifications == latest
    assert state._revoked_rule_ratifications == revoked
    assert state._verified_seal is seal
    assert state.quarantined_overlays == frozenset()


def _legacy_signed_events(key, intents):
    """Construct historical signed rows without the current append boundary."""
    events = []
    previous = None
    for sequence, (event_id, intent) in enumerate(intents, 1):
        provisional = AuthorityEvent(event_id, sequence, previous, intent, base_time(), "", "sha256:" + "0" * 64)
        signed = AuthorityEvent(event_id, sequence, previous, intent, base_time(), sign(key, event_unsigned_bytes(provisional)), "sha256:" + "0" * 64)
        complete = AuthorityEvent(event_id, sequence, previous, intent, base_time(), signed.signature, event_hash(signed))
        events.append(complete)
        previous = complete.event_hash
    return tuple(events)


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


# ------------------- overlay exige ratificación vigente (auditor de #377)

def test_overlay_a_corpus_no_ratificado_se_rechaza_antes_de_escribir():
    """Un overlay cuyo corpus objetivo jamás fue ratificado no entra al ledger:
    el replay previo de #377 lo niega antes de firmar hacia el storage."""
    from tests.policy.test_authority_ledger_events import ratification_intent
    store, root, key = setup_ledger()
    huerfano = OverlayPayload(
        "huerfano", OverlayType.EXCEPTION, "sha256:" + "b" * 64,
        OverlayScope(("ALICE",), ("READ",)), base_time(), base_time() + timedelta(days=1),
        target_rule_ids=("rule-a",), exception_code="HUERFANO",
    )
    with pytest.raises(AuthorityStateError, match="overlay exige ratificación"):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=huerfano))
    assert store.events() == ()


def test_overlay_a_corpus_con_ratificacion_revocada_tampoco_pasa():
    from tests.policy.test_authority_ledger_events import append_ratification
    store, root, key = setup_ledger()
    rat = append_ratification(store, root, key)
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.RATIFICATION_REVOKED, "human:fernando", ratification_event_id=rat.event_id))
    tras_revocacion = OverlayPayload(
        "tras-revocacion", OverlayType.EXCEPTION, rat.intent.policy_corpus_hash,
        OverlayScope(("ALICE",), ("READ",)), base_time(), base_time() + timedelta(days=1),
        target_rule_ids=("rule-a",), exception_code="TARDIO",
    )
    with pytest.raises(AuthorityStateError, match="overlay exige ratificación"):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=tras_revocacion))
    assert len(store.events()) == 2   # ratificación + revocación: el overlay no dejó rastro


def test_legacy_overlay_sin_ratificacion_queda_en_cuarentena_permanente_y_se_puede_revocar():
    """Old signed history replays, but later ratification never activates its overlay."""
    from tests.policy.test_authority_ledger_events import ratification_intent
    store, root, key = setup_ledger()
    ratification = ratification_intent()
    legacy = OverlayPayload(
        "legacy-huerfano", OverlayType.EXCEPTION, ratification.policy_corpus_hash,
        OverlayScope(("ALICE",), ("READ",)), base_time(), base_time() + timedelta(days=1),
        target_rule_ids=("rule-a",), exception_code="LEGACY",
    )
    overlay_id = "018cc251-f400-7000-8000-000000000001"
    ratification_id = "018cc251-f400-7000-8000-000000000002"
    activation_id = "018cc251-f400-7000-8000-000000000003"
    revoke_id = "018cc251-f400-7000-8000-000000000004"
    historical = _legacy_signed_events(key, (
        (overlay_id, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=legacy)),
        (ratification_id, ratification),
        (activation_id, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=ratification_id)),
    ))

    state = replay_authority_history(store.get_genesis(), historical, root)
    assert state.active_policy_corpus_hash == ratification.policy_corpus_hash
    assert state.overlays[legacy.overlay_id] == legacy
    assert state.quarantined_overlays == frozenset({legacy.overlay_id})
    assert _effective_overlays_for_history(state, context(), base_time()) == ()

    revoked = _legacy_signed_events(key, (
        (overlay_id, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=legacy)),
        (ratification_id, ratification),
        (activation_id, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=ratification_id)),
        (revoke_id, AuthorityEventIntent(AuthorityEventType.OVERLAY_REVOKED, "human:fernando", overlay_id=legacy.overlay_id)),
    ))
    revoked_state = replay_authority_history(store.get_genesis(), revoked, root)
    assert legacy.overlay_id in revoked_state.revoked_overlays
    assert legacy.overlay_id in revoked_state.quarantined_overlays
