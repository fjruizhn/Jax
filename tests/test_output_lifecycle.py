"""Focused F2-D core contract tests (no persistence or transport adapter)."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from policy.governance.governed_renderer import GovernedRenderer, RenderContext, WebChatGovernanceAdapter
from policy.governance.response import ContractState
from policy.governance.output_lifecycle import (
    OUTPUT_LIFECYCLE_API_VERSION,
    GovernedTransportUnit,
    OutputLifecycleError,
    OutputLifecycleState,
    mint_governed_transport_unit,
    record_delivery_acknowledgement,
    revalidate_for_transport,
    validate_lifecycle_transition,
    validate_lifecycle_version,
)

# Reuse the F2-C test-only, server-owned registry/receipt composition.  These
# helpers construct a sealed current claim and a trusted reference resolver.
from test_governed_renderer import NOW, receipt, scope, sealed_current


def unit_at(now=NOW):
    envelope, context = sealed_current()
    rendered = GovernedRenderer().render_text(envelope, context)
    return mint_governed_transport_unit(
        envelope, rendered, context, transport_kind="web-chat-http",
        idempotency_key="request-a:response-1:attempt-1", now=now,
    ), rendered


def test_mint_binds_exact_effective_f2c_projection_and_scope():
    unit, rendered = unit_at()
    assert unit.response_id == rendered.response_id
    assert unit.original_envelope_digest == rendered.source_envelope_digest
    assert unit.durable_projection()["effective_contract_state"] == "VALID"
    assert unit.durable_projection()["tenant_id"] == "tenant-a"
    assert unit.trace_id == "trace-a"
    assert unit.durable_projection()["scope_digest"] == unit.scope_digest
    assert unit.durable_projection()["schema_version"] == "f2-c.1"
    assert unit.durable_projection()["governance_reference_ids"] == ["receipt-1"]
    assert unit.contains_current_claim
    assert revalidate_for_transport(unit, NOW) == rendered


def test_mint_rejects_foreign_or_mutated_projection():
    envelope, context = sealed_current()
    rendered = GovernedRenderer().render_text(envelope, context)
    for changed in (
        replace(rendered, text="different text"),
        replace(rendered, contract_state=ContractState.UNAVAILABLE),
        replace(rendered, response_id="foreign-response"),
        replace(rendered, source_envelope_digest="sha256:" + "0" * 64),
    ):
        with pytest.raises(OutputLifecycleError):
            mint_governed_transport_unit(envelope, changed, context, transport_kind="web-chat-http", idempotency_key="key", now=NOW)


def test_current_claim_is_revalidated_at_preparation_and_transport():
    unit, _ = unit_at()
    with pytest.raises(OutputLifecycleError):
        revalidate_for_transport(unit, NOW + timedelta(seconds=61))
    envelope, context = sealed_current()
    rendered = GovernedRenderer().render_text(envelope, context)
    with pytest.raises(OutputLifecycleError):
        mint_governed_transport_unit(
            envelope, rendered, context, transport_kind="web-chat-http", idempotency_key="late", now=NOW + timedelta(seconds=61)
        )


def test_non_current_narrative_can_still_use_the_lifecycle_contract():
    response_scope = scope()
    envelope = WebChatGovernanceAdapter(response_scope, receipt()).seal_non_governed_candidate(
        response_id="ordinary-narrative", candidate_text="An ordinary non-current sentence."
    )
    context = RenderContext(None, now=lambda: NOW)
    rendered = GovernedRenderer().render_text(envelope, context, chunk_size=4)
    unit = mint_governed_transport_unit(
        envelope, rendered, context, transport_kind="web-chat-http", idempotency_key="ordinary", now=NOW
    )
    assert revalidate_for_transport(unit, NOW).text == rendered.text


def test_transport_unit_cannot_be_self_attested_or_unknown_version_accepted():
    with pytest.raises(OutputLifecycleError):
        GovernedTransportUnit()
    with pytest.raises(OutputLifecycleError):
        validate_lifecycle_version("f2-d.lifecycle.unknown")
    validate_lifecycle_version(OUTPUT_LIFECYCLE_API_VERSION)


def test_every_lifecycle_transition_is_strict_and_monotonic():
    allowed = {
        (OutputLifecycleState.OUTPUT_PREPARED, OutputLifecycleState.TRANSPORT_COMMITTING),
        (OutputLifecycleState.OUTPUT_PREPARED, OutputLifecycleState.FAILED_BEFORE_COMMIT),
        (OutputLifecycleState.OUTPUT_PREPARED, OutputLifecycleState.CANCELLED_BEFORE_COMMIT),
        (OutputLifecycleState.TRANSPORT_COMMITTING, OutputLifecycleState.OUTPUT_COMMITTED_TO_TRANSPORT),
        (OutputLifecycleState.TRANSPORT_COMMITTING, OutputLifecycleState.TRANSPORT_OUTCOME_UNKNOWN),
    }
    for current in OutputLifecycleState:
        for target in OutputLifecycleState:
            if (current, target) in allowed:
                validate_lifecycle_transition(current, target)
            else:
                with pytest.raises(OutputLifecycleError):
                    validate_lifecycle_transition(current, target)
    validate_lifecycle_transition(OutputLifecycleState.OUTPUT_PREPARED, OutputLifecycleState.OUTPUT_PREPARED, same_identity=True)


def test_committing_can_cancel_only_with_server_owned_pre_send_proof():
    with pytest.raises(OutputLifecycleError):
        validate_lifecycle_transition(
            OutputLifecycleState.TRANSPORT_COMMITTING,
            OutputLifecycleState.CANCELLED_BEFORE_COMMIT,
        )
    validate_lifecycle_transition(
        OutputLifecycleState.TRANSPORT_COMMITTING,
        OutputLifecycleState.CANCELLED_BEFORE_COMMIT,
        before_send=True,
    )


def test_delivery_acknowledgement_is_defined_but_unreachable_without_protocol():
    with pytest.raises(OutputLifecycleError):
        record_delivery_acknowledgement("response-1")
