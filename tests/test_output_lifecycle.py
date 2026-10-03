"""Focused F2-D core contract tests (no persistence or transport adapter)."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from policy.governance.governed_renderer import (
    GovernedDomainRegistry, GovernedRenderer, RenderContext, WebChatGovernanceAdapter,
)
from policy.governance.response import (
    ClaimRecord, ContentBlock, ContentBlockKind, ContractState,
    EpistemicStatus, SourceClass, ReferenceType,
)
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
from test_governed_renderer import NOW, receipt, ref, scope, sealed_current


def unit_at(now=NOW):
    envelope, context = sealed_current()
    rendered = GovernedRenderer().render_text(envelope, context)
    unit = mint_governed_transport_unit(
        envelope, rendered, context, transport_kind="web-chat-http",
        idempotency_key="request-a:response-1:attempt-1", now=now,
    )
    return unit, rendered, context


def test_mint_binds_exact_effective_f2c_projection_and_scope():
    unit, rendered, context = unit_at()
    assert unit.response_id == rendered.response_id
    assert unit.original_envelope_digest == rendered.source_envelope_digest
    assert unit.durable_projection()["effective_contract_state"] == "VALID"
    assert unit.durable_projection()["tenant_id"] == "tenant-a"
    assert unit.trace_id == "trace-a"
    assert unit.durable_projection()["scope_digest"] == unit.scope_digest
    assert unit.durable_projection()["schema_version"] == "f2-c.1"
    assert unit.durable_projection()["governance_reference_ids"] == ["receipt-1"]
    assert unit.contains_current_claim
    assert unit.current_not_after == context.receipts["receipt-1"].not_after
    assert unit.durable_projection()["current_not_after"] == unit.current_not_after.isoformat()
    assert revalidate_for_transport(unit, NOW) == rendered


def test_mint_rejects_foreign_or_mutated_projection():
    envelope, context = sealed_current()
    rendered = GovernedRenderer().render_text(envelope, context)
    for changed in (
        replace(rendered, text="different text"),
        replace(rendered, chunks=("UNBOUND BYTES",)),
        replace(rendered, contract_state=ContractState.UNAVAILABLE),
        replace(rendered, response_id="foreign-response"),
        replace(rendered, source_envelope_digest="sha256:" + "0" * 64),
    ):
        with pytest.raises(OutputLifecycleError):
            mint_governed_transport_unit(envelope, changed, context, transport_kind="web-chat-http", idempotency_key="key", now=NOW)


def test_transport_digest_binds_exact_chunk_projection():
    envelope, context = sealed_current()
    rendered = GovernedRenderer().render_text(envelope, context)
    forged = replace(rendered, chunks=("UNBOUND BYTES",))
    with pytest.raises(OutputLifecycleError):
        mint_governed_transport_unit(envelope, forged, context,
            transport_kind="web-chat-http", idempotency_key="forged-chunks", now=NOW)

    narrative = WebChatGovernanceAdapter(scope(), receipt()).seal_non_governed_candidate(
        response_id="chunked", candidate_text="canonical output")
    context = RenderContext(None, now=lambda: NOW)
    chunked = GovernedRenderer().render_text(narrative, context, chunk_size=4)
    unit = mint_governed_transport_unit(narrative, chunked, context,
        transport_kind="web-chat-http", idempotency_key="chunked", now=NOW)
    assert revalidate_for_transport(unit, NOW).chunks == chunked.chunks
    object.__setattr__(unit.rendered, "chunks", ("UNBOUND BYTES",))
    with pytest.raises(OutputLifecycleError):
        revalidate_for_transport(unit, NOW)


def test_current_claim_is_revalidated_at_preparation_and_transport():
    unit, _, _ = unit_at()
    with pytest.raises(OutputLifecycleError):
        revalidate_for_transport(unit, NOW + timedelta(seconds=61))
    envelope, context = sealed_current()
    rendered = GovernedRenderer().render_text(envelope, context)
    with pytest.raises(OutputLifecycleError):
        mint_governed_transport_unit(
            envelope, rendered, context, transport_kind="web-chat-http", idempotency_key="late", now=NOW + timedelta(seconds=61)
        )


def test_transport_revalidation_reuses_minted_projection_without_rerendering(monkeypatch):
    unit, rendered, _ = unit_at()

    def unexpected_render(*_args, **_kwargs):
        pytest.fail("transport revalidation must reuse the minted projection")

    monkeypatch.setattr(GovernedRenderer, "render_text", unexpected_render)
    assert revalidate_for_transport(unit, NOW) == rendered


def test_transport_reuse_still_revalidates_current_receipts(monkeypatch):
    unit, rendered, context = unit_at()
    original = type(context.registry).verify_receipt_for_claim
    verifications = []

    def counted_verification(registry, *args, **kwargs):
        verifications.append(args[0])
        return original(registry, *args, **kwargs)

    def unexpected_render(*_args, **_kwargs):
        pytest.fail("transport revalidation must reuse the minted projection")

    monkeypatch.setattr(type(context.registry), "verify_receipt_for_claim", counted_verification)
    monkeypatch.setattr(GovernedRenderer, "render_text", unexpected_render)

    assert revalidate_for_transport(unit, NOW) == rendered
    assert verifications == [context.receipts["receipt-1"]]
    with pytest.raises(OutputLifecycleError):
        revalidate_for_transport(unit, NOW + timedelta(seconds=61))
    assert verifications == [context.receipts["receipt-1"], context.receipts["receipt-1"]]


def test_transport_revalidation_preserves_attributed_user_quote_path():
    from policy.governance import response

    response_scope = scope()
    assertion_ref = ref("user-assertion", ReferenceType.USER_ASSERTION,
        response_scope, asserter="Fernando")
    claim = ClaimRecord("user-claim", "USER_REPORT", {"text": "A quoted statement."},
        response_scope, SourceClass.USER_INPUT, EpistemicStatus.USER_ASSERTED,
        basis_refs=(assertion_ref.ref_id,))
    quote = ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE, "A quoted statement.",
        claim_refs=(claim.claim_id,), attribution_ref=assertion_ref.ref_id,
        speaker="Fernando")
    candidate = response.GovernedResponseCandidate("f2-c.1", "quoted-response",
        response_scope.request_id, response_scope.trace_id, response_scope, "web-chat",
        (), (quote,), (claim,), (assertion_ref,))
    envelope = response._seal_candidate_for_server(candidate,
        contract_state=ContractState.VALID, governance_receipt=receipt())
    context = RenderContext(None, {}, {}, {}, domain_registry=GovernedDomainRegistry(),
        reference_validator=lambda _reference, _scope: True,
        now=lambda: NOW, user_assertion_content={assertion_ref.ref_id: "A quoted statement."})
    rendered = GovernedRenderer().render_text(envelope, context)
    assert rendered.text == "Fernando says: A quoted statement."
    unit = mint_governed_transport_unit(envelope, rendered, context,
        transport_kind="web-chat-http", idempotency_key="quoted-response", now=NOW)

    assert revalidate_for_transport(unit, NOW) == rendered


def test_transport_revalidation_rejects_revoked_user_quote_access():
    from policy.governance import response

    response_scope = scope()
    assertion_ref = ref("revocable-user-assertion", ReferenceType.USER_ASSERTION,
        response_scope, asserter="Fernando")
    claim = ClaimRecord("revocable-user-claim", "USER_REPORT", {"text": "A quoted statement."},
        response_scope, SourceClass.USER_INPUT, EpistemicStatus.USER_ASSERTED,
        basis_refs=(assertion_ref.ref_id,))
    quote = ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE, "A quoted statement.",
        claim_refs=(claim.claim_id,), attribution_ref=assertion_ref.ref_id, speaker="Fernando")
    candidate = response.GovernedResponseCandidate("f2-c.1", "revocable-response",
        response_scope.request_id, response_scope.trace_id, response_scope, "web-chat",
        (), (quote,), (claim,), (assertion_ref,))
    envelope = response._seal_candidate_for_server(candidate,
        contract_state=ContractState.VALID, governance_receipt=receipt())
    access = {"allowed": True}
    context = RenderContext(None, {}, {}, {}, domain_registry=GovernedDomainRegistry(),
        reference_validator=lambda _reference, _scope: access["allowed"],
        now=lambda: NOW,
        user_assertion_content={assertion_ref.ref_id: "A quoted statement."})
    rendered = GovernedRenderer().render_text(envelope, context)
    unit = mint_governed_transport_unit(envelope, rendered, context,
        transport_kind="web-chat-http", idempotency_key="revocable-response", now=NOW)
    access["allowed"] = False
    with pytest.raises(OutputLifecycleError):
        revalidate_for_transport(unit, NOW)


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
