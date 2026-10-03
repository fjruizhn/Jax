"""F2-E structured governed-output contract tests."""
from dataclasses import replace
from datetime import timedelta

import pytest

import policy.governance.response as response
from policy.governance.governed_renderer import RenderContext
from policy.governance.resolution import (
    AdapterKind, ConflictPolicy, PredicateAuthorityBinding, ReceiptAuthenticator,
    RegistryEntry, ResolutionObservation, ResolutionStatus, ScopeRule,
    SourceScopeClass, TrustedAdapterRegistration, _build_approved_registry_for_server,
    _runtime_status_evidence_from_server,
)
from policy.governance.response import (
    ClaimDisposition, ClaimRecord, ContentBlock, ContentBlockKind,
    ContractState, EpistemicStatus, GovernedResponseCandidate, ReferenceRef,
    ReferenceType, SourceClass, TemplateContract, TemporalClass,
)
from policy.governance.structured_output import (
    GovernedStructuredRenderer, OriginBinding, OutputOrigin,
    StructJSONNumberBinding, StructJSONNumberPattern, StructuredClaimSlot, StructuredLayout, StructuredLayoutRegistry,
    SlotProjectionMode, StructuredOutputError, STRUCTURED_OUTPUT_SCHEMA_VERSION, STRUCTURED_RENDERER_API_VERSION,
    expand_structured_number_patterns, pack_structured_float64_carriers, validate_structured_compatibility,
)
from policy.governance.runtime_status import RUNTIME_STATUS_API_VERSION
from policy.governance.output_lifecycle import (
    OutputLifecycleError, StructuredTransportMetadata,
    mint_structured_governed_transport_unit, revalidate_structured_for_transport,
)
from policy.governance.external_output import (
    CHANNELS, ExternalOutputChannelId, GovernedExternalOutputAdapter,
    GovernedExternalOutputSubmission,
)
from test_governed_renderer import NOW, receipt, ref, scope, trusted_receipt_reference, validates_trusted_receipt_reference


def _pipeline_envelope(*, payload=None, status="running", stale=False):
    s = scope()
    rule = ScopeRule(s.environment, s.tenant_id, s.project_id, s.subject_id,
        s.actor_id, s.audience, s.component_id)
    binding = PredicateAuthorityBinding(
        "PIPELINE_STATUS", RUNTIME_STATUS_API_VERSION, "jacobs:canonical-store",
        "authority:jacobs", s.environment, rule, rule, 60,
        ConflictPolicy.SINGLE_SOURCE_REQUIRED,
        "policy.governance.runtime_status:JacobsPipelineStatusResolver",
        "f2-e.runtime-status-resolver." + RUNTIME_STATUS_API_VERSION.rsplit(".", 1)[1], "sha256:" + "a" * 64,
        RUNTIME_STATUS_API_VERSION,
        source_scope_class=SourceScopeClass.EXACT_RESPONSE_SCOPE,
    )
    entry = RegistryEntry(binding, TrustedAdapterRegistration(
        AdapterKind.JACOBS_PIPELINE_STATUS,
        binding.resolver_implementation_identity, binding.resolver_version,
        binding.designated_source_identity, binding.source_configuration_digest,
    ), ("pipeline_id", "status"), "pipeline@1:en")
    registry = _build_approved_registry_for_server((entry,),
        authenticator=ReceiptAuthenticator.for_testing(b"s" * 32))
    observation = ResolutionObservation(ResolutionStatus.RESOLVED, NOW,
        "jacobs:pipeline:pipeline-1", {"pipeline_id": "pipeline-1", "status": status})
    evidence = _runtime_status_evidence_from_server(AdapterKind.JACOBS_PIPELINE_STATUS,
        observation, s, binding.source_configuration_digest)
    resolved = registry.resolve("PIPELINE_STATUS", {"pipeline_id": "pipeline-1", "status": status},
        s, validation_time=NOW, runtime_status_evidence=evidence)
    receipt_ref = replace(ref("receipt-1", ReferenceType.RESOLUTION_RECEIPT, s),
        revision_or_digest=resolved.receipt_id)
    claim = ClaimRecord("pipeline-claim", "PIPELINE_STATUS",
        {"pipeline_id": "pipeline-1", "status": status}, s,
        SourceClass.CURRENT_SOURCE, EpistemicStatus.CURRENT_OBSERVATION,
        resolution_receipt_ref="receipt-1", disposition=ClaimDisposition.ASSERTABLE,
        template_contract=TemplateContract("pipeline", "1", "en"))
    root = payload if payload is not None else {
        "name": "important pipeline", "steps": [
            {"pipeline_id": "pipeline-1", "status": status, "output_ref": "tool:42"},
        ],
    }
    candidate = GovernedResponseCandidate("f2-c.1", "structured-response", s.request_id,
        s.trace_id, s, "web-chat", (), (
            ContentBlock(ContentBlockKind.TOOL_DATA, root),
            ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK, claim_refs=("pipeline-claim",)),
        ), (claim,), (receipt_ref,))
    envelope = response._seal_candidate_for_server(candidate,
        contract_state=ContractState.VALID, governance_receipt=receipt())
    trusted = {"receipt-1": trusted_receipt_reference(receipt_ref)}
    context = RenderContext(registry, {"receipt-1": resolved},
        {("pipeline", "1", "en"): "Pipeline {pipeline_id} is {status}."}, {},
        reference_validator=lambda item, item_scope: validates_trusted_receipt_reference(item, item_scope, trusted),
        now=lambda: NOW + timedelta(seconds=61) if stale else NOW,
        receipt_reference_resolver=lambda item, item_scope: trusted.get(item.ref_id)
        if validates_trusted_receipt_reference(item, item_scope, trusted) else None)
    return envelope, context


def _layout(*, status_pointer="/steps/0/status", origins=None, presentation_map=False):
    return StructuredLayout("pipeline-view", "1", "PLATFORM_HTTP_JSON_V1", 0, (
        StructuredClaimSlot("PIPELINE_STATUS", "/steps/0", "pipeline-claim", {
            "pipeline_id": "/steps/0/pipeline_id", "status": status_pointer,
        }, projection_mode=SlotProjectionMode.PRESENTATION_MAP if presentation_map else SlotProjectionMode.EXACT,
        presentation_argument="status" if presentation_map else None,
        presentation_map_id="pipeline-ui" if presentation_map else None),
    ), origins or (
        OriginBinding("/name", OutputOrigin.TOOL),
        OriginBinding("/steps/0/pipeline_id", OutputOrigin.SYSTEM, "receipt-1"),
        OriginBinding("/steps/0/status", OutputOrigin.SYSTEM, "receipt-1"),
        OriginBinding("/steps/0/output_ref", OutputOrigin.TOOL),
    ), presentation_maps={"pipeline-ui": {"interrupted": "waiting_gate", "aborted": "failed", "expired": "failed"}} if presentation_map else {})


def _render(**kwargs):
    envelope, context = _pipeline_envelope(**kwargs)
    layout = _layout()
    return GovernedStructuredRenderer().render(envelope, context, layout,
        StructuredLayoutRegistry({layout.layout_id: layout}))


def test_current_pipeline_slot_preserves_native_schema_and_canonical_bytes():
    rendered = _render()
    assert rendered.value["steps"][0]["pipeline_id"] == "pipeline-1"
    assert rendered.value["steps"][0]["status"] == "running"
    assert rendered.value["steps"][0]["output_ref"] == "tool:42"
    assert rendered.canonical_bytes == b'{"name":"important pipeline","steps":[{"output_ref":"tool:42","pipeline_id":"pipeline-1","status":"running"}]}'
    assert rendered.claim_ids == ("pipeline-claim",)
    assert rendered.renderer_api_version == STRUCTURED_RENDERER_API_VERSION
    assert rendered.structured_schema_version == STRUCTURED_OUTPUT_SCHEMA_VERSION


@pytest.mark.parametrize("payload", (
    {"name": "x", "steps": [{"pipeline_id": "pipeline-1", "status": "failed", "output_ref": "tool:42"}]},
    {"name": "x", "steps": [{"pipeline_id": "pipeline-1", "status": "running", "output_ref": "tool:42"}], "leak": {"pipeline_id": "other", "status": "completed"}},
    {"name": "x", "steps": [{"pipeline_id": "pipeline-1", "status": "running", "output_ref": '{"pipeline_id":"x","status":"completed"}'}]},
))
def test_raw_or_unbound_runtime_status_never_escapes_structured_projection(payload):
    rendered = _render(payload=payload)
    assert rendered.contract_state is ContractState.UNAVAILABLE
    assert rendered.value == {"error": "governed_output_unavailable"}
    assert b"important pipeline" not in rendered.canonical_bytes


def test_stale_current_receipt_fails_closed_without_candidate_leakage():
    rendered = _render(stale=True)
    assert rendered.contract_state is ContractState.UNAVAILABLE
    assert rendered.value == {"error": "governed_output_unavailable"}


def test_layout_registry_origins_and_versions_are_exact():
    envelope, context = _pipeline_envelope()
    layout = _layout()
    renderer = GovernedStructuredRenderer()
    assert renderer.render(envelope, context, layout, StructuredLayoutRegistry({layout.layout_id: layout})).contract_state is ContractState.VALID
    other = replace(layout, layout_version="2")
    assert renderer.render(envelope, context, other, StructuredLayoutRegistry({layout.layout_id: layout})).contract_state is ContractState.UNAVAILABLE
    patterned = replace(layout, origins=(OriginBinding("", OutputOrigin.TOOL), OriginBinding("/name", OutputOrigin.SYSTEM)))
    assert renderer.render(envelope, context, patterned,
        StructuredLayoutRegistry({patterned.layout_id: patterned})).contract_state is ContractState.VALID


def test_post_render_mutation_changes_exact_canonical_bytes_and_digest():
    rendered = _render()
    changed = rendered.canonical_bytes.replace(b"running", b"failed")
    assert changed != rendered.canonical_bytes
    assert rendered.output_digest != "sha256:" + __import__("hashlib").sha256(changed).hexdigest()


def test_f2d_binds_exact_structured_body_and_server_transport_metadata():
    envelope, context = _pipeline_envelope()
    layout = _layout()
    layouts = StructuredLayoutRegistry({layout.layout_id: layout})
    rendered = GovernedStructuredRenderer().render(envelope, context, layout, layouts)
    metadata = StructuredTransportMetadata("PLATFORM_HTTP_JSON_V1", "application/json", "success", http_status=200)
    unit = mint_structured_governed_transport_unit(envelope, rendered, context, layout, layouts,
        metadata=metadata, idempotency_key="request-a:structured-response:1", now=NOW)
    assert revalidate_structured_for_transport(unit, NOW).canonical_bytes == rendered.canonical_bytes
    assert unit.durable_projection()["transport_metadata_digest"] == metadata.digest
    with pytest.raises(OutputLifecycleError):
        mint_structured_governed_transport_unit(envelope, replace(rendered, canonical_bytes=b"{}"), context,
            layout, layouts, metadata=metadata, idempotency_key="changed", now=NOW)


def test_server_owned_presentation_map_preserves_canonical_pipeline_claim():
    payload = {"name": "important pipeline", "steps": [
        {"pipeline_id": "pipeline-1", "status": "waiting_gate", "output_ref": "tool:42"},
    ]}
    envelope, context = _pipeline_envelope(payload=payload, status="interrupted")
    layout = _layout(presentation_map=True)
    rendered = GovernedStructuredRenderer().render(envelope, context, layout,
        StructuredLayoutRegistry({layout.layout_id: layout}))
    assert rendered.contract_state is ContractState.VALID
    assert rendered.value["steps"][0]["status"] == "waiting_gate"
    assert envelope.claims[0].typed_arguments["status"] == "interrupted"
    changed_map = replace(layout, presentation_maps={"pipeline-ui": {"interrupted": "wrong"}})
    assert GovernedStructuredRenderer().render(envelope, context, changed_map,
        StructuredLayoutRegistry({changed_map.layout_id: changed_map})).output_digest != rendered.output_digest


def test_closed_channel_adapter_owns_structured_transport_selection():
    envelope, context = _pipeline_envelope()
    layout = _layout()
    adapter = GovernedExternalOutputAdapter(context, StructuredLayoutRegistry({layout.layout_id: layout}),
        CHANNELS[ExternalOutputChannelId.PLATFORM_HTTP_JSON_V1])
    submission = GovernedExternalOutputSubmission(envelope, envelope.response_scope,
        OutputOrigin.TOOL, ExternalOutputChannelId.PLATFORM_HTTP_JSON_V1)
    unit = adapter.prepare_structured(submission, layout=layout,
        metadata=StructuredTransportMetadata("PLATFORM_HTTP_JSON_V1", "application/json", "success", http_status=200),
        idempotency_key="route:structured-response", now=NOW)
    assert unit.durable_projection()["channel_id"] == "PLATFORM_HTTP_JSON_V1"
    with pytest.raises(Exception):
        GovernedExternalOutputSubmission(envelope, envelope.response_scope, OutputOrigin.TOOL, "caller-made-channel")


def test_structured_compatibility_accepts_only_its_exact_tuple():
    validate_structured_compatibility(renderer_api_version="f2-c.structured-renderer.2",
        domain_spec_version="f2-c.domain.5", schema_versions=frozenset({"f2-c.structured-output.2"}))
    for renderer, domain, schemas in (
        ("f2-c.structured-renderer.0", "f2-c.domain.5", frozenset({"f2-c.structured-output.1"})),
        ("f2-c.structured-renderer.1", "f2-c.domain.5", frozenset({"f2-c.structured-output.1"})),
        ("f2-c.structured-renderer.2", "f2-c.domain.6", frozenset({"f2-c.structured-output.2"})),
        ("f2-c.structured-renderer.2", "f2-c.domain.5", frozenset({"f2-c.structured-output.1", "f2-c.structured-output.2"})),
    ):
        with pytest.raises(StructuredOutputError):
            validate_structured_compatibility(renderer_api_version=renderer, domain_spec_version=domain, schema_versions=schemas)


def test_variable_array_origin_patterns_bind_exact_manifest_and_f2d_revalidates():
    envelope, context = _pipeline_envelope(payload={"steps": [
        {"pipeline_id": "pipeline-1", "status": "running", "output_ref": "tool:1", "prompt": "user one"},
        {"pipeline_id": "tool-only", "status": "queued", "output_ref": "tool:2", "prompt": "user two"},
    ]})
    layout = StructuredLayout("array-patterns", "1", "PLATFORM_HTTP_JSON_V1", 0,
        _layout().slots, (
            OriginBinding("/steps/*/pipeline_id", OutputOrigin.SYSTEM),
            OriginBinding("/steps/*/status", OutputOrigin.SYSTEM),
            OriginBinding("/steps/*/output_ref", OutputOrigin.TOOL),
            OriginBinding("/steps/*/prompt", OutputOrigin.USER),
        ))
    rendered = GovernedStructuredRenderer().render(envelope, context, layout,
        StructuredLayoutRegistry({layout.layout_id: layout}))
    assert rendered.contract_state is ContractState.VALID
    assert rendered.origin_manifest_digest


@pytest.mark.parametrize("encoding,frame,event,expected", (
    ("http-json", None, None, b'{"name":"important pipeline","steps":[{"output_ref":"tool:42","pipeline_id":"pipeline-1","status":"running"}]}'),
    ("websocket-text", "text", None, b'{"name":"important pipeline","steps":[{"output_ref":"tool:42","pipeline_id":"pipeline-1","status":"running"}]}'),
    ("sse-data", None, "status", b'event:status\ndata:{"name":"important pipeline","steps":[{"output_ref":"tool:42","pipeline_id":"pipeline-1","status":"running"}]}\n\n'),
))
def test_closed_wire_encodings_bind_exact_bytes(encoding, frame, event, expected):
    envelope, context = _pipeline_envelope(); layout = _layout(); layouts = StructuredLayoutRegistry({layout.layout_id: layout})
    rendered = GovernedStructuredRenderer().render(envelope, context, layout, layouts)
    metadata = StructuredTransportMetadata("PLATFORM_HTTP_JSON_V1", "application/json", "success", frame_type=frame, event_type=event, wire_encoding=encoding)
    unit = mint_structured_governed_transport_unit(envelope, rendered, context, layout, layouts, metadata=metadata, idempotency_key=encoding, now=NOW)
    assert unit.wire_bytes == expected
    assert revalidate_structured_for_transport(unit, NOW).canonical_bytes == rendered.canonical_bytes


def test_unknown_or_unsafe_wire_encoding_fails_closed():
    with pytest.raises(OutputLifecycleError):
        StructuredTransportMetadata("PLATFORM_HTTP_JSON_V1", "application/json", "success", wire_encoding="latest")
    with pytest.raises(OutputLifecycleError):
        StructuredTransportMetadata("PLATFORM_SSE_JSON_V1", "text/event-stream", "success", event_type="bad\nname", wire_encoding="sse-data")


def test_bound_float64_carriers_materialize_native_json_numbers_and_nulls():
    bindings = tuple(StructJSONNumberBinding("/" + field, nullable=field != "created_at")
        for field in ("created_at", "started_at", "finished_at", "status_updated_at"))
    payload = pack_structured_float64_carriers({
        "created_at": 1.5, "started_at": None, "finished_at": 3.25,
        "status_updated_at": None, "steps": [{"pipeline_id": "pipeline-1", "status": "running", "output_ref": "tool:42"}],
    }, bindings)
    envelope, context = _pipeline_envelope(payload=payload)
    layout = StructuredLayout("numeric-pipeline", "1", "PLATFORM_HTTP_JSON_V1", 0, _layout().slots, (
        *(OriginBinding("/" + field, OutputOrigin.SYSTEM) for field in ("created_at", "started_at", "finished_at", "status_updated_at")),
        OriginBinding("/steps/0/pipeline_id", OutputOrigin.SYSTEM), OriginBinding("/steps/0/status", OutputOrigin.SYSTEM),
        OriginBinding("/steps/0/output_ref", OutputOrigin.TOOL),
    ), number_bindings=bindings)
    layouts = StructuredLayoutRegistry({layout.layout_id: layout})
    rendered = GovernedStructuredRenderer().render(envelope, context, layout, layouts)
    assert rendered.value["created_at"] == 1.5 and isinstance(rendered.value["created_at"], float)
    assert rendered.value["started_at"] is None and rendered.value["finished_at"] == 3.25
    assert b'"created_at":1.5' in rendered.canonical_bytes
    unit = mint_structured_governed_transport_unit(envelope, rendered, context, layout, layouts,
        metadata=StructuredTransportMetadata("PLATFORM_HTTP_JSON_V1", "application/json", "success", http_status=200),
        idempotency_key="numeric-carrier", now=NOW)
    assert revalidate_structured_for_transport(unit, NOW).canonical_bytes == rendered.canonical_bytes


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True))
def test_numeric_carrier_rejects_nonfinite_or_nonnumeric_bound_value(value):
    binding = (StructJSONNumberBinding("/created_at"),)
    with pytest.raises(StructuredOutputError):
        pack_structured_float64_carriers({"created_at": value}, binding)


def test_numeric_carrier_rejects_injection_misplacement_claim_overlap_and_old_tuple():
    with pytest.raises(StructuredOutputError):
        pack_structured_float64_carriers({"created_at": {"f2-e.float64": "0x1.0000000000000p+0"}},
            (StructJSONNumberBinding("/created_at"),))
    with pytest.raises(StructuredOutputError):
        StructuredLayout("overlap", "1", "PLATFORM_HTTP_JSON_V1", 0, _layout().slots, _layout().origins,
            number_bindings=(StructJSONNumberBinding("/steps/0/status"),))
    with pytest.raises(StructuredOutputError):
        validate_structured_compatibility(renderer_api_version="f2-c.structured-renderer.1",
            domain_spec_version="f2-c.domain.5", schema_versions=frozenset({"f2-c.structured-output.1"}))
    with pytest.raises(response.GovernanceContractError):
        ContentBlock(ContentBlockKind.TOOL_DATA, {"created_at": 1.5})


def test_renderer_fails_closed_on_misplaced_or_noncanonical_carrier():
    payload = {"created_at": {"f2-e.float64": "0x1.0p+0"},
        "steps": [{"pipeline_id": "pipeline-1", "status": "running", "output_ref": "tool:42"}]}
    envelope, context = _pipeline_envelope(payload=payload)
    layout = StructuredLayout("bad-carrier", "1", "PLATFORM_HTTP_JSON_V1", 0, _layout().slots, (
        OriginBinding("/created_at", OutputOrigin.SYSTEM), OriginBinding("/steps/0/pipeline_id", OutputOrigin.SYSTEM),
        OriginBinding("/steps/0/status", OutputOrigin.SYSTEM), OriginBinding("/steps/0/output_ref", OutputOrigin.TOOL),
    ), number_bindings=(StructJSONNumberBinding("/created_at"),))
    rendered = GovernedStructuredRenderer().render(envelope, context, layout,
        StructuredLayoutRegistry({layout.layout_id: layout}))
    assert rendered.contract_state is ContractState.UNAVAILABLE


@pytest.mark.parametrize("pattern", ("/*/started_at", "/steps/**/started_at", "/steps/foo*/started_at", "/steps/*/*/started_at"))
def test_numeric_patterns_reject_ambiguous_wildcards(pattern):
    with pytest.raises(StructuredOutputError):
        StructJSONNumberPattern(pattern)


def test_numeric_pattern_rejects_mapping_missing_leaf_and_expansion_budget():
    pattern = StructJSONNumberPattern("/steps/*/started_at", nullable=True)
    for value in ({"steps": {"one": {"started_at": 1.0}}}, {"steps": [{"other": 1.0}]}):
        with pytest.raises(StructuredOutputError):
            expand_structured_number_patterns(value, (pattern,))
    with pytest.raises(StructuredOutputError):
        expand_structured_number_patterns({"steps": [{"started_at": None} for _ in range(1025)]}, (pattern,))
