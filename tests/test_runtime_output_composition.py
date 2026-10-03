"""Server runtime-claim composition, without request/provider authority."""
import asyncio
from datetime import datetime, timezone

import pytest

from policy.governance.governed_renderer import GovernedRenderer
from policy.governance.response import GovernanceReceipt
from policy.governance.runtime_output_composition import (
    RuntimeClaimRequest, RuntimeOutputComposition, RuntimeOutputCompositionError,
    RuntimeSlotContract,
)
from policy.governance.runtime_status import (
    PlatformRuntimeStatusSnapshot, RUNTIME_STATUS_API_VERSION,
    build_owned_runtime_status_registry, platform_runtime_status_evidence,
)
from policy.governance.structured_output import (
    GovernedStructuredRenderer, OriginBinding, OutputOrigin, StructJSONNumberBinding, StructJSONNumberPattern,
    StructuredLayout, StructuredNumberBindingContract,
    StructuredLayoutRegistry,
)
from policy.governance.output_lifecycle import (
    StructuredTransportMetadata, mint_structured_governed_transport_unit,
    revalidate_structured_for_transport,
)
from policy.governance.resolution import ReceiptAuthenticator
from test_governed_renderer import NOW, scope


_PLATFORM_CONFIG = {
    "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState", "status_field": "status",
        "observed_at_field": "resolver_read_time", "allowed_statuses": ["idle", "thinking", "error", "offline"]},
    "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET", "path": "/internal/health",
        "service_authentication_identity": "plataforma", "service_authentication_header": "X-Jax-Credencial-Servicio",
        "timeout_seconds": 5, "poll_interval_seconds": 30, "success_status_code": 200},
}


def _receipt():
    return GovernanceReceipt("policy-v1", "vocab-v1", "sha256:" + "a" * 64, "f2-e", "runtime-output")


def _bridge(request, request_scope):
    snapshot = PlatformRuntimeStatusSnapshot(request.predicate, request.canonical_arguments, NOW,
        "platform-observation", _PLATFORM_CONFIG[request.predicate])
    return platform_runtime_status_evidence(snapshot, request.canonical_arguments, request_scope)


def _composition():
    return RuntimeOutputComposition(platform_source_configuration=_PLATFORM_CONFIG,
        governance_receipt=_receipt(), templates={("FACET_RUNTIME_STATUS", RUNTIME_STATUS_API_VERSION, "es"): "{name}: {status}"},
        platform_evidence_bridge=_bridge,
        slot_contracts={("FACET_RUNTIME_STATUS", "/status"): RuntimeSlotContract("FACET_RUNTIME_STATUS", "/status", {"name": "/facet", "status": "/status"})},
        clock=lambda: NOW,
        authenticator=ReceiptAuthenticator(b"r" * 32, key_id="runtime-output-test-process"))


def _fresh_composition():
    def bridge(request, request_scope):
        snapshot = PlatformRuntimeStatusSnapshot(request.predicate, request.canonical_arguments,
            datetime.now(timezone.utc), "platform-observation", _PLATFORM_CONFIG[request.predicate])
        return platform_runtime_status_evidence(snapshot, request.canonical_arguments, request_scope)
    contracts = {
        ("JOB_STATUS", "/status"): RuntimeSlotContract("JOB_STATUS", "/status", {"job_id": "/job_id", "status": "/status"}),
        ("PIPELINE_STATUS", "/status"): RuntimeSlotContract("PIPELINE_STATUS", "/status", {"pipeline_id": "/pipeline_id", "status": "/status"}),
        ("FACET_RUNTIME_STATUS", "/status"): RuntimeSlotContract("FACET_RUNTIME_STATUS", "/status", {"name": "/name", "status": "/status"}),
        ("ENGINE_STATUS", "/las_manos_alive"): RuntimeSlotContract("ENGINE_STATUS", "/las_manos_alive", {"status": "/las_manos_alive"}, {"name": "las_manos"}),
    }
    return RuntimeOutputComposition(platform_source_configuration=_PLATFORM_CONFIG,
        governance_receipt=_receipt(), templates={("FACET_RUNTIME_STATUS", RUNTIME_STATUS_API_VERSION, "es"): "{name}: {status}"},
        platform_evidence_bridge=bridge, slot_contracts=contracts, clock=lambda: datetime.now(timezone.utc),
        authenticator=ReceiptAuthenticator(b"f" * 32, key_id="runtime-output-fresh-process"))


def test_platform_runtime_composition_seals_exact_claim_and_render_context():
    current_scope = scope(component_id="facet-output")
    request = RuntimeClaimRequest("FACET_RUNTIME_STATUS", {"name": "hyde", "status": "idle"}, "/status")
    result = asyncio.run(_composition().compose(scope=current_scope, response_id="runtime-1",
        producer="facet-output", tool_data={"facet": "hyde", "status": "idle", "output_ref": "tool:42"},
        requests=(request,)))
    assert result.envelope.claims[0].predicate == "FACET_RUNTIME_STATUS"
    assert result.envelope.claims[0].typed_arguments == {"name": "hyde", "status": "idle"}
    assert result.slots[0].argument_pointers == {"name": "/facet", "status": "/status"}
    layout = StructuredLayout("facet-view", "1", "PLATFORM_HTTP_JSON_V1", 0, result.slots, (
        OriginBinding("/facet", OutputOrigin.TOOL), OriginBinding("/status", OutputOrigin.SYSTEM),
        OriginBinding("/output_ref", OutputOrigin.TOOL),
    ))
    rendered = GovernedStructuredRenderer().render(result.envelope, result.context, layout,
        StructuredLayoutRegistry({layout.layout_id: layout}))
    assert rendered.value["status"] == "idle" and rendered.claim_ids == ("runtime-claim:0",)


def test_composition_rejects_missing_bridge_unaccredited_predicate_and_test_authenticator():
    with pytest.raises(RuntimeOutputCompositionError):
        RuntimeOutputComposition(platform_source_configuration=_PLATFORM_CONFIG, governance_receipt=_receipt(), templates={},
            authenticator=ReceiptAuthenticator(b"r" * 32))
    with pytest.raises(RuntimeOutputCompositionError):
        RuntimeClaimRequest("CONFIG_VALUE", {"name": "x", "status": "idle"}, "/status")
    request = RuntimeClaimRequest("ENGINE_STATUS", {"name": "las_manos", "status": "alive"}, "/status")
    with pytest.raises(RuntimeOutputCompositionError):
        asyncio.run(RuntimeOutputComposition(platform_source_configuration=_PLATFORM_CONFIG,
            governance_receipt=_receipt(), templates={}, clock=lambda: NOW,
            authenticator=ReceiptAuthenticator(b"r" * 32, key_id="runtime-output-test-process")).compose(
                scope=scope(component_id="engine-output"), response_id="runtime-2", producer="engine-output",
                tool_data={"status": "alive"}, requests=(request,)))


def test_empty_requests_remain_typed_tool_data_without_fake_claim():
    result = asyncio.run(_composition().compose(scope=scope(component_id="tool-output"),
        response_id="tool-only", producer="tool-output",
        tool_data={"job_id": "tool-reference-only", "output_ref": "tool:42"}, requests=()))
    assert len(result.envelope.content_blocks) == 1
    assert result.envelope.content_blocks[0].kind.value == "TOOL_DATA"
    assert result.envelope.claims == () and result.context.registry is not None


def test_no_claim_path_derives_real_scope_bound_registry_receipt():
    auth = ReceiptAuthenticator(b"n" * 32, key_id="runtime-output-no-claim")
    composer = RuntimeOutputComposition(platform_source_configuration=_PLATFORM_CONFIG,
        governance_receipt=None, templates={}, platform_evidence_bridge=_bridge,
        clock=lambda: NOW, authenticator=auth)
    first_scope = scope(component_id="tool-output")
    first = asyncio.run(composer.compose(scope=first_scope, response_id="tool-one", producer="tool-output",
        tool_data={"output_ref": "tool:one"}, requests=()))
    second_scope = scope(component_id="tool-output", subject_id="another-user")
    second = asyncio.run(composer.compose(scope=second_scope, response_id="tool-two", producer="tool-output",
        tool_data={"output_ref": "tool:two"}, requests=()))
    assert first.envelope.governance_receipt.registry_snapshot_digest == first.context.registry.snapshot_digest
    assert second.envelope.governance_receipt.registry_snapshot_digest == second.context.registry.snapshot_digest
    assert first.context.registry.snapshot_digest != second.context.registry.snapshot_digest


def test_server_numeric_contract_registry_is_route_specific_and_closed():
    motor_bindings = tuple(StructJSONNumberBinding("/" + field, nullable=field != "created_at")
        for field in ("created_at", "started_at", "finished_at", "status_updated_at"))
    other_bindings = (StructJSONNumberBinding("/duration"),)
    composer = RuntimeOutputComposition(platform_source_configuration=_PLATFORM_CONFIG,
        governance_receipt=_receipt(), templates={}, platform_evidence_bridge=_bridge,
        number_binding_contracts={"motor-job-view": motor_bindings, "duration-view": other_bindings},
        clock=lambda: NOW, authenticator=ReceiptAuthenticator(b"c" * 32, key_id="runtime-output-contracts"))
    motor = asyncio.run(composer.compose(scope=scope(component_id="motor-view"), response_id="motor-view",
        producer="motor-view", tool_data={"created_at": 1.5, "started_at": None, "finished_at": 2.5,
            "status_updated_at": None}, requests=(), number_binding_contract_id="motor-job-view"))
    assert motor.number_bindings == motor_bindings
    assert motor.envelope.content_blocks[0].payload["created_at"] == {"f2-e.float64": "0x1.8000000000000p+0"}
    layout = StructuredLayout("motor-numeric", "1", "PLATFORM_HTTP_JSON_V1", 0, motor.slots,
        tuple(OriginBinding("/" + field, OutputOrigin.SYSTEM)
              for field in ("created_at", "started_at", "finished_at", "status_updated_at")),
        number_bindings=motor.number_bindings)
    rendered = GovernedStructuredRenderer().render(motor.envelope, motor.context, layout,
        StructuredLayoutRegistry({layout.layout_id: layout}))
    assert rendered.value == {"created_at": 1.5, "started_at": None, "finished_at": 2.5, "status_updated_at": None}
    duration = asyncio.run(composer.compose(scope=scope(component_id="duration-view"), response_id="duration-view",
        producer="duration-view", tool_data={"duration": 0.25}, requests=(),
        number_binding_contract_id="duration-view"))
    assert duration.number_bindings == other_bindings
    assert duration.envelope.content_blocks[0].payload["duration"] == {"f2-e.float64": "0x1.0000000000000p-2"}
    plain = asyncio.run(composer.compose(scope=scope(component_id="plain-view"), response_id="plain-view",
        producer="plain-view", tool_data={"result": "no numeric route"}, requests=()))
    assert plain.number_bindings == ()
    with pytest.raises(RuntimeOutputCompositionError):
        asyncio.run(composer.compose(scope=scope(component_id="bad-view"), response_id="bad-view",
            producer="bad-view", tool_data={"duration": 1.0}, requests=(), number_binding_contract_id="caller-made"))


@pytest.mark.parametrize("count", (0, 1, 20))
def test_numeric_array_pattern_expands_exact_existing_indices_and_revalidates(count):
    pattern = StructJSONNumberPattern("/steps/*/started_at", nullable=True)
    composer = RuntimeOutputComposition(platform_source_configuration=_PLATFORM_CONFIG,
        governance_receipt=_receipt(), templates={}, platform_evidence_bridge=_bridge,
        number_binding_contracts={"step-times": StructuredNumberBindingContract(patterns=(pattern,))},
        clock=lambda: NOW, authenticator=ReceiptAuthenticator(b"p" * 32, key_id="runtime-output-patterns"))
    steps = [{"name": str(index), "started_at": None if index % 2 else index + 0.5} for index in range(count)]
    result = asyncio.run(composer.compose(scope=scope(component_id="steps-view"), response_id="steps-" + str(count),
        producer="steps-view", tool_data={"steps": steps}, requests=(), number_binding_contract_id="step-times"))
    expected_paths = tuple(f"/steps/{index}/started_at" for index in range(count))
    assert result.number_binding_manifest.expanded["/steps/*/started_at"] == expected_paths
    assert tuple(item.json_pointer for item in result.number_bindings) == expected_paths
    layout = StructuredLayout("steps-" + str(count), "1", "PLATFORM_HTTP_JSON_V1", 0, result.slots,
        (OriginBinding("/steps", OutputOrigin.TOOL), OriginBinding("/steps/*/name", OutputOrigin.TOOL),
         OriginBinding("/steps/*/started_at", OutputOrigin.SYSTEM)),
        number_bindings=result.number_bindings, number_patterns=result.number_patterns,
        number_binding_manifest=result.number_binding_manifest)
    layouts = StructuredLayoutRegistry({layout.layout_id: layout})
    rendered = GovernedStructuredRenderer().render(result.envelope, result.context, layout, layouts)
    assert rendered.value["steps"] == tuple(steps)
    unit = mint_structured_governed_transport_unit(result.envelope, rendered, result.context, layout, layouts,
        metadata=StructuredTransportMetadata("PLATFORM_HTTP_JSON_V1", "application/json", "success", http_status=200),
        idempotency_key="patterns-" + str(count), now=NOW)
    assert revalidate_structured_for_transport(unit, NOW).canonical_bytes == rendered.canonical_bytes


def test_owned_jax_registry_factory_excludes_platform_predicates():
    auth = ReceiptAuthenticator(b"o" * 32, key_id="runtime-output-owned-process")
    factory = lambda current_scope, _auth: build_owned_runtime_status_registry(current_scope, authenticator=_auth)
    composer = RuntimeOutputComposition(platform_source_configuration=None, registry_factory=factory,
        governance_receipt=_receipt(), templates={}, clock=lambda: NOW, authenticator=auth)
    request = RuntimeClaimRequest("ENGINE_STATUS", {"name": "las_manos", "status": "alive"}, "/status")
    with pytest.raises(RuntimeOutputCompositionError):
        asyncio.run(composer.compose(scope=scope(component_id="owned"), response_id="owned", producer="owned",
            tool_data={"status": "alive"}, requests=(request,)))


def _render_and_mint(result, layout):
    now = datetime.now(timezone.utc)
    layouts = StructuredLayoutRegistry({layout.layout_id: layout})
    rendered = GovernedStructuredRenderer().render(result.envelope, result.context, layout, layouts)
    unit = mint_structured_governed_transport_unit(result.envelope, rendered, result.context,
        layout, layouts, metadata=StructuredTransportMetadata("PLATFORM_HTTP_JSON_V1", "application/json", "success", http_status=200),
        idempotency_key="integration:" + result.envelope.response_id, now=now)
    assert revalidate_structured_for_transport(unit, now).canonical_bytes == rendered.canonical_bytes
    return rendered


def test_real_path_job_pipeline_facet_engine_and_tool_data(monkeypatch, tmp_path):
    from motor_registry import routes
    from motor_registry.job_store import JobStore
    from jacobs import models, store as jacobs_store

    # JOB_STATUS: canonical JobStore plus canonical owner/scope check.
    job_store = JobStore(str(tmp_path / "jobs.jsonl"))
    monkeypatch.setattr(routes, "_STORE", job_store)
    job_store.create(caller="test", capability="read", motor="test", trace_id="trace", prompt="p",
        recursion_depth=0, tenant_id="tenant-a", user_id="user-a", job_id="job-1")
    job_scope = scope(component_id="job-output", project_id=None)
    job_request = RuntimeClaimRequest("JOB_STATUS", {"job_id": "job-1", "status": "pending"}, "/status")
    composer = _fresh_composition()
    job = asyncio.run(composer.compose(scope=job_scope, response_id="job", producer="job-output",
        tool_data={"job_id": "job-1", "status": "pending", "output_ref": "tool:job"}, requests=(job_request,)))
    job_layout = StructuredLayout("job", "1", "PLATFORM_HTTP_JSON_V1", 0, job.slots, (
        OriginBinding("/job_id", OutputOrigin.SYSTEM), OriginBinding("/status", OutputOrigin.SYSTEM),
        OriginBinding("/output_ref", OutputOrigin.TOOL),
    ))
    assert _render_and_mint(job, job_layout).value["status"] == "pending"

    # PIPELINE_STATUS: canonical persistence source, while presentation is map-bound.
    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    pipeline = models.Pipeline(pipeline_id="pipeline-1", name="pipeline", invoked_by="web", mode="supervised",
        status=models.PipelineStatus.interrupted, tenant_id="tenant-a", user_id="user-a", updated_at=NOW.timestamp())
    async def pipeline_get(identifier):
        return pipeline if identifier == "pipeline-1" else None
    monkeypatch.setattr(jacobs_store, "pipeline_get", pipeline_get)
    pipeline_scope = scope(component_id="pipeline-output", project_id=None)
    pipeline_request = RuntimeClaimRequest("PIPELINE_STATUS", {"pipeline_id": "pipeline-1", "status": "interrupted"}, "/status", presentation_map_id="pipeline-ui")
    composed_pipeline = asyncio.run(composer.compose(scope=pipeline_scope, response_id="pipeline", producer="pipeline-output",
        tool_data={"pipeline_id": "pipeline-1", "status": "waiting_gate", "output_ref": "tool:pipeline"}, requests=(pipeline_request,)))
    pipeline_layout = StructuredLayout("pipeline", "1", "PLATFORM_HTTP_JSON_V1", 0, composed_pipeline.slots, (
        OriginBinding("/pipeline_id", OutputOrigin.SYSTEM), OriginBinding("/status", OutputOrigin.SYSTEM),
        OriginBinding("/output_ref", OutputOrigin.TOOL),
    ), presentation_maps={"pipeline-ui": {"interrupted": "waiting_gate"}})
    rendered_pipeline = _render_and_mint(composed_pipeline, pipeline_layout)
    assert rendered_pipeline.value["status"] == "waiting_gate"
    assert composed_pipeline.envelope.claims[0].typed_arguments["status"] == "interrupted"

    # FACET_RUNTIME_STATUS through typed server bridge.
    facet_scope = scope(component_id="facet-output")
    facet_request = RuntimeClaimRequest("FACET_RUNTIME_STATUS", {"name": "hyde", "status": "idle"}, "/status")
    facet = asyncio.run(composer.compose(scope=facet_scope, response_id="facet", producer="facet-output",
        tool_data={"name": "hyde", "status": "idle"}, requests=(facet_request,)))
    facet_layout = StructuredLayout("facet", "1", "PLATFORM_HTTP_JSON_V1", 0, facet.slots,
        (OriginBinding("/name", OutputOrigin.SYSTEM), OriginBinding("/status", OutputOrigin.SYSTEM)))
    assert _render_and_mint(facet, facet_layout).value["status"] == "idle"

    # ENGINE_STATUS has no raw canonical name; its layout declares a server constant.
    engine_scope = scope(component_id="engine-output")
    engine_request = RuntimeClaimRequest("ENGINE_STATUS", {"name": "las_manos", "status": "alive"}, "/las_manos_alive", presentation_map_id="alive-bool")
    engine = asyncio.run(composer.compose(scope=engine_scope, response_id="engine", producer="engine-output",
        tool_data={"las_manos_alive": True}, requests=(engine_request,)))
    engine_layout = StructuredLayout("engine", "1", "PLATFORM_HTTP_JSON_V1", 0, engine.slots,
        (OriginBinding("/las_manos_alive", OutputOrigin.SYSTEM),),
        presentation_maps={"alive-bool": {"alive": True, "down": False}})
    assert _render_and_mint(engine, engine_layout).value["las_manos_alive"] is True

    # No status proposition: the same renderer/lifecycle path transports only typed TOOL_DATA.
    tool_scope = scope(component_id="tool-output")
    tool = asyncio.run(composer.compose(scope=tool_scope, response_id="tool", producer="tool-output",
        tool_data={"output_ref": "tool:opaque", "steps": ["queued"]}, requests=()))
    tool_layout = StructuredLayout("tool", "1", "PLATFORM_HTTP_JSON_V1", 0, (), (
        OriginBinding("/output_ref", OutputOrigin.TOOL), OriginBinding("/steps", OutputOrigin.TOOL)))
    assert _render_and_mint(tool, tool_layout).value["output_ref"] == "tool:opaque"
