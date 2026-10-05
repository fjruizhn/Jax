"""F2-E versioned structured projection contract tests."""
from dataclasses import FrozenInstanceError
from dataclasses import replace
from datetime import timedelta
import json

import pytest


def test_registered_pipeline_list_projection_binds_status_slots_to_f2b_claims(monkeypatch):
    from policy.governance.structured_projection import (
        STRUCTURED_PROJECTION_API_VERSION,
        PIPELINE_LIST_SCHEMA_VERSION,
        GovernedStructuredRenderer,
    )

    assert STRUCTURED_PROJECTION_API_VERSION == "f2-c.structured-projection.1"
    assert PIPELINE_LIST_SCHEMA_VERSION == "jax.pipeline-list.json.1"
    # The integration fixture composes a real sealed envelope, F2-B receipt,
    # and TOOL_DATA block below; status is intentionally absent from TOOL_DATA.
    envelope, context = _resolved_pipeline_envelope(monkeypatch)
    rendered = GovernedStructuredRenderer().render_json(envelope, context)

    assert rendered.schema_version == PIPELINE_LIST_SCHEMA_VERSION
    assert rendered.payload["pipelines"][0]["status"] == "running"
    assert rendered.payload["pipelines"][0]["costo_usd"] == 0.08
    assert rendered.claim_ids == ("claim-pipeline-1",)
    assert rendered.field_origins["/pipelines/0/name"].origin == "SYSTEM"
    assert rendered.field_origins["/pipelines/0/status"].origin == "SYSTEM"
    assert b'"status":"running"' in rendered.canonical_bytes
    with pytest.raises(TypeError):
        rendered.field_origins["/pipelines/0/name"] = None
    with pytest.raises((FrozenInstanceError, AttributeError)):
        rendered.canonical_bytes = b"forged"


def test_pipeline_list_projection_rejects_producer_status_or_wrong_claim(monkeypatch):
    from policy.governance.structured_projection import (
        GovernedStructuredRenderer,
        StructuredProjectionError,
    )

    envelope, context = _resolved_pipeline_envelope(monkeypatch, tool_status="failed")
    with pytest.raises(StructuredProjectionError):
        GovernedStructuredRenderer().render_json(envelope, context)

    envelope, context = _resolved_pipeline_envelope(monkeypatch, claim_status="completed")
    with pytest.raises(StructuredProjectionError):
        GovernedStructuredRenderer().render_json(envelope, context)


def test_pipeline_list_projection_rejects_nested_sr03_status_bypass(monkeypatch):
    from policy.governance.structured_projection import (
        GovernedStructuredRenderer,
        StructuredProjectionError,
    )

    envelope, context = _resolved_pipeline_envelope(
        monkeypatch,
        extra_tool_data={"nested": {"pipeline_id": "pipeline-1", "status": "completed"}},
    )
    with pytest.raises(StructuredProjectionError):
        GovernedStructuredRenderer().render_json(envelope, context)


def test_pipeline_list_projection_rejects_trusted_metadata_inside_json_arrays(monkeypatch):
    from policy.governance.structured_projection import (
        GovernedStructuredRenderer,
        StructuredProjectionError,
    )

    envelope, context = _resolved_pipeline_envelope(
        monkeypatch, extra_tool_data={"labels": [{"authority": "human"}]})
    with pytest.raises(StructuredProjectionError, match="trusted governance metadata"):
        GovernedStructuredRenderer().render_json(envelope, context)


def test_pipeline_list_projection_rejects_duplicate_json_keys(monkeypatch):
    from policy.governance import response
    from policy.governance.structured_projection import (
        GovernedStructuredRenderer,
        StructuredProjectionError,
    )

    envelope, context = _resolved_pipeline_envelope(monkeypatch)
    blocks = tuple(response.ContentBlock(block.kind,
        '{"pipelines":[{"pipeline_id":"pipeline-1","name":"one","name":"two"}],"has_more":false}'
        if block.kind is response.ContentBlockKind.TOOL_DATA else block.payload,
        claim_refs=block.claim_refs, attribution_ref=block.attribution_ref,
        speaker=block.speaker, notice_id=block.notice_id)
        for block in envelope.content_blocks)
    candidate = replace(envelope.candidate, content_blocks=blocks)
    changed = response._seal_candidate_for_server(candidate,
        contract_state=envelope.contract_state, governance_receipt=envelope.governance_receipt)
    with pytest.raises(StructuredProjectionError, match="duplicate JSON keys"):
        GovernedStructuredRenderer().render_json(changed, context)


def test_multiple_pipeline_rows_bind_status_by_id_not_array_position(monkeypatch):
    from policy.governance.structured_projection import GovernedStructuredRenderer

    envelope, context = _resolved_pipeline_envelope(monkeypatch, second_pipeline=True)
    rendered = GovernedStructuredRenderer().render_json(envelope, context)
    assert [row["status"] for row in rendered.payload["pipelines"]] == ["running", "completed"]
    assert rendered.field_claim_ids == {
        "/pipelines/0/status": "claim-pipeline-1",
        "/pipelines/1/status": "claim-pipeline-2",
    }


def test_structured_projection_rejects_unknown_schema_versions(monkeypatch):
    from policy.governance.structured_projection import (
        GovernedStructuredRenderer,
        StructuredProjectionError,
    )

    envelope, context = _resolved_pipeline_envelope(monkeypatch)
    with pytest.raises(StructuredProjectionError):
        GovernedStructuredRenderer().render_json(
            envelope, context, schema_version="jax.pipeline-list.json.2")


@pytest.mark.parametrize("row_overrides", [
    {"unreviewed": "field"}, {"name": {"nested": "object"}},
    {"costo_usd": True}, {"created_at": "not-a-timestamp"},
    {"causa": {"tipo": "fallo", "nested": {"unexpected": True}}},
])
def test_pipeline_projection_rejects_unknown_or_ill_typed_closed_rows(monkeypatch, row_overrides):
    from policy.governance.structured_projection import GovernedStructuredRenderer, StructuredProjectionError
    envelope, context = _resolved_pipeline_envelope(monkeypatch, row_overrides=row_overrides)
    with pytest.raises(StructuredProjectionError):
        GovernedStructuredRenderer().render_json(envelope, context)


def test_pipeline_projection_requires_consistent_cursor_and_page_bound(monkeypatch):
    from policy.governance.structured_projection import GovernedStructuredRenderer, StructuredProjectionError
    envelope, context = _resolved_pipeline_envelope(monkeypatch, extra_tool_data={"cursor_siguiente": "bad"})
    with pytest.raises(StructuredProjectionError, match="without more rows"):
        GovernedStructuredRenderer().render_json(envelope, context)
    envelope, context = _resolved_pipeline_envelope(monkeypatch, extra_tool_data={"has_more": True})
    with pytest.raises(StructuredProjectionError, match="requires a cursor"):
        GovernedStructuredRenderer().render_json(envelope, context)


def test_f2d_structured_transport_binds_exact_bytes_and_rejects_ack(monkeypatch):
    from policy.governance.structured_lifecycle import (
        STRUCTURED_BYTES_LIFECYCLE_API_VERSION,
        StructuredOutputLifecycleError,
        StructuredOutputChannelId,
        mint_structured_transport_unit,
        revalidate_structured_for_transport,
        record_structured_delivery_acknowledgement,
        validate_structured_lifecycle_transition,
    )
    from policy.governance.structured_projection import GovernedStructuredRenderer

    envelope, context = _resolved_pipeline_envelope(monkeypatch)
    projection = GovernedStructuredRenderer().render_json(envelope, context)
    unit = mint_structured_transport_unit(
        envelope, projection, context,
        idempotency_key="pipeline-list-request-1", now=context.now())

    assert STRUCTURED_BYTES_LIFECYCLE_API_VERSION == "f2-d.structured-bytes.1"
    assert unit.canonical_bytes is projection.canonical_bytes
    assert revalidate_structured_for_transport(unit, context.now()) is projection.canonical_bytes
    with pytest.raises(StructuredOutputLifecycleError):
        revalidate_structured_for_transport(unit, context.now() + timedelta(seconds=61))
    with pytest.raises(StructuredOutputLifecycleError):
        mint_structured_transport_unit(envelope, projection, context,
            channel_id="pipeline.list.unknown.v2", idempotency_key="unknown", now=context.now())
    with pytest.raises(StructuredOutputLifecycleError):
        validate_structured_lifecycle_transition(
            __import__("policy.governance.output_lifecycle", fromlist=["OutputLifecycleState"]).OutputLifecycleState.TRANSPORT_COMMITTING,
            __import__("policy.governance.output_lifecycle", fromlist=["OutputLifecycleState"]).OutputLifecycleState.DELIVERY_ACKNOWLEDGED)
    with pytest.raises(StructuredOutputLifecycleError):
        record_structured_delivery_acknowledgement(unit)


def _resolved_pipeline_envelope(monkeypatch, *, claim_status=None, tool_status=None,
                                extra_tool_data=None, second_pipeline=False, row_overrides=None):
    """Compose a real Jacobs PIPELINE_STATUS resolution receipt for the tests."""
    from datetime import datetime, timezone
    from jacobs import models, store
    from policy.governance import response
    from policy.governance.governed_renderer import GovernedDomainRegistry, RenderContext
    from policy.governance.response import (
        ClaimRecord, ContentBlock, ContentBlockKind, ContractState, EpistemicStatus,
        GovernanceReceipt, ReferenceRef, ReferenceType, SourceClass, TemporalClass,
        ExistenceState, TemplateContract, ResponseScope,
    )
    from policy.governance.resolution import ReceiptAuthenticator, ReferenceLookupRecord
    from policy.governance.runtime_status import (
        JacobsPipelineStatusResolver,
        build_runtime_status_registry,
    )

    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    now = datetime.now(timezone.utc)
    scope = ResponseScope("production", "tenant-a", None, "user-a", "service:platform",
        "human:user-a", "pipeline-list", "request-1", "trace-1")
    pipeline = models.Pipeline(pipeline_id="pipeline-1", name="Pipeline one",
        invoked_by="plataforma", mode="supervised", status=models.PipelineStatus.running,
        tenant_id="tenant-a", user_id="user-a", updated_at=now.timestamp())
    pipelines = [pipeline]
    if second_pipeline:
        pipelines.append(models.Pipeline(pipeline_id="pipeline-2", name="Pipeline two",
            invoked_by="plataforma", mode="supervised", status=models.PipelineStatus.completed,
            tenant_id="tenant-a", user_id="user-a", updated_at=now.timestamp()))
    canonical_by_id = {item.pipeline_id: item for item in pipelines}

    async def pipeline_status_snapshots(pipeline_ids):
        from datetime import datetime, timezone
        from types import MappingProxyType
        return MappingProxyType({pipeline_id: store.PipelineStatusSnapshot(
            pipeline_id=pipeline_id, tenant_id=item.tenant_id, user_id=item.user_id,
            status=item.status, observed_at=datetime.now(timezone.utc))
            for pipeline_id in pipeline_ids if (item := canonical_by_id.get(pipeline_id)) is not None})

    monkeypatch.setattr(store, "pipeline_status_snapshots", pipeline_status_snapshots)
    platform_config = {
        "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState",
            "status_field": "status", "observed_at_field": "resolver_read_time",
            "allowed_statuses": ["idle", "thinking", "error", "offline"]},
        "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET",
            "path": "/health", "timeout_seconds": 5, "poll_interval_seconds": 30,
            "success_status_code": 200},
    }
    authenticator = ReceiptAuthenticator.for_testing(b"p" * 32)
    registry = build_runtime_status_registry(scope, authenticator=authenticator,
        platform_source_configuration=platform_config)
    claims = []
    receipt_refs = []
    receipts = {}
    claim_refs = []
    for index, pipeline_value in enumerate(pipelines, start=1):
        current_status = getattr(pipeline_value.status, "value", pipeline_value.status)
        arguments = {"pipeline_id": pipeline_value.pipeline_id,
            "status": claim_status if claim_status is not None else current_status}
        evidence = __import__("asyncio").run(JacobsPipelineStatusResolver().evidence(arguments, scope))
        resolution_receipt = registry.resolve("PIPELINE_STATUS", arguments, scope,
            validation_time=evidence.observation.observed_at, runtime_status_evidence=evidence)
        receipt_id = f"receipt-pipeline-{index}"
        claim = ClaimRecord(f"claim-pipeline-{index}", "PIPELINE_STATUS", arguments, scope,
            SourceClass.CURRENT_SOURCE, EpistemicStatus.CURRENT_OBSERVATION,
            resolution_receipt_ref=receipt_id, disposition=response.ClaimDisposition.ASSERTABLE,
            template_contract=TemplateContract("PIPELINE_STATUS", "f2-e.runtime-status.3", "es"))
        receipt_ref = ReferenceRef(receipt_id, ReferenceType.RESOLUTION_RECEIPT,
            f"axioma://{receipt_id}", f"immutable:{receipt_id}",
            resolution_receipt.receipt_id, scope.scope_digest, TemporalClass.CURRENT,
            ExistenceState.PRESENT)
        claims.append(claim)
        receipt_refs.append(receipt_ref)
        receipts[receipt_id] = resolution_receipt
        claim_refs.append(claim.claim_id)
    rows = [{"pipeline_id": item.pipeline_id, "name": item.name,
        "created_at": now.timestamp(), "updated_at": now.timestamp(),
        "duracion_s": None, "costo_usd": 0.08, "causa": None}
        for item in pipelines]
    row = rows[0]
    if row_overrides:
        row.update(row_overrides)
    if tool_status is not None:
        row["status"] = tool_status
    data = {"pipelines": rows, "has_more": False, "cursor_siguiente": None}
    if extra_tool_data:
        data.update(extra_tool_data)
    candidate = response.GovernedResponseCandidate("f2-c.1", "response-1",
        scope.request_id, scope.trace_id, scope, "pipeline-list", (),
        (ContentBlock(ContentBlockKind.TOOL_DATA, json.dumps(data, separators=(",", ":"))),
         ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK, claim_refs=tuple(claim_refs))),
        tuple(claims), tuple(receipt_refs))
    envelope = response._seal_candidate_for_server(candidate,
        contract_state=ContractState.VALID,
        governance_receipt=GovernanceReceipt("policy-v1", "vocab-v1",
            "sha256:" + "a" * 64, "f2-c", "renderer-plan-f2-c"))
    lookup_by_id = {ref.ref_id: ReferenceLookupRecord(ref.ref_id, ref.ref_type,
        ref.canonical_locator, ref.immutable_identity, ref.revision_or_digest,
        ref.scope_digest, ref.temporal_class, ref.existence_state, True)
        for ref in receipt_refs}
    context = RenderContext(registry, receipts, {}, {},
        GovernedDomainRegistry(), reference_validator=lambda _ref, _scope: True,
        now=lambda: datetime.now(timezone.utc),
        receipt_reference_resolver=lambda ref, _scope: lookup_by_id.get(ref.ref_id))
    return envelope, context
