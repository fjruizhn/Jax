"""JAX route boundary tests for F2-E structured HTTP output."""
import asyncio
import json
from datetime import datetime, timezone

import pytest

from jax.external_output import (
    GovernedJaxHTTPAdapter,
    JaxExternalOutputChannel,
    StructuredRouteContract,
    response_scope,
)
from policy.governance.response import GovernanceReceipt
from policy.governance.runtime_output_composition import RuntimeClaimRequest, RuntimeOutputComposition
from policy.governance.runtime_status import build_owned_runtime_status_registry
from policy.governance.resolution import ReceiptAuthenticator
from policy.governance.structured_output import OriginBinding, OutputOrigin


def _adapter() -> GovernedJaxHTTPAdapter:
    authenticator = ReceiptAuthenticator(b"a" * 32, key_id="runtime-output-tool-data-test")
    composition = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda scope, current_auth: build_owned_runtime_status_registry(
            scope, authenticator=current_auth),
        governance_receipt=GovernanceReceipt("f2-e.test", "f2-e.test", "sha256:" + "a" * 64,
                                             "f2-e.test", "f2-e.test"),
        templates={},
        clock=lambda: datetime(2026, 10, 3, tzinfo=timezone.utc),
        authenticator=authenticator,
    )
    return GovernedJaxHTTPAdapter(composition)


def _contract() -> StructuredRouteContract:
    return StructuredRouteContract(
        "test-tool-dto", "1", "las-manos.test", JaxExternalOutputChannel.LAS_MANOS_HTTP,
        (OriginBinding("/job_id", OutputOrigin.TOOL), OriginBinding("/output_ref", OutputOrigin.TOOL)),
        {}, 202,
    )


@pytest.mark.asyncio
async def test_untrusted_tool_dto_crosses_exact_structured_f2c_f2d_without_claim() -> None:
    adapter = _adapter()
    scope = response_scope(environment="test", tenant_id="tenant-a", subject_id="user-a",
                           component_id="las-manos.test")
    prepared = await adapter.prepare(contract=_contract(), scope=scope,
        payload={"job_id": "job-1", "output_ref": "tool:untrusted"})

    assert prepared.status_code == 202
    assert prepared.canonical_bytes == b'{"job_id":"job-1","output_ref":"tool:untrusted"}'
    assert prepared.bytes_for_transport_commit(now=datetime(2026, 10, 3, tzinfo=timezone.utc)) == prepared.canonical_bytes


@pytest.mark.asyncio
async def test_unregistered_post_render_field_fails_closed_before_transport() -> None:
    adapter = _adapter()
    scope = response_scope(environment="test", tenant_id="tenant-a", subject_id="user-a",
                           component_id="las-manos.test")
    prepared = await adapter.prepare(contract=_contract(), scope=scope,
        payload={"job_id": "job-1", "output_ref": "tool:untrusted", "leak": "raw"})

    assert prepared.canonical_bytes == b'{"error":"governed_output_unavailable"}'



def test_motor_dispatch_dto_boundary_uses_canonical_job_status_through_f2b_f2c_f2d(
        monkeypatch, tmp_path) -> None:
    """The registered Motor dispatch DTO does not trust its own status field."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from jax.external_output import (
        GovernedJaxHTTPAdapter,
        configure_jax_external_http_adapter,
        reset_jax_external_http_adapter_for_testing,
    )
    from motor_registry import routes
    from motor_registry.governed_output import MOTOR_RUNTIME_SLOT_CONTRACTS, governed_motor_job_response
    from motor_registry.job_store import JobStore
    from motor_registry.models import JobStatus, MotorDispatchResponse
    from policy.governance.resolution import ReceiptAuthenticator
    from policy.governance.runtime_status import build_owned_runtime_status_registry

    store = JobStore(str(tmp_path / "motor-jobs.jsonl"))
    store.create(caller="service:test", capability="read", motor="test", trace_id="trace-1",
                 prompt="untrusted", recursion_depth=0, tenant_id="tenant-a", user_id="user-a",
                 job_id="job-1")
    monkeypatch.setattr(routes, "_STORE", store)
    monkeypatch.setenv("JAX_GOVERNANCE_ENVIRONMENT", "test")

    authenticator = ReceiptAuthenticator(b"m" * 32, key_id="runtime-output-route-test")
    composition = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda current_scope, current_auth: build_owned_runtime_status_registry(
            current_scope, authenticator=current_auth),
        governance_receipt=GovernanceReceipt(
            "test-policy", "test-vocabulary", "sha256:" + "c" * 64,
            "test-validator", "test-renderer-plan"),
        templates={}, slot_contracts=MOTOR_RUNTIME_SLOT_CONTRACTS,
        clock=lambda: datetime.now(timezone.utc), authenticator=authenticator,
    )
    configure_jax_external_http_adapter(GovernedJaxHTTPAdapter(composition))
    native_dto = MotorDispatchResponse(job_id="job-1", status=JobStatus.PENDING,
        motor="test", capability="read", trace_id="trace-1")

    app = FastAPI()

    @app.post("/motor/governed-dispatch", status_code=202)
    async def dispatch_view():
        return await governed_motor_job_response(store=store, value=native_dto, status_code=202)

    try:
        with TestClient(app) as client:
            response = client.post("/motor/governed-dispatch")
    finally:
        reset_jax_external_http_adapter_for_testing()

    assert response.status_code == 202
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "capability": "read", "job_id": "job-1", "motor": "test",
        "rejected_reason": None, "status": "pending", "trace_id": "trace-1",
    }


def test_registered_motor_job_route_commits_governed_full_native_dto(monkeypatch, tmp_path) -> None:
    """The real registered GET route cannot bypass F2-C/F2-D serialization."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from jax.external_output import (
        GovernedJaxHTTPAdapter,
        configure_jax_external_http_adapter,
        reset_jax_external_http_adapter_for_testing,
    )
    from motor_registry import routes
    from motor_registry.governed_output import (
        MOTOR_JOB_NUMBER_BINDINGS,
        MOTOR_RUNTIME_SLOT_CONTRACTS,
        _MOTOR_JOB_NUMBER_BINDING_CONTRACT_ID,
    )
    from motor_registry.job_store import JobStore
    from policy.governance.resolution import ReceiptAuthenticator
    from policy.governance.runtime_status import build_owned_runtime_status_registry

    store = JobStore(str(tmp_path / "motor-jobs.jsonl"))
    store.create(caller="service:test", capability="read", motor="test", trace_id="trace-1",
                 prompt="untrusted", recursion_depth=0, tenant_id="tenant-a", user_id="user-a",
                 job_id="job-1")
    monkeypatch.setattr(routes, "_STORE", store)
    monkeypatch.setenv("JAX_GOVERNANCE_ENVIRONMENT", "test")
    authenticator = ReceiptAuthenticator(b"g" * 32, key_id="runtime-output-real-route-test")
    composition = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda current_scope, current_auth: build_owned_runtime_status_registry(
            current_scope, authenticator=current_auth),
        governance_receipt=GovernanceReceipt(
            "test-policy", "test-vocabulary", "sha256:" + "e" * 64,
            "test-validator", "test-renderer-plan"),
        templates={}, slot_contracts=MOTOR_RUNTIME_SLOT_CONTRACTS,
        number_binding_contracts={
            _MOTOR_JOB_NUMBER_BINDING_CONTRACT_ID: MOTOR_JOB_NUMBER_BINDINGS,
        }, clock=lambda: datetime.now(timezone.utc), authenticator=authenticator,
    )
    configure_jax_external_http_adapter(GovernedJaxHTTPAdapter(composition))
    app = FastAPI()
    app.include_router(routes.router)
    try:
        with TestClient(app) as client:
            response = client.get("/motor/job/job-1")
    finally:
        reset_jax_external_http_adapter_for_testing()

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "job-1"
    assert body["status"] == "pending"
    assert isinstance(body["created_at"], float)


def test_registered_motor_job_route_rejects_status_that_disagrees_with_canonical_store(
        monkeypatch, tmp_path) -> None:
    """The registered route cannot serialize a handler-mutated job view."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from jax.external_output import (
        GovernedJaxHTTPAdapter,
        configure_jax_external_http_adapter,
        reset_jax_external_http_adapter_for_testing,
    )
    from motor_registry import routes
    from motor_registry.governed_output import MOTOR_RUNTIME_SLOT_CONTRACTS
    from motor_registry.job_store import JobStore
    from motor_registry.models import JobStatus
    from policy.governance.resolution import ReceiptAuthenticator
    from policy.governance.runtime_status import build_owned_runtime_status_registry

    store = JobStore(str(tmp_path / "motor-jobs.jsonl"))
    store.create(caller="service:test", capability="read", motor="test", trace_id="trace-1",
                 prompt="untrusted", recursion_depth=0, tenant_id="tenant-a", user_id="user-a",
                 job_id="job-1")
    canonical_get = store.get
    monkeypatch.setattr(store, "get", lambda job_id: canonical_get(job_id).model_copy(
        update={"status": JobStatus.COMPLETED}))
    monkeypatch.setattr(routes, "_STORE", store)
    monkeypatch.setenv("JAX_GOVERNANCE_ENVIRONMENT", "test")
    authenticator = ReceiptAuthenticator(b"h" * 32, key_id="runtime-output-real-route-mismatch")
    composition = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda current_scope, current_auth: build_owned_runtime_status_registry(
            current_scope, authenticator=current_auth),
        governance_receipt=GovernanceReceipt(
            "test-policy", "test-vocabulary", "sha256:" + "f" * 64,
            "test-validator", "test-renderer-plan"),
        templates={}, slot_contracts=MOTOR_RUNTIME_SLOT_CONTRACTS,
        clock=lambda: datetime.now(timezone.utc), authenticator=authenticator,
    )
    configure_jax_external_http_adapter(GovernedJaxHTTPAdapter(composition))
    app = FastAPI()
    app.include_router(routes.router)
    try:
        with TestClient(app) as client:
            response = client.get("/motor/job/job-1")
    finally:
        reset_jax_external_http_adapter_for_testing()

    assert response.status_code == 503
    assert response.content == b'{"error":"governed_output_unavailable"}'


def test_registered_jacobs_pipeline_route_rejects_a_stale_handler_snapshot(monkeypatch) -> None:
    """The real GET path re-accredits status against Jacobs' canonical store."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from jax.external_output import (
        GovernedJaxHTTPAdapter,
        configure_jax_external_http_adapter,
        reset_jax_external_http_adapter_for_testing,
    )
    from jacobs import routes
    from jacobs.governed_output import JACOBS_RUNTIME_SLOT_CONTRACTS
    from jacobs.models import Pipeline, PipelineStatus
    from policy.governance.resolution import ReceiptAuthenticator
    from policy.governance.runtime_status import build_owned_runtime_status_registry

    reads = iter((
        Pipeline(pipeline_id="pipeline-1", name="test", invoked_by="plataforma", mode="supervised",
                 status=PipelineStatus.running, tenant_id="tenant-a", user_id="user-a", created_at=1.0,
                 updated_at=2.0),
        Pipeline(pipeline_id="pipeline-1", name="test", invoked_by="plataforma", mode="supervised",
                 status=PipelineStatus.interrupted, tenant_id="tenant-a", user_id="user-a", created_at=1.0,
                 updated_at=2.0),
    ))

    async def pipeline_get(_pipeline_id):
        return next(reads)

    async def steps_by_pipeline(_pipeline_id):
        return []

    monkeypatch.setattr(routes.store, "pipeline_get", pipeline_get)
    monkeypatch.setattr(routes.store, "steps_by_pipeline", steps_by_pipeline)
    monkeypatch.setenv("JAX_GOVERNANCE_ENVIRONMENT", "test")
    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    authenticator = ReceiptAuthenticator(b"i" * 32, key_id="runtime-output-jacobs-route-mismatch")
    composition = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda current_scope, current_auth: build_owned_runtime_status_registry(
            current_scope, authenticator=current_auth),
        governance_receipt=GovernanceReceipt(
            "test-policy", "test-vocabulary", "sha256:" + "9" * 64,
            "test-validator", "test-renderer-plan"),
        templates={}, slot_contracts=JACOBS_RUNTIME_SLOT_CONTRACTS,
        clock=lambda: datetime.now(timezone.utc), authenticator=authenticator,
    )
    configure_jax_external_http_adapter(GovernedJaxHTTPAdapter(composition))
    app = FastAPI()
    app.include_router(routes.router)
    try:
        with TestClient(app) as client:
            response = client.get("/jacobs/pipeline/pipeline-1")
    finally:
        reset_jax_external_http_adapter_for_testing()

    assert response.status_code == 503
    assert response.content == b'{"error":"governed_output_unavailable"}'


def test_registered_jacobs_pipeline_route_fails_closed_for_unaccredited_nested_step_status(
        monkeypatch) -> None:
    """A native Step status cannot bypass F2-E while no STEP_STATUS claim exists."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from jax.external_output import (
        GovernedJaxHTTPAdapter,
        configure_jax_external_http_adapter,
        reset_jax_external_http_adapter_for_testing,
    )
    from jacobs import routes
    from jacobs.governed_output import (
        JACOBS_PIPELINE_DETAIL_NUMBER_BINDINGS,
        JACOBS_RUNTIME_SLOT_CONTRACTS,
        _PIPELINE_DETAIL_NUMBER_BINDING_CONTRACT_ID,
    )
    from jacobs.models import Pipeline, PipelineStatus, Step
    from policy.governance.resolution import ReceiptAuthenticator
    from policy.governance.runtime_status import build_owned_runtime_status_registry

    pipeline = Pipeline(pipeline_id="pipeline-1", name="test", invoked_by="plataforma",
        mode="supervised", status=PipelineStatus.running, tenant_id="tenant-a", user_id="user-a",
        created_at=1.25, updated_at=2.5)
    steps = [Step(pipeline_id="pipeline-1", step_index=0, facet="hyde", capability="read",
        started_at=3.75, finished_at=None, output_ref="tool:opaque")]

    async def pipeline_get(_pipeline_id):
        return pipeline

    async def steps_by_pipeline(_pipeline_id):
        return steps

    monkeypatch.setattr(routes.store, "pipeline_get", pipeline_get)
    monkeypatch.setattr(routes.store, "steps_by_pipeline", steps_by_pipeline)
    monkeypatch.setenv("JAX_GOVERNANCE_ENVIRONMENT", "test")
    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    authenticator = ReceiptAuthenticator(b"j" * 32, key_id="runtime-output-jacobs-route-full")
    composition = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda current_scope, current_auth: build_owned_runtime_status_registry(
            current_scope, authenticator=current_auth),
        governance_receipt=GovernanceReceipt(
            "test-policy", "test-vocabulary", "sha256:" + "8" * 64,
            "test-validator", "test-renderer-plan"),
        templates={}, slot_contracts=JACOBS_RUNTIME_SLOT_CONTRACTS,
        number_binding_contracts={
            _PIPELINE_DETAIL_NUMBER_BINDING_CONTRACT_ID: JACOBS_PIPELINE_DETAIL_NUMBER_BINDINGS,
        }, clock=lambda: datetime.now(timezone.utc), authenticator=authenticator,
    )
    configure_jax_external_http_adapter(GovernedJaxHTTPAdapter(composition))
    app = FastAPI()
    app.include_router(routes.router)
    try:
        with TestClient(app) as client:
            response = client.get("/jacobs/pipeline/pipeline-1")
    finally:
        reset_jax_external_http_adapter_for_testing()

    assert response.status_code == 503
    assert response.content == b'{"error":"governed_output_unavailable"}'


def test_motor_dispatch_dto_cannot_replace_canonical_status(monkeypatch, tmp_path) -> None:
    """A producer-supplied DTO status differing from JobStore fails closed."""
    from jax.external_output import (
        GovernedJaxHTTPAdapter,
        configure_jax_external_http_adapter,
        reset_jax_external_http_adapter_for_testing,
    )
    from motor_registry import routes
    from motor_registry.governed_output import MOTOR_RUNTIME_SLOT_CONTRACTS, governed_motor_job_response
    from motor_registry.job_store import JobStore
    from motor_registry.models import JobStatus, MotorDispatchResponse
    from policy.governance.resolution import ReceiptAuthenticator
    from policy.governance.runtime_status import build_owned_runtime_status_registry

    store = JobStore(str(tmp_path / "motor-jobs.jsonl"))
    store.create(caller="service:test", capability="read", motor="test", trace_id="trace-1",
                 prompt="untrusted", recursion_depth=0, tenant_id="tenant-a", user_id="user-a",
                 job_id="job-1")
    monkeypatch.setattr(routes, "_STORE", store)
    monkeypatch.setenv("JAX_GOVERNANCE_ENVIRONMENT", "test")
    authenticator = ReceiptAuthenticator(b"n" * 32, key_id="runtime-output-mismatch-test")
    composition = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda current_scope, current_auth: build_owned_runtime_status_registry(
            current_scope, authenticator=current_auth),
        governance_receipt=GovernanceReceipt(
            "test-policy", "test-vocabulary", "sha256:" + "d" * 64,
            "test-validator", "test-renderer-plan"),
        templates={}, slot_contracts=MOTOR_RUNTIME_SLOT_CONTRACTS,
        clock=lambda: datetime.now(timezone.utc), authenticator=authenticator,
    )
    configure_jax_external_http_adapter(GovernedJaxHTTPAdapter(composition))
    try:
        response = asyncio.run(governed_motor_job_response(
            store=store,
            value=MotorDispatchResponse(job_id="job-1", status=JobStatus.COMPLETED,
                motor="test", capability="read", trace_id="trace-1"),
            status_code=202,
        ))
    finally:
        reset_jax_external_http_adapter_for_testing()
    assert response.status_code == 503


def test_runtime_templates_are_loaded_from_the_verified_closed_policy_file() -> None:
    from jax.external_output import runtime_status_templates
    from policy.governance.runtime_status import RUNTIME_STATUS_API_VERSION

    templates = runtime_status_templates()
    assert set(templates) == {
        ("JOB_STATUS", RUNTIME_STATUS_API_VERSION, "es"),
        ("PIPELINE_STATUS", RUNTIME_STATUS_API_VERSION, "es"),
        ("FACET_RUNTIME_STATUS", RUNTIME_STATUS_API_VERSION, "es"),
        ("ENGINE_STATUS", RUNTIME_STATUS_API_VERSION, "es"),
    }

@pytest.mark.asyncio
async def test_full_motor_job_dto_preserves_finite_float_schema_inside_f2c(monkeypatch, tmp_path) -> None:
    from motor_registry import routes
    from motor_registry.governed_output import (
        MOTOR_JOB_NUMBER_BINDINGS, MOTOR_RUNTIME_SLOT_CONTRACTS,
        _JOB_CONTRACT, _MOTOR_JOB_NUMBER_BINDING_CONTRACT_ID,
    )
    from motor_registry.job_store import JobStore
    from policy.governance.resolution import ReceiptAuthenticator

    store = JobStore(str(tmp_path / "motor-jobs.jsonl"))
    store.create(caller="service:test", capability="read", motor="test", trace_id="trace-1",
                 prompt="untrusted", recursion_depth=0, tenant_id="tenant-a", user_id="user-a",
                 job_id="job-1")
    monkeypatch.setattr(routes, "_STORE", store)
    authenticator = ReceiptAuthenticator(b"f" * 32, key_id="runtime-output-float-test")
    composer = RuntimeOutputComposition(
        platform_source_configuration=None,
        registry_factory=lambda scope, current_auth: build_owned_runtime_status_registry(
            scope, authenticator=current_auth),
        governance_receipt=None, templates={}, slot_contracts=MOTOR_RUNTIME_SLOT_CONTRACTS,
        number_binding_contracts={
            _MOTOR_JOB_NUMBER_BINDING_CONTRACT_ID: MOTOR_JOB_NUMBER_BINDINGS,
        }, authenticator=authenticator,
    )
    view = store.get("job-1")
    adapter = GovernedJaxHTTPAdapter(composer)
    prepared = await adapter.prepare(
        contract=_JOB_CONTRACT,
        scope=response_scope(environment="test", tenant_id="tenant-a", subject_id="user-a",
                             component_id="las-manos.motor-registry"),
        payload=view.model_dump(mode="json"),
        claims=(RuntimeClaimRequest("JOB_STATUS", {"job_id": "job-1", "status": "pending"}, "/status"),),
    )
    body = json.loads(prepared.bytes_for_transport_commit())
    assert isinstance(body["created_at"], float)
    assert body["created_at"] == view.created_at
    assert body["status"] == "pending"
