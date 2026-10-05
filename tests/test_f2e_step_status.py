import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from jacobs import models, store
from policy.governance.response import ResponseScope
from policy.governance.resolution import ResolutionStatus
from policy.governance.runtime_status import JacobsStepStatusResolver
from policy.governance.runtime_status import (
    build_runtime_status_registry, runtime_status_source_configuration_digest,
)
from policy.governance.resolution import ReceiptAuthenticator


NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _jacobs_db_config(monkeypatch):
    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")


def _scope(**changes):
    values = dict(environment="production", tenant_id="tenant-a", project_id=None,
        subject_id="user-a", actor_id="service:platform", audience="human:user",
        component_id="jacobs", request_id="request-1", trace_id="trace-1")
    values.update(changes)
    return ResponseScope(**values)


def _snapshot(**changes):
    values = dict(step_id="step-1", status="blocked_human_gate", pipeline_id="pipe-1",
        tenant_id="tenant-a", user_id="user-a", owner_ack_at=1.0,
        pipeline_status="running", observed_at=NOW)
    values.update(changes)
    return store.StepStatusSnapshot(**values)


def test_step_status_enum_tripwire_matches_canonical_jacobs_enum():
    from policy.governance.governed_domain import (
        _STRUCTURED_STEP_STATUS_VALUES, GovernedDomainSpecification,
    )

    assert _STRUCTURED_STEP_STATUS_VALUES == frozenset(status.value for status in models.StepStatus)
    domain = GovernedDomainSpecification()
    assert domain.registered_proposition("Jacobs step step-1 is blocked_human_gate.") == "STEP_STATUS"
    assert domain.registered_proposition(
        "Jacobs registra actualmente que el paso step-1 está blocked_human_gate.") == "STEP_STATUS"
    assert domain.structured_runtime_status_predicate({
        "nested": [{"step_id": "step-1", "status": "blocked_human_gate"}]
    }) == "STEP_STATUS"
    encoded_twice = json.dumps(json.dumps({"step_id": "step-1", "status": "blocked_human_gate"}))
    assert domain.structured_runtime_status_predicate(encoded_twice) == "STEP_STATUS"
    assert domain.structured_runtime_status_predicate(
        '{"step_id":"step-1","status":"waiting_gate"}') is None


def test_step_snapshot_is_one_joined_read_and_returns_immutable_observation(monkeypatch):
    class Cursor:
        row = {"step_id": "step-1", "status": "completed", "pipeline_id": "pipe-1",
            "tenant_id": "tenant-a", "user_id": "user-a", "owner_ack_at": 1.0,
            "pipeline_status": "running"}
        calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, query, params):
            self.calls.append((query, params))

        async def fetchone(self):
            return self.row

    cursor = Cursor()

    class Connection:
        def cursor(self, *_args, **_kwargs):
            return cursor

    @asynccontextmanager
    async def connection():
        yield Connection()

    monkeypatch.setattr(store, "conexion_del_pool", connection)
    snapshot = asyncio.run(store.step_status_snapshot("step-1"))

    assert len(cursor.calls) == 1
    query, params = cursor.calls[0]
    assert "FROM jacobs_steps" in query
    assert "JOIN jacobs_pipelines" in query
    assert "ON p.pipeline_id = s.pipeline_id" in query
    assert "s.step_id = %s" in query
    assert params == ("step-1",)
    assert isinstance(snapshot, store.StepStatusSnapshot)
    assert snapshot.status == models.StepStatus.completed.value
    assert snapshot.observed_at.tzinfo is not None
    with pytest.raises((AttributeError, TypeError)):
        snapshot.status = "failed"


def test_step_snapshot_timestamps_the_read_before_async_resource_close(monkeypatch):
    events = []

    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            events.append("cursor_closed")
            return False

        async def execute(self, _query, _params):
            return None

        async def fetchone(self):
            events.append("row_fetched")
            return {"step_id": "step-1", "status": "completed", "pipeline_id": "pipe-1",
                "tenant_id": "tenant-a", "user_id": "user-a", "owner_ack_at": 1.0,
                "pipeline_status": "running"}

    class Connection:
        def cursor(self, *_args, **_kwargs):
            return Cursor()

    @asynccontextmanager
    async def connection():
        try:
            yield Connection()
        finally:
            events.append("connection_closed")

    class ObservationClock:
        @staticmethod
        def now(_tz):
            events.append("observed_at")
            return NOW

    monkeypatch.setattr(store, "conexion_del_pool", connection)
    monkeypatch.setattr(store, "datetime", ObservationClock)

    snapshot = asyncio.run(store.step_status_snapshot("step-1"))

    assert snapshot.observed_at == NOW
    assert events == ["row_fetched", "observed_at", "cursor_closed", "connection_closed"]


@pytest.mark.parametrize("canonical_status", [status.value for status in models.StepStatus])
def test_step_resolver_uses_canonical_status_and_owner_snapshot(monkeypatch, canonical_status):
    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot(status=canonical_status)))
    evidence = asyncio.run(JacobsStepStatusResolver().evidence(
        {"step_id": "step-1", "status": canonical_status}, _scope()))

    assert evidence.adapter_kind.value == "JACOBS_STEP_STATUS"
    assert evidence.observation.status is ResolutionStatus.RESOLVED
    assert evidence.observation.result == {"step_id": "step-1", "status": canonical_status}
    assert evidence.observation.observed_at == NOW


@pytest.mark.parametrize("overrides", [
    {"tenant_id": None}, {"user_id": None}, {"owner_ack_at": None},
    {"pipeline_status": "hidden"}, {"pipeline_status": "discarded"},
    {"status": "waiting_gate"}, {"status": "mystery"}, {"step_id": "other-step"},
])
def test_step_resolver_fails_closed_for_unowned_hidden_or_noncanonical_rows(monkeypatch, overrides):
    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot(**overrides)))
    evidence = asyncio.run(JacobsStepStatusResolver().evidence(
        {"step_id": "step-1", "status": "completed"}, _scope()))

    assert evidence.observation.status is ResolutionStatus.UNAVAILABLE


def test_step_resolver_returns_wrong_scope_for_other_owner(monkeypatch):
    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot(user_id="user-b")))
    evidence = asyncio.run(JacobsStepStatusResolver().evidence(
        {"step_id": "step-1", "status": "blocked_human_gate"}, _scope()))

    assert evidence.observation.status is ResolutionStatus.WRONG_SCOPE


def test_step_resolver_returns_wrong_scope_for_same_user_in_other_tenant(monkeypatch):
    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot(tenant_id="tenant-b")))
    evidence = asyncio.run(JacobsStepStatusResolver().evidence(
        {"step_id": "step-1", "status": "blocked_human_gate"}, _scope()))

    assert evidence.observation.status is ResolutionStatus.WRONG_SCOPE


def test_step_resolver_does_not_hide_programming_errors_as_unavailable(monkeypatch):
    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(side_effect=AssertionError("bug")))

    with pytest.raises(AssertionError, match="bug"):
        asyncio.run(JacobsStepStatusResolver().evidence(
            {"step_id": "step-1", "status": "blocked_human_gate"}, _scope()))


def test_step_resolver_does_not_hide_source_configuration_programming_errors(monkeypatch):
    monkeypatch.setattr(store, "_db_cfg", lambda: (_ for _ in ()).throw(AssertionError("bug")))

    with pytest.raises(AssertionError, match="bug"):
        asyncio.run(JacobsStepStatusResolver().evidence(
            {"step_id": "step-1", "status": "blocked_human_gate"}, _scope()))


def test_step_resolver_rejects_project_scope_and_noncanonical_arguments(monkeypatch):
    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot()))
    resolver = JacobsStepStatusResolver()

    with pytest.raises(ValueError):
        asyncio.run(resolver.evidence({"step_id": "step-1", "status": "blocked_human_gate"},
            _scope(project_id="project-a")))
    with pytest.raises(ValueError):
        asyncio.run(resolver.evidence({"step_id": "step-1", "status": "waiting_gate"}, _scope()))


def test_step_status_uses_its_own_versioned_source_identity_and_exact_arguments(monkeypatch):
    from policy.governance.runtime_status import _jacobs_step_source_configuration

    step_config = _jacobs_step_source_configuration(require_config=True)
    step_digest = runtime_status_source_configuration_digest("STEP_STATUS", step_config)
    pipeline_digest = runtime_status_source_configuration_digest("PIPELINE_STATUS", {
        "database_engine": "mariadb", "host": "mariadb.test", "port": 3308,
        "database": "jax_memory_test", "store_contract": "jacobs-pipeline-store-v1",
        "table": "jacobs_pipelines",
    })
    assert step_digest != pipeline_digest

    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot()))
    evidence = asyncio.run(JacobsStepStatusResolver().evidence(
        {"step_id": "step-1", "status": "blocked_human_gate"}, _scope()))
    registry = build_runtime_status_registry(_scope(),
        authenticator=ReceiptAuthenticator.for_testing(b"s" * 32),
        platform_source_configuration={
            "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState",
                "status_field": "status", "observed_at_field": "resolver_read_time",
                "allowed_statuses": ["idle", "thinking", "error", "offline"]},
            "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET",
                "path": "/health", "timeout_seconds": 5, "poll_interval_seconds": 30,
                "success_status_code": 200},
        })
    assert {row["predicate"] for row in registry.status_table()} == {
        "JOB_STATUS", "PROCESSING_JOB_STATUS", "PIPELINE_STATUS", "STEP_STATUS",
        "FACET_RUNTIME_STATUS", "ENGINE_STATUS"}
    receipt = registry.resolve("STEP_STATUS", {"step_id": "step-1", "status": "blocked_human_gate"},
        _scope(), validation_time=NOW, runtime_status_evidence=evidence)
    assert receipt.status is ResolutionStatus.RESOLVED

    mismatch = registry.resolve("STEP_STATUS", {"step_id": "step-1", "status": "completed"},
        _scope(), validation_time=NOW, runtime_status_evidence=evidence)
    assert mismatch.status is ResolutionStatus.SOURCE_MISMATCH


def test_step_receipt_stales_without_a_new_authoritative_read(monkeypatch):
    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot()))
    evidence = asyncio.run(JacobsStepStatusResolver().evidence(
        {"step_id": "step-1", "status": "blocked_human_gate"}, _scope()))
    registry = build_runtime_status_registry(_scope(),
        authenticator=ReceiptAuthenticator.for_testing(b"t" * 32),
        platform_source_configuration={
            "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState",
                "status_field": "status", "observed_at_field": "resolver_read_time",
                "allowed_statuses": ["idle", "thinking", "error", "offline"]},
            "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET",
                "path": "/health", "timeout_seconds": 5, "poll_interval_seconds": 30,
                "success_status_code": 200},
        })
    stale = registry.resolve("STEP_STATUS", {"step_id": "step-1", "status": "blocked_human_gate"},
        _scope(), validation_time=NOW.replace(minute=2), runtime_status_evidence=evidence)
    assert stale.status is ResolutionStatus.STALE


def test_step_status_claim_requires_b2_receipt_and_uses_server_template(monkeypatch):
    from policy.governance.governed_domain import GovernedDomainSpecification
    from policy.governance.governed_renderer import GovernedDomainRegistry, GovernedRenderer, RenderContext
    from policy.governance.loaders import load_templates
    from policy.governance.response import (
        ClaimDisposition, ClaimRecord, ContentBlock, ContentBlockKind, ContractState,
        EpistemicStatus, ExistenceState, GovernanceReceipt, ReferenceRef, ReferenceType,
        SourceClass, TemporalClass, TemplateContract, _seal_candidate_for_server,
    )
    from policy.governance.resolution import ReferenceLookupRecord

    monkeypatch.setattr(store, "step_status_snapshot", AsyncMock(return_value=_snapshot()))
    evidence = asyncio.run(JacobsStepStatusResolver().evidence(
        {"step_id": "step-1", "status": "blocked_human_gate"}, _scope()))
    auth = ReceiptAuthenticator.for_testing(b"r" * 32)
    registry = build_runtime_status_registry(_scope(), authenticator=auth,
        platform_source_configuration={
            "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState",
                "status_field": "status", "observed_at_field": "resolver_read_time",
                "allowed_statuses": ["idle", "thinking", "error", "offline"]},
            "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET",
                "path": "/health", "timeout_seconds": 5, "poll_interval_seconds": 30,
                "success_status_code": 200},
        })
    arguments = {"step_id": "step-1", "status": "blocked_human_gate"}
    resolution_receipt = registry.resolve("STEP_STATUS", arguments, _scope(),
        validation_time=NOW, runtime_status_evidence=evidence)
    ref = ReferenceRef("receipt-ref", ReferenceType.RESOLUTION_RECEIPT,
        "test://step-receipt", "receipt-identity", resolution_receipt.receipt_id,
        _scope().scope_digest, TemporalClass.CURRENT, ExistenceState.PRESENT)
    claim = ClaimRecord("step-claim", "STEP_STATUS", arguments, _scope(),
        SourceClass.CURRENT_SOURCE, EpistemicStatus.CURRENT_OBSERVATION,
        resolution_receipt_ref=ref.ref_id, disposition=ClaimDisposition.ASSERTABLE,
        template_contract=TemplateContract("STEP_STATUS", "f2-e.runtime-status.4", "es"))
    candidate = __import__("policy.governance.response", fromlist=["GovernedResponseCandidate"]).GovernedResponseCandidate(
        "f2-c.1", "step-response", _scope().request_id, _scope().trace_id,
        _scope(), "jacobs", (),
        (ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK, claim_refs=(claim.claim_id,)),),
        (claim,), (ref,))
    governance_receipt = GovernanceReceipt("test-policy", "test-vocabulary",
        registry.snapshot_digest, "test-validator", "f2-c.1")
    envelope = _seal_candidate_for_server(candidate, contract_state=ContractState.VALID,
        governance_receipt=governance_receipt)
    lookup = ReferenceLookupRecord(ref.ref_id, ref.ref_type, ref.canonical_locator,
        ref.immutable_identity, ref.revision_or_digest, ref.scope_digest,
        ref.temporal_class, ref.existence_state, True)
    templates = load_templates()
    context = RenderContext(registry, {ref.ref_id: resolution_receipt},
        {("STEP_STATUS", "f2-e.runtime-status.4", "es"): templates["STEP_STATUS"].template},
        {}, GovernedDomainRegistry(specification=GovernedDomainSpecification()),
        reference_validator=lambda _ref, _scope: True,
        now=lambda: NOW,
        receipt_reference_resolver=lambda _ref, _scope: lookup)

    rendered = GovernedRenderer().render_text(envelope, context)
    assert rendered.contract_state is ContractState.VALID
    assert rendered.text == "Jacobs registra actualmente que el paso step-1 está blocked_human_gate."
