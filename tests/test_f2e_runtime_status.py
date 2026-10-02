import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from policy.governance.response import ResponseScope
from policy.governance.resolution import (
    AdapterKind, ConflictPolicy, PredicateAuthorityBinding, ResolutionObservation,
    ResolutionStatus, ScopeRule, SourceScopeClass, TrustedAdapterRegistration,
    RegistryEntry, ReceiptAuthenticator, _build_approved_registry_for_server,
    RuntimeStatusEvidence, _runtime_status_evidence_from_server,
)
from policy.governance.runtime_status import (JacobsPipelineStatusResolver,
    MotorJobStatusResolver, build_runtime_status_registry)
from policy.governance.governed_domain import GovernedDomainSpecification
from motor_registry.job_store import JobStore


NOW = datetime(2026, 10, 2, tzinfo=timezone.utc)


def _scope(**changes):
    base = ResponseScope("production", "tenant-a", None, "user-a", "service:jax",
                         "human:fernando", "governance", "request-a", "trace-a")
    return replace(base, **changes)


def _entry():
    scope = _scope()
    rule = ScopeRule(scope.environment, scope.tenant_id, scope.project_id, scope.subject_id,
                     scope.actor_id, scope.audience, scope.component_id)
    binding = PredicateAuthorityBinding(
        "JOB_STATUS", "f2-e.1", "motor:job-store", "authority:motor",
        "production", rule, rule, 60, ConflictPolicy.SINGLE_SOURCE_REQUIRED,
        "policy.governance.runtime_status:MotorJobStatusResolver", "f2-e.1", None,
        "f2-e.1", source_scope_class=SourceScopeClass.EXACT_RESPONSE_SCOPE,
    )
    adapter = TrustedAdapterRegistration(
        AdapterKind.MOTOR_JOB_STATUS, binding.resolver_implementation_identity,
        binding.resolver_version, binding.designated_source_identity,
    )
    return RegistryEntry(binding, adapter, ("job_id", "status"))


def test_source_scope_class_changes_binding_identity():
    entry = _entry()
    global_binding = replace(entry.binding,
        source_scope_class=SourceScopeClass.INSTALLATION_GLOBAL)
    assert global_binding.digest != entry.binding.digest
    legacy_shape = replace(entry.binding, source_scope_class=None)
    assert "source_scope_class" not in legacy_shape.projection()


def test_job_status_rejects_observation_for_different_canonical_arguments():
    entry = _entry()
    registry = _build_approved_registry_for_server(
        (entry,), authenticator=ReceiptAuthenticator.for_testing(b"x" * 32))
    observation = ResolutionObservation(
        ResolutionStatus.RESOLVED, NOW, "motor-job:job-1",
        {"job_id": "job-1", "status": "failed"},
    )
    evidence = _runtime_status_evidence_from_server(AdapterKind.MOTOR_JOB_STATUS, observation, _scope())
    receipt = registry.resolve("JOB_STATUS", {"job_id": "job-1", "status": "completed"},
        _scope(), validation_time=NOW, runtime_status_evidence=evidence)
    assert receipt.status is ResolutionStatus.SOURCE_MISMATCH


def test_job_status_rejects_project_scope_without_authoritative_mapping():
    entry = _entry()
    registry = _build_approved_registry_for_server(
        (entry,), authenticator=ReceiptAuthenticator.for_testing(b"x" * 32))
    observation = ResolutionObservation(
        ResolutionStatus.RESOLVED, NOW, "motor-job:job-1",
        {"job_id": "job-1", "status": "completed"},
    )
    evidence = _runtime_status_evidence_from_server(AdapterKind.MOTOR_JOB_STATUS, observation, _scope())
    receipt = registry.resolve("JOB_STATUS", {"job_id": "job-1", "status": "completed"},
        _scope(project_id="project-a"), validation_time=NOW, runtime_status_evidence=evidence)
    assert receipt.status is ResolutionStatus.WRONG_SCOPE


def test_motor_status_ownerless_legacy_job_fails_closed(tmp_path):
    store = JobStore(str(tmp_path / "jobs.jsonl"))
    job_id = store.create(caller="jax", capability="x", motor="m", trace_id="t",
        prompt="p", recursion_depth=0)
    evidence = MotorJobStatusResolver(store).evidence(
        {"job_id": job_id, "status": "pending"}, _scope())
    assert evidence.observation.status is ResolutionStatus.UNAVAILABLE


def test_motor_status_persists_governed_owner_and_uses_coherent_view(tmp_path):
    store = JobStore(str(tmp_path / "jobs.jsonl"))
    job_id = store.create(caller="jax", capability="x", motor="m", trace_id="t",
        prompt="p", recursion_depth=0, tenant_id="tenant-a", user_id="user-a", project_id=None)
    evidence = MotorJobStatusResolver(store).evidence(
        {"job_id": job_id, "status": "pending"}, _scope())
    assert evidence.observation.result == {"job_id": job_id, "status": "pending"}


def test_runtime_registry_is_exactly_the_four_authorized_predicates():
    registry = build_runtime_status_registry(_scope(), authenticator=ReceiptAuthenticator.for_testing(b"x" * 32))
    assert {row["predicate"] for row in registry.status_table()} == {
        "JOB_STATUS", "PIPELINE_STATUS", "FACET_RUNTIME_STATUS", "ENGINE_STATUS"}
    assert all(row["freshness_sla_seconds"] in {15, 60} for row in registry.status_table())
    assert all(row["source_owner"] for row in registry.status_table())


def test_installation_global_evidence_cannot_replay_across_response_scope():
    registry = build_runtime_status_registry(_scope(), authenticator=ReceiptAuthenticator.for_testing(b"x" * 32))
    observation = ResolutionObservation(ResolutionStatus.RESOLVED, NOW, "platform:health:x", {"name": "x", "status": "healthy"})
    evidence = _runtime_status_evidence_from_server(AdapterKind.ENGINE_STATUS, observation, _scope())
    receipt = registry.resolve("ENGINE_STATUS", {"name": "x", "status": "healthy"},
        _scope(request_id="other"), validation_time=NOW, runtime_status_evidence=evidence)
    assert receipt.status is ResolutionStatus.WRONG_SCOPE


def test_each_status_claim_is_bound_to_its_own_predicate_and_arguments():
    registry = build_runtime_status_registry(_scope(), authenticator=ReceiptAuthenticator.for_testing(b"x" * 32))
    observation = ResolutionObservation(ResolutionStatus.RESOLVED, NOW, "platform:facet:hyde",
        {"name": "hyde", "status": "thinking"})
    evidence = _runtime_status_evidence_from_server(AdapterKind.FACET_RUNTIME_STATUS, observation, _scope())
    wrong_kind = registry.resolve("ENGINE_STATUS", {"name": "hyde", "status": "thinking"},
        _scope(), validation_time=NOW, runtime_status_evidence=evidence)
    wrong_id = registry.resolve("FACET_RUNTIME_STATUS", {"name": "jekyll", "status": "thinking"},
        _scope(), validation_time=NOW, runtime_status_evidence=evidence)
    wrong_status = registry.resolve("FACET_RUNTIME_STATUS", {"name": "hyde", "status": "offline"},
        _scope(), validation_time=NOW, runtime_status_evidence=evidence)
    assert wrong_kind.status is ResolutionStatus.UNAVAILABLE
    assert wrong_id.status is ResolutionStatus.SOURCE_MISMATCH
    assert wrong_status.status is ResolutionStatus.SOURCE_MISMATCH


def test_runtime_observation_timestamp_is_source_owned_and_stale_or_future_fails():
    registry = build_runtime_status_registry(_scope(), authenticator=ReceiptAuthenticator.for_testing(b"x" * 32))
    arguments = {"name": "hyde", "status": "idle"}
    stale = ResolutionObservation(ResolutionStatus.RESOLVED, NOW.replace(year=2020), "platform:facet:hyde", arguments)
    future = ResolutionObservation(ResolutionStatus.RESOLVED, NOW.replace(year=2030), "platform:facet:hyde", arguments)
    for observation in (stale, future):
        evidence = _runtime_status_evidence_from_server(AdapterKind.FACET_RUNTIME_STATUS, observation, _scope())
        receipt = registry.resolve("FACET_RUNTIME_STATUS", arguments, _scope(), validation_time=NOW,
            runtime_status_evidence=evidence)
        assert receipt.status is ResolutionStatus.STALE


def test_status_evidence_cannot_be_constructed_from_serialized_request_fields():
    observation = ResolutionObservation(ResolutionStatus.RESOLVED, NOW, "platform:facet:hyde",
        {"name": "hyde", "status": "idle"})
    try:
        RuntimeStatusEvidence(AdapterKind.FACET_RUNTIME_STATUS, observation, _scope())
    except Exception as exc:
        assert "server pathway" in str(exc)
    else:
        raise AssertionError("caller constructed trusted status evidence")


def test_governed_domain_distinguishes_pipeline_facet_and_engine_status():
    domain = GovernedDomainSpecification()
    assert domain.registered_proposition("Jacobs pipeline p-1 is running.") == "PIPELINE_STATUS"
    assert domain.registered_proposition("The facet hyde is thinking.") == "FACET_RUNTIME_STATUS"
    assert domain.registered_proposition("LAS MANOS is healthy.") == "ENGINE_STATUS"
    assert domain.registered_proposition("FACET_EXISTS(hyde) is true.") == "FACET_EXISTS"


def test_jacobs_pipeline_resolver_uses_canonical_store_and_exact_owner(monkeypatch):
    from jacobs import models, store

    pipeline = models.Pipeline(
        pipeline_id="same-looking-id", name="test", invoked_by="plataforma", mode="supervised",
        status=models.PipelineStatus.running, tenant_id="tenant-a", user_id="user-a",
        updated_at=NOW.timestamp(),
    )

    async def pipeline_get(pipeline_id):
        assert pipeline_id == "same-looking-id"
        return pipeline

    monkeypatch.setattr(store, "pipeline_get", pipeline_get)
    resolver = JacobsPipelineStatusResolver()
    evidence = asyncio.run(resolver.evidence(
        {"pipeline_id": "same-looking-id", "status": "running"}, _scope()))
    assert evidence.adapter_kind is AdapterKind.JACOBS_PIPELINE_STATUS
    assert evidence.observation.observed_at == NOW
    assert evidence.observation.result == {"pipeline_id": "same-looking-id", "status": "running"}

    other = replace(_scope(), tenant_id="tenant-b", request_id="request-b")
    wrong_tenant = asyncio.run(resolver.evidence(
        {"pipeline_id": "same-looking-id", "status": "running"}, other))
    assert wrong_tenant.observation.status is ResolutionStatus.WRONG_SCOPE


def test_unknown_jacobs_pipeline_never_resolves_and_ignores_caller_timestamps(monkeypatch):
    from jacobs import store

    async def pipeline_get(_pipeline_id):
        return None

    monkeypatch.setattr(store, "pipeline_get", pipeline_get)
    resolver = JacobsPipelineStatusResolver()
    # There is intentionally no observed_at/source argument to this API.
    try:
        asyncio.run(resolver.evidence(
            {"pipeline_id": "missing", "status": "running", "observed_at": NOW}, _scope()))
    except Exception as exc:
        assert "arguments invalid" in str(exc)
    else:
        raise AssertionError("caller timestamp/source fields were accepted")
    evidence = asyncio.run(resolver.evidence(
        {"pipeline_id": "missing", "status": "running"}, _scope()))
    assert evidence.observation.status is ResolutionStatus.UNAVAILABLE
