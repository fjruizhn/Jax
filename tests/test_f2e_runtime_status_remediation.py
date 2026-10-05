import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from motor_registry.job_store import JobStore
from policy.governance.runtime_status import MotorJobStatusResolver
from policy.governance.resolution import ResolutionStatus
from policy.governance.response import ResponseScope


def _scope():
    return ResponseScope("production", "tenant-a", None, "user-a", "service:jax",
                         "human:fernando", "governance", "request-a", "trace-a")


def _status_batch(*pipelines):
    by_id = {pipeline.pipeline_id: pipeline for pipeline in pipelines}
    async def snapshots(ids):
        from jacobs import store
        observed_at = datetime.now(timezone.utc)
        return MappingProxyType({pipeline_id: store.PipelineStatusSnapshot(
            pipeline_id=pipeline_id, tenant_id=pipeline.tenant_id, user_id=pipeline.user_id,
            status=pipeline.status, observed_at=observed_at)
            for pipeline_id in ids if (pipeline := by_id.get(pipeline_id)) is not None})
    return snapshots


def _job_event(job_id="job-a", status="completed", **overrides):
    return {"job_id": job_id, "status": status, "created_at": 1.0,
            "started_at": 2.0, "finished_at": 3.0, "status_updated_at": 3.0,
            "motor": "m", "capability": "x", "caller": "jax", "trace_id": "t",
            "recursion_depth": 0,
            "tenant_id": "tenant-a", "user_id": "user-a", "project_id": None,
            **overrides}


def test_corrupt_latest_job_event_disables_authoritative_snapshot(tmp_path, monkeypatch):
    from motor_registry import routes
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps(_job_event()) + "\n" + '{"job_id":"job-a","status":', encoding="utf-8")
    store = JobStore(str(path))
    assert store.get("job-a").status.value == "completed"  # legacy polling remains best effort
    assert store.authoritative_snapshot("job-a") is None
    monkeypatch.setattr(routes, "_STORE", store)
    evidence = MotorJobStatusResolver().evidence({"job_id": "job-a", "status": "completed"}, _scope())
    assert evidence.observation.status is ResolutionStatus.UNAVAILABLE


@pytest.mark.parametrize("bad_event", [
    '{"job_id":"job-a","status":',
    '{"job_id":"job-a","status":"failed"',
    '{"status":"completed"}',
    json.dumps({**_job_event(), "status": "invented"}),
    json.dumps({"job_id": "job-a", "status": "completed"}),
    json.dumps({**_job_event(), "finished_at": float("nan")}),
])
def test_any_uncertain_job_history_prevents_authoritative_reconstruction(tmp_path, bad_event):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps(_job_event()) + "\n" + bad_event + "\n" +
        json.dumps(_job_event(status="failed")) + "\n", encoding="utf-8")
    store = JobStore(str(path))
    assert store.authoritative_snapshot("job-a") is None


def test_unterminated_complete_json_tail_is_not_authoritative(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps(_job_event()), encoding="utf-8")
    assert JobStore(str(path)).authoritative_snapshot("job-a") is None


def test_clean_job_history_has_fresh_observation_separate_from_transition(tmp_path, monkeypatch):
    from motor_registry import routes
    import motor_registry.job_store as module
    path = tmp_path / "jobs.jsonl"
    store = JobStore(str(path))
    monkeypatch.setattr(routes, "_STORE", store)
    job_id = store.create(caller="jax", capability="x", motor="m", trace_id="t", prompt="p",
        recursion_depth=0, tenant_id="tenant-a", user_id="user-a")
    store.update(job_id, status="running", started_at=1.0)
    store.update(job_id, status="completed", finished_at=2.0)
    resolver = MotorJobStatusResolver()
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    evidence = resolver.evidence({"job_id": job_id, "status": "completed"}, _scope())
    assert evidence.observation.status is ResolutionStatus.RESOLVED
    assert evidence.observation.observed_at > old
    assert store.get(job_id).finished_at == 2.0


def test_job_source_configuration_changes_identity(tmp_path):
    first = JobStore(str(tmp_path / "one.jsonl"))
    second = JobStore(str(tmp_path / "two.jsonl"))
    assert first.source_configuration() != second.source_configuration()


def test_jsonl_append_fsyncs_before_authoritative_index_update(tmp_path, monkeypatch):
    import motor_registry.job_store as module
    synced = []
    monkeypatch.setattr(module.os, "fsync", lambda descriptor: synced.append(descriptor))
    store = JobStore(str(tmp_path / "jobs.jsonl"))
    job_id = store.create(caller="jax", capability="x", motor="m", trace_id="t", prompt="p",
        recursion_depth=0, tenant_id="tenant-a", user_id="user-a")
    assert synced
    assert store.authoritative_snapshot(job_id) is not None


def test_fsync_failure_does_not_advance_authoritative_job_index(tmp_path, monkeypatch):
    import motor_registry.job_store as module
    store = JobStore(str(tmp_path / "jobs.jsonl"))
    monkeypatch.setattr(module.os, "fsync", lambda _descriptor: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        store.create(caller="jax", capability="x", motor="m", trace_id="t", prompt="p",
            recursion_depth=0, tenant_id="tenant-a", user_id="user-a", job_id="job-fsync")
    assert store.get("job-fsync") is None
    assert store.authoritative_snapshot("job-fsync") is None


def test_jacobs_stable_status_gets_observation_time_not_transition_time(monkeypatch):
    from jacobs import models, store as jacobs_store
    from policy.governance.runtime_status import JacobsPipelineStatusResolver
    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    old_transition = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
    pipeline = models.Pipeline(pipeline_id="pipeline-old", name="p", invoked_by="web", mode="supervised",
        status=models.PipelineStatus.completed, tenant_id="tenant-a", user_id="user-a",
        updated_at=old_transition)
    monkeypatch.setattr(jacobs_store, "pipeline_status_snapshots", _status_batch(pipeline))
    evidence = asyncio.run(JacobsPipelineStatusResolver().evidence(
        {"pipeline_id": "pipeline-old", "status": "completed"}, _scope()))
    assert evidence.observation.status is ResolutionStatus.RESOLVED
    assert evidence.observation.observed_at > datetime.fromtimestamp(old_transition, timezone.utc)


def test_job_and_jacobs_configuration_identity_is_nonsecret_and_changes_digest(tmp_path):
    from policy.governance.runtime_status import runtime_status_source_configuration_digest
    job_a = JobStore(str(tmp_path / "a.jsonl")).source_configuration()
    job_b = JobStore(str(tmp_path / "b.jsonl")).source_configuration()
    assert runtime_status_source_configuration_digest("JOB_STATUS", job_a) != runtime_status_source_configuration_digest("JOB_STATUS", job_b)
    jacobs_a = {"database_engine": "mariadb", "host": "db-a", "port": 3308,
        "database": "jax_memory", "store_contract": "jacobs-pipeline-store-v1", "table": "jacobs_pipelines"}
    jacobs_b = {**jacobs_a, "database": "jax_memory_test"}
    assert runtime_status_source_configuration_digest("PIPELINE_STATUS", jacobs_a) != runtime_status_source_configuration_digest("PIPELINE_STATUS", jacobs_b)


def test_jacobs_database_change_invalidates_previously_minted_receipt(monkeypatch):
    from jacobs import models, store as jacobs_store
    from policy.governance.runtime_status import build_runtime_status_registry, JacobsPipelineStatusResolver
    from policy.governance.resolution import ReceiptAuthenticator
    monkeypatch.setenv("JAX_DB_HOST", "db-a.internal")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory")
    pipeline = models.Pipeline(pipeline_id="pipeline-a", name="p", invoked_by="web", mode="supervised",
        status=models.PipelineStatus.completed, tenant_id="tenant-a", user_id="user-a",
        updated_at=1.0)
    monkeypatch.setattr(jacobs_store, "pipeline_status_snapshots", _status_batch(pipeline))
    platform_config = {
        "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState", "status_field": "status",
            "observed_at_field": "resolver_read_time", "allowed_statuses": ["idle", "thinking", "error", "offline"]},
        "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET", "path": "/health",
            "timeout_seconds": 5, "poll_interval_seconds": 30, "success_status_code": 200},
    }
    auth = ReceiptAuthenticator.for_testing(b"j" * 32)
    registry_a = build_runtime_status_registry(_scope(), authenticator=auth,
        platform_source_configuration=platform_config)
    args = {"pipeline_id": "pipeline-a", "status": "completed"}
    evidence = asyncio.run(JacobsPipelineStatusResolver().evidence(args, _scope()))
    receipt = registry_a.resolve("PIPELINE_STATUS", args, _scope(),
        validation_time=evidence.observation.observed_at, runtime_status_evidence=evidence)
    assert receipt.status is ResolutionStatus.RESOLVED
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_other")
    registry_b = build_runtime_status_registry(_scope(), authenticator=auth,
        platform_source_configuration=platform_config)
    assert not registry_b.verify_receipt(receipt, _scope(), validation_time=receipt.observed_at)


def test_old_engine_health_probe_remains_stale_without_new_probe():
    from policy.governance.runtime_status import build_runtime_status_registry, runtime_status_source_configuration_digest
    from policy.governance.resolution import ReceiptAuthenticator, AdapterKind, ResolutionObservation, _runtime_status_evidence_from_server
    platform_config = {
        "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState", "status_field": "status",
            "observed_at_field": "resolver_read_time", "allowed_statuses": ["idle", "thinking", "error", "offline"]},
        "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET", "path": "/health",
            "timeout_seconds": 5, "poll_interval_seconds": 30, "success_status_code": 200},
    }
    registry = build_runtime_status_registry(_scope(),
        authenticator=ReceiptAuthenticator.for_testing(b"e" * 32),
        platform_source_configuration=platform_config)
    probe_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    evidence = _runtime_status_evidence_from_server(AdapterKind.ENGINE_STATUS,
        ResolutionObservation(ResolutionStatus.RESOLVED, probe_time, "platform:las-manos-health",
            {"name": "las_manos", "status": "alive"}), _scope(),
        runtime_status_source_configuration_digest("ENGINE_STATUS", platform_config["ENGINE_STATUS"]))
    receipt = registry.resolve("ENGINE_STATUS", {"name": "las_manos", "status": "alive"},
        _scope(), validation_time=datetime.now(timezone.utc), runtime_status_evidence=evidence)
    assert receipt.status is ResolutionStatus.STALE


def test_pipeline_read_keeps_one_scope_and_status_snapshot_during_transition(monkeypatch):
    from jacobs import models, store as jacobs_store
    from policy.governance.runtime_status import JacobsPipelineStatusResolver
    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    source = {"status": models.PipelineStatus.running, "tenant_id": "tenant-a", "user_id": "user-a"}
    captured = asyncio.Event()
    continue_read = asyncio.Event()
    async def snapshots(ids):
        row = dict(source)
        captured.set()
        await continue_read.wait()
        from jacobs import store
        return MappingProxyType({pipeline_id: store.PipelineStatusSnapshot(
            pipeline_id=pipeline_id, tenant_id=row["tenant_id"], user_id=row["user_id"],
            status=row["status"], observed_at=datetime.now(timezone.utc)) for pipeline_id in ids})
    monkeypatch.setattr(jacobs_store, "pipeline_status_snapshots", snapshots)
    async def scenario():
        task = asyncio.create_task(JacobsPipelineStatusResolver().evidence(
            {"pipeline_id": "pipeline-race", "status": "running"}, _scope()))
        await captured.wait()
        source.update(status=models.PipelineStatus.interrupted, tenant_id="tenant-b", user_id="user-b")
        continue_read.set()
        return await task
    evidence = asyncio.run(scenario())
    assert evidence.observation.result == {"pipeline_id": "pipeline-race", "status": "running"}
    assert evidence.observation.status is ResolutionStatus.RESOLVED


def test_job_source_change_invalidates_previously_minted_receipt(tmp_path, monkeypatch):
    from motor_registry import routes
    from policy.governance.runtime_status import build_runtime_status_registry
    from policy.governance.resolution import ReceiptAuthenticator
    platform_config = {
        "FACET_RUNTIME_STATUS": {"state_contract": "JAXEngineState.FacetState", "status_field": "status",
            "observed_at_field": "resolver_read_time", "allowed_statuses": ["idle", "thinking", "error", "offline"]},
        "ENGINE_STATUS": {"endpoint_sha256": "sha256:" + "e" * 64, "method": "GET", "path": "/health",
            "timeout_seconds": 5, "poll_interval_seconds": 30, "success_status_code": 200},
    }
    store_a = JobStore(str(tmp_path / "jobs-a.jsonl"))
    job_id = store_a.create(caller="jax", capability="x", motor="m", trace_id="t", prompt="p",
        recursion_depth=0, tenant_id="tenant-a", user_id="user-a")
    monkeypatch.setattr(routes, "_STORE", store_a)
    auth = ReceiptAuthenticator.for_testing(b"r" * 32)
    registry_a = build_runtime_status_registry(_scope(), authenticator=auth,
        platform_source_configuration=platform_config)
    evidence = MotorJobStatusResolver().evidence({"job_id": job_id, "status": "pending"}, _scope())
    receipt = registry_a.resolve("JOB_STATUS", {"job_id": job_id, "status": "pending"},
        _scope(), validation_time=evidence.observation.observed_at, runtime_status_evidence=evidence)
    assert receipt.status is ResolutionStatus.RESOLVED
    monkeypatch.setattr(routes, "_STORE", JobStore(str(tmp_path / "jobs-b.jsonl")))
    registry_b = build_runtime_status_registry(_scope(), authenticator=auth,
        platform_source_configuration=platform_config)
    assert not registry_b.verify_receipt(receipt, _scope(), validation_time=receipt.observed_at)
