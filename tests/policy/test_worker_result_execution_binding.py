"""B7 worker-result binding: only an exact governed Motor completion is evidence."""
from datetime import datetime, timezone

import pytest

from policy.enforcement_evidence.evidence_store import EvidenceStore
from policy.enforcement_evidence.errors import EvidenceArtifactIntegrityError
from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
from policy.enforcement_evidence.models import (ClaimEnvironment, ClaimScope,
    ImplementationIdentity, SourceState)
from policy.enforcement_evidence.trusted_lifecycle import (EvidenceLifecycleService,
    RuntimeEvidenceRecorder)
from policy.enforcement_evidence.worker_results import WorkerResultIngestor
from policy.execution_control.canonical import execution_record_hash
from policy.execution_control.ids import new_authorization_id, new_execution_id
from policy.execution_control.models import ExecutionEnvironment, ExecutionRecord
from policy.execution_control.storage import ExecutionEvent, InMemoryExecutionStore


def _record(execution_id: str) -> ExecutionRecord:
    value = {
        "schema_version": "1.0", "kind": "JAX_GOVERNED_EXECUTION_RECORD",
        "execution_id": execution_id, "decision_id": new_execution_id(),
        "decision_record_hash": "sha256:" + "1" * 64,
        "execution_request_hash": "sha256:" + "2" * 64,
        "execution_authorization_hash": "sha256:" + "3" * 64,
        "authorization_id": new_authorization_id(), "capability": "implementation",
        "authenticated_caller_id": "worker", "motor": "sandbox",
        "environment": "SANDBOX", "timeout_seconds": 60,
        "created_at_utc": "2026-01-01T00:00:00Z",
    }
    return ExecutionRecord("1.0", value["kind"], execution_id, value["decision_id"],
        value["decision_record_hash"], value["execution_request_hash"],
        value["execution_authorization_hash"], value["authorization_id"],
        value["capability"], value["authenticated_caller_id"], value["motor"],
        ExecutionEnvironment.SANDBOX, 60, datetime(2026, 1, 1, tzinfo=timezone.utc),
        execution_record_hash(value))


def _ingestor():
    evidence = EvidenceStore()
    manifest = evidence.put_evidence_blob(b'{"sources":[]}')
    identity = ImplementationIdentity("fjruizhn/Jax", "a" * 40, "b" * 40,
        SourceState.CLEAN, manifest.evidence_hash)
    recorder = RuntimeEvidenceRecorder(
        EvidenceLifecycleService(evidence, _ControlledTestIdentityProvider(identity)),
        "worker", ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME))
    store = InMemoryExecutionStore()
    return WorkerResultIngestor(recorder, store), store


def test_worker_result_requires_exact_authoritative_execution_and_job(tmp_path):
    ingestor, store = _ingestor()
    result = tmp_path / "worker-result"
    result.write_bytes(b"governed result")
    execution_id = new_execution_id()
    record = _record(execution_id)
    store._records[execution_id] = record
    store._events[execution_id] = [ExecutionEvent(execution_id, "DISPATCHED", "MOTOR_DISPATCHED",
        datetime.now(timezone.utc), "job-a")]

    artifact = ingestor.ingest_governed_completion(result, execution_id=execution_id, job_id="job-a")
    assert artifact.execution_id == execution_id

    with pytest.raises(EvidenceArtifactIntegrityError):
        ingestor.ingest_governed_completion(result, execution_id=execution_id, job_id="job-b")
    with pytest.raises(EvidenceArtifactIntegrityError):
        ingestor.ingest_governed_completion(result, execution_id=new_execution_id(), job_id="job-a")
