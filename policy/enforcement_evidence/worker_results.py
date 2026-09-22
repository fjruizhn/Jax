"""Composition-owned import of a worker result into the evidence CAS.

A path is deliberately only an input to this adapter.  It is never returned
as evidence identity and callers cannot provide the producer, trust domain or
implementation identity used for the resulting artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .errors import EvidenceBlobTooLargeError, EvidenceArtifactIntegrityError
from .evidence_store import MAX_BLOB_BYTES
from .models import (EvidenceArtifact, EvidenceBlobRef, EvidenceClass,
                     EvidenceSubject, EvidenceSubjectType, EvidenceTrustDomain,
                     EvidenceType)
from .control_registry import load_control_definition


class WorkerResultIngestor:
    """Trusted startup dependency for one worker/result producer."""
    def __init__(self, recorder, execution_store):
        self._recorder = recorder
        # A structural ``load_execution`` method is not authoritative.  The
        # execution boundary is fixed at composition and is intentionally a
        # concrete Block 6 store, so a request cannot substitute a fake store.
        from policy.execution_control.storage import InMemoryExecutionStore, MariaDBExecutionStore
        if not isinstance(execution_store, (InMemoryExecutionStore, MariaDBExecutionStore)):
            raise EvidenceArtifactIntegrityError("authoritative execution store required")
        self._execution_store = execution_store

    def ingest_governed_completion(self, path: str | Path, *, execution_id: str, job_id: str):
        """Ingest only the completion of the exact durably dispatched job.

        ``execution_id`` and ``job_id`` come from the fixed Motor adapter;
        neither is caller-provided evidence metadata.  The authoritative B6
        event is the durable link which prevents a result for execution A
        being rebound to execution B.
        """
        if not execution_id:
            raise EvidenceArtifactIntegrityError("execution_id required")
        if not job_id:
            raise EvidenceArtifactIntegrityError("job_id required")
        # The execution binding comes from authoritative Block 6 storage, not
        # from caller metadata.  This rejects invented and cross-bound IDs.
        try:
            execution = self._execution_store.load_execution(execution_id)
        except Exception as exc:
            raise EvidenceArtifactIntegrityError("unknown authoritative execution") from exc
        from policy.execution_control.models import ExecutionRecord
        if not isinstance(execution, ExecutionRecord) or execution.execution_id != execution_id:
            raise EvidenceArtifactIntegrityError("execution binding mismatch")
        try:
            events = self._execution_store.events(execution_id)
        except Exception as exc:
            raise EvidenceArtifactIntegrityError("execution event binding unavailable") from exc
        if not any(event.event_type == "MOTOR_DISPATCHED" and event.job_id == job_id
                   for event in events):
            raise EvidenceArtifactIntegrityError("worker job is not the authoritative dispatch")
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise EvidenceArtifactIntegrityError("worker result unreadable") from exc
        if len(data) > MAX_BLOB_BYTES:
            raise EvidenceBlobTooLargeError("worker result > 1 MiB")
        # This artifact always attests to the fixed governed-dispatch control;
        # result senders cannot retarget it to another control.
        observation = self._recorder._record_worker_result(execution_id, data, "CTL.B6.GOVERNED_DISPATCH")
        return self._recorder._store.load_evidence_artifact(observation.evidence_artifact_hashes[0])
