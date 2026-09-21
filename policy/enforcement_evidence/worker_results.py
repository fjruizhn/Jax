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
    def __init__(self, recorder):
        self._recorder = recorder

    def ingest(self, path: str | Path, *, execution_id: str,
               control_id: str = "CTL.B6.GOVERNED_DISPATCH"):
        if not execution_id:
            raise EvidenceArtifactIntegrityError("execution_id required")
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise EvidenceArtifactIntegrityError("worker result unreadable") from exc
        if len(data) > MAX_BLOB_BYTES:
            raise EvidenceBlobTooLargeError("worker result > 1 MiB")
        definition = load_control_definition(control_id)
        blob = self._recorder._store.put_evidence_blob(data)
        identity = self._recorder._identity
        artifact = EvidenceArtifact(
            EvidenceType.WORKER_RESULT, EvidenceClass.RUNTIME_OBSERVATION,
            definition.control_id, definition.control_version,
            definition.control_definition_hash,
            EvidenceSubject(EvidenceSubjectType.EXECUTION, execution_id),
            (EvidenceBlobRef(blob.evidence_hash, "worker_result",
                             "application/octet-stream", "binary"),),
            EvidenceTrustDomain.JAX_RUNTIME, self._recorder._producer,
            identity.implementation_identity_hash, datetime.now(timezone.utc),
            execution_id=execution_id,
        )
        return self._recorder.record_artifact(artifact)
