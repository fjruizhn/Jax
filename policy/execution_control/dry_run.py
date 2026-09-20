"""Immutable exact-request dry-run artifacts."""
from dataclasses import dataclass
from datetime import datetime, timezone
from .canonical import canonical_bytes
from .ids import new_dry_run_artifact_id
import hashlib

@dataclass(frozen=True)
class DryRunArtifact:
    dry_run_id: str; execution_id: str; decision_id: str; execution_request_hash: str
    execution_authorization_hash: str; capability: str; motor: str; environment: str
    status: str; result: object; recorded_at_utc: datetime; dry_run_artifact_hash: str

def build_dry_run_artifact(record, authorization, *, status: str, result: object, recorded_at_utc: datetime) -> DryRunArtifact:
    timestamp = recorded_at_utc.astimezone(timezone.utc)
    projection = {"schema_version":"1.0","kind":"JAX_DRY_RUN_ARTIFACT","execution_id":record.execution_id,
      "decision_id":record.decision_id,"execution_request_hash":record.execution_request_hash,
      "execution_authorization_hash":record.execution_authorization_hash,"capability":record.capability,
      "motor":record.motor,"environment":record.environment.value,"status":status,"result":result,
      "recorded_at_utc":timestamp.isoformat().replace("+00:00","Z")}
    digest = "sha256:" + hashlib.sha256(b"JAX-DRY-RUN-ARTIFACT/1.0\0" + canonical_bytes(projection)).hexdigest()
    return DryRunArtifact(new_dry_run_artifact_id(), record.execution_id, record.decision_id, record.execution_request_hash,
      record.execution_authorization_hash, record.capability, record.motor, record.environment.value, status, result, timestamp, digest)
