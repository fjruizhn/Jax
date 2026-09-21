import json
from datetime import datetime
from .models import EnforcementObservation, EvidenceSubject, EvidenceSubjectType, ObservationOutcome, ClaimScope, ClaimEnvironment, Coverage
from .errors import ObservationIntegrityError
def verify_enforcement_observation_content(value):
    if not value.observation_hash: raise ObservationIntegrityError("hash")
    return value
def deserialize_enforcement_observation(data):
    try:
        p=json.loads(data.decode() if isinstance(data,bytes) else data) if not isinstance(data,dict) else data; s=p["subject"]; q=p["scope"]
        return EnforcementObservation(p["observation_id"],p["control_id"],p["control_version"],p["control_definition_hash"],EvidenceSubject(EvidenceSubjectType(s["subject_type"]),s["identity"]),p["implementation_identity_hash"],ObservationOutcome(p["outcome"]),p["reason_code"],datetime.fromisoformat(p["occurred_at_utc"].replace("Z","+00:00")),ClaimScope(ClaimEnvironment(q["environment"]),q.get("deployment_id"),q.get("database_scope_id"),Coverage(q["coverage"])),tuple(p["evidence_artifact_hashes"]),p.get("decision_id"),p.get("execution_id"))
    except Exception as exc: raise ObservationIntegrityError("canonical observation inválida") from exc
