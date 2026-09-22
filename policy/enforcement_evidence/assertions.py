import json
from datetime import datetime
from .models import (EnforcementAssertion, EvidenceSubject, EvidenceSubjectType, ClaimLevel,
                     AssertionVerdict, ClaimScope, ClaimEnvironment, Coverage, EvidenceTrustDomain)
from .errors import AssertionIntegrityError
def verify_enforcement_assertion_content(value):
    if not value.assertion_hash: raise AssertionIntegrityError("hash")
    return value
def deserialize_enforcement_assertion(data):
    try:
        p=json.loads(data.decode() if isinstance(data,bytes) else data) if not isinstance(data,dict) else data; q=p["scope"]
        subjects=tuple(EvidenceSubject(EvidenceSubjectType(x["subject_type"]),x["identity"]) for x in p["subject_set"])
        return EnforcementAssertion(p["control_id"],p["control_version"],p["control_definition_hash"],ClaimLevel(p["claim_level"]),AssertionVerdict(p["verdict"]),p["implementation_identity_hash"],ClaimScope(ClaimEnvironment(q["environment"]),q.get("deployment_id"),q.get("database_scope_id"),Coverage(q["coverage"])),subjects,tuple(p["evidence_artifact_hashes"]),tuple(p["observation_ids"]),datetime.fromisoformat(p["as_of_utc"].replace("Z","+00:00")),datetime.fromisoformat(p["evidence_window_start_utc"].replace("Z","+00:00")),datetime.fromisoformat(p["evidence_window_end_utc"].replace("Z","+00:00")),tuple(EvidenceTrustDomain(x) for x in p.get("trust_domains_used",())),tuple(p.get("reason_codes",())))
    except Exception as exc: raise AssertionIntegrityError("canonical assertion inválida") from exc
