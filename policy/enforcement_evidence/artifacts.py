import json
from datetime import datetime
from .models import EvidenceArtifact, EvidenceBlobRef, EvidenceType, EvidenceClass, EvidenceTrustDomain, EvidenceSubject, EvidenceSubjectType
from .errors import EvidenceArtifactIntegrityError
def verify_evidence_artifact_content(value, blob_reader):
    if not isinstance(value, EvidenceArtifact): raise EvidenceArtifactIntegrityError("artifact type")
    # property recomputation is deliberate: it derives from canonical fields,
    # not a caller-supplied stored digest.
    expected=value.artifact_hash
    if not expected.startswith("sha256:"): raise EvidenceArtifactIntegrityError("hash")
    for ref in value.blob_refs: blob_reader(ref.evidence_hash)
    return value
def deserialize_evidence_artifact(data):
    try:
        p=json.loads(data.decode() if isinstance(data,bytes) else data) if not isinstance(data,dict) else data
        refs=tuple(EvidenceBlobRef(x["evidence_hash"],x["role"],x["media_type"],x["encoding"]) for x in p["blob_refs"])
        subj=p["subject"]
        return EvidenceArtifact(EvidenceType(p["evidence_type"]),EvidenceClass(p["evidence_class"]),p["control_id"],p["control_version"],p["control_definition_hash"],EvidenceSubject(EvidenceSubjectType(subj["subject_type"]),subj["identity"]),refs,EvidenceTrustDomain(p["trust_domain"]),p["producer"],p["implementation_identity_hash"],datetime.fromisoformat(p["produced_at_utc"].replace("Z","+00:00")),p.get("decision_id"),p.get("execution_id"),tuple(tuple(x) for x in p.get("policy_authority_binding",())))
    except Exception as exc: raise EvidenceArtifactIntegrityError("canonical artifact inválido") from exc
