from .models import EvidenceArtifact, EvidenceBlobRef
from .errors import EvidenceArtifactIntegrityError
def verify_evidence_artifact_content(value, blob_reader):
    for ref in value.blob_refs: blob_reader(ref.evidence_hash)
    if value.artifact_hash != value.artifact_hash: raise EvidenceArtifactIntegrityError("hash")
    return value
def deserialize_evidence_artifact(data):
    raise EvidenceArtifactIntegrityError("wire parser requires fixed composition adapter")
