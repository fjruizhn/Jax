"""Canonical, closed test evidence manifest validation without GitHub I/O."""
from __future__ import annotations
import json
from .ids import require_hash
from .errors import EvidenceBindingError
def ingest_test_evidence_manifest(store, manifest_bytes:bytes, raw_output:bytes|None=None):
 data=json.loads(manifest_bytes)
 required={"schema_version","kind","provider","repository_id","commit_sha","implementation_identity_hash","workflow","run_id","job_id","environment","started_at_utc","completed_at_utc","tests","counts","raw_output_blob_hash"}
 if set(data)!=required or data["kind"]!="JAX_TEST_EVIDENCE_MANIFEST": raise EvidenceBindingError("manifest inválido")
 require_hash(data["raw_output_blob_hash"], "raw_output_blob_hash")
 if raw_output is not None and store.put_evidence_blob(raw_output).evidence_hash!=data["raw_output_blob_hash"]: raise EvidenceBindingError("raw output mismatch")
 if not isinstance(data["tests"], list) or any(not isinstance(x,dict) or set(x)!={"test_id","bindings","result"} for x in data["tests"]): raise EvidenceBindingError("tests inválidos")
 return data
