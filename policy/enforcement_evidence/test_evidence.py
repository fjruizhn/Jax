"""Canonical, closed test evidence manifest validation without GitHub I/O."""
from __future__ import annotations
import json
from .errors import EvidenceBindingError
def ingest_test_evidence_manifest(store, manifest_bytes:bytes, raw_output:bytes|None=None):
 data=json.loads(manifest_bytes)
 required={"schema_version","kind","provider","repository_id","commit_sha","implementation_identity_hash","workflow","run_id","job_id","environment","started_at_utc","completed_at_utc","tests","counts","raw_output_blob_hash"}
 if set(data)!=required or data["kind"]!="JAX_TEST_EVIDENCE_MANIFEST": raise EvidenceBindingError("manifest inválido")
 if raw_output is not None and store.put_evidence_blob(raw_output)!=data["raw_output_blob_hash"]: raise EvidenceBindingError("raw output mismatch")
 return data
