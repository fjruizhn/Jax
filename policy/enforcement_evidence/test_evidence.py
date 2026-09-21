"""Canonical, closed test evidence manifest validation without GitHub I/O."""
from __future__ import annotations
import json
from .ids import require_hash
from .errors import EvidenceBindingError
from .control_registry import load_control_definition

_TEST_CONTROL_MAP = {
 "tests.policy.test_enforcement_evidence_core": ("CTL.B6.GOVERNED_DISPATCH",),
 "tests.policy.test_enforcement_evidence_mariadb_api": ("CTL.B6.ONE_DECISION_ONE_EXECUTION",),
}
def ingest_test_evidence_manifest(store, manifest_bytes:bytes, raw_output:bytes|None=None):
 data=json.loads(manifest_bytes)
 required={"schema_version","kind","provider","repository_id","commit_sha","implementation_identity_hash","workflow","run_id","job_id","environment","started_at_utc","completed_at_utc","tests","counts","raw_output_blob_hash"}
 if set(data)!=required or data["kind"]!="JAX_TEST_EVIDENCE_MANIFEST": raise EvidenceBindingError("manifest inválido")
 require_hash(data["raw_output_blob_hash"], "raw_output_blob_hash")
 if raw_output is not None and store.put_evidence_blob(raw_output).evidence_hash!=data["raw_output_blob_hash"]: raise EvidenceBindingError("raw output mismatch")
 if not isinstance(data["tests"], list) or not data["tests"] or any(not isinstance(x,dict) or set(x)!={"test_id","bindings","result"} for x in data["tests"]): raise EvidenceBindingError("tests inválidos")
 if any(x["result"] not in {"PASSED","FAILED","ERROR","SKIPPED"} or x["test_id"] not in _TEST_CONTROL_MAP for x in data["tests"]): raise EvidenceBindingError("resultado/test_id no confiable")
 for item in data["tests"]:
  expected=_TEST_CONTROL_MAP[item["test_id"]]
  bound=tuple((x.get("control_id"),x.get("control_version")) for x in item["bindings"] if isinstance(x,dict))
  if bound != tuple((cid,1) for cid in expected): raise EvidenceBindingError("binding de control inválido")
  for cid in expected: load_control_definition(cid,1)
 if not isinstance(data["counts"],dict) or data["counts"].get("passed") != sum(x["result"]=="PASSED" for x in data["tests"]): raise EvidenceBindingError("counts inválidos")
 return data
