"""Canonical, closed test evidence manifest validation without GitHub I/O."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from .ids import require_hash
from .errors import EvidenceBindingError
from .control_registry import load_control_definition

_TEST_CONTROL_MAP = {
 "tests/policy/test_enforcement_evidence_core.py::test_blob_hash_and_dedup_and_missing": ("CTL.B6.GOVERNED_DISPATCH",),
 "tests/policy/test_enforcement_evidence_core.py::test_artifact_load_is_only_trusted_lifecycle": ("CTL.B5.DECISION_PROVENANCE",),
 "tests/policy/test_enforcement_evidence_core.py::test_trust_composition_has_no_reusable_writer_or_caller_selected_ci_context": ("CTL.B6.AUTHORIZATION_PROVENANCE",),
 "tests/policy/test_enforcement_evidence_core.py::test_database_profile_rederives_scope_and_rejects_relabelled_payload": ("CTL.B6.ONE_DECISION_ONE_EXECUTION",),
 "tests/policy/test_enforcement_evidence_core.py::test_real_las_manos_startup_composes_required_b7_recorder": ("CTL.B6.GOVERNED_DISPATCH",),
 "tests/policy/test_enforcement_evidence_core.py::test_cross_control_and_subject_never_support": ("CTL.B6.SANDBOX_ONLY",),
 "tests/policy/test_enforcement_evidence_core.py::test_failed_dominates_and_staleness": ("CTL.B6.HUMAN_APPROVAL_BINDING",),
 "tests/policy/test_enforcement_evidence_core.py::test_untrusted_definition_rejected": ("CTL.B5.DECISION_PROVENANCE",),
 "tests/policy/test_enforcement_evidence_core.py::test_fixed_runtime_recorder_persists_bounded_denial_without_prompt": ("CTL.B6.KILL_SWITCH",),
 "tests/policy/test_enforcement_evidence_core.py::test_no_evidence_never_supports_enforced_and_identity_drift_isolated": ("CTL.B6.AUTHORIZATION_EXPIRY",),
 "tests/policy/test_enforcement_evidence_core.py::test_tested_is_not_inferred_from_runtime_observation": ("CTL.B6.TIMEOUT_CEILING",),
 "tests/policy/test_enforcement_evidence_core.py::test_one_runtime_observation_never_mints_enforced": ("CTL.B6.GOVERNED_DISPATCH",),
 "tests/policy/test_enforcement_evidence_core.py::test_canonical_ci_manifest_checks_raw_bytes_and_closed_shape": ("CTL.B6.GOVERNED_DISPATCH",),
 "tests/policy/test_enforcement_evidence_core.py::test_artifact_aggregate_and_envelope_bounds_are_fail_closed": ("CTL.B6.GOVERNED_DISPATCH",),
 "tests/policy/test_enforcement_evidence_core.py::test_assertion_relationship_bounds_and_worker_result_ingestion": ("CTL.B6.GOVERNED_DISPATCH",),
 "tests/policy/test_enforcement_evidence_mariadb_api.py::test_repeatable_read_is_explicit_store_boundary": ("CTL.B6.ONE_DECISION_ONE_EXECUTION",),
 "tests/policy/test_worker_result_execution_binding.py::test_worker_result_requires_exact_authoritative_execution_and_job": ("CTL.B6.GOVERNED_DISPATCH",),
}
def ingest_test_evidence_manifest(store, manifest_bytes:bytes, raw_output:bytes|None=None, *, repository_id=None, commit_sha=None, implementation_identity_hash=None):
 data=json.loads(manifest_bytes)
 required={"schema_version","kind","provider","repository_id","commit_sha","implementation_identity_hash","workflow","run_id","job_id","environment","started_at_utc","completed_at_utc","tests","counts","raw_output_blob_hash"}
 if set(data)!=required or data["kind"]!="JAX_TEST_EVIDENCE_MANIFEST": raise EvidenceBindingError("manifest inválido")
 if data["provider"] not in {"github-actions","local-test-runner"}: raise EvidenceBindingError("provider no confiable")
 if not isinstance(data["commit_sha"], str) or len(data["commit_sha"]) != 40 or any(c not in "0123456789abcdef" for c in data["commit_sha"].lower()): raise EvidenceBindingError("commit inválido")
 if repository_id is not None and data["repository_id"] != repository_id: raise EvidenceBindingError("repositorio incorrecto")
 if commit_sha is not None and data["commit_sha"] != commit_sha: raise EvidenceBindingError("commit incorrecto")
 if implementation_identity_hash is not None and data["implementation_identity_hash"] != implementation_identity_hash: raise EvidenceBindingError("identity incorrecta")
 require_hash(data["raw_output_blob_hash"], "raw_output_blob_hash")
 if raw_output is not None and store.put_evidence_blob(raw_output).evidence_hash!=data["raw_output_blob_hash"]: raise EvidenceBindingError("raw output mismatch")
 if not isinstance(data["tests"], list) or not data["tests"] or any(not isinstance(x,dict) or set(x)!={"test_id","bindings","result"} for x in data["tests"]): raise EvidenceBindingError("tests inválidos")
 if any(x["result"] not in {"PASSED","FAILED","ERROR","SKIPPED"} or x["test_id"] not in _TEST_CONTROL_MAP for x in data["tests"]): raise EvidenceBindingError("resultado/test_id no confiable")
 if len({x["test_id"] for x in data["tests"]}) != len(data["tests"]) or {x["test_id"] for x in data["tests"]} != set(_TEST_CONTROL_MAP): raise EvidenceBindingError("cobertura de tests incompleta")
 for item in data["tests"]:
  expected=_TEST_CONTROL_MAP[item["test_id"]]
  bound=tuple((x.get("control_id"),x.get("control_version")) for x in item["bindings"] if isinstance(x,dict))
  if bound != tuple((cid,1) for cid in expected): raise EvidenceBindingError("binding de control inválido")
  for cid in expected: load_control_definition(cid,1)
 if not isinstance(data["counts"],dict) or set(data["counts"]) != {"passed","failed","error","skipped","total"}: raise EvidenceBindingError("counts inválidos")
 expected_counts={"passed":sum(x["result"]=="PASSED" for x in data["tests"]),"failed":sum(x["result"]=="FAILED" for x in data["tests"]),"error":sum(x["result"]=="ERROR" for x in data["tests"]),"skipped":sum(x["result"]=="SKIPPED" for x in data["tests"]),"total":len(data["tests"])}
 if data["counts"] != expected_counts: raise EvidenceBindingError("counts inválidos")
 try:
  started=datetime.fromisoformat(data["started_at_utc"].replace("Z","+00:00")); completed=datetime.fromisoformat(data["completed_at_utc"].replace("Z","+00:00"))
  if started.tzinfo is None or completed.tzinfo is None or completed < started or completed > datetime.now(timezone.utc): raise ValueError()
 except Exception as exc: raise EvidenceBindingError("timestamps inválidos") from exc
 return data

class TestEvidenceIngester:
 """Fixed CI composition. Parsed manifest bytes remain untrusted until here."""
 def __init__(self, lifecycle):
  """Bind CI facts from the fixed GitHub Actions composition only.

  Repository/commit/provider are not constructor arguments so ordinary
  application code cannot turn an arbitrary JSON manifest into TESTED proof.
  """
  repository_id=os.environ.get("GITHUB_REPOSITORY")
  commit_sha=os.environ.get("GITHUB_SHA")
  if not repository_id or not commit_sha:
   raise EvidenceBindingError("trusted GitHub Actions context required")
  self._store=lifecycle._store; self._repository_id=repository_id; self._commit_sha=commit_sha
  self._lifecycle=lifecycle
  self._identity=lifecycle._identity.implementation_identity_hash; self._workflow="policy.yml"; self._environment="CI"; self._provider="github-actions"
 def ingest(self, manifest_bytes, raw_output=None):
  value=ingest_test_evidence_manifest(self._store, manifest_bytes, raw_output, repository_id=self._repository_id, commit_sha=self._commit_sha, implementation_identity_hash=self._identity)
  if value["provider"] != self._provider or value["workflow"] != self._workflow or value["environment"] != self._environment: raise EvidenceBindingError("CI context incorrecto")
  # The ingester owns the fixed CI context; parsing a manifest never grants a
  # caller a persistence/provenance capability.
  return self._lifecycle._EvidenceLifecycleService__ingest_test_manifest(value)
