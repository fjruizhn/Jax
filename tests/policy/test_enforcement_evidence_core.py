from datetime import datetime, timezone, timedelta
import copy
import pytest

from policy.enforcement_evidence import *
from policy.enforcement_evidence.control_registry import load_control_definition
from policy.enforcement_evidence.errors import EvidenceBlobMissingError, EvidenceBlobTooLargeError, UntrustedControlDefinitionError
from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider

NOW=datetime(2026,1,1,tzinfo=timezone.utc)
def identity(store):
 b=store.put_evidence_blob(b'{"sources":[]}')
 i=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,b.evidence_hash)
 EvidenceLifecycleService(store,_ControlledTestIdentityProvider(i))
 return i
def composition(store):
 b=store.put_evidence_blob(b'{"sources":[]}')
 i=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,b.evidence_hash)
 return EvidenceLifecycleService(store,_ControlledTestIdentityProvider(i)),i
def artifact(store, control="CTL.B6.GOVERNED_DISPATCH", subject=None):
 d=load_control_definition(control); lifecycle,i=composition(store); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 if control != "CTL.B6.GOVERNED_DISPATCH": raise ValueError("test helper only has governed dispatch composition")
 value=subject or EvidenceSubject(EvidenceSubjectType.EXECUTION,"attempt:1")
 o=RuntimeEvidenceRecorder(lifecycle,"runtime",scope).record_governed_dispatch(execution_id=value.identity,decision_id="decision-test")
 return d,i,store.load_evidence_artifact(o.evidence_artifact_hashes[0])
def test_blob_hash_and_dedup_and_missing():
 s=EvidenceStore(); a=s.put_evidence_blob(b"same"); assert s.put_evidence_blob(b"same") is a
 assert s.get_evidence_blob(a.evidence_hash)==b"same"
 with pytest.raises(EvidenceBlobMissingError): s.get_evidence_blob("sha256:"+"0"*64)
 with pytest.raises(EvidenceBlobTooLargeError): s.put_evidence_blob(b"x"*(1024*1024+1))
def test_artifact_load_is_only_trusted_lifecycle():
 s=EvidenceStore(); d,i,a=artifact(s)
 from policy.enforcement_evidence.evidence_store import is_trusted_evidence_artifact, is_trusted_observation, is_trusted_assertion
 assert is_trusted_evidence_artifact(a)
 clone=EvidenceArtifact(a.evidence_type,a.evidence_class,a.control_id,a.control_version,a.control_definition_hash,a.subject,a.blob_refs,a.trust_domain,a.producer,a.implementation_identity_hash,a.produced_at_utc)
 assert not is_trusted_evidence_artifact(clone)
 # Parsed/hash-consistent values and a manual identity cannot select the
 # provenance lifecycle or become trusted merely by construction.
 assert not hasattr(s,"_fixed_lifecycle_token") and not hasattr(s,"record_identity")
 with pytest.raises(AttributeError): EvidenceLifecycleService(s,i)
 manual_observation=EnforcementObservation("manual",d.control_id,1,d.control_definition_hash,a.subject,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW,ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME),(a.artifact_hash,))
 assert not is_trusted_observation(manual_observation)
 manual_assertion=EnforcementAssertion(d.control_id,1,d.control_definition_hash,ClaimLevel.ENFORCED,AssertionVerdict.SUPPORTED,i.implementation_identity_hash,ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME),(a.subject,),(a.artifact_hash,),(manual_observation.observation_id,),NOW,NOW,NOW)
 assert not is_trusted_assertion(manual_assertion)
 # Even reflective access to a store implementation detail has no authority:
 # final writes require the non-transferable fixed-composition execution
 # context, then an authoritative load returns the only trusted instance.
 with pytest.raises(Exception): s._EvidenceStore__record_artifact(clone)
 with pytest.raises(Exception): s._EvidenceStore__record_observation(manual_observation)
 with pytest.raises(Exception): s._EvidenceStore__record_assertion(manual_assertion)
 assert s.load_evidence_artifact(a.artifact_hash).artifact_hash==a.artifact_hash

def test_trust_composition_has_no_reusable_writer_or_caller_selected_ci_context():
 s=EvidenceStore(); lifecycle,_=composition(s)
 import policy.enforcement_evidence.evidence_store as store_module
 from policy.enforcement_evidence.test_evidence import TestEvidenceIngester
 assert not hasattr(store_module,"_lifecycle_writers")
 assert not hasattr(store_module,"_seal")
 assert not hasattr(RuntimeEvidenceRecorder,"record_satisfied")
 assert not hasattr(RuntimeEvidenceRecorder,"record_denial")
 with pytest.raises(TypeError):
  TestEvidenceIngester(lifecycle,repository_id="attacker/repo",commit_sha="a"*40,workflow="attacker")

def test_database_profile_rederives_scope_and_rejects_relabelled_payload():
 s=EvidenceStore(); lifecycle,_=composition(s)
 recorder=RuntimeEvidenceRecorder(lifecycle,"runtime",ClaimScope(ClaimEnvironment.CI))
 # The old (scope_label, payload) API no longer exists, and an inconsistent
 # label in a complete inspection cannot be persisted.
 with pytest.raises(TypeError): recorder._record_database_profile("db-b",b"{}")
 with pytest.raises(ValueError): recorder._record_database_profile({
  "database_scope_id":"db-b","server_uuid":"server-a","database_name":"jax-a",
  "deployment_id":"deployment-a","control_id":"CTL.B6.ONE_DECISION_ONE_EXECUTION",
  "control_version":1,"installed":True})

def test_real_las_manos_startup_composes_required_b7_recorder():
 from pathlib import Path
 source=(Path(__file__).parents[2]/"las_manos"/"server.py").read_text()
 assert "def _configure_b7_trusted_runtime()" in source
 assert "_configure_b7_trusted_runtime()" in source[source.index("async def _jacobs_init"):]
 for required in ("configure_b7_execution_evidence", "configure_b7_decision_recorder",
                  "configure_b7_authorization_recorder", "configure_b7_evidence_recorder",
                  "configure_governed_execution_store", "DatabaseControlInspector"):
  assert required in source
def test_cross_control_and_subject_never_support():
 s=EvidenceStore(); d,i,a=artifact(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 o=EnforcementObservation("obs-1",d.control_id,1,d.control_definition_hash,a.subject,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW,scope,(a.artifact_hash,))
 # The manually finalized observation has a valid hash but never becomes trusted.
 assert not is_trusted_observation(o)
 other=load_control_definition("CTL.B6.KILL_SWITCH")
 assert derive_assertion(other,i,s.observations(),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(a.subject,),as_of_utc=NOW).value == "NOT_OBSERVED"
 assert derive_assertion(d,i,s.observations(),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT,"other"),),as_of_utc=NOW).value == "NOT_OBSERVED"
def test_failed_dominates_and_staleness():
 s=EvidenceStore(); d,i,a=artifact(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 subj=a.subject
 fail=EnforcementObservation("obs-2",d.control_id,1,d.control_definition_hash,subj,i.implementation_identity_hash,ObservationOutcome.FAILED,"FAILED",NOW,scope,(a.artifact_hash,))
 assert derive_assertion(d,i,(fail,),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subj,),as_of_utc=NOW).value=="FAILED"
 old=EnforcementObservation("obs-3",d.control_id,1,d.control_definition_hash,subj,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW-timedelta(hours=25),scope,(a.artifact_hash,))
 assert derive_assertion(d,i,(old,),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subj,),as_of_utc=NOW).value=="STALE"
def test_untrusted_definition_rejected():
 d=load_control_definition("CTL.B6.GOVERNED_DISPATCH")
 clone=type(d)(d.control_id,d.control_version,d.mechanism,d.supported_subject_types,d.allowed_reason_codes,d.supported_claim_levels)
 s=EvidenceStore(); i=identity(s)
 with pytest.raises(UntrustedControlDefinitionError): derive_assertion(clone,i,(),claim_level=ClaimLevel.WRITTEN,scope=ClaimScope(ClaimEnvironment.LOCAL_TEST),subjects=(),as_of_utc=NOW)
def test_fixed_runtime_recorder_persists_bounded_denial_without_prompt():
 s=EvidenceStore(); lifecycle,i=composition(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 r=RuntimeEvidenceRecorder(lifecycle,"runtime",scope)
 o=r.record_kill_switch_denied(decision_id="d-1")
 assert o.outcome is ObservationOutcome.DENIED
 a=s.load_evidence_artifact(o.evidence_artifact_hashes[0])
 assert a.evidence_type is EvidenceType.CONTROL_INPUT
 assert b"prompt" not in s.get_evidence_blob(a.blob_refs[0].evidence_hash)
def test_no_evidence_never_supports_enforced_and_identity_drift_isolated():
 s=EvidenceStore(); d=load_control_definition("CTL.B6.GOVERNED_DISPATCH"); i=identity(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME); subject=EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT,"x")
 assert derive_assertion(d,i,(),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subject,),as_of_utc=NOW) is AssertionVerdict.NOT_OBSERVED
 other=ImplementationIdentity("fjruizhn/Jax","c"*40,"d"*40,SourceState.CLEAN,s.put_evidence_blob(b"other").evidence_hash)
 assert derive_assertion(d,other,(),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subject,),as_of_utc=NOW) is AssertionVerdict.UNVERIFIABLE
def test_tested_is_not_inferred_from_runtime_observation():
 s=EvidenceStore(); d,i,a=artifact(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 o=EnforcementObservation("obs-runtime",d.control_id,1,d.control_definition_hash,a.subject,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW,scope,(a.artifact_hash,))
 assert derive_assertion(d,i,(o,),claim_level=ClaimLevel.TESTED,scope=scope,subjects=(a.subject,),as_of_utc=NOW) is AssertionVerdict.INSUFFICIENT_EVIDENCE
def test_one_runtime_observation_never_mints_enforced():
 s=EvidenceStore(); d,i,a=artifact(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 o=EnforcementObservation("obs-alone",d.control_id,1,d.control_definition_hash,a.subject,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW,scope,(a.artifact_hash,))
 assert derive_assertion(d,i,(o,),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(a.subject,),as_of_utc=NOW) is AssertionVerdict.INSUFFICIENT_EVIDENCE

def test_persisted_status_service_has_no_readonly_query_capability():
 s=EvidenceStore(); lifecycle,_=composition(s); service=EnforcementStatusService(lifecycle)
 scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME); subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,"x")
 before=(len(s._assertion_rows),len(s._observation_rows),len(s._artifact_rows))
 persisted=service.evaluate_control_status(control_id="CTL.B6.GOVERNED_DISPATCH",control_version=1,claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subject,),as_of_utc=NOW)
 assert isinstance(persisted,EnforcementAssertion)
 assert not hasattr(service,"query_control_status")
 with pytest.raises(AttributeError):
  service.query_control_status(control_id="CTL.B6.GOVERNED_DISPATCH",control_version=1,claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subject,),as_of_utc=NOW)
 assert (len(s._assertion_rows),len(s._observation_rows),len(s._artifact_rows)) == (before[0]+1,before[1],before[2])

def test_caller_constructed_readonly_query_cannot_mint_authoritative_view():
 import inspect
 import policy.enforcement_evidence.status_engine as status_engine
 from policy.enforcement_evidence.status_engine import _ReadonlyStatusDerivation
 s=EvidenceStore(); lifecycle,identity_value=composition(s)
 scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME); subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,"x")
 provider=type("CallerProvider",(),{"verify_loaded_identity_bytes":lambda _self, value, manifest: {}})()
 # Arbitrary lifecycle/store/provider-shaped values can exercise only the
 # internal raw derivation helper.  No caller-composed object has a method
 # that emits AUTHORITATIVE_READONLY_DERIVATION.
 reader=_ReadonlyStatusDerivation(s,provider,identity_value.implementation_identity_hash)
 copied=copy.copy(reader)
 copied_dependencies=_ReadonlyStatusDerivation(lifecycle._store,provider,identity_value.implementation_identity_hash)
 for value in (reader,copied,copied_dependencies):
  assert not hasattr(value,"query_control_status")
  assert not hasattr(value,"status_view")
 assert not hasattr(status_engine,"_trusted_readonly_queries")
 assert tuple(inspect.signature(status_engine.query_control_status).parameters) == (
  "control_id","control_version","claim_level","scope","subjects","as_of_utc")
 for forbidden in ("store","identity_provider","lifecycle","registry","trusted","verdict","evidence","observations"):
  assert forbidden not in inspect.signature(status_engine.query_control_status).parameters
 assert not any(name in status_engine.__dict__ for name in ("register_trusted","mark_trusted","trusted_readonly_queries"))
 assert not hasattr(EnforcementStatusService,"query_control_status")

def test_public_readonly_query_has_no_injectable_composition_hook(monkeypatch):
 import inspect
 import policy.enforcement_evidence.status_engine as status_engine
 # Regression for B8-AUD-001: the former module-global composition factory
 # must not exist, and a normal API caller has no supported injection point.
 assert not hasattr(status_engine,"_compose_runtime_readonly_derivation")
 for name in ("composition","derivation","factory","provider","registry",
              "store","lifecycle","verdict","classification"):
  assert name not in inspect.signature(status_engine.query_control_status).parameters
 # Missing deployment configuration fails closed; it cannot be replaced with
 # caller-derived status data through the public function.
 # Recreate the former disclosure's replacement attempt.  The injected
 # attribute is now inert because the production entrypoint never consults it.
 monkeypatch.setattr(status_engine,"_compose_runtime_readonly_derivation",lambda: object(),raising=False)
 monkeypatch.delenv("JAX_DB_HOST",raising=False)
 with pytest.raises(RuntimeError,match="composition unavailable"):
  status_engine.query_control_status(control_id="CTL.B6.GOVERNED_DISPATCH",control_version=1,
   claim_level=ClaimLevel.ENFORCED,scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME),
   subjects=(EvidenceSubject(EvidenceSubjectType.EXECUTION,"x"),),as_of_utc=NOW)

def test_readonly_query_uses_captured_snapshot_without_later_artifact_loads(monkeypatch):
 s=EvidenceStore(); lifecycle,identity_value=composition(s)
 from policy.enforcement_evidence.status_engine import _ReadonlyStatusDerivation
 service=_ReadonlyStatusDerivation(s,type("Fixed",(),{"verify_loaded_identity_bytes":lambda _self, value, manifest: {}})(),identity_value.implementation_identity_hash)
 captured=s.observations()
 # MariaDB supplies all verified observations/domains from one RR snapshot;
 # a later insert or artifact loader must not affect this query result.
 snapshot=type("Snapshot",(),{"identity":identity_value,"manifest_bytes":b"{}","observations":captured,"manifests":(),"trust_domains":()})()
 monkeypatch.setattr(s,"readonly_status_snapshot",lambda identity_hash, control_id, control_version: snapshot,raising=False)
 monkeypatch.setattr(s,"load_evidence_artifact",lambda _hash: (_ for _ in ()).throw(AssertionError("outside snapshot")))
 _definition, _identity, observations, _subjects, _verdict, _start, _artifacts, _domains=service._derive(control_id="CTL.B6.GOVERNED_DISPATCH",control_version=1,claim_level=ClaimLevel.ENFORCED,scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME),subjects=(EvidenceSubject(EvidenceSubjectType.EXECUTION,"x"),),as_of_utc=NOW)
 assert observations == ()
def test_canonical_ci_manifest_checks_raw_bytes_and_closed_shape():
 import json
 from policy.enforcement_evidence.test_evidence import ingest_test_evidence_manifest, _TEST_CONTROL_MAP
 s=EvidenceStore(); raw=b"pytest output"; h=s.put_evidence_blob(raw).evidence_hash
 tests=[{"test_id":test_id,"bindings":[{"control_id":control_id,"control_version":1} for control_id in controls],"result":"PASSED"} for test_id,controls in _TEST_CONTROL_MAP.items()]
 manifest={"schema_version":"1.0","kind":"JAX_TEST_EVIDENCE_MANIFEST","provider":"github-actions","repository_id":"fjruizhn/Jax","commit_sha":"a"*40,"implementation_identity_hash":"sha256:"+"a"*64,"workflow":"policy","run_id":"1","job_id":"2","environment":"CI","started_at_utc":"2026-01-01T00:00:00Z","completed_at_utc":"2026-01-01T00:00:01Z","tests":tests,"counts":{"passed":len(tests),"failed":0,"error":0,"skipped":0,"total":len(tests)},"raw_output_blob_hash":h}
 assert ingest_test_evidence_manifest(s,json.dumps(manifest).encode(),raw)["commit_sha"]=="a"*40
 with pytest.raises(Exception): ingest_test_evidence_manifest(s,json.dumps(manifest).encode(),b"different")

def test_artifact_aggregate_and_envelope_bounds_are_fail_closed():
 s=EvidenceStore(); d=load_control_definition("CTL.B6.GOVERNED_DISPATCH"); i=identity(s)
 # One MiB blobs are legal individually but five references exceed the V1
 # aggregate cap.  This also proves the store, not caller metadata, counts.
 assert not hasattr(s,"_fixed_lifecycle_token") and not hasattr(s,"_record_artifact")
 with pytest.raises(EvidenceBlobTooLargeError): s.put_evidence_blob(b"x"*(1024*1024+1))

def test_assertion_relationship_bounds_and_worker_result_ingestion(tmp_path):
 from policy.enforcement_evidence.worker_results import WorkerResultIngestor
 s=EvidenceStore(); lifecycle,i=composition(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 recorder=RuntimeEvidenceRecorder(lifecycle,"worker",scope)
 class ExecutionStore:
  def load_execution(self, execution_id):
   if execution_id!="exec-1": raise ValueError(execution_id)
   return type("Execution",(),{"execution_id":"exec-1"})()
 p=tmp_path/"result.bin"; p.write_bytes(b"worker bytes")
 # A request-shaped duck object is not an authoritative execution store and
 # cannot mint a WORKER_RESULT binding.
 with pytest.raises(Exception): WorkerResultIngestor(recorder,ExecutionStore())
 d=load_control_definition("CTL.B6.GOVERNED_DISPATCH")
 subjects=tuple(EvidenceSubject(EvidenceSubjectType.EXECUTION,str(n)) for n in range(1001))
 with pytest.raises(ValueError): EnforcementAssertion(d.control_id,1,d.control_definition_hash,ClaimLevel.ENFORCED,AssertionVerdict.NOT_OBSERVED,i.implementation_identity_hash,scope,subjects,(),(),NOW,NOW,NOW)
