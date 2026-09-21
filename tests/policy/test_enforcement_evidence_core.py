from datetime import datetime, timezone, timedelta
import pytest

from policy.enforcement_evidence import *
from policy.enforcement_evidence.control_registry import load_control_definition
from policy.enforcement_evidence.errors import EvidenceBlobMissingError, EvidenceBlobTooLargeError, UntrustedControlDefinitionError

NOW=datetime(2026,1,1,tzinfo=timezone.utc)
def identity(store):
 b=store.put_evidence_blob(b'{"sources":[]}')
 return store.record_identity(ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,b.evidence_hash))
def artifact(store, control="CTL.B6.GOVERNED_DISPATCH", subject=None):
 d=load_control_definition(control); i=identity(store); b=store.put_evidence_blob(b"evidence")
 a=EvidenceArtifact(EvidenceType.CONTROL_INPUT,EvidenceClass.RUNTIME_OBSERVATION,control,1,d.control_definition_hash,subject or EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT,"attempt:1"),(EvidenceBlobRef(b.evidence_hash,"input"),),EvidenceTrustDomain.JAX_RUNTIME,"runtime",i.implementation_identity_hash,NOW)
 return d,i,store._record_artifact(a, _token=store._fixed_lifecycle_token())
def test_blob_hash_and_dedup_and_missing():
 s=EvidenceStore(); a=s.put_evidence_blob(b"same"); assert s.put_evidence_blob(b"same") is a
 assert s.get_evidence_blob(a.evidence_hash)==b"same"
 with pytest.raises(EvidenceBlobMissingError): s.get_evidence_blob("sha256:"+"0"*64)
 with pytest.raises(EvidenceBlobTooLargeError): s.put_evidence_blob(b"x"*(1024*1024+1))
def test_artifact_load_is_only_trusted_lifecycle():
 s=EvidenceStore(); d,i,a=artifact(s)
 from policy.enforcement_evidence.evidence_store import is_trusted_evidence_artifact
 assert is_trusted_evidence_artifact(a)
 clone=EvidenceArtifact(a.evidence_type,a.evidence_class,a.control_id,a.control_version,a.control_definition_hash,a.subject,a.blob_refs,a.trust_domain,a.producer,a.implementation_identity_hash,a.produced_at_utc)
 assert not is_trusted_evidence_artifact(clone)
 assert s.load_evidence_artifact(a.artifact_hash).artifact_hash==a.artifact_hash
def test_cross_control_and_subject_never_support():
 s=EvidenceStore(); d,i,a=artifact(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 o=EnforcementObservation("obs-1",d.control_id,1,d.control_definition_hash,a.subject,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW,scope,(a.artifact_hash,))
 s._record_observation(o, _token=s._fixed_lifecycle_token())
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
 from policy.enforcement_evidence.trusted_lifecycle import RuntimeEvidenceRecorder
 s=EvidenceStore(); i=identity(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 r=RuntimeEvidenceRecorder(s,i,"runtime",scope)
 o=r.record_denial(control_id="CTL.B6.KILL_SWITCH",reason_code="DENIED",decision_id="d-1")
 assert o.outcome is ObservationOutcome.DENIED
 a=s.load_evidence_artifact(o.evidence_artifact_hashes[0])
 assert a.evidence_type is EvidenceType.CONTROL_INPUT
 assert b"prompt" not in s.get_evidence_blob(a.blob_refs[0].evidence_hash)
def test_no_evidence_never_supports_enforced_and_identity_drift_isolated():
 s=EvidenceStore(); d=load_control_definition("CTL.B6.GOVERNED_DISPATCH"); i=identity(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME); subject=EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT,"x")
 assert derive_assertion(d,i,(),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subject,),as_of_utc=NOW) is AssertionVerdict.NOT_OBSERVED
 other=s.record_identity(ImplementationIdentity("fjruizhn/Jax","c"*40,"d"*40,SourceState.CLEAN,s.put_evidence_blob(b"other").evidence_hash))
 assert derive_assertion(d,other,(),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(subject,),as_of_utc=NOW) is AssertionVerdict.NOT_OBSERVED
def test_tested_is_not_inferred_from_runtime_observation():
 s=EvidenceStore(); d,i,a=artifact(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 o=EnforcementObservation("obs-runtime",d.control_id,1,d.control_definition_hash,a.subject,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW,scope,(a.artifact_hash,))
 assert derive_assertion(d,i,(o,),claim_level=ClaimLevel.TESTED,scope=scope,subjects=(a.subject,),as_of_utc=NOW) is AssertionVerdict.INSUFFICIENT_EVIDENCE
def test_one_runtime_observation_never_mints_enforced():
 s=EvidenceStore(); d,i,a=artifact(s); scope=ClaimScope(ClaimEnvironment.SANDBOX_RUNTIME)
 o=EnforcementObservation("obs-alone",d.control_id,1,d.control_definition_hash,a.subject,i.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",NOW,scope,(a.artifact_hash,))
 assert derive_assertion(d,i,(o,),claim_level=ClaimLevel.ENFORCED,scope=scope,subjects=(a.subject,),as_of_utc=NOW) is AssertionVerdict.INSUFFICIENT_EVIDENCE
def test_canonical_ci_manifest_checks_raw_bytes_and_closed_shape():
 import json
 from policy.enforcement_evidence.test_evidence import ingest_test_evidence_manifest
 s=EvidenceStore(); raw=b"pytest output"; h=s.put_evidence_blob(raw).evidence_hash
 manifest={"schema_version":"1.0","kind":"JAX_TEST_EVIDENCE_MANIFEST","provider":"github-actions","repository_id":"fjruizhn/Jax","commit_sha":"a"*40,"implementation_identity_hash":"sha256:"+"a"*64,"workflow":"policy","run_id":"1","job_id":"2","environment":"CI","started_at_utc":"2026-01-01T00:00:00Z","completed_at_utc":"2026-01-01T00:00:01Z","tests":[{"test_id":"tests.policy.test_enforcement_evidence_core","bindings":[{"control_id":"CTL.B6.GOVERNED_DISPATCH","control_version":1}],"result":"PASSED"}],"counts":{"passed":1},"raw_output_blob_hash":h}
 assert ingest_test_evidence_manifest(s,json.dumps(manifest).encode(),raw)["commit_sha"]=="a"*40
 with pytest.raises(Exception): ingest_test_evidence_manifest(s,json.dumps(manifest).encode(),b"different")
