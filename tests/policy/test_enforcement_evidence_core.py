from datetime import datetime, timezone, timedelta
import pytest

from policy.enforcement_evidence import *
from policy.enforcement_evidence.control_registry import load_control_definition
from policy.enforcement_evidence.errors import EvidenceBlobMissingError, EvidenceBlobTooLargeError, UntrustedControlDefinitionError

NOW=datetime(2026,1,1,tzinfo=timezone.utc)
def identity(store):
 b=store.put_evidence_blob(b'{"sources":[]}')
 return ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,b.evidence_hash)
def artifact(store, control="CTL.B6.GOVERNED_DISPATCH", subject=None):
 d=load_control_definition(control); i=identity(store); b=store.put_evidence_blob(b"evidence")
 a=EvidenceArtifact(EvidenceType.CONTROL_INPUT,EvidenceClass.RUNTIME_OBSERVATION,control,1,d.control_definition_hash,subject or EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT,"attempt:1"),(EvidenceBlobRef(b.evidence_hash,"input"),),EvidenceTrustDomain.JAX_RUNTIME,"runtime",i.implementation_identity_hash,NOW)
 return d,i,store.record_artifact(a)
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
 s.record_observation(o)
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
