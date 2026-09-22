"""Closed immutable values.  Their construction never establishes provenance."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from .ids import domain_hash, require_hash
from .canonical import utc_text

# These are protocol limits, rather than database implementation details.  They
# are deliberately repeated at each boundary that accepts the corresponding
# value so an in-memory test double cannot accidentally accept a value which a
# production store rejects.
MAX_BLOB_REFS_PER_ARTIFACT = 32
MAX_REFERENCED_BYTES_PER_ARTIFACT = 4 * 1024 * 1024
MAX_ARTIFACT_ENVELOPE_BYTES = 64 * 1024
MAX_OBSERVATION_ENVELOPE_BYTES = 128 * 1024
MAX_ASSERTION_ENVELOPE_BYTES = 128 * 1024
MAX_ASSERTION_SUBJECTS = 1000
MAX_ASSERTION_OBSERVATIONS = 1000

class EvidenceType(str, Enum):
 SOURCE_SNAPSHOT="SOURCE_SNAPSHOT"; IMPLEMENTATION_MANIFEST="IMPLEMENTATION_MANIFEST"; DECISION_RECORD="DECISION_RECORD"; EXECUTION_REQUEST="EXECUTION_REQUEST"; EXECUTION_AUTHORIZATION="EXECUTION_AUTHORIZATION"; EXECUTION_RECORD="EXECUTION_RECORD"; HUMAN_APPROVAL="HUMAN_APPROVAL"; DRY_RUN_RESULT="DRY_RUN_RESULT"; EXECUTION_EVENT_SNAPSHOT="EXECUTION_EVENT_SNAPSHOT"; CONTROL_INPUT="CONTROL_INPUT"; CONFIG_SNAPSHOT="CONFIG_SNAPSHOT"; TEST_RESULT="TEST_RESULT"; CI_RUN_REFERENCE="CI_RUN_REFERENCE"; DB_SCHEMA_OBSERVATION="DB_SCHEMA_OBSERVATION"; WORKER_RESULT="WORKER_RESULT"
class EvidenceClass(str, Enum): INTERNAL_STATE="INTERNAL_STATE"; RUNTIME_OBSERVATION="RUNTIME_OBSERVATION"; TEST_EVIDENCE="TEST_EVIDENCE"; EXTERNAL_CITED="EXTERNAL_CITED"
class EvidenceTrustDomain(str, Enum): JAX_CONTROL_REGISTRY="JAX_CONTROL_REGISTRY"; JAX_DECISION_STORE="JAX_DECISION_STORE"; JAX_EXECUTION_STORE="JAX_EXECUTION_STORE"; JAX_RUNTIME="JAX_RUNTIME"; JAX_DB_INTROSPECTION="JAX_DB_INTROSPECTION"; JAX_OPERATIONAL_APPROVER="JAX_OPERATIONAL_APPROVER"; LOCAL_TEST_RUNNER="LOCAL_TEST_RUNNER"; EXTERNAL_GITHUB_CI="EXTERNAL_GITHUB_CI"
class EvidenceSubjectType(str, Enum): CONTROL_IMPLEMENTATION="CONTROL_IMPLEMENTATION"; POLICY_RULE="POLICY_RULE"; DECISION="DECISION"; AUTHORIZATION="AUTHORIZATION"; EXECUTION="EXECUTION"; OPERATION_ATTEMPT="OPERATION_ATTEMPT"; DATABASE_SCHEMA="DATABASE_SCHEMA"; CI_COMMIT="CI_COMMIT"
class ObservationOutcome(str, Enum): SATISFIED="SATISFIED"; DENIED="DENIED"; FAILED="FAILED"; ERROR="ERROR"
class ClaimLevel(str, Enum): WRITTEN="WRITTEN"; TESTED="TESTED"; ENFORCED="ENFORCED"
class AssertionVerdict(str, Enum): SUPPORTED="SUPPORTED"; FAILED="FAILED"; INSUFFICIENT_EVIDENCE="INSUFFICIENT_EVIDENCE"; NOT_OBSERVED="NOT_OBSERVED"; STALE="STALE"; UNVERIFIABLE="UNVERIFIABLE"
class SourceState(str, Enum): CLEAN="CLEAN"; DIRTY="DIRTY"
class ClaimEnvironment(str, Enum): CI="CI"; LOCAL_TEST="LOCAL_TEST"; SANDBOX_RUNTIME="SANDBOX_RUNTIME"
class Coverage(str, Enum): OBSERVED_SUBJECTS_ONLY="OBSERVED_SUBJECTS_ONLY"; POINT_IN_TIME_CONFIGURATION="POINT_IN_TIME_CONFIGURATION"
@dataclass(frozen=True)
class EvidenceSubject:
 subject_type: EvidenceSubjectType; identity: str
 def __post_init__(self):
  if not isinstance(self.subject_type,EvidenceSubjectType) or not isinstance(self.identity,str) or not self.identity: raise ValueError("subject inválido")
 def projection(self): return {"subject_type":self.subject_type.value,"identity":self.identity}
@dataclass(frozen=True)
class EvidenceBlobRef:
 evidence_hash:str; role:str; media_type:str="application/json"; encoding:str="utf-8"
 def __post_init__(self): require_hash(self.evidence_hash,"evidence_hash")
@dataclass(frozen=True)
class ImplementationIdentity:
 repository_id:str; git_commit_sha:str; git_tree_id:str; source_state:SourceState; build_manifest_blob_hash:str; schema_versions:tuple[tuple[str,str],...]=(); build_id:str|None=None; schema_version:str="1.0"; kind:str="JAX_IMPLEMENTATION_IDENTITY"
 def __post_init__(self):
  if (self.schema_version,self.kind)!=("1.0","JAX_IMPLEMENTATION_IDENTITY") or len(self.git_commit_sha)!=40: raise ValueError("ImplementationIdentity inválida")
  require_hash(self.build_manifest_blob_hash,"build_manifest_blob_hash")
 @property
 def implementation_identity_hash(self): return domain_hash("JAX-IMPLEMENTATION-IDENTITY/1",self.projection())
 def projection(self): return {"schema_version":self.schema_version,"kind":self.kind,"repository_id":self.repository_id,"source_revision_kind":"GIT","git_commit_sha":self.git_commit_sha,"git_tree_id":self.git_tree_id,"source_state":self.source_state.value,"build_manifest_blob_hash":self.build_manifest_blob_hash,"schema_versions":[list(x) for x in self.schema_versions],"build_id":self.build_id}
@dataclass(frozen=True)
class ClaimScope:
 environment:ClaimEnvironment; deployment_id:str|None=None; database_scope_id:str|None=None; coverage:Coverage=Coverage.OBSERVED_SUBJECTS_ONLY
@dataclass(frozen=True)
class EvidenceArtifact:
 evidence_type:EvidenceType; evidence_class:EvidenceClass; control_id:str; control_version:int; control_definition_hash:str; subject:EvidenceSubject; blob_refs:tuple[EvidenceBlobRef,...]; trust_domain:EvidenceTrustDomain; producer:str; implementation_identity_hash:str; produced_at_utc:datetime; decision_id:str|None=None; execution_id:str|None=None; policy_authority_binding:tuple[tuple[str,str],...]=(); schema_version:str="1.0"; kind:str="JAX_EVIDENCE_ARTIFACT"
 def __post_init__(self):
  if (self.schema_version,self.kind)!=("1.0","JAX_EVIDENCE_ARTIFACT") or not self.blob_refs or len(self.blob_refs)>MAX_BLOB_REFS_PER_ARTIFACT: raise ValueError("EvidenceArtifact inválido")
  require_hash(self.control_definition_hash); require_hash(self.implementation_identity_hash)
 @property
 def artifact_hash(self): return domain_hash("JAX-EVIDENCE-ARTIFACT/1",self.projection())
 def projection(self): return {"schema_version":self.schema_version,"kind":self.kind,"evidence_type":self.evidence_type.value,"evidence_class":self.evidence_class.value,"control_id":self.control_id,"control_version":self.control_version,"control_definition_hash":self.control_definition_hash,"subject":self.subject.projection(),"blob_refs":[{"evidence_hash":x.evidence_hash,"role":x.role,"media_type":x.media_type,"encoding":x.encoding} for x in self.blob_refs],"trust_domain":self.trust_domain.value,"producer":self.producer,"implementation_identity_hash":self.implementation_identity_hash,"decision_id":self.decision_id,"execution_id":self.execution_id,"produced_at_utc":utc_text(self.produced_at_utc),"policy_authority_binding":[list(x) for x in self.policy_authority_binding]}
@dataclass(frozen=True)
class EnforcementObservation:
 observation_id:str; control_id:str; control_version:int; control_definition_hash:str; subject:EvidenceSubject; implementation_identity_hash:str; outcome:ObservationOutcome; reason_code:str; occurred_at_utc:datetime; scope:ClaimScope; evidence_artifact_hashes:tuple[str,...]; decision_id:str|None=None; execution_id:str|None=None; schema_version:str="1.0"; kind:str="JAX_ENFORCEMENT_OBSERVATION"
 def __post_init__(self):
  require_hash(self.control_definition_hash); require_hash(self.implementation_identity_hash); [require_hash(x) for x in self.evidence_artifact_hashes]
  # An observation is an envelope, not an arbitrary log transport.
  from .canonical import canonical_bytes
  if len(canonical_bytes(self.projection())) > MAX_OBSERVATION_ENVELOPE_BYTES: raise ValueError("observation envelope too large")
 @property
 def observation_hash(self): return domain_hash("JAX-ENFORCEMENT-OBSERVATION/1",self.projection())
 def projection(self): return {"schema_version":self.schema_version,"kind":self.kind,"observation_id":self.observation_id,"control_id":self.control_id,"control_version":self.control_version,"control_definition_hash":self.control_definition_hash,"subject":self.subject.projection(),"implementation_identity_hash":self.implementation_identity_hash,"outcome":self.outcome.value,"reason_code":self.reason_code,"occurred_at_utc":utc_text(self.occurred_at_utc),"scope":{"environment":self.scope.environment.value,"deployment_id":self.scope.deployment_id,"database_scope_id":self.scope.database_scope_id,"coverage":self.scope.coverage.value},"evidence_artifact_hashes":list(self.evidence_artifact_hashes),"decision_id":self.decision_id,"execution_id":self.execution_id}
@dataclass(frozen=True)
class EnforcementAssertion:
 control_id:str; control_version:int; control_definition_hash:str; claim_level:ClaimLevel; verdict:AssertionVerdict; implementation_identity_hash:str; scope:ClaimScope; subject_set:tuple[EvidenceSubject,...]; evidence_artifact_hashes:tuple[str,...]; observation_ids:tuple[str,...]; as_of_utc:datetime; evidence_window_start_utc:datetime; evidence_window_end_utc:datetime; trust_domains_used:tuple[EvidenceTrustDomain,...]=(); reason_codes:tuple[str,...]=(); schema_version:str="1.0"; kind:str="JAX_ENFORCEMENT_ASSERTION"
 def __post_init__(self):
  require_hash(self.control_definition_hash); require_hash(self.implementation_identity_hash)
  [require_hash(x) for x in self.evidence_artifact_hashes]
  if len(self.subject_set)>MAX_ASSERTION_SUBJECTS or len(self.observation_ids)>MAX_ASSERTION_OBSERVATIONS: raise ValueError("assertion relationship bound exceeded")
  from .canonical import canonical_bytes
  if len(canonical_bytes(self.projection())) > MAX_ASSERTION_ENVELOPE_BYTES: raise ValueError("assertion envelope too large")
 @property
 def assertion_hash(self): return domain_hash("JAX-ENFORCEMENT-ASSERTION/1",self.projection())
 def projection(self): return {"schema_version":self.schema_version,"kind":self.kind,"control_id":self.control_id,"control_version":self.control_version,"control_definition_hash":self.control_definition_hash,"claim_level":self.claim_level.value,"verdict":self.verdict.value,"implementation_identity_hash":self.implementation_identity_hash,"scope":{"environment":self.scope.environment.value,"deployment_id":self.scope.deployment_id,"database_scope_id":self.scope.database_scope_id,"coverage":self.scope.coverage.value},"subject_set":[x.projection() for x in self.subject_set],"evidence_artifact_hashes":list(self.evidence_artifact_hashes),"observation_ids":list(self.observation_ids),"as_of_utc":utc_text(self.as_of_utc),"evidence_window_start_utc":utc_text(self.evidence_window_start_utc),"evidence_window_end_utc":utc_text(self.evidence_window_end_utc),"trust_domains_used":[x.value for x in self.trust_domains_used],"reason_codes":list(self.reason_codes)}

@dataclass(frozen=True)
class ControlStatusView:
 """Ephemeral result of an authoritative, read-only B7 derivation.

 This is deliberately not an ``EnforcementAssertion``: it has no assertion
 identity/hash and never represents persisted historical evidence.
 """
 control_id:str; control_version:int; claim_level:ClaimLevel; verdict:AssertionVerdict
 implementation_identity_hash:str; scope:ClaimScope; subjects:tuple[EvidenceSubject,...]
 as_of_utc:datetime; evidence_window_start_utc:datetime; evidence_window_end_utc:datetime
 reason_codes:tuple[str,...]; trust_domains_used:tuple[EvidenceTrustDomain,...]
 supporting_artifact_hashes:tuple[str,...]; supporting_observation_ids:tuple[str,...]
 observed_at:datetime; classification:str="AUTHORITATIVE_READONLY_DERIVATION"; persisted:bool=False
 def __post_init__(self):
  if self.persisted is not False or self.classification != "AUTHORITATIVE_READONLY_DERIVATION": raise ValueError("ControlStatusView must be read-only")
  require_hash(self.implementation_identity_hash)
