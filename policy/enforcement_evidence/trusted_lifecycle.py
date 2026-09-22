"""Composition-owned evidence lifecycle; final values never come from requests."""
from __future__ import annotations
from datetime import datetime, timezone
from dataclasses import replace
import uuid
from .control_registry import load_control_definition
from .models import (EvidenceArtifact, EvidenceBlobRef, EvidenceClass, EvidenceSubject,
 EvidenceSubjectType, EvidenceTrustDomain, EvidenceType, EnforcementObservation, ObservationOutcome)

class EvidenceLifecycleService:
 def __init__(self, store, identity_provider):
  # A duck-typed provider is not a trust boundary: only the fixed production
  # provider (or the deliberately non-exported test-composition provider) may
  # enter the lifecycle.
  from .implementation_identity import TrustedImplementationIdentityProvider, _ControlledTestIdentityProvider
  if not isinstance(identity_provider, (TrustedImplementationIdentityProvider, _ControlledTestIdentityProvider)):
   raise AttributeError("identity provider must be fixed composition")
  self._store=store; self.__identity_provider=identity_provider; self._identity=identity_provider.load()
  self.__mariadb=self._is_mariadb(store)
  self.__record_identity(self._identity)
  # The in-process lifecycle uses the same object identity for its
  # authoritative row.  Establish provenance only by re-loading it, never by
  # accepting the object handed to the service as already trusted.
  self._identity=self._store.load_implementation_identity(self._identity.implementation_identity_hash)
 def _verify_composed_identity_manifest(self):
  return self.__identity_provider.verify_loaded_identity(self._identity)
 def _is_mariadb(self, store):
  try:
   from .mariadb_store import MariaDBEvidenceStore
   return isinstance(store, MariaDBEvidenceStore)
  except ImportError: return False
 def __record_identity(self, identity):
  from .evidence_store import _fixed_composition_write
  with _fixed_composition_write():
   if self.__mariadb: return self._store._MariaDBEvidenceStore__record_identity(identity)
   return self._store._EvidenceStore__record_identity(identity)
 def __record_artifact(self, artifact):
  from .evidence_store import _fixed_composition_write
  with _fixed_composition_write():
   if self.__mariadb: return self._store._MariaDBEvidenceStore__record_artifact(artifact)
   return self._store._EvidenceStore__record_artifact(artifact)
 def __record_observation(self, observation):
  from .evidence_store import _fixed_composition_write
  with _fixed_composition_write():
   if self.__mariadb: return self._store._MariaDBEvidenceStore__record_observation(observation)
   return self._store._EvidenceStore__record_observation(observation)
 def __record_assertion(self, assertion):
  from .evidence_store import _fixed_composition_write
  with _fixed_composition_write():
   if self.__mariadb: return self._store._MariaDBEvidenceStore__record_assertion(assertion)
   return self._store._EvidenceStore__record_assertion(assertion)
 def __persist_assertion(self, assertion):
  self.__record_assertion(assertion); return self._store.load_assertion(assertion.assertion_hash)
 def __persist_artifact(self, artifact):
  self.__record_artifact(artifact); return self._store.load_evidence_artifact(artifact.artifact_hash)
 def __persist_observation(self, observation):
  self.__record_observation(observation); return self._store.load_observation(observation.observation_id)
 def __ingest_test_manifest(self, manifest):
  from .evidence_store import _fixed_composition_write
  with _fixed_composition_write():
   if self.__mariadb:
    return self._store._MariaDBEvidenceStore__ingest_test_manifest(manifest)
   return self._store._EvidenceStore__ingest_test_manifest(manifest)
 def __write_observation_in_transaction(self, cursor, observation):
  if not self.__mariadb: raise RuntimeError("transactional evidence requires MariaDB lifecycle")
  from .evidence_store import _fixed_composition_write
  with _fixed_composition_write():
   return self._store._MariaDBEvidenceStore__write_observation_in_transaction(cursor, observation)

class RuntimeEvidenceRecorder:
 """Typed runtime emitter; it deliberately exposes no generic final-object writer."""
 def __init__(self, lifecycle: EvidenceLifecycleService, producer, scope):
  self.__lifecycle=lifecycle; self._store=lifecycle._store; self._identity=lifecycle._identity; self._producer=producer; self._scope=scope
 def _emit(self, *, control_id, subject, outcome, reason_code, payload, decision_id=None, execution_id=None, evidence_type=EvidenceType.CONTROL_INPUT, trust_domain=EvidenceTrustDomain.JAX_RUNTIME, scope=None):
  definition=load_control_definition(control_id)
  if reason_code not in definition.allowed_reason_codes: raise ValueError("reason code no declarado")
  now=datetime.now(timezone.utc); blob=self._store.put_evidence_blob(payload)
  artifact=EvidenceArtifact(evidence_type,EvidenceClass.RUNTIME_OBSERVATION,definition.control_id,definition.control_version,definition.control_definition_hash,subject,(EvidenceBlobRef(blob.evidence_hash,"bounded_input","application/octet-stream","binary"),),trust_domain,self._producer,self._identity.implementation_identity_hash,now,decision_id=decision_id,execution_id=execution_id)
  artifact=self.__lifecycle._EvidenceLifecycleService__persist_artifact(artifact)
  observation=EnforcementObservation(str(uuid.uuid4()),definition.control_id,definition.control_version,definition.control_definition_hash,subject,self._identity.implementation_identity_hash,outcome,reason_code,now,scope or self._scope,(artifact.artifact_hash,),decision_id=decision_id,execution_id=execution_id)
  return self.__lifecycle._EvidenceLifecycleService__persist_observation(observation)
 def _record_denial(self, *, control_id, reason_code="DENIED", decision_id=None):
  return self._emit(control_id=control_id,subject=EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT,"denial:"+(decision_id or str(uuid.uuid4()))),outcome=ObservationOutcome.DENIED,reason_code=reason_code,payload=("control="+control_id+";reason="+reason_code+";decision="+(decision_id or "")).encode(),decision_id=decision_id)
 def record_decision_provenance_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B5.DECISION_PROVENANCE",decision_id=decision_id)
 def record_decision_provenance(self, *, decision_id):
  return self._emit(control_id="CTL.B5.DECISION_PROVENANCE",subject=EvidenceSubject(EvidenceSubjectType.DECISION,decision_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("decision="+decision_id).encode(),decision_id=decision_id)
 def record_authorization_provenance_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.AUTHORIZATION_PROVENANCE",decision_id=decision_id)
 def record_one_decision_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.ONE_DECISION_ONE_EXECUTION",decision_id=decision_id)
 def record_human_approval_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.HUMAN_APPROVAL_BINDING",decision_id=decision_id)
 def record_governed_dispatch_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.GOVERNED_DISPATCH",decision_id=decision_id)
 def record_kill_switch_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.KILL_SWITCH",decision_id=decision_id)
 def record_authorization_expired(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.AUTHORIZATION_EXPIRY",decision_id=decision_id)
 def record_sandbox_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.SANDBOX_ONLY",decision_id=decision_id)
 def record_timeout_denied(self, *, decision_id=None): return self._record_denial(control_id="CTL.B6.TIMEOUT_CEILING",decision_id=decision_id)
 def record_authorization_provenance(self, *, authorization_id, decision_id):
  return self._emit(control_id="CTL.B6.AUTHORIZATION_PROVENANCE",subject=EvidenceSubject(EvidenceSubjectType.AUTHORIZATION,authorization_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("authorization="+authorization_id).encode(),decision_id=decision_id)
 def record_sandbox_validated(self, *, authorization_id, decision_id):
  return self._emit(control_id="CTL.B6.SANDBOX_ONLY",subject=EvidenceSubject(EvidenceSubjectType.AUTHORIZATION,authorization_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("authorization="+authorization_id).encode(),decision_id=decision_id)
 def record_timeout_validated(self, *, authorization_id, decision_id):
  return self._emit(control_id="CTL.B6.TIMEOUT_CEILING",subject=EvidenceSubject(EvidenceSubjectType.AUTHORIZATION,authorization_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("authorization="+authorization_id).encode(),decision_id=decision_id)
 def record_kill_switch_clear(self, *, execution_id, decision_id):
  return self._emit(control_id="CTL.B6.KILL_SWITCH",subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,execution_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("execution="+execution_id).encode(),decision_id=decision_id,execution_id=execution_id)
 def record_authorization_current(self, *, execution_id, decision_id):
  return self._emit(control_id="CTL.B6.AUTHORIZATION_EXPIRY",subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,execution_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("execution="+execution_id).encode(),decision_id=decision_id,execution_id=execution_id)
 def record_execution_created(self, *, execution_id, decision_id):
  return self._emit(control_id="CTL.B6.ONE_DECISION_ONE_EXECUTION",subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,execution_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("execution="+execution_id).encode(),decision_id=decision_id,execution_id=execution_id)
 def record_human_approval_bound(self, *, execution_id, decision_id):
  return self._emit(control_id="CTL.B6.HUMAN_APPROVAL_BINDING",subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,execution_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("execution="+execution_id).encode(),decision_id=decision_id,execution_id=execution_id)
 def record_governed_dispatch(self, *, execution_id, decision_id):
  return self._emit(control_id="CTL.B6.GOVERNED_DISPATCH",subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,execution_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=("execution="+execution_id).encode(),decision_id=decision_id,execution_id=execution_id)
 def _record_database_profile(self, installed):
  """Record only a complete live-inspector result, never a scope label.

  ``DatabaseControlInspector`` derives this identity from its fixed endpoint;
  a request cannot relabel evidence from one database as another.
  """
  import hashlib, json
  required={"database_scope_id","server_id","hostname","database_name","deployment_id",
            "control_id","control_version","installed"}
  if not isinstance(installed, dict) or not required.issubset(installed):
   raise ValueError("complete live database inspection required")
  derived="dbscope:sha256:"+hashlib.sha256(
   (str(installed["deployment_id"])+"|"+str(installed["server_id"])+"|"+str(installed["hostname"])+"|"+str(installed["database_name"])).encode()).hexdigest()
  if installed["database_scope_id"] != derived or installed["control_id"] != "CTL.B6.ONE_DECISION_ONE_EXECUTION" or installed["control_version"] != 1 or installed["installed"] is not True:
   raise ValueError("database inspection identity mismatch")
  payload=json.dumps(installed,sort_keys=True,separators=(",",":"),default=str).encode()
  return self._emit(control_id="CTL.B6.ONE_DECISION_ONE_EXECUTION",subject=EvidenceSubject(EvidenceSubjectType.DATABASE_SCHEMA,derived),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=payload,trust_domain=EvidenceTrustDomain.JAX_DB_INTROSPECTION,evidence_type=EvidenceType.DB_SCHEMA_OBSERVATION,scope=replace(self._scope,database_scope_id=derived))
 def _record_worker_result(self, execution_id, payload, control_id):
  return self._emit(control_id=control_id,subject=EvidenceSubject(EvidenceSubjectType.EXECUTION,execution_id),outcome=ObservationOutcome.SATISFIED,reason_code="SATISFIED",payload=payload,execution_id=execution_id,evidence_type=EvidenceType.WORKER_RESULT)
 def _write_transaction_observation(self, cursor, *, control_id, execution_id, decision_id, occurred_at_utc):
  definition=load_control_definition(control_id)
  observation=EnforcementObservation(str(uuid.uuid4()),definition.control_id,definition.control_version,definition.control_definition_hash,EvidenceSubject(EvidenceSubjectType.EXECUTION,execution_id),self._identity.implementation_identity_hash,ObservationOutcome.SATISFIED,"SATISFIED",occurred_at_utc,self._scope,(),decision_id=decision_id,execution_id=execution_id)
  return self.__lifecycle._EvidenceLifecycleService__write_observation_in_transaction(cursor, observation)
