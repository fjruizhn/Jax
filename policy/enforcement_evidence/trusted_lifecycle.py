"""Composition-owned recorder; caller requests cannot select its trust roots."""
class RuntimeEvidenceRecorder:
 def __init__(self, store, implementation_identity, producer, scope):
  self._store=store; self._identity=implementation_identity; self._producer=producer; self._scope=scope
  self._token=store._fixed_lifecycle_token()
  store.record_identity(implementation_identity)
 def record_artifact(self,value):
  if value.implementation_identity_hash != self._identity.implementation_identity_hash or value.producer != self._producer: raise ValueError("untrusted artifact composition")
  return self._store._record_artifact(value, _token=self._token)
 def record_observation(self,value):
  if value.implementation_identity_hash != self._identity.implementation_identity_hash or value.scope != self._scope: raise ValueError("untrusted observation composition")
  return self._store._record_observation(value, _token=self._token)
 def record_denial(self, *, control_id, reason_code, decision_id=None):
  """Composition hook. Concrete deployments provide a typed artifact draft.

  Keeping this operation on the startup-owned recorder prevents a web caller
  from selecting a store, registry, producer, identity, or verdict.
  """
  from .control_registry import load_control_definition
  definition=load_control_definition(control_id)
  if reason_code not in definition.allowed_reason_codes: raise ValueError("reason code no declarado")
  from datetime import datetime, timezone
  import uuid
  from .models import (EvidenceArtifact, EvidenceType, EvidenceClass, EvidenceSubject, EvidenceSubjectType,
                       EvidenceBlobRef, EvidenceTrustDomain, EnforcementObservation,
                       ObservationOutcome)
  # Deliberately bounded projection: it names the rejected control/decision,
  # never serializes a prompt or arbitrary request context.
  raw=("control="+control_id+";reason="+reason_code+";decision="+(decision_id or "")).encode("utf-8")
  blob=self._store.put_evidence_blob(raw); now=datetime.now(timezone.utc)
  subject=EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT, "denial:"+(decision_id or str(uuid.uuid4())))
  artifact=EvidenceArtifact(EvidenceType.CONTROL_INPUT,EvidenceClass.RUNTIME_OBSERVATION,control_id,definition.control_version,definition.control_definition_hash,subject,(EvidenceBlobRef(blob.evidence_hash,"bounded_input","text/plain","utf-8"),),EvidenceTrustDomain.JAX_RUNTIME,self._producer,self._identity.implementation_identity_hash,now,decision_id=decision_id)
  trusted=self.record_artifact(artifact)
  oid=str(uuid.uuid7()) if hasattr(uuid,"uuid7") else str(uuid.uuid4())
  observation=EnforcementObservation(oid,control_id,definition.control_version,definition.control_definition_hash,subject,self._identity.implementation_identity_hash,ObservationOutcome.DENIED,reason_code,now,self._scope,(trusted.artifact_hash,),decision_id=decision_id)
  return self.record_observation(observation)
