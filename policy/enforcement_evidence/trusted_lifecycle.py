"""Composition-owned recorder; caller requests cannot select its trust roots."""
class RuntimeEvidenceRecorder:
 def __init__(self, store, implementation_identity, producer, scope):
  self._store=store; self._identity=implementation_identity; self._producer=producer; self._scope=scope
  store.record_identity(implementation_identity)
 def record_artifact(self,value):
  if value.implementation_identity_hash != self._identity.implementation_identity_hash or value.producer != self._producer: raise ValueError("untrusted artifact composition")
  return self._store.record_artifact(value)
 def record_observation(self,value):
  if value.implementation_identity_hash != self._identity.implementation_identity_hash or value.scope != self._scope: raise ValueError("untrusted observation composition")
  return self._store.record_observation(value)
 def record_denial(self, *, control_id, reason_code, decision_id=None):
  """Composition hook. Concrete deployments provide a typed artifact draft.

  Keeping this operation on the startup-owned recorder prevents a web caller
  from selecting a store, registry, producer, identity, or verdict.
  """
  from .control_registry import load_control_definition
  definition=load_control_definition(control_id)
  if reason_code not in definition.allowed_reason_codes: raise ValueError("reason code no declarado")
  # The recorder intentionally requires the deployment adapter to supply an
  # operation-attempt artifact; a bare exception is not asserted as evidence.
  raise RuntimeError("DENIAL_EVIDENCE_UNAVAILABLE: operation-attempt artifact required")
