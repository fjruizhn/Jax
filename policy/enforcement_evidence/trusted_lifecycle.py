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
