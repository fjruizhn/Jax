"""Small authoritative lifecycle with an in-memory implementation for pure use/tests.

MariaDB deployments use the same boundary; arbitrary parsed values are never trusted.
"""
from __future__ import annotations
import weakref
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from .ids import sha256_bytes, require_hash
from .errors import EvidenceBlobMissingError, EvidenceBlobHashMismatchError, EvidenceBlobTooLargeError, EvidenceArtifactIntegrityError, EvidenceArtifactUntrustedError, ObservationIntegrityError, AssertionIntegrityError, EvidenceBindingError
from .models import (EvidenceArtifact, EnforcementObservation,
                     MAX_REFERENCED_BYTES_PER_ARTIFACT,
                     MAX_ARTIFACT_ENVELOPE_BYTES)
from .canonical import canonical_bytes
MAX_BLOB_BYTES=1024*1024
@dataclass(frozen=True)
class EvidenceBlob: evidence_hash:str; size_bytes:int; bytes:bytes
# Provenance is intentionally not represented by a token, callback, or a
# mutable public registry.  The only way an object enters these weak maps is
# an authoritative store load.  Persisting a value is deliberately not enough
# to make the caller's instance trusted.
_artifacts:dict[int,weakref.ReferenceType]={}; _observations:dict[int,weakref.ReferenceType]={}; _assertions:dict[int,weakref.ReferenceType]={}; _identities:dict[int,weakref.ReferenceType]={}
_lifecycle_write = ContextVar("jax_evidence_lifecycle_write", default=False)
@contextmanager
def _fixed_composition_write():
 """Internal execution context, not an object capability handed to callers."""
 marker=_lifecycle_write.set(True)
 try: yield
 finally: _lifecycle_write.reset(marker)
def _require_fixed_composition_write():
 if not _lifecycle_write.get(): raise EvidenceArtifactUntrustedError("only fixed lifecycle may persist final evidence")
def _loaded(reg,obj):
 reg[id(obj)]=weakref.ref(obj); return obj
def _sealed(reg,obj):
 r=reg.get(id(obj)); return r is not None and r() is obj
class EvidenceStore:
 def __init__(self): self._blobs={}; self._artifact_rows={}; self._observation_rows={}; self._assertion_rows={}; self._identity_rows={}; self._test_manifests={}
 def put_evidence_blob(self,data:bytes)->EvidenceBlob:
  if not isinstance(data,bytes): raise TypeError("bytes requeridos")
  if len(data)>MAX_BLOB_BYTES: raise EvidenceBlobTooLargeError("blob > 1 MiB")
  h=sha256_bytes(data); prior=self._blobs.get(h)
  if prior is not None and (prior.bytes!=data or prior.size_bytes!=len(data)): raise EvidenceBlobHashMismatchError("colisión/mismatch")
  blob=prior or EvidenceBlob(h,len(data),data); self._blobs[h]=blob; return blob
 def __record_identity(self, identity):
  _require_fixed_composition_write()
  h=identity.implementation_identity_hash; old=self._identity_rows.get(h)
  if old is not None and old!=identity: raise EvidenceArtifactIntegrityError("identity collision")
  self._identity_rows[h]=identity; return identity
 def load_implementation_identity(self, identity_hash):
  value=self._identity_rows.get(identity_hash)
  if value is None or value.implementation_identity_hash!=identity_hash: raise EvidenceBlobMissingError(identity_hash)
  return _loaded(_identities,value)
 def get_evidence_blob(self,evidence_hash:str)->bytes:
  require_hash(evidence_hash); blob=self._blobs.get(evidence_hash)
  if blob is None: raise EvidenceBlobMissingError(evidence_hash)
  if blob.size_bytes!=len(blob.bytes) or sha256_bytes(blob.bytes)!=evidence_hash: raise EvidenceBlobHashMismatchError(evidence_hash)
  return blob.bytes
 def __record_artifact(self,artifact:EvidenceArtifact)->EvidenceArtifact:
  _require_fixed_composition_write()
  if len(canonical_bytes(artifact.projection())) > MAX_ARTIFACT_ENVELOPE_BYTES:
   raise EvidenceArtifactIntegrityError("artifact envelope too large")
  total=0
  for ref in artifact.blob_refs:
   data=self.get_evidence_blob(ref.evidence_hash); total += len(data)
  if total > MAX_REFERENCED_BYTES_PER_ARTIFACT:
   raise EvidenceArtifactIntegrityError("artifact referenced bytes too large")
  h=artifact.artifact_hash; old=self._artifact_rows.get(h)
  if old is not None and old!=artifact: raise EvidenceArtifactIntegrityError("artifact hash conflict")
  self._artifact_rows[h]=artifact; return artifact
 def load_evidence_artifact(self,h:str)->EvidenceArtifact:
  art=self._artifact_rows.get(h)
  if art is None: raise EvidenceBlobMissingError(h)
  if art.artifact_hash!=h: raise EvidenceArtifactIntegrityError("artifact hash")
  for ref in art.blob_refs:self.get_evidence_blob(ref.evidence_hash)
  from .control_registry import load_control_definition
  definition=load_control_definition(art.control_id,art.control_version)
  if definition.control_definition_hash != art.control_definition_hash: raise EvidenceArtifactIntegrityError("artifact control binding")
  self.load_implementation_identity(art.implementation_identity_hash)
  return _loaded(_artifacts,art)
 def __record_observation(self,value:EnforcementObservation)->EnforcementObservation:
  _require_fixed_composition_write()
  for h in value.evidence_artifact_hashes:self.load_evidence_artifact(h)
  old=self._observation_rows.get(value.observation_id)
  if old is not None and old.observation_hash!=value.observation_hash: raise ObservationIntegrityError("observation collision")
  self._observation_rows[value.observation_id]=value; return value
 def load_observation(self,oid:str)->EnforcementObservation:
  v=self._observation_rows.get(oid)
  if v is None or not v.observation_hash: raise ObservationIntegrityError("observation missing/corrupt")
  from .control_registry import load_control_definition
  definition=load_control_definition(v.control_id,v.control_version)
  if definition.control_definition_hash != v.control_definition_hash: raise ObservationIntegrityError("observation control binding")
  self.load_implementation_identity(v.implementation_identity_hash)
  for h in v.evidence_artifact_hashes: self.load_evidence_artifact(h)
  return _loaded(_observations,v)
 def __record_assertion(self, value):
  _require_fixed_composition_write()
  h=value.assertion_hash; old=self._assertion_rows.get(h)
  if old is not None and old!=value: raise AssertionIntegrityError("assertion collision")
  self._assertion_rows[h]=value; return value
 def load_assertion(self,h):
  value=self._assertion_rows.get(h)
  if value is None or value.assertion_hash!=h: raise AssertionIntegrityError("assertion missing/corrupt")
  from .control_registry import load_control_definition
  definition=load_control_definition(value.control_id,value.control_version)
  if definition.control_definition_hash != value.control_definition_hash: raise AssertionIntegrityError("assertion control binding")
  self.load_implementation_identity(value.implementation_identity_hash)
  for ref in value.evidence_artifact_hashes: self.load_evidence_artifact(ref)
  for oid in value.observation_ids: self.load_observation(oid)
  return _loaded(_assertions,value)
 def observations(self): return tuple(self._observation_rows.values())
 def __ingest_test_manifest(self, manifest):
  _require_fixed_composition_write()
  key=(manifest["repository_id"],manifest["commit_sha"],manifest["implementation_identity_hash"],manifest["job_id"])
  self._test_manifests[key]=manifest
  return manifest
 def _test_manifests_for(self, identity_hash):
  return tuple(x for x in self._test_manifests.values() if x["implementation_identity_hash"]==identity_hash)

def is_trusted_evidence_artifact(v): return _sealed(_artifacts,v)
def is_trusted_observation(v): return _sealed(_observations,v)
def is_trusted_assertion(v): return _sealed(_assertions,v)
def is_trusted_implementation_identity(v): return _sealed(_identities,v)
class EvidenceStoreProvider:
 def __init__(self,store:EvidenceStore): self._store=store
 def read(self,evidence_ref:str)->bytes: return self._store.get_evidence_blob(evidence_ref)
