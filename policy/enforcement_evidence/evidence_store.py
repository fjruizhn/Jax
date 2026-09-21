"""Small authoritative lifecycle with an in-memory implementation for pure use/tests.

MariaDB deployments use the same boundary; arbitrary parsed values are never trusted.
"""
from __future__ import annotations
import weakref
from dataclasses import dataclass
from .ids import sha256_bytes, require_hash
from .errors import EvidenceBlobMissingError, EvidenceBlobHashMismatchError, EvidenceBlobTooLargeError, EvidenceArtifactIntegrityError, EvidenceArtifactUntrustedError, ObservationIntegrityError, AssertionIntegrityError
from .models import EvidenceArtifact, EnforcementObservation
MAX_BLOB_BYTES=1024*1024
@dataclass(frozen=True)
class EvidenceBlob: evidence_hash:str; size_bytes:int; bytes:bytes
_artifacts:dict[int,weakref.ReferenceType]={}; _observations:dict[int,weakref.ReferenceType]={}; _assertions:dict[int,weakref.ReferenceType]={}; _identities:dict[int,weakref.ReferenceType]={}
def _seal(reg,obj):
 reg[id(obj)]=weakref.ref(obj); return obj
def _sealed(reg,obj):
 r=reg.get(id(obj)); return r is not None and r() is obj
class EvidenceStore:
 def __init__(self): self._blobs={}; self._artifact_rows={}; self._observation_rows={}; self._assertion_rows={}; self._identity_rows={}; self.__lifecycle_token=object()
 def _fixed_lifecycle_token(self): return self.__lifecycle_token
 def put_evidence_blob(self,data:bytes)->EvidenceBlob:
  if not isinstance(data,bytes): raise TypeError("bytes requeridos")
  if len(data)>MAX_BLOB_BYTES: raise EvidenceBlobTooLargeError("blob > 1 MiB")
  h=sha256_bytes(data); prior=self._blobs.get(h)
  if prior is not None and (prior.bytes!=data or prior.size_bytes!=len(data)): raise EvidenceBlobHashMismatchError("colisión/mismatch")
  blob=prior or EvidenceBlob(h,len(data),data); self._blobs[h]=blob; return blob
 def record_identity(self, identity):
  h=identity.implementation_identity_hash; old=self._identity_rows.get(h)
  if old is not None and old!=identity: raise EvidenceArtifactIntegrityError("identity collision")
  self._identity_rows[h]=identity; return _seal(_identities, identity)
 def get_evidence_blob(self,evidence_hash:str)->bytes:
  require_hash(evidence_hash); blob=self._blobs.get(evidence_hash)
  if blob is None: raise EvidenceBlobMissingError(evidence_hash)
  if blob.size_bytes!=len(blob.bytes) or sha256_bytes(blob.bytes)!=evidence_hash: raise EvidenceBlobHashMismatchError(evidence_hash)
  return blob.bytes
 def _record_artifact(self,artifact:EvidenceArtifact, *, _token)->EvidenceArtifact:
  if _token is not self.__lifecycle_token: raise EvidenceArtifactUntrustedError("fixed lifecycle required")
  for ref in artifact.blob_refs: self.get_evidence_blob(ref.evidence_hash)
  h=artifact.artifact_hash; old=self._artifact_rows.get(h)
  if old is not None and old!=artifact: raise EvidenceArtifactIntegrityError("artifact hash conflict")
  self._artifact_rows[h]=artifact; return _seal(_artifacts,artifact)
 def load_evidence_artifact(self,h:str)->EvidenceArtifact:
  art=self._artifact_rows.get(h)
  if art is None: raise EvidenceBlobMissingError(h)
  if art.artifact_hash!=h: raise EvidenceArtifactIntegrityError("artifact hash")
  for ref in art.blob_refs:self.get_evidence_blob(ref.evidence_hash)
  return _seal(_artifacts,art)
 def _record_observation(self,value:EnforcementObservation, *, _token)->EnforcementObservation:
  if _token is not self.__lifecycle_token: raise ObservationIntegrityError("fixed lifecycle required")
  for h in value.evidence_artifact_hashes:self.load_evidence_artifact(h)
  old=self._observation_rows.get(value.observation_id)
  if old is not None and old.observation_hash!=value.observation_hash: raise ObservationIntegrityError("observation collision")
  self._observation_rows[value.observation_id]=value; return _seal(_observations,value)
 def load_observation(self,oid:str)->EnforcementObservation:
  v=self._observation_rows.get(oid)
  if v is None or not v.observation_hash: raise ObservationIntegrityError("observation missing/corrupt")
  return _seal(_observations,v)
 def _record_assertion(self, value, *, _token):
  if _token is not self.__lifecycle_token: raise AssertionIntegrityError("fixed lifecycle required")
  h=value.assertion_hash; old=self._assertion_rows.get(h)
  if old is not None and old!=value: raise AssertionIntegrityError("assertion collision")
  self._assertion_rows[h]=value; return _seal(_assertions,value)
 def load_assertion(self,h):
  value=self._assertion_rows.get(h)
  if value is None or value.assertion_hash!=h: raise AssertionIntegrityError("assertion missing/corrupt")
  return _seal(_assertions,value)
 def observations(self): return tuple(self._observation_rows.values())
def is_trusted_evidence_artifact(v): return _sealed(_artifacts,v)
def is_trusted_observation(v): return _sealed(_observations,v)
def is_trusted_assertion(v): return _sealed(_assertions,v)
class EvidenceStoreProvider:
 def __init__(self,store:EvidenceStore): self._store=store
 def read(self,evidence_ref:str)->bytes: return self._store.get_evidence_blob(evidence_ref)
