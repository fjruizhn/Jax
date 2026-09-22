"""Fixed-composition implementation identity loading."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from .models import ImplementationIdentity, SourceState
from .errors import UntrustedImplementationIdentityError
def load_runtime_implementation_identity(path="/etc/jax/build/implementation-identity.json"):
 data=json.loads(Path(path).read_text(encoding="utf-8"))
 return implementation_identity_from_projection(data)

_DEPLOYMENT_IDENTITY_PATH = "/etc/jax/build/implementation-identity.json"
_DEPLOYMENT_REPOSITORY_ROOT = "/srv/jax"
_V1_REQUIRED_SOURCE_PATHS = frozenset({
 "policy/enforcement_evidence/controls/v1.json",
 "policy/enforcement_evidence/migrations/001_enforcement_evidence.sql",
 "policy/enforcement_evidence/control_registry.py",
 "policy/enforcement_evidence/status_engine.py",
 "policy/enforcement_evidence/trusted_lifecycle.py",
 "policy/enforcement_evidence/mariadb_store.py",
 "policy/enforcement_evidence/test_evidence.py",
 "policy/enforcement_evidence/worker_results.py",
 "policy/execution_control/authorization.py",
 "policy/execution_control/service.py",
 "policy/execution_control/storage.py",
})

class TrustedImplementationIdentityProvider:
 """Fixed deployment composition for the implementation identity.

 The provider owns both the deployment identity file and source root.  A
 request can never select either of them, nor submit an Identity instance.
 """
 def __init__(self, store):
  # These are deployment constants, not an API. Tests use the separate
  # controlled provider below; accepting a path/root here would let a caller
  # redefine the bytes whose integrity is being claimed.
  self.__store=store
 def load(self):
  value=load_runtime_implementation_identity(_DEPLOYMENT_IDENTITY_PATH)
  verify_build_manifest(self.__store,value,repository_root=_DEPLOYMENT_REPOSITORY_ROOT)
  return value
 def verify_loaded_identity(self, value):
  """Re-verify against the root captured by deployment composition.

  Status callers deliberately receive no repository-root parameter.
  """
  return verify_build_manifest(self.__store,value,repository_root=_DEPLOYMENT_REPOSITORY_ROOT)

class _ControlledTestIdentityProvider:
 """Test composition seam; deliberately not exported from package API."""
 def __init__(self, identity): self.__identity=identity
 def load(self): return self.__identity
 def verify_loaded_identity(self, value):
  # Test composition has no deployment filesystem; it is intentionally never
  # a production WRITTEN source.
  raise UntrustedImplementationIdentityError("test identity has no deployment manifest")

def implementation_identity_from_projection(data: dict) -> ImplementationIdentity:
 """Strict parser only; provenance is deliberately not established here."""
 try:
  expected={"schema_version","kind","repository_id","source_revision_kind","git_commit_sha","git_tree_id","source_state","build_manifest_blob_hash","schema_versions","build_id"}
  if set(data) != expected or data["source_revision_kind"] != "GIT": raise ValueError()
  return ImplementationIdentity(data["repository_id"], data["git_commit_sha"],
    data["git_tree_id"], SourceState(data["source_state"]),
    data["build_manifest_blob_hash"], tuple(tuple(x) for x in data["schema_versions"]),
    data["build_id"], data["schema_version"], data["kind"])
 except Exception as exc:
  raise UntrustedImplementationIdentityError("identity inválida") from exc

def verify_build_manifest(store, identity: ImplementationIdentity, *, repository_root: str) -> dict:
 """Verify the immutable manifest bytes, not a mutable version label.

 V1 deliberately accepts only a closed `{schema_version, kind, files}` form;
 every path is relative and every listed hash must match current source bytes.
 """
 try:
  data=json.loads(store.get_evidence_blob(identity.build_manifest_blob_hash))
  if set(data)!={"schema_version","kind","files"} or data["schema_version"]!="1.0" or data["kind"]!="JAX_BUILD_MANIFEST" or not isinstance(data["files"],dict) or not data["files"]: raise ValueError()
  root=Path(repository_root).resolve()
  for name,digest in data["files"].items():
   path=(root/name).resolve()
   if not str(path).startswith(str(root)+"/") or not path.is_file() or not isinstance(digest,str) or not digest.startswith("sha256:"): raise ValueError()
   if "sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError()
  if not _V1_REQUIRED_SOURCE_PATHS.issubset(data["files"]): raise ValueError()
  return data
 except Exception as exc:
  raise UntrustedImplementationIdentityError("build manifest inválido o drift") from exc
