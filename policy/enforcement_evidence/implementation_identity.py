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
  return data
 except Exception as exc:
  raise UntrustedImplementationIdentityError("build manifest inválido o drift") from exc
