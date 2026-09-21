"""Fixed-composition implementation identity loading."""
from __future__ import annotations
import json
from pathlib import Path
from .models import ImplementationIdentity
def load_runtime_implementation_identity(path="/etc/jax/build/implementation-identity.json"):
 data=json.loads(Path(path).read_text(encoding="utf-8"))
 return ImplementationIdentity(data["repository_id"],data["git_commit_sha"],data["git_tree_id"],data["source_state"],data["build_manifest_blob_hash"],tuple(tuple(x) for x in data.get("schema_versions",())),data.get("build_id"))
