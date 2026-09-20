"""External trusted approver-root loader.  The root is never sourced from DB."""
from __future__ import annotations
import base64, json
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from ..errors import HumanApprovalBindingError

DEFAULT_TRUSTED_APPROVERS_PATH = "/etc/jax/execution/trusted-approvers.json"

def load_trusted_approver(key_id: str, *, path: str = DEFAULT_TRUSTED_APPROVERS_PATH) -> Ed25519PublicKey:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = payload["approvers"][key_id]
        if entry["actor_id"] != "human:fernando": raise ValueError("actor")
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(entry["public_key"], validate=True))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HumanApprovalBindingError("trusted approver root inválido") from exc
