import base64
import json
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.execution_control.human_approval import HumanApprovalArtifact
from policy.execution_control.adapters.trusted_approver import load_trusted_approver
from policy.execution_control.errors import HumanApprovalBindingError
def test_human_approval_artifact_is_closed_value(): assert HumanApprovalArtifact.__dataclass_params__.frozen


def test_trusted_approver_root_accepts_only_configured_actor_key(tmp_path):
    trusted = Ed25519PrivateKey.generate().public_key()
    root = tmp_path / "trusted-approvers.json"
    root.write_text(json.dumps({"approvers": {"fernando-1": {
        "actor_id": "human:fernando",
        "public_key": base64.b64encode(trusted.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode(),
    }}}))
    assert load_trusted_approver("human:fernando", "fernando-1", path=str(root)).public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw) == trusted.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    with pytest.raises(HumanApprovalBindingError):
        load_trusted_approver("human:fernando", "attacker-key", path=str(root))
    with pytest.raises(HumanApprovalBindingError):
        load_trusted_approver("human:attacker", "fernando-1", path=str(root))
