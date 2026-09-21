"""Operation-bound signed human approvals; separate from constitutional keys."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from .canonical import canonical_bytes, utc_text
from .errors import HumanApprovalBindingError, HumanApprovalExpiredError
from .ids import new_human_approval_id

@dataclass(frozen=True)
class HumanApprovalArtifact:
    human_approval_id: str; approver_actor_id: str; approver_key_id: str
    decision_id: str; decision_record_hash: str; authorization_id: str
    execution_authorization_hash: str; execution_request_hash: str; capability: str
    caller: str; motor: str; environment: str; target: str
    issued_at_utc: datetime; expires_at_utc: datetime; signature: str; human_approval_hash: str
    def projection(self) -> dict:
        return {"schema_version":"1.0","kind":"JAX_EXECUTION_HUMAN_APPROVAL","human_approval_id":self.human_approval_id,
            "approver_actor_id":self.approver_actor_id,"approver_key_id":self.approver_key_id,"decision_id":self.decision_id,
            "decision_record_hash":self.decision_record_hash,"authorization_id":self.authorization_id,
            "execution_authorization_hash":self.execution_authorization_hash,"execution_request_hash":self.execution_request_hash,
            "capability":self.capability,"caller":self.caller,"motor":self.motor,"environment":self.environment,
            "target":self.target,"issued_at_utc":utc_text(self.issued_at_utc),"expires_at_utc":utc_text(self.expires_at_utc)}

def _message(projection: dict) -> bytes:
    return b"JAX-EXECUTION-HUMAN-APPROVAL/1\0" + canonical_bytes(projection)

def issue_human_approval(authorization, *, approver_actor_id: str, approver_key_id: str,
                         private_key: Ed25519PrivateKey, issued_at_utc: datetime) -> HumanApprovalArtifact:
    if approver_actor_id != "human:fernando": raise HumanApprovalBindingError("approver no permitido")
    issued = issued_at_utc.astimezone(timezone.utc)
    expires = min(authorization.expires_at_utc, issued + (authorization.expires_at_utc - issued))
    provisional = HumanApprovalArtifact(new_human_approval_id(), approver_actor_id, approver_key_id,
        authorization.decision_id, authorization.decision_record_hash, authorization.authorization_id,
        authorization.execution_authorization_hash, authorization.execution_request.execution_request_hash,
        authorization.execution_request.capability, authorization.execution_request.authenticated_caller_id,
        authorization.execution_request.motor, authorization.execution_request.environment.value,
        authorization.execution_request.target_value, issued, expires, "", "")
    signature = base64.b64encode(private_key.sign(_message(provisional.projection()))).decode("ascii")
    digest = "sha256:" + hashlib.sha256(_message(provisional.projection()) + b"\0" + signature.encode()).hexdigest()
    return HumanApprovalArtifact(provisional.human_approval_id, provisional.approver_actor_id,
        provisional.approver_key_id, provisional.decision_id, provisional.decision_record_hash,
        provisional.authorization_id, provisional.execution_authorization_hash,
        provisional.execution_request_hash, provisional.capability, provisional.caller,
        provisional.motor, provisional.environment, provisional.target,
        provisional.issued_at_utc, provisional.expires_at_utc, signature, digest)

def verify_human_approval(approval: HumanApprovalArtifact, authorization, public_key: Ed25519PublicKey, *, now_utc: datetime) -> None:
    if not isinstance(approval, HumanApprovalArtifact) or now_utc.astimezone(timezone.utc) > approval.expires_at_utc:
        raise HumanApprovalExpiredError("approval vencido")
    bound = (approval.decision_id, approval.decision_record_hash, approval.authorization_id,
      approval.execution_authorization_hash, approval.execution_request_hash, approval.capability,
      approval.caller, approval.motor, approval.environment, approval.target)
    req = authorization.execution_request
    expected = (authorization.decision_id, authorization.decision_record_hash, authorization.authorization_id,
      authorization.execution_authorization_hash, req.execution_request_hash, req.capability,
      req.authenticated_caller_id, req.motor, req.environment.value, req.target_value)
    if bound != expected: raise HumanApprovalBindingError("approval no corresponde a authorization")
    try: public_key.verify(base64.b64decode(approval.signature, validate=True), _message(approval.projection()))
    except (InvalidSignature, ValueError) as exc: raise HumanApprovalBindingError("firma approval inválida") from exc
