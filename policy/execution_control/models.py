"""Closed immutable Block 6 artifacts.  Provenance is kept out of fields."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
import unicodedata
from typing import Any, Mapping

from .canonical import (canonical_value, execution_authorization_hash,
                        execution_record_hash, execution_request_hash, utc_text)
from .errors import ExecutionIntegrityError

_UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")

def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or unicodedata.normalize("NFC", value) != value:
        raise ExecutionIntegrityError(f"{name} debe ser string NFC no vacío")
    return value

def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ExecutionIntegrityError(f"{name} inválido")
    return value

def _uuid(value: object, name: str) -> str:
    if not isinstance(value, str) or not _UUID7.fullmatch(value):
        raise ExecutionIntegrityError(f"{name} debe ser UUIDv7")
    return value

def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ExecutionIntegrityError(f"{name} debe ser UTC timezone-aware")
    return value.astimezone(timezone.utc)

def _freeze(value: object) -> object:
    if isinstance(value, dict): return tuple((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, list): return tuple(_freeze(item) for item in value)
    return value

def _thaw(value: object) -> object:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {item[0]: _thaw(item[1]) for item in value}
        return [_thaw(item) for item in value]
    return value

class ExecutionEnvironment(str, Enum):
    LOCAL = "LOCAL"
    SANDBOX = "SANDBOX"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"

class ExecutionState(str, Enum):
    WAITING_HUMAN_APPROVAL = "WAITING_HUMAN_APPROVAL"
    READY_FOR_DRY_RUN = "READY_FOR_DRY_RUN"
    READY_TO_DISPATCH = "READY_TO_DISPATCH"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    EXPIRED = "EXPIRED"

@dataclass(frozen=True)
class CapabilityPolicyProjection:
    capability: str
    allowed_callers: tuple[str, ...]
    allowed_motors: tuple[str, ...]
    sandbox_only: bool
    requires_human_gate: bool
    max_execution_minutes: int
    max_recursion_depth: int
    mode: str
    risk_level: str

    def projection(self) -> dict:
        return {"capability": self.capability, "allowed_callers": list(self.allowed_callers),
                "allowed_motors": list(self.allowed_motors), "sandbox_only": self.sandbox_only,
                "requires_human_gate": self.requires_human_gate,
                "max_execution_minutes": self.max_execution_minutes,
                "max_recursion_depth": self.max_recursion_depth, "mode": self.mode,
                "risk_level": self.risk_level}

@dataclass(frozen=True)
class ExecutionRequest:
    schema_version: str
    kind: str
    decision_id: str
    decision_record_hash: str
    capability: str
    authenticated_caller_id: str
    motor: str
    environment: ExecutionEnvironment
    target_kind: str
    target_value: str
    prompt: str
    context: object
    timeout_seconds: int
    sandbox_required: bool
    tenant_id: str | None
    user_id: str | None

    def __post_init__(self):
        if (self.schema_version, self.kind) != ("1.0", "JAX_EXECUTION_REQUEST"):
            raise ExecutionIntegrityError("schema ExecutionRequest inválido")
        for name in ("decision_id",): _uuid(getattr(self, name), name)
        _hash(self.decision_record_hash, "decision_record_hash")
        for name in ("capability", "authenticated_caller_id", "motor", "target_kind", "target_value", "prompt"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.environment is not ExecutionEnvironment.SANDBOX:
            raise ExecutionIntegrityError("B6 V1 sólo permite SANDBOX")
        if (self.target_kind, self.target_value) != ("JAX_WORKSPACE", "JAX_WORKSPACE"):
            raise ExecutionIntegrityError("target B6 V1 debe ser JAX_WORKSPACE")
        if type(self.timeout_seconds) is not int or self.timeout_seconds < 1:
            raise ExecutionIntegrityError("timeout_seconds inválido")
        if self.sandbox_required is not True:
            raise ExecutionIntegrityError("sandbox_required debe ser true")
        if not isinstance(self.context, Mapping): raise ExecutionIntegrityError("context debe ser mapping")
        object.__setattr__(self, "context", _freeze(canonical_value(self.context)))
        for name in ("tenant_id", "user_id"):
            value = getattr(self, name)
            if value is not None: object.__setattr__(self, name, _text(value, name))

    def projection(self) -> dict:
        return {"schema_version": self.schema_version, "kind": self.kind,
          "decision_id": self.decision_id, "decision_record_hash": self.decision_record_hash,
          "capability": self.capability, "authenticated_caller_id": self.authenticated_caller_id,
          "motor": self.motor, "environment": self.environment.value,
          "target": {"kind": self.target_kind, "value": self.target_value},
          "prompt": self.prompt, "context": _thaw(self.context), "timeout_seconds": self.timeout_seconds,
          "sandbox_required": self.sandbox_required, "tenant_id": self.tenant_id, "user_id": self.user_id}

    @property
    def execution_request_hash(self) -> str: return execution_request_hash(self.projection())
    def _is_trusted(self) -> bool:
        # The issuer registry lives in the trusted factory module, not in a
        # public model/serialization registration surface.
        from .authorization import _request_issued_by_factory
        return _request_issued_by_factory(self)

@dataclass(frozen=True)
class ExecutionAuthorization:
    schema_version: str; kind: str; authorization_id: str; decision_id: str
    decision_record_hash: str; execution_request: ExecutionRequest
    capability_policy: CapabilityPolicyProjection
    policy_corpus_hash: str; effective_authority_context_hash: str; authority_ledger_checkpoint_hash: str
    requires_human_approval: bool; requires_dry_run: bool
    issued_at_utc: datetime; expires_at_utc: datetime; execution_authorization_hash: str

    def __post_init__(self):
        if (self.schema_version, self.kind) != ("1.0", "JAX_EXECUTION_AUTHORIZATION"):
            raise ExecutionIntegrityError("schema authorization inválido")
        for name in ("authorization_id", "decision_id"): _uuid(getattr(self, name), name)
        for name in ("decision_record_hash", "policy_corpus_hash", "effective_authority_context_hash", "authority_ledger_checkpoint_hash", "execution_authorization_hash"): _hash(getattr(self, name), name)
        if not isinstance(self.execution_request, ExecutionRequest) or not self.execution_request._is_trusted(): raise ExecutionIntegrityError("request no confiable")
        if self.decision_id != self.execution_request.decision_id or self.decision_record_hash != self.execution_request.decision_record_hash: raise ExecutionIntegrityError("binding decision/request inválido")
        issued, expires = _utc(self.issued_at_utc, "issued_at_utc"), _utc(self.expires_at_utc, "expires_at_utc")
        object.__setattr__(self, "issued_at_utc", issued); object.__setattr__(self, "expires_at_utc", expires)
        if expires <= issued: raise ExecutionIntegrityError("expiración inválida")
        if self.execution_authorization_hash != execution_authorization_hash(self.projection_without_hash()): raise ExecutionIntegrityError("hash authorization inválido")
    def projection_without_hash(self) -> dict:
        return {"schema_version": self.schema_version, "kind": self.kind, "authorization_id": self.authorization_id,
          "decision_id": self.decision_id, "decision_record_hash": self.decision_record_hash,
          "execution_request": self.execution_request.projection(), "capability_policy": self.capability_policy.projection(),
          "policy_corpus_hash": self.policy_corpus_hash, "effective_authority_context_hash": self.effective_authority_context_hash,
          "authority_ledger_checkpoint_hash": self.authority_ledger_checkpoint_hash,
          "requires_human_approval": self.requires_human_approval, "requires_dry_run": self.requires_dry_run,
          "issued_at_utc": utc_text(self.issued_at_utc), "expires_at_utc": utc_text(self.expires_at_utc)}
    def _is_trusted(self) -> bool:
        from .authorization import _authorization_issued_by_factory
        return _authorization_issued_by_factory(self)

@dataclass(frozen=True)
class ExecutionRecord:
    schema_version: str; kind: str; execution_id: str; decision_id: str; decision_record_hash: str
    execution_request_hash: str; execution_authorization_hash: str; authorization_id: str
    capability: str; authenticated_caller_id: str; motor: str; environment: ExecutionEnvironment
    timeout_seconds: int; created_at_utc: datetime; execution_record_hash: str
    def __post_init__(self):
        if (self.schema_version, self.kind) != ("1.0", "JAX_GOVERNED_EXECUTION_RECORD"): raise ExecutionIntegrityError("schema record inválido")
        for name in ("execution_id", "decision_id", "authorization_id"): _uuid(getattr(self, name), name)
        for name in ("decision_record_hash", "execution_request_hash", "execution_authorization_hash", "execution_record_hash"): _hash(getattr(self, name), name)
        object.__setattr__(self, "created_at_utc", _utc(self.created_at_utc, "created_at_utc"))
        if self.execution_record_hash != execution_record_hash(self.projection_without_hash()): raise ExecutionIntegrityError("hash record inválido")
    def projection_without_hash(self) -> dict:
        return {"schema_version":self.schema_version,"kind":self.kind,"execution_id":self.execution_id,"decision_id":self.decision_id,"decision_record_hash":self.decision_record_hash,"execution_request_hash":self.execution_request_hash,"execution_authorization_hash":self.execution_authorization_hash,"authorization_id":self.authorization_id,"capability":self.capability,"authenticated_caller_id":self.authenticated_caller_id,"motor":self.motor,"environment":self.environment.value,"timeout_seconds":self.timeout_seconds,"created_at_utc":utc_text(self.created_at_utc)}
