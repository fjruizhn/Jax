"""Trusted service boundary for immutable governed execution artifacts."""
from __future__ import annotations
from datetime import datetime, timezone

from .authorization import authorize_execution, build_execution_request
from .canonical import execution_record_hash
from .dry_run import build_dry_run_artifact
from .errors import (AuthorizationExpiredError, DryRunFailedError, DryRunRequiredError,
                     ExecutionRequestScopeError, HumanApprovalRequiredError,
                     KillSwitchActiveError)
from .ids import new_execution_id
from .models import ExecutionAuthorization, ExecutionRecord, ExecutionState
from .state_machine import initial_state, transition
from .storage import ExecutionEvent

def _now(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None: raise ExecutionRequestScopeError("now_utc explícito requerido")
    return value.astimezone(timezone.utc)

def _record(authorization: ExecutionAuthorization, *, now_utc: datetime) -> ExecutionRecord:
    request = authorization.execution_request; execution_id = new_execution_id()
    projection = {"schema_version":"1.0","kind":"JAX_GOVERNED_EXECUTION_RECORD","execution_id":execution_id,
      "decision_id":authorization.decision_id,"decision_record_hash":authorization.decision_record_hash,
      "execution_request_hash":request.execution_request_hash,"execution_authorization_hash":authorization.execution_authorization_hash,
      "authorization_id":authorization.authorization_id,"capability":request.capability,
      "authenticated_caller_id":request.authenticated_caller_id,"motor":request.motor,
      "environment":request.environment.value,"timeout_seconds":request.timeout_seconds,
      "created_at_utc":now_utc.isoformat().replace("+00:00","Z")}
    return ExecutionRecord("1.0", "JAX_GOVERNED_EXECUTION_RECORD", execution_id,
      authorization.decision_id, authorization.decision_record_hash, request.execution_request_hash,
      authorization.execution_authorization_hash, authorization.authorization_id, request.capability,
      request.authenticated_caller_id, request.motor, request.environment, request.timeout_seconds,
      now_utc, execution_record_hash(projection))

def create_execution(store, authorization: ExecutionAuthorization, *, now_utc: datetime, kill_switch_active: bool = False) -> ExecutionRecord:
    now = _now(now_utc)
    if kill_switch_active: raise KillSwitchActiveError("kill switch activo")
    if not isinstance(authorization, ExecutionAuthorization) or not authorization._is_trusted(): raise ExecutionRequestScopeError("authorization no sellada")
    if now > authorization.expires_at_utc: raise AuthorizationExpiredError("authorization vencida")
    record = _record(authorization, now_utc=now)
    state = initial_state(requires_human_approval=authorization.requires_human_approval, requires_dry_run=authorization.requires_dry_run)
    return store.create_execution(authorization, record, ExecutionEvent(record.execution_id, state.value, "EXECUTION_CREATED", now))

def consume_human_approval(store, record, authorization, approval, *, now_utc: datetime,
                           trusted_approver_resolver=None) -> None:
    from .human_approval import verify_human_approval
    from .adapters.trusted_approver import load_trusted_approver
    now = _now(now_utc)
    if not authorization.requires_human_approval: return
    resolver = trusted_approver_resolver or load_trusted_approver
    # Resolver is an injected trusted root for tests, never a verification
    # key supplied by the approval caller.
    public_key = resolver(approval.approver_actor_id, approval.approver_key_id)
    verify_human_approval(approval, authorization, public_key, now_utc=now); store.consume_approval(approval.human_approval_id)
    store.append_event(ExecutionEvent(record.execution_id,
      (ExecutionState.READY_FOR_DRY_RUN if authorization.requires_dry_run else ExecutionState.READY_TO_DISPATCH).value,
      "HUMAN_APPROVAL_CONSUMED", now))

def record_dry_run(store, record, authorization, *, status: str, result: object, now_utc: datetime):
    now = _now(now_utc); artifact = build_dry_run_artifact(record, authorization, status=status, result=result, recorded_at_utc=now)
    store.save_dry_run(artifact)
    if status != "SUCCEEDED":
        store.append_event(ExecutionEvent(record.execution_id, ExecutionState.FAILED.value, "DRY_RUN_FAILED", now)); raise DryRunFailedError("dry-run falló")
    store.append_event(ExecutionEvent(record.execution_id, ExecutionState.READY_TO_DISPATCH.value, "DRY_RUN_SUCCEEDED", now)); return artifact

def dispatch_execution(store, record, authorization, *, now_utc: datetime, kill_switch_active: bool = False, job_id: str | None = None):
    now = _now(now_utc)
    if kill_switch_active: raise KillSwitchActiveError("kill switch activo")
    if now > authorization.expires_at_utc: raise AuthorizationExpiredError("authorization vencida")
    states = store.events(record.execution_id); current = ExecutionState(states[-1].state)
    if authorization.requires_human_approval and current is ExecutionState.WAITING_HUMAN_APPROVAL: raise HumanApprovalRequiredError("approval requerida")
    if authorization.requires_dry_run:
        artifact = store.load_dry_run(record.execution_id)
        if artifact is None: raise DryRunRequiredError("dry-run requerida")
        if artifact.status != "SUCCEEDED" or artifact.execution_request_hash != record.execution_request_hash: raise DryRunFailedError("dry-run no vinculado")
    transition(current, ExecutionState.DISPATCHED)
    event = ExecutionEvent(record.execution_id, ExecutionState.DISPATCHED.value, "MOTOR_DISPATCHED", now, job_id)
    store.append_event(event); return event

def cancel_execution(store, execution_id: str, *, now_utc: datetime):
    events = store.events(execution_id); current = ExecutionState(events[-1].state)
    transition(current, ExecutionState.CANCELLED); store.append_event(ExecutionEvent(execution_id, ExecutionState.CANCELLED.value, "CANCELLED", _now(now_utc)))
