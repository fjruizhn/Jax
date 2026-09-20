"""Pure governed-execution authorization from sealed Block 5 records."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from policy.decision_record import is_verified_decision_record
from policy.decision_record.models import DecisionFactValueType, DecisionRecord

from .canonical import execution_authorization_hash, parameters_hash
from .errors import (CallerNotAuthorizedError, ExecutionRequestScopeError,
                     ExecutionTimeoutError, MotorNotAuthorizedError,
                     SandboxViolationError, UnknownCapabilityError,
                     UnsupportedExecutionEnvironmentError)
from .ids import new_authorization_id
from .models import (CapabilityPolicyProjection, ExecutionAuthorization,
                     ExecutionEnvironment, ExecutionRequest, register_authorization,
                     register_request)


_FACTS = {
    "EXECUTION_CAPABILITY": DecisionFactValueType.STRING,
    "EXECUTION_CALLER": DecisionFactValueType.STRING,
    "EXECUTION_MOTOR": DecisionFactValueType.STRING,
    "EXECUTION_ENVIRONMENT": DecisionFactValueType.STRING,
    "EXECUTION_TARGET": DecisionFactValueType.STRING,
    "EXECUTION_PARAMETERS_HASH": DecisionFactValueType.STRING,
    "EXECUTION_TIMEOUT_SECONDS": DecisionFactValueType.INTEGER,
    "EXECUTION_SANDBOX_REQUIRED": DecisionFactValueType.BOOLEAN,
    "EXECUTION_DRY_RUN_REQUIRED": DecisionFactValueType.BOOLEAN,
    "EXECUTION_HUMAN_APPROVAL_REQUIRED": DecisionFactValueType.BOOLEAN,
}


def _facts(record: DecisionRecord) -> dict[str, object]:
    if record.decision_input.evaluation_context.action != "EXECUTION":
        raise ExecutionRequestScopeError("la acción de decisión debe ser EXECUTION")
    values = {fact.id: fact for fact in record.decision_input.facts}
    if set(values) != set(_FACTS):
        raise ExecutionRequestScopeError("facts de ejecución incompletos o adicionales")
    result: dict[str, object] = {}
    for key, kind in _FACTS.items():
        fact = values[key]
        if fact.value_type is not kind:
            raise ExecutionRequestScopeError(f"tipo inválido para {key}")
        result[key] = fact.value
    return result


def build_execution_request(record: DecisionRecord, *, authenticated_caller_id: str,
                            capability: str, motor: str, environment: ExecutionEnvironment,
                            target_kind: str, target_value: str, prompt: str, context: object,
                            timeout_seconds: int, sandbox_required: bool = True,
                            tenant_id: str | None = None, user_id: str | None = None) -> ExecutionRequest:
    """Creates the only request shape accepted by the authorization boundary."""
    if not is_verified_decision_record(record):
        from .errors import UnverifiedDecisionRecordError
        raise UnverifiedDecisionRecordError("DecisionRecord no verificado")
    return register_request(ExecutionRequest("1.0", "JAX_EXECUTION_REQUEST", record.decision_id,
        record.decision_record_hash, capability, authenticated_caller_id, motor, environment,
        target_kind, target_value, prompt, context, timeout_seconds, sandbox_required,
        tenant_id, user_id))


def _projection(capability) -> CapabilityPolicyProjection:
    return CapabilityPolicyProjection(capability.name, tuple(sorted(capability.allowed_callers)),
        tuple(sorted(capability.allowed_motors)), capability.sandbox_only,
        capability.requires_human_gate, capability.max_execution_minutes,
        capability.max_recursion_depth, capability.mode, capability.risk_level)


def authorize_execution(record: DecisionRecord, request: ExecutionRequest, catalog, *, now_utc: datetime) -> ExecutionAuthorization:
    """Applies the decision facts and the catalog's operational ceilings."""
    if not is_verified_decision_record(record):
        from .errors import UnverifiedDecisionRecordError
        raise UnverifiedDecisionRecordError("DecisionRecord no verificado")
    if not isinstance(request, ExecutionRequest) or not request._is_trusted():
        raise ExecutionRequestScopeError("ExecutionRequest no sellada")
    if request.decision_id != record.decision_id or request.decision_record_hash != record.decision_record_hash:
        raise ExecutionRequestScopeError("request no corresponde a la decisión")
    facts = _facts(record)
    expected = {
        "EXECUTION_CAPABILITY": request.capability,
        "EXECUTION_CALLER": request.authenticated_caller_id,
        "EXECUTION_MOTOR": request.motor,
        "EXECUTION_ENVIRONMENT": request.environment.value,
        "EXECUTION_TARGET": request.target_value,
        "EXECUTION_PARAMETERS_HASH": parameters_hash(request.prompt, request.projection()["context"]),
        "EXECUTION_TIMEOUT_SECONDS": request.timeout_seconds,
        "EXECUTION_SANDBOX_REQUIRED": request.sandbox_required,
    }
    for key, value in expected.items():
        if facts[key] != value:
            raise ExecutionRequestScopeError(f"{key} no coincide con la decisión")
    if request.environment is not ExecutionEnvironment.SANDBOX:
        raise UnsupportedExecutionEnvironmentError("B6 V1 sólo autoriza SANDBOX")
    cap = catalog.get_capability(request.capability)
    if cap is None:
        raise UnknownCapabilityError(request.capability)
    if request.authenticated_caller_id != "jacobs" or request.authenticated_caller_id not in cap.allowed_callers:
        raise CallerNotAuthorizedError(request.authenticated_caller_id)
    motor = catalog.get_motor(request.motor)
    if motor is None or not motor.enabled or request.motor not in cap.allowed_motors:
        raise MotorNotAuthorizedError(request.motor)
    if not cap.sandbox_only or not motor.sandbox_only or request.sandbox_required is not True:
        raise SandboxViolationError("capability/motor/request debe ser sandbox-only")
    if request.timeout_seconds > cap.max_execution_minutes * 60:
        raise ExecutionTimeoutError("timeout excede capability")
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise ExecutionRequestScopeError("now_utc explícito timezone-aware requerido")
    issued = now_utc.astimezone(timezone.utc)
    expires = issued + timedelta(seconds=300)
    policy = _projection(cap)
    auth_id = new_authorization_id()
    payload = {"schema_version": "1.0", "kind": "JAX_EXECUTION_AUTHORIZATION",
        "authorization_id": auth_id, "decision_id": record.decision_id,
        "decision_record_hash": record.decision_record_hash,
        "execution_request": request.projection(), "capability_policy": policy.projection(),
        "policy_corpus_hash": record.authority_binding.active_policy_corpus_hash,
        "effective_authority_context_hash": record.authority_binding.effective_authority_context_hash,
        "authority_ledger_checkpoint_hash": record.authority_binding.authority_ledger_checkpoint_hash,
        "requires_human_approval": facts["EXECUTION_HUMAN_APPROVAL_REQUIRED"],
        "requires_dry_run": facts["EXECUTION_DRY_RUN_REQUIRED"],
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": expires.isoformat().replace("+00:00", "Z")}
    digest = execution_authorization_hash(payload)
    return register_authorization(ExecutionAuthorization("1.0", "JAX_EXECUTION_AUTHORIZATION", auth_id,
        record.decision_id, record.decision_record_hash, request, policy,
        record.authority_binding.active_policy_corpus_hash,
        record.authority_binding.effective_authority_context_hash,
        record.authority_binding.authority_ledger_checkpoint_hash,
        facts["EXECUTION_HUMAN_APPROVAL_REQUIRED"], facts["EXECUTION_DRY_RUN_REQUIRED"],
        issued, expires, digest))
