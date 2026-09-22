"""Pure governed-execution authorization from sealed Block 5 records."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import weakref

from policy.decision_record import is_verified_decision_record
from policy.decision_record.models import DecisionFactValueType, DecisionRecord

from .canonical import execution_authorization_hash, parameters_hash
from .errors import (CallerNotAuthorizedError, ExecutionRequestScopeError,
                     ExecutionTimeoutError, MotorNotAuthorizedError,
                     SandboxViolationError, UnknownCapabilityError,
                     UnsupportedExecutionEnvironmentError)
from .ids import new_authorization_id
from .models import (CapabilityPolicyProjection, ExecutionAuthorization,
                     ExecutionEnvironment, ExecutionRequest)

# Exact-instance provenance remains wholly inside the two real issuance
# factories below.  There is deliberately no helper accepting an artifact or
# callback to register from another module.
_issued_requests: dict[int, weakref.ReferenceType] = {}
_issued_authorizations: dict[int, weakref.ReferenceType] = {}
_B7_DECISION_RECORDER = None
_B7_AUTHORIZATION_RECORDER = None

def configure_b7_decision_recorder(recorder) -> None:
    """Application-startup seam; request callers never supply this recorder."""
    global _B7_DECISION_RECORDER
    _B7_DECISION_RECORDER = recorder

def configure_b7_authorization_recorder(recorder) -> None:
    """Application-startup seam; request callers cannot replace it."""
    global _B7_AUTHORIZATION_RECORDER
    _B7_AUTHORIZATION_RECORDER = recorder

def _b7_authorization_outcome(control_id, *, denied=False, decision_id=None, subject_identity=None):
    recorder=_B7_AUTHORIZATION_RECORDER
    if recorder is None: return
    try:
        if denied:
            typed = {
                "CTL.B6.AUTHORIZATION_PROVENANCE": recorder.record_authorization_provenance_denied,
                "CTL.B6.SANDBOX_ONLY": recorder.record_sandbox_denied,
                "CTL.B6.TIMEOUT_CEILING": recorder.record_timeout_denied,
            }.get(control_id)
            if typed is None: raise ValueError("unsupported authorization evidence control")
            typed(decision_id=decision_id)
        else:
            typed = {
                "CTL.B6.AUTHORIZATION_PROVENANCE": recorder.record_authorization_provenance,
                "CTL.B6.SANDBOX_ONLY": recorder.record_sandbox_validated,
                "CTL.B6.TIMEOUT_CEILING": recorder.record_timeout_validated,
            }.get(control_id)
            if typed is None: raise ValueError("unsupported authorization evidence control")
            typed(authorization_id=subject_identity or decision_id, decision_id=decision_id)
    except Exception:  # fail-soft: optional observation cannot alter B6 authorization.
        # B7 does not replace the governing authorization decision.
        pass

def _record_unverified_decision(record) -> None:
    if _B7_DECISION_RECORDER is None:
        return
    try:
        _B7_DECISION_RECORDER.record_decision_provenance_denied(
            decision_id=getattr(record, "decision_id", None))
    except Exception:  # fail-soft: rejected DecisionRecord never becomes eligible on evidence outage.
        pass

def _issued(registry, value):
    key = id(value)
    registry[key] = weakref.ref(value, lambda _r, k=key, r=registry: r.pop(k, None))
    return value

def _request_issued_by_factory(value) -> bool:
    ref = _issued_requests.get(id(value))
    return ref is not None and ref() is value

def _authorization_issued_by_factory(value) -> bool:
    ref = _issued_authorizations.get(id(value))
    return ref is not None and ref() is value


def deserialize_execution_authorization(data: dict) -> ExecutionAuthorization:
    """Parse and verify a canonical authorization without granting provenance.

    Canonical bytes establish only internal consistency.  The authoritative
    store loader is the sole path which may subsequently trust this instance.
    """
    request_data = data["execution_request"]
    target = request_data["target"]
    request = ExecutionRequest(
        request_data["schema_version"], request_data["kind"], request_data["decision_id"],
        request_data["decision_record_hash"], request_data["capability"],
        request_data["authenticated_caller_id"], request_data["motor"],
        ExecutionEnvironment(request_data["environment"]), target["kind"], target["value"],
        request_data["prompt"], request_data["context"], request_data["timeout_seconds"],
        request_data["sandbox_required"], request_data.get("tenant_id"), request_data.get("user_id"))
    policy = data["capability_policy"]
    projection = CapabilityPolicyProjection(policy["capability"], tuple(policy["allowed_callers"]),
        tuple(policy["allowed_motors"]), policy["sandbox_only"], policy["requires_human_gate"],
        policy["max_execution_minutes"], policy["max_recursion_depth"], policy["mode"], policy["risk_level"])
    return ExecutionAuthorization(
        data["schema_version"], data["kind"], data["authorization_id"], data["decision_id"],
        data["decision_record_hash"], request, projection, data["policy_corpus_hash"],
        data["effective_authority_context_hash"], data["authority_ledger_checkpoint_hash"],
        data["requires_human_approval"], data["requires_dry_run"],
        datetime.fromisoformat(data["issued_at_utc"].replace("Z", "+00:00")),
        datetime.fromisoformat(data["expires_at_utc"].replace("Z", "+00:00")),
        data["execution_authorization_hash"])


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
        _record_unverified_decision(record); raise UnverifiedDecisionRecordError("DecisionRecord no verificado")
    if _B7_DECISION_RECORDER is not None:
        try: _B7_DECISION_RECORDER.record_decision_provenance(decision_id=record.decision_id)
        except Exception: pass  # evidence cannot make an unverified request eligible
    return _issued(_issued_requests, ExecutionRequest("1.0", "JAX_EXECUTION_REQUEST", record.decision_id,
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
        _record_unverified_decision(record); raise UnverifiedDecisionRecordError("DecisionRecord no verificado")
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
        _b7_authorization_outcome("CTL.B6.SANDBOX_ONLY",denied=True,decision_id=record.decision_id)
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
        _b7_authorization_outcome("CTL.B6.SANDBOX_ONLY",denied=True,decision_id=record.decision_id)
        raise SandboxViolationError("capability/motor/request debe ser sandbox-only")
    if request.timeout_seconds > cap.max_execution_minutes * 60:
        _b7_authorization_outcome("CTL.B6.TIMEOUT_CEILING",denied=True,decision_id=record.decision_id)
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
    result=_issued(_issued_authorizations, ExecutionAuthorization("1.0", "JAX_EXECUTION_AUTHORIZATION", auth_id,
        record.decision_id, record.decision_record_hash, request, policy,
        record.authority_binding.active_policy_corpus_hash,
        record.authority_binding.effective_authority_context_hash,
        record.authority_binding.authority_ledger_checkpoint_hash,
        facts["EXECUTION_HUMAN_APPROVAL_REQUIRED"], facts["EXECUTION_DRY_RUN_REQUIRED"],
        issued, expires, digest))
    _b7_authorization_outcome("CTL.B6.SANDBOX_ONLY",decision_id=record.decision_id,subject_identity=auth_id)
    _b7_authorization_outcome("CTL.B6.TIMEOUT_CEILING",decision_id=record.decision_id,subject_identity=auth_id)
    return result
