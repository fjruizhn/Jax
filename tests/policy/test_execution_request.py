from datetime import datetime, timezone
import pytest

from las_manos.motor_registry.catalog import MotorCatalog
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_resolution.models import EvaluationContext
from policy.decision_record import DecisionFact, DecisionFactValueType, build_decision_input, is_verified_decision_record
from policy.decision_record.authority_binding import evaluate_decision_input
from policy.decision_record.ids import new_decision_id
from policy.decision_record.service import build_decision_record
from policy.execution_control.authorization import authorize_execution, build_execution_request
from policy.execution_control.canonical import parameters_hash
from policy.execution_control.models import ExecutionEnvironment
from tests.policy.test_authority_ledger_events import setup_ledger, ratification_intent

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

def catalog():
    return MotorCatalog({"motors":{"m":{"enabled":True,"sandbox_only":True}}, "capabilities":{"CAP":{"allowed_callers":["jacobs"],"allowed_motors":["m"],"sandbox_only":True,"requires_human_gate":False,"max_execution_minutes":2,"max_recursion_depth":0,"mode":"mutating"}}})

def record():
    store, root, key = setup_ledger(); rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    state = verify_authority_ledger(store.get_genesis(), store.events(), root)
    facts = (DecisionFact("EXECUTION_CAPABILITY", DecisionFactValueType.STRING, "CAP"), DecisionFact("EXECUTION_CALLER", DecisionFactValueType.STRING, "jacobs"), DecisionFact("EXECUTION_MOTOR", DecisionFactValueType.STRING, "m"), DecisionFact("EXECUTION_ENVIRONMENT", DecisionFactValueType.STRING, "SANDBOX"), DecisionFact("EXECUTION_TARGET", DecisionFactValueType.STRING, "JAX_WORKSPACE"), DecisionFact("EXECUTION_PARAMETERS_HASH", DecisionFactValueType.STRING, parameters_hash("p", {"x": 1})), DecisionFact("EXECUTION_TIMEOUT_SECONDS", DecisionFactValueType.INTEGER, 60), DecisionFact("EXECUTION_SANDBOX_REQUIRED", DecisionFactValueType.BOOLEAN, True), DecisionFact("EXECUTION_DRY_RUN_REQUIRED", DecisionFactValueType.BOOLEAN, False), DecisionFact("EXECUTION_HUMAN_APPROVAL_REQUIRED", DecisionFactValueType.BOOLEAN, False))
    input_ = build_decision_input(EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "S", "EXECUTION", ()), NOW, facts=facts)
    return build_decision_record(evaluate_decision_input(state, input_), decision_id=new_decision_id(), recorded_at_utc=NOW)

def request(): return build_execution_request(record(), authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)

def test_sealed_request_and_authorization():
    r = request(); assert r._is_trusted(); assert r.execution_request_hash.startswith("sha256:")

def test_context_is_deeply_immutable():
    r = request()
    with pytest.raises((TypeError, AttributeError)): r.context[0] = "bad"

def test_authorization_binds_decision_and_catalog():
    rec = record(); req = build_execution_request(rec, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)
    assert authorize_execution(rec, req, catalog(), now_utc=NOW).decision_id == rec.decision_id

@pytest.mark.parametrize("field,value", [("capability","OTHER"), ("motor","other"), ("timeout_seconds",121)])
def test_scope_or_ceiling_substitution_is_rejected(field, value):
    rec = record(); values = {"authenticated_caller_id":"jacobs", "capability":"CAP", "motor":"m", "environment":ExecutionEnvironment.SANDBOX, "target_kind":"JAX_WORKSPACE", "target_value":"JAX_WORKSPACE", "prompt":"p", "context":{"x":1}, "timeout_seconds":60}
    values[field] = value
    req = build_execution_request(rec, **values)
    with pytest.raises(Exception): authorize_execution(rec, req, catalog(), now_utc=NOW)
