from datetime import datetime, timezone
from dataclasses import replace
import pytest

from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_resolution.models import EvaluationContext
from policy.decision_record import DecisionFact, DecisionFactValueType, build_decision_input, is_verified_decision_record
from policy.decision_record.authority_binding import evaluate_decision_input
from policy.decision_record.ids import new_decision_id
from policy.decision_record.service import record_decision
from policy.decision_record.storage import InMemoryDecisionRecordStore
from policy.execution_control.authorization import (authorize_execution, build_execution_request,
    deserialize_execution_authorization)
from policy.execution_control.canonical import parameters_hash
from policy.execution_control.models import ExecutionEnvironment
from policy.execution_control.service import create_execution
from policy.execution_control.storage import InMemoryExecutionStore
from policy.execution_control.storage import MariaDBExecutionStore
from tests.policy.test_authority_ledger_events import setup_ledger, ratification_intent

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

class _Capability:
    name = "CAP"; allowed_callers = ["jacobs"]; allowed_motors = ["m"]
    sandbox_only = True; requires_human_gate = False; max_execution_minutes = 2
    max_recursion_depth = 0; mode = "mutating"; risk_level = "high"

class _Motor:
    enabled = True; sandbox_only = True

class _Catalog:
    def get_capability(self, name): return _Capability() if name == "CAP" else None
    def get_motor(self, name): return _Motor() if name == "m" else None

def catalog(): return _Catalog()

def record():
    store, root, key = setup_ledger(); rat = append_authority_event(store, root, key, ratification_intent())
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", (), ratification_event_id=rat.event_id))
    state = verify_authority_ledger(store.get_genesis(), store.events(), root)
    facts = (DecisionFact("EXECUTION_CAPABILITY", DecisionFactValueType.STRING, "CAP"), DecisionFact("EXECUTION_CALLER", DecisionFactValueType.STRING, "jacobs"), DecisionFact("EXECUTION_MOTOR", DecisionFactValueType.STRING, "m"), DecisionFact("EXECUTION_ENVIRONMENT", DecisionFactValueType.STRING, "SANDBOX"), DecisionFact("EXECUTION_TARGET", DecisionFactValueType.STRING, "JAX_WORKSPACE"), DecisionFact("EXECUTION_PARAMETERS_HASH", DecisionFactValueType.STRING, parameters_hash("p", {"x": 1})), DecisionFact("EXECUTION_TIMEOUT_SECONDS", DecisionFactValueType.INTEGER, 60), DecisionFact("EXECUTION_SANDBOX_REQUIRED", DecisionFactValueType.BOOLEAN, True), DecisionFact("EXECUTION_DRY_RUN_REQUIRED", DecisionFactValueType.BOOLEAN, False), DecisionFact("EXECUTION_HUMAN_APPROVAL_REQUIRED", DecisionFactValueType.BOOLEAN, False))
    input_ = build_decision_input(EvaluationContext("1.0", "JAX_AUTHORITY_EVALUATION_CONTEXT", "JAX", "S", "EXECUTION", ()), NOW, facts=facts)
    return record_decision(InMemoryDecisionRecordStore(), evaluate_decision_input(state, input_), decision_id=new_decision_id(), recorded_at_utc=NOW)

def request(): return build_execution_request(record(), authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)

def test_sealed_request_and_authorization():
    r = request(); assert r._is_trusted(); assert r.execution_request_hash.startswith("sha256:")

def test_context_is_deeply_immutable():
    r = request()
    with pytest.raises((TypeError, AttributeError)): r.context[0] = "bad"


def test_reconstructed_record_cannot_enter_execution_boundary():
    from policy.decision_record.serialization import canonical_decision_record_bytes
    from policy.decision_record.service import verify_decision_record
    from policy.execution_control.errors import UnverifiedDecisionRecordError
    reconstructed = verify_decision_record(canonical_decision_record_bytes(record()))
    assert not is_verified_decision_record(reconstructed)
    with pytest.raises(UnverifiedDecisionRecordError):
        build_execution_request(reconstructed, authenticated_caller_id="jacobs", capability="CAP", motor="m",
            environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE",
            prompt="p", context={"x": 1}, timeout_seconds=60)


def test_manual_request_and_authorization_are_not_trusted():
    rec = record()
    request = build_execution_request(rec, authenticated_caller_id="jacobs", capability="CAP", motor="m",
        environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE",
        prompt="p", context={"x": 1}, timeout_seconds=60)
    with pytest.raises(Exception):
        authorize_execution(rec, replace(request), catalog(), now_utc=NOW)
    authorization = authorize_execution(rec, request, catalog(), now_utc=NOW)
    with pytest.raises(Exception):
        create_execution(InMemoryExecutionStore(), replace(authorization), now_utc=NOW)


@pytest.mark.parametrize("field,value", [("motor", "attacker-motor"),
    ("capability", "ATTACKER")])
def test_canonical_authorization_parser_never_mints_provenance(field, value):
    """A self-consistent caller projection is data, not an issuance event."""
    rec = record(); req = build_execution_request(rec, authenticated_caller_id="jacobs", capability="CAP", motor="m",
        environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)
    auth = authorize_execution(rec, req, catalog(), now_utc=NOW)
    projection = auth.projection_without_hash()
    request = projection["execution_request"]
    request[field] = value
    policy = projection["capability_policy"]
    if field == "motor": policy["allowed_motors"] = [value]
    if field == "capability": policy["capability"] = value
    from policy.execution_control.canonical import execution_authorization_hash
    projection["execution_authorization_hash"] = execution_authorization_hash(projection)
    parsed = deserialize_execution_authorization(projection)
    assert not parsed._is_trusted()
    assert not parsed.execution_request._is_trusted()
    with pytest.raises(Exception):
        InMemoryExecutionStore().insert_authorization(parsed)
    with pytest.raises(Exception):
        create_execution(InMemoryExecutionStore(), parsed, now_utc=NOW)


def test_changed_environment_cannot_be_reconstructed_into_an_authorization():
    rec = record(); req = build_execution_request(rec, authenticated_caller_id="jacobs", capability="CAP", motor="m",
        environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)
    auth = authorize_execution(rec, req, catalog(), now_utc=NOW)
    projection = auth.projection_without_hash()
    projection["execution_request"]["environment"] = "PRODUCTION"
    from policy.execution_control.canonical import execution_authorization_hash
    projection["execution_authorization_hash"] = execution_authorization_hash(projection)
    with pytest.raises(Exception):
        deserialize_execution_authorization(projection)


def test_authoritative_store_load_is_the_only_authorization_provenance_boundary():
    store = InMemoryExecutionStore()
    rec = record(); req = build_execution_request(rec, authenticated_caller_id="jacobs", capability="CAP", motor="m",
        environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)
    auth = authorize_execution(rec, req, catalog(), now_utc=NOW)
    assert store.insert_authorization(auth)._is_trusted()
    assert store.load_authorization(auth.authorization_id)._is_trusted()


def test_models_do_not_publish_trust_registration_symbols():
    import policy.execution_control.models as models
    assert not hasattr(models, "register_request")
    assert not hasattr(models, "register_authorization")


def test_mariadb_dbapi_create_persists_canonical_record_and_event():
    class Cursor:
        def __init__(self, db): self.db, self.last = db, None
        def execute(self, sql, args=()):
            self.last = sql
            if sql.startswith("INSERT INTO jax_execution.execution_authorizations"):
                self.db.auth = args
            elif sql.startswith("INSERT INTO jax_execution.execution_records"):
                self.db.record = args
            elif sql.startswith("INSERT INTO jax_execution.execution_events"):
                self.db.event = args
        def fetchone(self):
            if "canonical_authorization_hash" in self.last: return (self.db.auth[3],)
            return None
    class Conn:
        def __init__(self): self.auth = self.record = self.event = None; self.commits = self.rollbacks = 0
        def cursor(self): return Cursor(self)
        def commit(self): self.commits += 1
        def rollback(self): self.rollbacks += 1
        def close(self): pass
    conn = Conn(); store = MariaDBExecutionStore(lambda: conn)
    rec = record(); request = build_execution_request(rec, authenticated_caller_id="jacobs", capability="CAP", motor="m",
        environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)
    auth = authorize_execution(rec, request, catalog(), now_utc=NOW)
    store.insert_authorization(auth)
    execution = create_execution(store, auth, now_utc=NOW)
    assert execution.execution_id == conn.record[0]
    assert '"execution_record_hash"' in conn.record[3]
    assert conn.event[0] == execution.execution_id

def test_authorization_binds_decision_and_catalog():
    rec = record(); req = build_execution_request(rec, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x":1}, timeout_seconds=60)
    assert authorize_execution(rec, req, catalog(), now_utc=NOW).decision_id == rec.decision_id

@pytest.mark.parametrize("field,value", [("capability","OTHER"), ("motor","other"), ("timeout_seconds",121)])
def test_scope_or_ceiling_substitution_is_rejected(field, value):
    rec = record(); values = {"authenticated_caller_id":"jacobs", "capability":"CAP", "motor":"m", "environment":ExecutionEnvironment.SANDBOX, "target_kind":"JAX_WORKSPACE", "target_value":"JAX_WORKSPACE", "prompt":"p", "context":{"x":1}, "timeout_seconds":60}
    values[field] = value
    req = build_execution_request(rec, **values)
    with pytest.raises(Exception): authorize_execution(rec, req, catalog(), now_utc=NOW)
