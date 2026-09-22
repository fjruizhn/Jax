"""Block 6 evidence against the real MariaDB adapter (CI service only)."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import pytest

from policy.execution_control.authorization import (authorize_execution,
    build_execution_request, deserialize_execution_authorization)
from policy.execution_control.errors import (DecisionExecutionConflictError,
    UnknownExecutionError)
from policy.execution_control.service import create_execution
from policy.execution_control.service import _record
from policy.execution_control.storage import ExecutionEvent, MariaDBExecutionStore
from policy.execution_control.models import ExecutionEnvironment
from tests.policy.test_execution_request import catalog, record


pytestmark = pytest.mark.skipif(
    not os.environ.get("JAX_EXECUTION_TEST_MARIADB"),
    reason="requiere MariaDB aislada del job governed-execution-mariadb",
)


def _connection():
    import pymysql
    return pymysql.connect(host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ["JAX_DB_NAME"], charset="utf8mb4", autocommit=False)


def _apply_migration() -> None:
    sql = (Path(__file__).parents[2] / "policy/execution_control/migrations/001_governed_execution.sql").read_text()
    tables, triggers = sql.split("DELIMITER //", 1)
    triggers, _ = triggers.split("DELIMITER ;", 1)
    connection = _connection()
    try:
        cursor = connection.cursor()
        # CI runs this focused suite against one service database.  The
        # migration contains immutable trigger declarations, so replaying it
        # after the first test is not a valid migration operation.
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='jax_execution' AND table_name='execution_records'")
        if cursor.fetchone()[0]:
            return
        for statement in tables.split(";"):
            if statement.strip(): cursor.execute(statement)
        for statement in triggers.split("//"):
            if statement.strip(): cursor.execute(statement)
        connection.commit()
    finally:
        connection.close()

def _apply_evidence_migration() -> None:
    sql = (Path(__file__).parents[2] / "policy/enforcement_evidence/migrations/001_enforcement_evidence.sql").read_text()
    tables, triggers = sql.split("DELIMITER //", 1)
    triggers, _ = triggers.split("DELIMITER ;", 1)
    connection = _connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='jax_evidence' AND table_name='evidence_blobs'")
        if cursor.fetchone()[0]:
            return
        for statement in tables.split(";"):
            if statement.strip(): cursor.execute(statement)
        for statement in triggers.split("//"):
            if statement.strip(): cursor.execute(statement)
        connection.commit()
    finally: connection.close()

def test_b7_evidence_blob_real_mariadb_immutability():
    """CI-only real DB proof: bytes deduplicate and trigger blocks mutation."""
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    _apply_evidence_migration()
    store = MariaDBEvidenceStore(_connection); blob = store.put_evidence_blob(b"b7-real-db")
    assert store.put_evidence_blob(b"b7-real-db").evidence_hash == blob.evidence_hash
    connection = _connection()
    try:
        with pytest.raises(Exception):
            cursor=connection.cursor(); cursor.execute("UPDATE jax_evidence.evidence_blobs SET size_bytes=1 WHERE evidence_hash=%s", (blob.evidence_hash,)); connection.commit()
        connection.rollback()
    finally: connection.close()

def test_b7_real_mariadb_artifact_and_observation_relations():
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.control_registry import load_control_definition
    from policy.enforcement_evidence.models import (ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment)
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    _apply_evidence_migration(); store=MariaDBEvidenceStore(_connection); now=datetime.now(timezone.utc)
    manifest=store.put_evidence_blob(b"b7-manifest"); definition=load_control_definition("CTL.B6.GOVERNED_DISPATCH")
    identity=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,manifest.evidence_hash)
    lifecycle=EvidenceLifecycleService(store,_ControlledTestIdentityProvider(identity)); recorder=RuntimeEvidenceRecorder(lifecycle,"ci",ClaimScope(ClaimEnvironment.CI))
    observation=recorder.record_governed_dispatch_denied()
    artifact=store.load_evidence_artifact(observation.evidence_artifact_hashes[0])
    loaded=store.load_evidence_artifact(artifact.artifact_hash)
    assert loaded.artifact_hash == artifact.artifact_hash
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.observation_artifacts WHERE observation_id=%s",(observation.observation_id,)) == 1
    assert store.load_observation(observation.observation_id).observation_hash == observation.observation_hash
    connection=_connection()
    try:
        with pytest.raises(Exception):
            cursor=connection.cursor(); cursor.execute("DELETE FROM jax_evidence.evidence_artifacts WHERE artifact_hash=%s",(artifact.artifact_hash,)); connection.commit()
        connection.rollback()
    finally: connection.close()

def test_b7_real_mariadb_transaction_rollback_leaves_no_artifact():
    _apply_evidence_migration(); marker = "sha256:" + "f" * 64
    connection = _connection()
    try:
        cursor=connection.cursor()
        with pytest.raises(Exception):
            cursor.execute("INSERT INTO jax_evidence.evidence_artifacts(artifact_hash,canonical_artifact) VALUES (%s,%s)",(marker,"{}"))
            # A missing blob FK makes the relationship insertion fail; rollback
            # must erase the otherwise valid artifact row as one transaction.
            cursor.execute("INSERT INTO jax_evidence.evidence_artifact_blobs(artifact_hash,evidence_hash) VALUES (%s,%s)",(marker,"sha256:"+"e"*64))
            connection.commit()
        connection.rollback()
        cursor.execute("SELECT COUNT(*) FROM jax_evidence.evidence_artifacts WHERE artifact_hash=%s",(marker,))
        assert cursor.fetchone()[0] == 0
    finally: connection.close()

def test_b7_writer_failure_rolls_back_real_governed_execution():
    _apply_migration(); now=datetime.now(timezone.utc); decision=record()
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now)
    store=MariaDBExecutionStore(_connection); store.insert_authorization(auth)
    def fail(_cursor, _record): raise RuntimeError("forced B7 persistence failure")
    store.execution_evidence_writer=fail
    with pytest.raises(RuntimeError): create_execution(store, auth, now_utc=now)
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s",(auth.authorization_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE decision_id=%s",(decision.decision_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE event_type='EXECUTION_CREATED'") == 0

def test_b7_dispatch_writer_failure_rolls_back_dispatch_event():
    from policy.execution_control.service import dispatch_execution
    _apply_migration(); now=datetime.now(timezone.utc); decision=record()
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now)
    store=MariaDBExecutionStore(_connection); store.insert_authorization(auth); execution=create_execution(store, auth, now_utc=now)
    def fail(_cursor, _event): raise RuntimeError("forced B7 dispatch evidence failure")
    store.dispatch_evidence_writer=fail
    with pytest.raises(RuntimeError): dispatch_execution(store, execution, auth, now_utc=now, job_id="b7-job")
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s AND event_type='MOTOR_DISPATCHED'",(execution.execution_id,)) == 0

def test_b7_composed_writer_persists_with_same_create_transaction():
    """The installed writer uses the B6 cursor; no second B7 commit exists."""
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    from policy.execution_control.service import configure_b7_execution_evidence
    _apply_migration(); _apply_evidence_migration(); now=datetime.now(timezone.utc); decision=record()
    evidence=MariaDBEvidenceStore(_connection); manifest=evidence.put_evidence_blob(b"runtime-manifest")
    identity=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,manifest.evidence_hash)
    recorder=RuntimeEvidenceRecorder(EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity)),"ci",ClaimScope(ClaimEnvironment.CI,database_scope_id="ci-mariadb"))
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now)
    store=MariaDBExecutionStore(_connection); store.insert_authorization(auth)
    configure_b7_execution_evidence(store,evidence,recorder)
    execution=create_execution(store,auth,now_utc=now)
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.enforcement_observations WHERE canonical_observation LIKE %s",("%"+execution.execution_id+"%",)) == 1

def test_b7_composed_writer_persists_with_same_dispatch_transaction():
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    from policy.execution_control.service import configure_b7_execution_evidence, dispatch_execution
    _apply_migration(); _apply_evidence_migration(); now=datetime.now(timezone.utc); decision=record()
    evidence=MariaDBEvidenceStore(_connection); manifest=evidence.put_evidence_blob(b"dispatch-runtime-manifest")
    identity=ImplementationIdentity("fjruizhn/Jax","c"*40,"d"*40,SourceState.CLEAN,manifest.evidence_hash)
    recorder=RuntimeEvidenceRecorder(EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity)),"ci",ClaimScope(ClaimEnvironment.CI,database_scope_id="ci-mariadb"))
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now); store=MariaDBExecutionStore(_connection); store.insert_authorization(auth)
    configure_b7_execution_evidence(store,evidence,recorder); execution=create_execution(store,auth,now_utc=now)
    dispatch_execution(store,execution,auth,now_utc=now,job_id="b7-shared")
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s AND event_type='MOTOR_DISPATCHED'",(execution.execution_id,)) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.enforcement_observations WHERE canonical_observation LIKE %s",("%MOTOR_DISPATCHED%",)) == 0 # event text is deliberately not evidence payload
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.enforcement_observations WHERE canonical_observation LIKE %s",("%"+execution.execution_id+"%",)) == 2

def test_b7_live_inspector_observes_installed_execution_schema():
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    from policy.enforcement_evidence.database_evidence import DatabaseControlInspector
    _apply_migration(); _apply_evidence_migration()
    evidence=MariaDBEvidenceStore(_connection); manifest=evidence.put_evidence_blob(b"inspection-manifest")
    identity=ImplementationIdentity("fjruizhn/Jax","e"*40,"f"*40,SourceState.CLEAN,manifest.evidence_hash)
    recorder=RuntimeEvidenceRecorder(EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity)),"ci",ClaimScope(ClaimEnvironment.CI,database_scope_id="ci-mariadb"))
    observation=DatabaseControlInspector(_connection,recorder,deployment_id="ci").inspect_one_decision_one_execution()
    assert observation.subject.identity.startswith("dbscope:sha256:")
    assert observation.outcome.value == "SATISFIED"


def _b8_runtime_identity_fixture(tmp_path):
    """Install a test deployment identity only through fixed composition."""
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState
    from policy.enforcement_evidence.control_registry import _controls, load_control_definition
    from policy.enforcement_evidence.implementation_identity import _V1_REQUIRED_SOURCE_PATHS
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    _apply_evidence_migration()
    evidence=MariaDBEvidenceStore(_connection)
    root=Path(__file__).parents[2]
    files={name:"sha256:"+hashlib.sha256((root/name).read_bytes()).hexdigest() for name in _V1_REQUIRED_SOURCE_PATHS}
    manifest=evidence.put_evidence_blob(json.dumps({"schema_version":"1.0","kind":"JAX_BUILD_MANIFEST","files":files},sort_keys=True,separators=(",",":")).encode())
    identity=ImplementationIdentity("fjruizhn/Jax","1"*40,"2"*40,SourceState.CLEAN,manifest.evidence_hash)
    EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity))
    for definition in _controls.values():
        # Persist only the packaged registry instance; raw module values do
        # not themselves carry the trusted-definition provenance marker.
        evidence._MariaDBEvidenceStore__persist_control_definition(
            load_control_definition(definition.control_id, definition.control_version))
    path=tmp_path/"implementation-identity.json"; path.write_text(json.dumps(identity.projection()),encoding="utf-8")
    return evidence, path

def _b7_counts():
    tables=("evidence_blobs","evidence_artifacts","enforcement_observations","enforcement_assertions","assertion_artifacts","assertion_observations")
    return {name:_scalar("SELECT COUNT(*) FROM jax_evidence."+name) for name in tables}

def test_b8_jaxctl_control_real_mariadb_is_zero_write(monkeypatch, tmp_path, capsys):
    """The actual CLI/runtime/read-only B7 path cannot alter authoritative rows."""
    evidence, identity_path=_b8_runtime_identity_fixture(tmp_path)
    import policy.enforcement_evidence.implementation_identity as implementation_identity
    from jaxctl.commands import run
    monkeypatch.setattr(implementation_identity,"_DEPLOYMENT_IDENTITY_PATH",str(identity_path))
    before=_b7_counts()
    status=run(["control","CTL.B6.GOVERNED_DISPATCH","--version","1","--claim","ENFORCED","--scope",'{"environment":"SANDBOX_RUNTIME"}',"--subjects",'[{"subject_type":"EXECUTION","identity":"b8-readonly"}]',"--json"])
    assert status == 0 and '"persisted":false' in capsys.readouterr().out
    assert _b7_counts() == before

def test_b8_jaxctl_control_unavailable_is_zero_write(monkeypatch, tmp_path, capsys):
    _evidence, identity_path=_b8_runtime_identity_fixture(tmp_path)
    import policy.enforcement_evidence.implementation_identity as implementation_identity
    from jaxctl.commands import run
    monkeypatch.setattr(implementation_identity,"_DEPLOYMENT_IDENTITY_PATH",str(identity_path)+".missing")
    before=_b7_counts()
    status=run(["control","CTL.B6.GOVERNED_DISPATCH","--version","1","--claim","ENFORCED","--scope",'{"environment":"SANDBOX_RUNTIME"}',"--subjects",'[{"subject_type":"EXECUTION","identity":"b8-unavailable"}]',"--json"])
    assert status == 2 and '"status":"UNAVAILABLE"' in capsys.readouterr().out
    assert _b7_counts() == before


def _scalar(sql, args=()):
    connection = _connection()
    try:
        cursor = connection.cursor(); cursor.execute(sql, args)
        return cursor.fetchone()[0]
    finally:
        connection.close()


def test_governed_execution_authoritative_mariadb_contract():
    _apply_migration()
    now = datetime.now(timezone.utc)
    decision = record()
    def bound_request(for_decision):
        return build_execution_request(for_decision, authenticated_caller_id="jacobs", capability="CAP", motor="m",
            environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE",
            prompt="p", context={"x": 1}, timeout_seconds=60)
    auth = authorize_execution(decision, bound_request(decision), catalog(), now_utc=now)
    store = MariaDBExecutionStore(_connection)
    store.insert_authorization(auth)

    loaded = store.load_authorization(auth.authorization_id)
    assert loaded._is_trusted() and loaded.execution_request._is_trusted()
    parsed = deserialize_execution_authorization(loaded.projection_without_hash() | {
        "execution_authorization_hash": loaded.execution_authorization_hash})
    assert not parsed._is_trusted()

    execution = create_execution(store, loaded, now_utc=now)
    assert execution.execution_id
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE execution_id=%s", (execution.execution_id,)) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s", (auth.authorization_id,)) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s", (execution.execution_id,)) == 1
    assert _scalar("SELECT canonical_record FROM jax_execution.execution_records WHERE execution_id=%s", (execution.execution_id,))

    duplicate = authorize_execution(decision, bound_request(decision), catalog(), now_utc=now)
    store.insert_authorization(duplicate)
    with pytest.raises(DecisionExecutionConflictError):
        create_execution(store, store.load_authorization(duplicate.authorization_id), now_utc=now)

    # Independent sessions racing on one decision must leave one durable row.
    concurrent_decision = record()
    first = authorize_execution(concurrent_decision, bound_request(concurrent_decision), catalog(), now_utc=now)
    second = authorize_execution(concurrent_decision, bound_request(concurrent_decision), catalog(), now_utc=now)
    store.insert_authorization(first); store.insert_authorization(second)
    def create(auth_id):
        try:
            return create_execution(MariaDBExecutionStore(_connection),
                MariaDBExecutionStore(_connection).load_authorization(auth_id), now_utc=now)
        except Exception:  # fail-soft: the losing concurrent transaction is the expected uniqueness proof.
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(create, (first.authorization_id, second.authorization_id)))
    assert sum(item is not None for item in outcomes) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE decision_id=%s", (concurrent_decision.decision_id,)) == 1

    # An error after the consumption insert rolls the whole transaction back.
    failing_decision = record()
    failing = authorize_execution(failing_decision, bound_request(failing_decision), catalog(), now_utc=now)
    store.insert_authorization(failing)
    failing_loaded = store.load_authorization(failing.authorization_id)
    failing_record = _record(failing_loaded, now_utc=now)
    with pytest.raises(Exception):
        store.create_execution(failing_loaded, failing_record,
            ExecutionEvent(failing_record.execution_id, "READY_TO_DISPATCH", "EXECUTION_CREATED", now, "x" * 129))
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s", (failing.authorization_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE execution_id=%s", (failing_record.execution_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s", (failing_record.execution_id,)) == 0

    connection = _connection()
    try:
        cursor = connection.cursor()
        for table in ("execution_records", "execution_events"):
            with pytest.raises(Exception):
                cursor.execute(f"UPDATE jax_execution.{table} SET event_type='X'" if table == "execution_events"
                    else f"UPDATE jax_execution.{table} SET decision_id=decision_id")
            connection.rollback()
            with pytest.raises(Exception):
                cursor.execute(f"DELETE FROM jax_execution.{table}")
            connection.rollback()
    finally:
        connection.close()

    # Tampered canonical bytes and duplicated row keys are both fail closed.
    connection = _connection()
    try:
        cursor = connection.cursor()
        cursor.execute("DROP TRIGGER jax_execution.execution_authorizations_no_update")
        cursor.execute("UPDATE jax_execution.execution_authorizations SET canonical_authorization=JSON_SET(canonical_authorization, '$.execution_request.motor', 'tampered') WHERE authorization_id=%s", (duplicate.authorization_id,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(Exception):
        store.load_authorization(duplicate.authorization_id)

    connection = _connection()
    try:
        cursor = connection.cursor()
        cursor.execute("UPDATE jax_execution.execution_authorizations SET execution_request_hash='sha256:" + "0" * 64 + "' WHERE authorization_id=%s", (auth.authorization_id,))
        cursor.execute("DROP TRIGGER jax_execution.execution_records_no_update")
        cursor.execute("UPDATE jax_execution.execution_records SET canonical_record=JSON_SET(canonical_record, '$.motor', 'tampered') WHERE execution_id=%s", (execution.execution_id,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(UnknownExecutionError):
        store.load_authorization(auth.authorization_id)
    with pytest.raises(Exception):
        store.load_execution(execution.execution_id)
