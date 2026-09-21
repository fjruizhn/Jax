"""Block 6 evidence against the real MariaDB adapter (CI service only)."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
    from policy.enforcement_evidence.models import (ImplementationIdentity, SourceState, EvidenceArtifact,
        EvidenceType, EvidenceClass, EvidenceTrustDomain, EvidenceSubject, EvidenceSubjectType,
        EvidenceBlobRef, ClaimScope, ClaimEnvironment, EnforcementObservation, ObservationOutcome)
    _apply_evidence_migration(); store=MariaDBEvidenceStore(_connection); now=datetime.now(timezone.utc)
    manifest=store.put_evidence_blob(b"b7-manifest"); definition=load_control_definition("CTL.B6.GOVERNED_DISPATCH")
    identity=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,manifest.evidence_hash)
    raw=store.put_evidence_blob(b"control=CTL.B6.GOVERNED_DISPATCH;reason=DENIED")
    subject=EvidenceSubject(EvidenceSubjectType.OPERATION_ATTEMPT,"b7-db-attempt")
    artifact=EvidenceArtifact(EvidenceType.CONTROL_INPUT,EvidenceClass.RUNTIME_OBSERVATION,definition.control_id,1,definition.control_definition_hash,subject,(EvidenceBlobRef(raw.evidence_hash,"bounded_input","text/plain","utf-8"),),EvidenceTrustDomain.JAX_RUNTIME,"ci",identity.implementation_identity_hash,now)
    store._record_artifact(artifact, _token=store._fixed_lifecycle_token())
    loaded=store.load_evidence_artifact(artifact.artifact_hash)
    assert loaded.artifact_hash == artifact.artifact_hash
    observation=EnforcementObservation("b7-db-observation",definition.control_id,1,definition.control_definition_hash,subject,identity.implementation_identity_hash,ObservationOutcome.DENIED,"DENIED",now,ClaimScope(ClaimEnvironment.CI),(artifact.artifact_hash,))
    store._record_observation(observation, _token=store._fixed_lifecycle_token())
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.observation_artifacts WHERE observation_id=%s",(observation.observation_id,)) == 1
    assert store.load_observation(observation.observation_id).observation_hash == observation.observation_hash
    from policy.enforcement_evidence.models import EnforcementAssertion, ClaimLevel, AssertionVerdict
    assertion=EnforcementAssertion(definition.control_id,1,definition.control_definition_hash,ClaimLevel.ENFORCED,AssertionVerdict.NOT_OBSERVED,identity.implementation_identity_hash,ClaimScope(ClaimEnvironment.CI),(subject,),(artifact.artifact_hash,),(observation.observation_id,),now,now,now)
    store._record_assertion(assertion, _token=store._fixed_lifecycle_token())
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.enforcement_assertions WHERE assertion_hash=%s",(assertion.assertion_hash,)) == 1
    assert store.load_assertion(assertion.assertion_hash).assertion_hash == assertion.assertion_hash
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
