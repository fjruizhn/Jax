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
