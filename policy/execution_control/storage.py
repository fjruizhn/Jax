"""Authoritative governed-execution stores; in-memory is strictly test support."""
from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
import json
from datetime import datetime, timezone
from .models import ExecutionEnvironment, ExecutionRecord
from .canonical import execution_record_hash

from .errors import (AuthorizationConsumedError, DecisionExecutionConflictError,
                     HumanApprovalConsumedError, UnknownExecutionError)

@dataclass(frozen=True)
class ExecutionEvent:
    execution_id: str; state: str; event_type: str; at_utc: object; job_id: str | None = None

class InMemoryExecutionStore:
    """Test-only transactional model with the same uniqueness invariants as MariaDB."""
    def __init__(self) -> None:
        self._lock = Lock(); self._authorizations = {}; self._records = {}; self._by_decision = {}
        self._events = {}; self._auth_consumed = set(); self._approval_consumed = set(); self._dry_runs = {}
    def insert_authorization(self, value):
        with self._lock: self._authorizations[value.authorization_id] = value
        return value
    def load_authorization(self, authorization_id):
        try: return self._authorizations[authorization_id]
        except KeyError as exc: raise UnknownExecutionError(authorization_id) from exc
    def create_execution(self, authorization, record, initial_event):
        with self._lock:
            if authorization.authorization_id in self._auth_consumed: raise AuthorizationConsumedError("authorization ya consumida")
            if record.decision_id in self._by_decision: raise DecisionExecutionConflictError("decision ya tiene execution")
            self._auth_consumed.add(authorization.authorization_id); self._records[record.execution_id] = record
            self._by_decision[record.decision_id] = record.execution_id; self._events[record.execution_id] = [initial_event]
        return record
    def load_execution(self, execution_id):
        try: return self._records[execution_id]
        except KeyError as exc: raise UnknownExecutionError(execution_id) from exc
    def events(self, execution_id): return tuple(self._events.get(execution_id, ()))
    def append_event(self, event):
        with self._lock:
            if event.execution_id not in self._records: raise UnknownExecutionError(event.execution_id)
            self._events[event.execution_id].append(event)
    def consume_approval(self, approval_id: str):
        with self._lock:
            if approval_id in self._approval_consumed: raise HumanApprovalConsumedError("approval ya consumida")
            self._approval_consumed.add(approval_id)
    def save_dry_run(self, artifact):
        with self._lock: self._dry_runs[artifact.execution_id] = artifact
    def load_dry_run(self, execution_id): return self._dry_runs.get(execution_id)

class MariaDBExecutionStore:
    """Synchronous DB-API production adapter.

    The governed service/store protocol is deliberately synchronous for both
    its test and MariaDB implementations.  Callers running async web code
    invoke it in their normal DB worker boundary; an aiomysql coroutine must
    not be substituted for this DB-API factory.
    """
    def __init__(self, connection_factory): self._connection_factory = connection_factory
    def _canonical(self, value):
        if hasattr(value, "execution_record_hash"):
            data = value.projection_without_hash() | {"execution_record_hash": value.execution_record_hash}
        elif hasattr(value, "execution_authorization_hash"):
            data = value.projection_without_hash() | {"execution_authorization_hash": value.execution_authorization_hash}
        elif hasattr(value, "projection"):
            data = value.projection()
        else:
            raise ValueError("artifact canonical no soportado")
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def insert_authorization(self, value):
        connection = self._connection_factory()
        try:
            cur = connection.cursor()
            cur.execute("INSERT INTO jax_execution.execution_authorizations (authorization_id,decision_id,canonical_authorization_hash,canonical_authorization,created_at_utc) VALUES (%s,%s,%s,%s,%s)",
                        (value.authorization_id, value.decision_id, value.execution_authorization_hash,
                         self._canonical(value), value.issued_at_utc))
            connection.commit()
            return value
        except Exception:
            connection.rollback(); raise
        finally:
            connection.close()

    def create_execution(self, authorization, record, initial_event):
        connection = self._connection_factory()
        try:
            cur = connection.cursor()
            # Lock the authoritative authorization row, then consume and
            # create the record/event in one transaction.
            cur.execute("SELECT canonical_authorization_hash FROM jax_execution.execution_authorizations WHERE authorization_id=%s FOR UPDATE", (authorization.authorization_id,))
            row = cur.fetchone()
            if row is None or row[0] != authorization.execution_authorization_hash:
                raise UnknownExecutionError(authorization.authorization_id)
            cur.execute("SELECT authorization_id FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s FOR UPDATE", (authorization.authorization_id,))
            if cur.fetchone():
                raise AuthorizationConsumedError("authorization ya consumida")
            cur.execute("SELECT execution_id FROM jax_execution.execution_records WHERE decision_id=%s FOR UPDATE", (record.decision_id,))
            if cur.fetchone():
                raise DecisionExecutionConflictError("decision ya tiene execution")
            cur.execute("INSERT INTO jax_execution.execution_authorization_consumptions (authorization_id) VALUES (%s)", (authorization.authorization_id,))
            cur.execute("INSERT INTO jax_execution.execution_records (execution_id,decision_id,canonical_record_hash,canonical_record,created_at_utc) VALUES (%s,%s,%s,%s,%s)",
                        (record.execution_id, record.decision_id, record.execution_record_hash,
                         self._canonical(record), record.created_at_utc))
            cur.execute("INSERT INTO jax_execution.execution_events (execution_id,state,event_type,event_at_utc,job_id) VALUES (%s,%s,%s,%s,%s)",
                        (initial_event.execution_id, initial_event.state, initial_event.event_type,
                         initial_event.at_utc, initial_event.job_id))
            connection.commit()
        except Exception:
            connection.rollback(); raise
        finally:
            connection.close()
        return record

    def load_authorization(self, authorization_id):
        connection = self._connection_factory()
        try:
            cur = connection.cursor()
            cur.execute("SELECT canonical_authorization,canonical_authorization_hash FROM jax_execution.execution_authorizations WHERE authorization_id=%s", (authorization_id,))
            row = cur.fetchone()
            if row is None: raise UnknownExecutionError(authorization_id)
            raw = row[0].decode() if isinstance(row[0], bytes) else row[0]
            from .authorization import _load_authorization_from_authoritative_projection
            value = _load_authorization_from_authoritative_projection(json.loads(raw))
            if value.execution_authorization_hash != row[1] or value.authorization_id != authorization_id:
                raise UnknownExecutionError("authorization canonical corrupta")
            return value
        finally: connection.close()

    def load_execution(self, execution_id):
        connection = self._connection_factory()
        try:
            cur = connection.cursor()
            cur.execute("SELECT canonical_record,canonical_record_hash,decision_id FROM jax_execution.execution_records WHERE execution_id=%s", (execution_id,))
            row = cur.fetchone()
            if row is None: raise UnknownExecutionError(execution_id)
            data = json.loads(row[0].decode() if isinstance(row[0], bytes) else row[0])
            value = ExecutionRecord(data["schema_version"], data["kind"], data["execution_id"], data["decision_id"],
                data["decision_record_hash"], data["execution_request_hash"], data["execution_authorization_hash"],
                data["authorization_id"], data["capability"], data["authenticated_caller_id"], data["motor"],
                ExecutionEnvironment(data["environment"]), data["timeout_seconds"],
                datetime.fromisoformat(data["created_at_utc"].replace("Z", "+00:00")), data["execution_record_hash"])
            if value.execution_id != execution_id or value.decision_id != row[2] or value.execution_record_hash != row[1]:
                raise UnknownExecutionError("execution canonical corrupta")
            return value
        finally: connection.close()

    def events(self, execution_id):
        connection = self._connection_factory()
        try:
            cur = connection.cursor(); cur.execute("SELECT state,event_type,event_at_utc,job_id FROM jax_execution.execution_events WHERE execution_id=%s ORDER BY event_sequence", (execution_id,))
            return tuple(ExecutionEvent(execution_id, *row) for row in cur.fetchall())
        finally: connection.close()

    def append_event(self, event):
        connection = self._connection_factory()
        try:
            cur = connection.cursor(); cur.execute("INSERT INTO jax_execution.execution_events (execution_id,state,event_type,event_at_utc,job_id) VALUES (%s,%s,%s,%s,%s)", (event.execution_id,event.state,event.event_type,event.at_utc,event.job_id)); connection.commit()
        except Exception: connection.rollback(); raise
        finally: connection.close()

    def consume_approval(self, approval_id):
        connection=self._connection_factory()
        try:
            cur=connection.cursor(); cur.execute("INSERT INTO jax_execution.human_approval_consumptions (human_approval_id) VALUES (%s)",(approval_id,)); connection.commit()
        except Exception: connection.rollback(); raise
        finally: connection.close()

    def save_dry_run(self, artifact):
        connection=self._connection_factory()
        try:
            data = {"dry_run_id": artifact.dry_run_id, "execution_id": artifact.execution_id,
                "decision_id": artifact.decision_id, "execution_request_hash": artifact.execution_request_hash,
                "execution_authorization_hash": artifact.execution_authorization_hash, "capability": artifact.capability,
                "motor": artifact.motor, "environment": artifact.environment, "status": artifact.status,
                "result": artifact.result, "recorded_at_utc": artifact.recorded_at_utc.isoformat().replace("+00:00", "Z"),
                "dry_run_artifact_hash": artifact.dry_run_artifact_hash}
            cur=connection.cursor(); cur.execute("INSERT INTO jax_execution.dry_run_artifacts (dry_run_id,execution_id,artifact_hash,canonical_artifact) VALUES (%s,%s,%s,%s)",(artifact.dry_run_id,artifact.execution_id,artifact.dry_run_artifact_hash,json.dumps(data,sort_keys=True,separators=(",",":")))); connection.commit()
        except Exception: connection.rollback(); raise
        finally: connection.close()

    def load_dry_run(self, execution_id):
        # Dispatch only needs a reverified binding/status; full artifact is
        # reconstructed by the service's stored canonical contract in CI.
        connection=self._connection_factory()
        try:
            cur=connection.cursor(); cur.execute("SELECT canonical_artifact FROM jax_execution.dry_run_artifacts WHERE execution_id=%s",(execution_id,)); row=cur.fetchone()
            if row is None: return None
            from .dry_run import DryRunArtifact
            data=json.loads(row[0].decode() if isinstance(row[0],bytes) else row[0])
            return DryRunArtifact(data["dry_run_id"],data["execution_id"],data["decision_id"],data["execution_request_hash"],data["execution_authorization_hash"],data["capability"],data["motor"],data["environment"],data["status"],data["result"],datetime.fromisoformat(data["recorded_at_utc"].replace("Z","+00:00")),data["dry_run_artifact_hash"])
        finally: connection.close()
