"""Authoritative governed-execution stores; in-memory is strictly test support."""
from __future__ import annotations
from dataclasses import dataclass
from threading import Lock

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
    """Production adapter.  The supplied connection factory owns credentials."""
    def __init__(self, connection_factory): self._connection_factory = connection_factory
    # The async service adapter intentionally keeps SQL explicit: identity artifacts are INSERT/SELECT only.
    async def create_execution(self, authorization, record, initial_event):
        connection = await self._connection_factory()
        try:
            async with connection.cursor() as cur:
                await cur.execute("SELECT authorization_id FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s FOR UPDATE", (authorization.authorization_id,))
                if await cur.fetchone(): raise AuthorizationConsumedError("authorization ya consumida")
                await cur.execute("INSERT INTO jax_execution.execution_authorization_consumptions (authorization_id) VALUES (%s)", (authorization.authorization_id,))
                await cur.execute("INSERT INTO jax_execution.execution_records (execution_id, decision_id, canonical_record_hash) VALUES (%s,%s,%s)", (record.execution_id, record.decision_id, record.execution_record_hash))
                await cur.execute("INSERT INTO jax_execution.execution_events (execution_id, state, event_type, event_at_utc) VALUES (%s,%s,%s,%s)", (initial_event.execution_id, initial_event.state, initial_event.event_type, initial_event.at_utc))
            await connection.commit()
        except (OSError, RuntimeError, ValueError):
            await connection.rollback(); raise
        finally: connection.close()
        return record
