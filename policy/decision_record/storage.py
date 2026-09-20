"""Immutable decision record storage; in-memory is only for unit tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Protocol

from .errors import DecisionIdConflictError, DecisionStorageError
from .serialization import canonical_decision_record_bytes, decision_record_from_bytes
from .models import DecisionRecord


class DecisionRecordStore(Protocol):
    def insert(self, record: DecisionRecord) -> DecisionRecord: ...
    def load_canonical_bytes(self, decision_id: str) -> bytes: ...


@dataclass
class InMemoryDecisionRecordStore:
    _records: dict[str, bytes] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def insert(self, record: DecisionRecord) -> DecisionRecord:
        raw = canonical_decision_record_bytes(record)
        with self._lock:
            existing = self._records.get(record.decision_id)
            if existing is None:
                self._records[record.decision_id] = raw
                return record
            if existing != raw:
                raise DecisionIdConflictError("decision_id ya posee contenido distinto")
            return decision_record_from_bytes(existing)

    def load_canonical_bytes(self, value: str) -> bytes:
        with self._lock:
            try: return self._records[value]
            except KeyError as exc: raise DecisionStorageError("DecisionRecord no encontrado") from exc


class MariaDBDecisionRecordStore:
    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connect = connection_factory

    def insert(self, record: DecisionRecord) -> DecisionRecord:
        try:
            from pymysql.err import IntegrityError
        except ImportError as exc:
            raise DecisionStorageError("MariaDB adapter requiere driver DB-API") from exc
        raw = canonical_decision_record_bytes(record)
        b = record.authority_binding
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("INSERT INTO jax_decisions.decision_records (decision_id,decision_input_hash,policy_corpus_hash,effective_authority_context_hash,authority_ledger_checkpoint_hash,resolver_identity,resolver_version,decision_record_hash,canonical_record,recorded_at_utc) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (record.decision_id, record.decision_input_hash, b.active_policy_corpus_hash, b.effective_authority_context_hash,
                 b.authority_ledger_checkpoint_hash, b.resolver_identity, b.resolver_version, record.decision_record_hash, raw, record.recorded_at_utc))
            conn.commit()
            return record
        except IntegrityError as exc:
            conn.rollback()
            # Only duplicate-key handling is translated. A query/programming
            # failure below propagates rather than being disguised as a record error.
            cur = conn.cursor()
            cur.execute("SELECT canonical_record FROM jax_decisions.decision_records WHERE decision_id=%s", (record.decision_id,))
            row = cur.fetchone()
            if row is not None:
                raw_existing = row[0] if isinstance(row[0], bytes) else row[0].encode("utf-8")
                if raw_existing == raw:
                    return decision_record_from_bytes(raw_existing)
                raise DecisionIdConflictError("decision_id ya posee contenido distinto")
            raise DecisionStorageError("violación de integridad sin record existente") from exc
        finally:
            conn.close()

    def load_canonical_bytes(self, value: str) -> bytes:
        conn = self._connect()
        try:
            cur = conn.cursor(); cur.execute("SELECT canonical_record,decision_input_hash,policy_corpus_hash,effective_authority_context_hash,authority_ledger_checkpoint_hash,resolver_identity,resolver_version,decision_record_hash FROM jax_decisions.decision_records WHERE decision_id=%s", (value,)); row = cur.fetchone()
            if row is None: raise DecisionStorageError("DecisionRecord no encontrado")
            raw = row[0] if isinstance(row[0], bytes) else row[0].encode("utf-8")
            record = decision_record_from_bytes(raw)
            expected = (record.decision_input_hash, record.authority_binding.active_policy_corpus_hash,
                        record.authority_binding.effective_authority_context_hash, record.authority_binding.authority_ledger_checkpoint_hash,
                        record.authority_binding.resolver_identity, record.authority_binding.resolver_version, record.decision_record_hash)
            if tuple(row[1:]) != expected: raise DecisionStorageError("índices denormalizados no coinciden")
            return raw
        finally:
            conn.close()
