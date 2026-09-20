"""Append-only authority event storage boundary.

The in-memory implementation is for deterministic unit tests.  A production
MariaDB adapter must implement the same locked append contract; replay, not a
cache, remains authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol, Callable, Any

from .errors import AuthorityStateError
from .models import AuthorityEvent, AuthorityLedgerGenesis


class AuthorityLedgerStore(Protocol):
    def get_genesis(self) -> AuthorityLedgerGenesis: ...
    def events(self) -> tuple[AuthorityEvent, ...]: ...
    def append(self, event: AuthorityEvent) -> None: ...


@dataclass
class InMemoryAuthorityLedgerStore:
    """Test-only transactional append model with one linear predecessor."""
    genesis: AuthorityLedgerGenesis
    _events: list[AuthorityEvent] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def get_genesis(self) -> AuthorityLedgerGenesis:
        return self.genesis

    def events(self) -> tuple[AuthorityEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def append(self, event: AuthorityEvent) -> None:
        with self._lock:
            expected_sequence = len(self._events) + 1
            predecessor = self._events[-1].event_hash if self._events else None
            if event.sequence != expected_sequence or event.previous_event_hash != predecessor:
                raise AuthorityStateError("append fuera de secuencia/predecesor")
            if any(existing.event_id == event.event_id for existing in self._events):
                raise AuthorityStateError("event_id duplicado")
            self._events.append(event)


class MariaDBAuthorityLedgerStore:
    """Production DB-API adapter. Connection factory owns repository deployment wiring.

    The adapter deliberately uses only INSERT for events and locks the singleton
    head row while checking the signed caller's expected predecessor.
    """
    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connect = connection_factory

    def get_genesis(self) -> AuthorityLedgerGenesis:
        from .serialization import genesis_from_projection
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT canonical_genesis FROM jax_authority.authority_ledger_genesis WHERE singleton=1")
            row = cursor.fetchone()
            if row is None:
                raise AuthorityStateError("genesis ausente")
            return genesis_from_projection(row[0])

    def events(self) -> tuple[AuthorityEvent, ...]:
        from .serialization import event_from_storage_row
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT sequence,event_id,canonical_event FROM jax_authority.authority_events ORDER BY sequence ASC")
            return tuple(event_from_storage_row(row) for row in cursor.fetchall())

    def append(self, event: AuthorityEvent) -> None:
        from .canonical import canonical_bytes
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT sequence,head_event_hash FROM jax_authority.authority_ledger_head WHERE singleton=1 FOR UPDATE")
            row = cursor.fetchone()
            sequence, predecessor = row if row else (0, None)
            if event.sequence != sequence + 1 or event.previous_event_hash != predecessor:
                raise AuthorityStateError("append concurrente/stale")
            cursor.execute("INSERT INTO jax_authority.authority_events (sequence,event_id,event_type,actor_id,canonical_event,previous_event_hash,event_hash,signature,recorded_at_utc) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", (event.sequence,event.event_id,event.intent.event_type.value,event.intent.actor_id,canonical_bytes(event),event.previous_event_hash,event.event_hash,event.signature,event.recorded_at_utc))
            cursor.execute("UPDATE jax_authority.authority_ledger_head SET sequence=%s,head_event_id=%s,head_event_hash=%s WHERE singleton=1", (event.sequence,event.event_id,event.event_hash))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
