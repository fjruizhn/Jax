"""Append-only authority event storage boundary.

The in-memory implementation is for deterministic unit tests.  A production
MariaDB adapter must implement the same locked append contract; replay, not a
cache, remains authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol

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
