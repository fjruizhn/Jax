"""Strict persistence codecs for immutable ledger records."""
from __future__ import annotations

import json
from datetime import datetime

from .models import AuthorityEvent, AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis


def genesis_from_projection(value):
    if isinstance(value, bytes): value = value.decode("utf-8")
    if isinstance(value, str): value = json.loads(value)
    if not isinstance(value, dict): raise TypeError("canonical_genesis inválido")
    return AuthorityLedgerGenesis(**value)


def event_from_storage_row(row):
    sequence, event_id, value = row
    if isinstance(value, bytes): value = value.decode("utf-8")
    if isinstance(value, str): value = json.loads(value)
    # Stored canonical event uses the closed projection; unknown shapes fail.
    intent_data = value["intent"]
    intent = AuthorityEventIntent(AuthorityEventType(intent_data["event_type"]), intent_data["actor_id"], tuple(intent_data["evidence_refs"]))
    return AuthorityEvent(event_id, sequence, value["previous_event_hash"], intent, datetime.fromisoformat(value["recorded_at_utc"]), value["signature"], value["event_hash"])
