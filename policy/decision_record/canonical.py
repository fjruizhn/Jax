"""Canonical projections and domain hashes for Block 5."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from policy.authority_ledger.canonical import domain_hash
from policy.canonicalization.canonical_json import canonical_json_bytes


def utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError("datetime UTC requerido")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return utc_text(value)
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            item.name: plain(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    return value


def canonical_bytes(value: Any) -> bytes:
    return canonical_json_bytes(plain(value))


def decision_input_hash(projection: Mapping[str, Any]) -> str:
    return domain_hash("JAX-DECISION-INPUT", "1.0", projection)


def decision_record_hash(projection: Mapping[str, Any]) -> str:
    return domain_hash("JAX-DECISION-RECORD", "1.0", projection)
