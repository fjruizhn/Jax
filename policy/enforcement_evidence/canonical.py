"""Canonical projections for the Block 7 value objects."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from policy.canonicalization.canonical_json import canonical_json_bytes
def utc_text(v: datetime) -> str:
    if not isinstance(v, datetime) or v.tzinfo is None: raise TypeError("UTC requerido")
    return v.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
def plain(v: Any) -> Any:
    if isinstance(v, Enum): return v.value
    if isinstance(v, datetime): return utc_text(v)
    if isinstance(v, tuple): return [plain(x) for x in v]
    if isinstance(v, Mapping): return {str(k): plain(x) for k,x in v.items()}
    if is_dataclass(v): return {f.name: plain(getattr(v,f.name)) for f in fields(v) if not f.name.endswith("hash")}
    return v
def canonical_bytes(value: Any) -> bytes: return canonical_json_bytes(plain(value))
