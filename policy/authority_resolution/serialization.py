"""Canonical, non-hashing representation of resolver results."""
from __future__ import annotations

from dataclasses import asdict
from enum import Enum

from policy.canonicalization.canonical_json import canonical_json_bytes

from .models import StaticAuthorityResolution

def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    if isinstance(value, list):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    return value

def canonical_resolution_dict(result: StaticAuthorityResolution) -> dict:
    return _plain(asdict(result))

def canonical_resolution_bytes(result: StaticAuthorityResolution) -> bytes:
    return canonical_json_bytes(canonical_resolution_dict(result))
