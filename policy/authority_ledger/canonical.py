"""Canonical bytes and domain-separated SHA-256 primitives for Block 4."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from enum import Enum
from typing import Any

from policy.canonicalization.canonical_json import canonical_json_bytes


def plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return plain(asdict(value))
    return value


def canonical_bytes(value: Any) -> bytes:
    return canonical_json_bytes(plain(value))


def domain_hash(domain: str, version: str, projection: Any) -> str:
    data = domain.encode("ascii") + b"\0" + version.encode("ascii") + b"\0" + canonical_bytes(projection)
    return "sha256:" + hashlib.sha256(data).hexdigest()
