"""Serialización JSON canónica de los valores semánticos aceptados."""

from __future__ import annotations

import json
from typing import Any

from .normalize import normalize_semantic


def canonical_json_bytes(value: Any) -> bytes:
    normalized = normalize_semantic(value)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
