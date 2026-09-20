"""Normalización semántica independiente de representación YAML."""

from __future__ import annotations

import unicodedata
from typing import Any

from .bootstrap import PINNED_UNICODE_VERSION
from .errors import CanonicalizationError


def normalize_semantic(value: Any) -> Any:
    if unicodedata.unidata_version != PINNED_UNICODE_VERSION:
        raise CanonicalizationError(
            "UNSUPPORTED_UNICODE_VERSION: "
            f"runtime={unicodedata.unidata_version}, pinned={PINNED_UNICODE_VERSION}"
        )
    if isinstance(value, float):
        raise CanonicalizationError("los floats no pertenecen al modelo canónico")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalize_semantic(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("todas las claves deben ser strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise CanonicalizationError(
                    f"claves duplicadas después de NFC: {normalized_key!r}"
                )
            normalized[normalized_key] = normalize_semantic(item)
        return normalized
    raise CanonicalizationError(f"tipo no canónico: {type(value).__name__}")
