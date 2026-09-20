"""Proyección normativa explícita gobernada por el bootstrap pinneado."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .bootstrap import VerifiedBootstrap, verified_bootstrap
from .errors import SchemaValidationError
from .schemas import validate_legacy_rule


def normative_projection(
    rule: dict[str, Any], bootstrap: VerifiedBootstrap | None = None
) -> dict[str, Any]:
    trusted = bootstrap or verified_bootstrap()
    validate_legacy_rule(rule, trusted)
    classifications = trusted.field_classes["fields"]
    forbidden = [
        field
        for field, field_class in classifications.items()
        if field_class == "FACTUAL_FORBIDDEN" and field in rule
    ]
    if forbidden:
        raise SchemaValidationError(
            f"fuente contiene campos FACTUAL_FORBIDDEN: {sorted(forbidden)}"
        )
    return {
        field: deepcopy(rule[field])
        for field, field_class in classifications.items()
        if field_class == "NORMATIVE" and field in rule
    }
