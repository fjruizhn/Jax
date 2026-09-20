"""Identifiers accepted by the authority ledger."""
from __future__ import annotations

import re
import uuid

from .errors import AuthorityEventValidationError

_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EVIDENCE = _SHA


def sha256_id(value: object, field: str = "hash") -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise AuthorityEventValidationError(f"{field} debe ser sha256 lowercase")
    return value


def uuid7_text(value: object, field: str = "event_id") -> str:
    if not isinstance(value, str):
        raise AuthorityEventValidationError(f"{field} inválido")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise AuthorityEventValidationError(f"{field} inválido") from exc
    # Python's uuid.UUID accepts all UUID versions; v7 is an explicit contract.
    if parsed.version != 7:
        raise AuthorityEventValidationError(f"{field} debe ser UUIDv7")
    return str(parsed)


def canonical_evidence_refs(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise AuthorityEventValidationError("evidence_refs debe ser array")
    refs = tuple(sha256_id(value, "evidence_ref") for value in values)
    if len(refs) != len(set(refs)):
        raise AuthorityEventValidationError("evidence_refs duplicadas")
    return tuple(sorted(refs))
