"""Closed identifiers and evidence references for Block 5."""
from __future__ import annotations

import uuid

from policy.authority_ledger.ids import canonical_evidence_refs, sha256_id, uuid7_text
from policy.authority_ledger.errors import AuthorityEventValidationError

from .errors import DecisionContractError


def decision_id(value: object, field: str = "decision_id") -> str:
    try:
        return uuid7_text(value, field)
    except AuthorityEventValidationError as exc:
        raise DecisionContractError(f"{field} debe ser UUIDv7") from exc


def new_decision_id() -> str:
    creator = getattr(uuid, "uuid7", None)
    if creator is None:
        raise DecisionContractError("runtime no provee UUIDv7")
    return decision_id(str(creator()))


__all__ = ["canonical_evidence_refs", "decision_id", "new_decision_id", "sha256_id"]
