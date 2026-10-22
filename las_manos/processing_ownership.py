"""Immutable Platform-authenticated ownership for Processing jobs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


OWNER_VERSION = "processing-owner.1"
HEADER_OWNER_VERSION = "x-jax-processing-owner-version"
HEADER_TENANT_ID = "x-jax-processing-tenant-id"
HEADER_USER_ID = "x-jax-processing-user-id"
HEADER_PROJECT_ID = "x-jax-processing-project-id"
PROCESSING_OWNER_HEADERS = (
    HEADER_OWNER_VERSION, HEADER_TENANT_ID, HEADER_USER_ID, HEADER_PROJECT_ID,
)


class ProcessingOwnershipError(ValueError):
    pass


def _canonical_id(value: object) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or value == "0" or value.startswith("0"):
        raise ProcessingOwnershipError("processing ownership identifier is noncanonical")
    return value


@dataclass(frozen=True)
class ProcessingOwnershipContext:
    version: Literal["processing-owner.1"]
    tenant_id: str
    user_id: str
    project_id: str

    def __post_init__(self) -> None:
        if self.version != OWNER_VERSION:
            raise ProcessingOwnershipError("processing ownership version is unsupported")
        for value in (self.tenant_id, self.user_id, self.project_id):
            _canonical_id(value)


def processing_ownership_from_headers(headers: Sequence[tuple[bytes, bytes]]) -> ProcessingOwnershipContext:
    values: dict[str, str] = {}
    for raw_name, raw_value in headers:
        name = raw_name.decode("latin-1").lower()
        if name.startswith("x-jax-processing-") and name not in PROCESSING_OWNER_HEADERS:
            raise ProcessingOwnershipError("unknown processing ownership header")
        if name in PROCESSING_OWNER_HEADERS:
            if name in values:
                raise ProcessingOwnershipError("duplicate processing ownership header")
            try:
                values[name] = raw_value.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ProcessingOwnershipError("processing ownership header is not ASCII") from exc
    if set(values) != set(PROCESSING_OWNER_HEADERS):
        raise ProcessingOwnershipError("missing processing ownership header")
    return ProcessingOwnershipContext(
        values[HEADER_OWNER_VERSION], values[HEADER_TENANT_ID],
        values[HEADER_USER_ID], values[HEADER_PROJECT_ID],
    )


def processing_ownership_from_scope(scope) -> ProcessingOwnershipContext:
    state = scope.get("state", {})
    value = state.get("processing_ownership")
    if not isinstance(value, ProcessingOwnershipContext):
        raise ProcessingOwnershipError("processing ownership context is absent")
    return value


def processing_owner_headers(context: ProcessingOwnershipContext) -> dict[str, str]:
    return {
        "X-Jax-Processing-Owner-Version": context.version,
        "X-Jax-Processing-Tenant-Id": context.tenant_id,
        "X-Jax-Processing-User-Id": context.user_id,
        "X-Jax-Processing-Project-Id": context.project_id,
    }
