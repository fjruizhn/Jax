"""Pure, opaque Office V1 execution contract.

This module deliberately contains no HTTP, catalog lookup, or persistence.  It
is the narrow boundary consumed by the store in slice 4A; prompts and provider
payloads are not representable here.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OFFICE_EXECUTION_CONTRACT = "OFFICE_V1"
OFFICE_WORK_CONTRACT_VERSION = "office-work.1"
CLASSIC_EXECUTION_CONTRACT = "CLASSIC"


class OfficePhase(str, Enum):
    RESERVED = "RESERVED"
    PREFLIGHT_IN_FLIGHT = "PREFLIGHT_IN_FLIGHT"
    READY = "READY"
    DELIVERY_IN_FLIGHT = "DELIVERY_IN_FLIGHT"
    RECEIPT_STORED = "RECEIPT_STORED"
    FAILED = "FAILED"
    CANCELLED_REVOKED = "CANCELLED_REVOKED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class OfficeRunRequest(BaseModel):
    """Only binding/accounting claims supplied after JXP authorization.

    Extra keys fail closed, so untrusted request bodies cannot smuggle a prompt,
    URL, model, recipient group, or authority selector into Jacobs.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: str
    office_job_id: uuid.UUID
    office_stage_id: uuid.UUID
    client_request_id: uuid.UUID
    project_id: int = Field(gt=0)
    tenant_id: int = Field(gt=0)
    usage_user_id: int = Field(gt=0)
    facet: str = Field(min_length=1, max_length=64)
    capability: str = Field(min_length=1, max_length=128)
    timeout_seconds: int = Field(gt=0)
    accepted_cost_usd: Decimal = Field(ge=Decimal("0"), max_digits=14, decimal_places=6)
    request_sha256: str

    @field_validator("contract_version", "facet", "capability", mode="before")
    @classmethod
    def _strict_text(cls, value: object, info) -> str:
        if type(value) is not str:
            raise ValueError("must be a JSON string")
        if info.field_name == "contract_version" and value != OFFICE_WORK_CONTRACT_VERSION:
            raise ValueError("unsupported Office contract version")
        return value

    @field_validator("request_sha256", mode="before")
    @classmethod
    def _sha(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError("request_sha256 must be a JSON string")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("request_sha256 must be lowercase sha256")
        return value

    @field_validator("project_id", "tenant_id", "usage_user_id", "timeout_seconds", mode="before")
    @classmethod
    def _strict_positive_int(cls, value: object) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("must be a positive JSON integer")
        return value

    @field_validator("office_job_id", "office_stage_id", "client_request_id", mode="before")
    @classmethod
    def _uuid_json(cls, value: object) -> uuid.UUID | str:
        if not isinstance(value, (str, uuid.UUID)):
            raise ValueError("must be a UUID string")
        return value

    @field_validator("accepted_cost_usd", mode="before")
    @classmethod
    def _exact_money(cls, value: object) -> Decimal:
        if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (str, Decimal)):
            raise ValueError("accepted_cost_usd must be an exact decimal string")
        try:
            amount = Decimal(str(value))
        except Exception as exc:
            raise ValueError("invalid accepted_cost_usd") from exc
        if (not amount.is_finite() or amount < 0 or -amount.as_tuple().exponent > 6
                or (amount and amount.adjusted() > 7)):
            raise ValueError("accepted_cost_usd is not exactly representable")
        return amount

    @model_validator(mode="after")
    def _digest_matches(self) -> "OfficeRunRequest":
        if self.request_sha256 != office_request_digest(self):
            raise ValueError("request_sha256 does not bind Office request")
        return self


def canonical_usd(value: Decimal) -> str:
    """Canonical six-place USD used in cost binding, never float formatting."""
    # Exactness is already validated above; formatting must never round an
    # accepted budget upward. This is the existing canonical Jacobs formatter.
    from jacobs.prevuelo_reglas import formatear_usd
    return formatear_usd(value)


def _digest_claims(request: OfficeRunRequest | Mapping[str, object]) -> dict[str, object]:
    data = request.model_dump(mode="json") if isinstance(request, OfficeRunRequest) else dict(request)
    data.pop("request_sha256", None)
    # Decimal/UUID conversion is explicit; JSON's default coercion is not a
    # stable authority binding.
    for key in ("office_job_id", "office_stage_id", "client_request_id"):
        data[key] = str(uuid.UUID(str(data[key])))
    for key in ("project_id", "tenant_id", "usage_user_id", "timeout_seconds"):
        data[key] = int(data[key])
    data["accepted_cost_usd"] = canonical_usd(Decimal(str(data["accepted_cost_usd"])))
    return data


def office_request_digest(request: OfficeRunRequest | Mapping[str, object]) -> str:
    payload = json.dumps(_digest_claims(request), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def deterministic_step_id(pipeline_id: str) -> str:
    return str(uuid.uuid5(uuid.UUID(pipeline_id), f"{OFFICE_EXECUTION_CONTRACT}:step:0"))


def deterministic_correlation_id(pipeline_id: str) -> str:
    return str(uuid.uuid5(uuid.UUID(pipeline_id), f"{OFFICE_EXECUTION_CONTRACT}:correlation:0"))


@dataclass(frozen=True)
class OfficeRuntimePolicy:
    contract: str = OFFICE_EXECUTION_CONTRACT
    tool_mode: str = "NONE"
    generation_calls: int = 1
    allowed_transports: frozenset[str] = frozenset({"http_gemini", "http_openai_compat", "ollama"})
    error_detail: str = "STABLE_CODE_ONLY"

    def __post_init__(self) -> None:
        if type(self.allowed_transports) is not frozenset or type(self.generation_calls) is not int or (self.contract, self.tool_mode, self.generation_calls, self.error_detail) != (
            OFFICE_EXECUTION_CONTRACT, "NONE", 1, "STABLE_CODE_ONLY"
        ) or self.allowed_transports != frozenset({"http_gemini", "http_openai_compat", "ollama"}):
            raise ValueError("Office runtime policy is fixed by OFFICE_V1")


OFFICE_RUNTIME_POLICY = OfficeRuntimePolicy()
