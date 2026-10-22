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
from typing import Literal, Mapping

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


class OfficeRejectionCode(str, Enum):
    OFFICE_INPUT_STALE = "OFFICE_INPUT_STALE"
    OFFICE_RUNTIME_POLICY_DENIED = "OFFICE_RUNTIME_POLICY_DENIED"
    OFFICE_PROVIDER_REJECTED = "OFFICE_PROVIDER_REJECTED"
    OFFICE_PROVIDER_UNAVAILABLE = "OFFICE_PROVIDER_UNAVAILABLE"
    OFFICE_PROVIDER_TIMEOUT = "OFFICE_PROVIDER_TIMEOUT"
    OFFICE_PROVIDER_RESPONSE_INVALID = "OFFICE_PROVIDER_RESPONSE_INVALID"
    OFFICE_OUTPUT_TRUNCATED = "OFFICE_OUTPUT_TRUNCATED"
    OFFICE_PREFLIGHT_FAILED = "OFFICE_PREFLIGHT_FAILED"
    OFFICE_PREFLIGHT_UNKNOWN = "OFFICE_PREFLIGHT_UNKNOWN"
    OFFICE_DELIVERY_OUTCOME_UNKNOWN = "OFFICE_DELIVERY_OUTCOME_UNKNOWN"
    OFFICE_EXECUTION_FAILED = "OFFICE_EXECUTION_FAILED"
    OFFICE_TRANSPORT_NOT_ALLOWED = "OFFICE_TRANSPORT_NOT_ALLOWED"
    OFFICE_REVOKED = "OFFICE_REVOKED"


OFFICE_REJECTION_CODES = tuple(code.value for code in OfficeRejectionCode)


def office_utf8_sha256(value: str) -> str:
    """Digest an opaque provider/JXP string without interpreting its content."""
    if type(value) is not str:
        raise ValueError("Office content must be a JSON string")
    try:
        return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()
    except UnicodeError:
        raise ValueError("Office content is not strict UTF-8") from None


def _canonical_uuid(value: object, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical UUID string")
    return value


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be lowercase sha256")
    return value


class _OfficeEnvelope(BaseModel):
    """Strict, immutable callback binding shared by every callback DTO."""
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    contract_version: Literal["office-work.1"]
    pipeline_id: str
    run_epoch: int = Field(ge=0)
    step_id: str
    correlation_id: str
    request_sha256: str

    @field_validator("pipeline_id", "step_id", "correlation_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> str:
        return _canonical_uuid(value, info.field_name)

    @field_validator("run_epoch", mode="before")
    @classmethod
    def _epoch(cls, value: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("run_epoch must be a nonnegative JSON integer")
        return value

    @field_validator("request_sha256", mode="before")
    @classmethod
    def _request_hash(cls, value: object) -> str:
        return _sha256(value, "request_sha256")

    @model_validator(mode="after")
    def _deterministic_ids(self) -> "_OfficeEnvelope":
        if self.step_id != deterministic_step_id(self.pipeline_id) or self.correlation_id != deterministic_correlation_id(self.pipeline_id):
            raise ValueError("Office callback IDs do not bind the pipeline")
        return self


class OfficeMaterializeRequest(_OfficeEnvelope):
    phase: Literal["PREFLIGHT", "EXECUTION"]


class OfficeMaterializedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    outcome: Literal["MATERIALIZED"]
    prompt: str
    materialization_sha256: str
    binding_version: int = Field(gt=0)
    contract_version: Literal["office-work.1"]

    @field_validator("prompt", mode="before")
    @classmethod
    def _prompt(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError("prompt must be a JSON string")
        # Force strict UTF-8 now; no normalization or parsing occurs.
        office_utf8_sha256(value)
        return value

    @field_validator("materialization_sha256", mode="before")
    @classmethod
    def _materialization_hash(cls, value: object) -> str:
        return _sha256(value, "materialization_sha256")

    @field_validator("binding_version", mode="before")
    @classmethod
    def _binding_version(cls, value: object) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("binding_version must be a positive JSON integer")
        return value

    @model_validator(mode="after")
    def _digest_matches_prompt(self) -> "OfficeMaterializedResponse":
        if self.materialization_sha256 != office_utf8_sha256(self.prompt):
            raise ValueError("materialization digest does not match prompt")
        return self


class OfficeMaterializeCancelledResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    outcome: Literal["CANCELLED_REVOKED", "INPUT_STALE"]
    contract_version: Literal["office-work.1"]


class OfficeRunResponse(BaseModel):
    """The only successful/in-progress public Office run representation."""
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    contract_version: Literal["office-work.1"]
    pipeline_id: str
    run_epoch: int = Field(ge=0)
    phase: OfficePhase

    @field_validator("pipeline_id", mode="before")
    @classmethod
    def _pipeline_id(cls, value: object) -> str:
        return _canonical_uuid(value, "pipeline_id")

    @field_validator("run_epoch", mode="before")
    @classmethod
    def _strict_epoch(cls, value: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("run_epoch must be a nonnegative JSON integer")
        return value


class OfficeCompleteRequest(_OfficeEnvelope):
    output_json: str
    output_sha256: str
    modelo_real: str

    @field_validator("output_json", mode="before")
    @classmethod
    def _opaque_output(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError("output_json must be a JSON string")
        office_utf8_sha256(value)
        return value

    @field_validator("output_sha256", mode="before")
    @classmethod
    def _output_hash(cls, value: object) -> str:
        return _sha256(value, "output_sha256")

    @field_validator("modelo_real", mode="before")
    @classmethod
    def _model_token(cls, value: object) -> str:
        if type(value) is not str or not value or len(value) > 128 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:/-" for c in value):
            raise ValueError("modelo_real must be a safe model token")
        return value

    @model_validator(mode="after")
    def _digest_matches_output(self) -> "OfficeCompleteRequest":
        if self.output_sha256 != office_utf8_sha256(self.output_json):
            raise ValueError("output digest does not match output_json")
        return self


class OfficeReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    receipt_id: str
    effect_kind: str
    effect_id: str
    output_sha256: str
    office_stage_version: int = Field(gt=0)

    @field_validator("receipt_id", mode="before")
    @classmethod
    def _receipt_id(cls, value: object) -> str:
        return _canonical_uuid(value, "receipt_id")

    @field_validator("effect_kind", mode="before")
    @classmethod
    def _effect_kind(cls, value: object) -> str:
        if type(value) is not str or not value or len(value) > 48 or not value[0].isupper() or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in value):
            raise ValueError("effect_kind must be an ASCII stable token")
        return value

    @field_validator("effect_id", mode="before")
    @classmethod
    def _effect_id(cls, value: object) -> str:
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
        if type(value) is not str or not value or len(value) > 128 or value[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" or any(c not in allowed for c in value):
            raise ValueError("effect_id must be an ASCII receipt id")
        return value

    @field_validator("output_sha256", mode="before")
    @classmethod
    def _receipt_hash(cls, value: object) -> str:
        return _sha256(value, "output_sha256")

    @field_validator("office_stage_version", mode="before")
    @classmethod
    def _stage_version(cls, value: object) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("office_stage_version must be a positive JSON integer")
        return value


class OfficeCompleteStoredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    outcome: Literal["STORED"]
    receipt: OfficeReceipt
    contract_version: Literal["office-work.1"]


class OfficeCompleteCancelledResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    outcome: Literal["CANCELLED_REVOKED"]
    contract_version: Literal["office-work.1"]


class OfficeFailRequest(_OfficeEnvelope):
    phase: Literal["PREFLIGHT", "GENERATION", "DELIVERY"]
    code: OfficeRejectionCode
    model_invoked: bool
    output_sha256: str | None = None

    @field_validator("model_invoked", mode="before")
    @classmethod
    def _invoked(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("model_invoked must be a JSON boolean")
        return value

    @field_validator("code", mode="before")
    @classmethod
    def _code(cls, value: object) -> OfficeRejectionCode:
        if type(value) is not str:
            raise ValueError("code must be a stable JSON string")
        try:
            return OfficeRejectionCode(value)
        except ValueError:
            raise ValueError("code is not a registered Office stable code") from None

    @field_validator("output_sha256", mode="before")
    @classmethod
    def _optional_output_hash(cls, value: object) -> str | None:
        if value is None:
            return None
        return _sha256(value, "output_sha256")


class OfficeFailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    outcome: Literal["FAILED", "CANCELLED_REVOKED", "OUTCOME_UNKNOWN"]
    contract_version: Literal["office-work.1"]


class OfficeAlreadyCompletedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    outcome: Literal["ALREADY_COMPLETED"]
    receipt: OfficeReceipt
    contract_version: Literal["office-work.1"]


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
