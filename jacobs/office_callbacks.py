"""Bounded, fail-closed callback transport for the paired Office service."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from cliente_http_compartido import obtener_cliente_http
from las_manos.auth_servicio import IDENTIDAD_JACOBS, encabezado_propio
from jacobs.office_contract import (
    OFFICE_WORK_CONTRACT_VERSION, OfficeAlreadyCompletedResponse,
    OfficeCompleteCancelledResponse, OfficeCompleteRequest, OfficeCompleteStoredResponse,
    OfficeFailRequest, OfficeFailResponse, OfficeMaterializeCancelledResponse,
    OfficeMaterializeRequest, OfficeMaterializedResponse,
)
from jacobs.office_runtime import ensure_office_http_log_privacy, office_http_log_scope


class OfficeCallbackConfigurationError(RuntimeError):
    pass


class OfficeCallbackError(RuntimeError):
    """Content-free callback failure with a stable classification."""
    def __init__(self, code: str, *, ambiguous: bool = False,
                 effect_may_have_committed: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.ambiguous = ambiguous
        self.effect_may_have_committed = effect_may_have_committed


@dataclass(frozen=True)
class OfficeCallbackConfig:
    base_url: str
    timeout_seconds: int
    max_response_bytes: int


def _positive(name: str) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw or "")
    except ValueError:
        value = 0
    if value <= 0:
        raise OfficeCallbackConfigurationError("OFFICE_CALLBACK_CONFIG_INVALID")
    return value


@lru_cache(maxsize=1)
def office_callback_config() -> OfficeCallbackConfig:
    if os.getenv("JAX_OFFICE_EXECUTION_ENABLED") != "true":
        raise OfficeCallbackConfigurationError("OFFICE_EXECUTION_DISABLED")
    if os.getenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION") != OFFICE_WORK_CONTRACT_VERSION:
        raise OfficeCallbackConfigurationError("OFFICE_PAIR_GATE_DISABLED")
    base = os.getenv("JAX_OFFICE_CALLBACK_BASE_URL", "")
    try:
        parsed = urlsplit(base)
        hostname, port = parsed.hostname, parsed.port
    except ValueError:
        raise OfficeCallbackConfigurationError("OFFICE_CALLBACK_CONFIG_INVALID") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or not hostname or port is not None and port <= 0
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise OfficeCallbackConfigurationError("OFFICE_CALLBACK_CONFIG_INVALID")
    try:
        encabezado_propio(IDENTIDAD_JACOBS)
    except Exception:
        raise OfficeCallbackConfigurationError("OFFICE_CALLBACK_CREDENTIAL_UNAVAILABLE") from None
    return OfficeCallbackConfig(base.rstrip("/"), _positive("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS"),
                                _positive("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES"))


def clear_office_callback_config_cache() -> None:
    office_callback_config.cache_clear()


def _reject_constant(value: str) -> None:
    raise ValueError("invalid callback JSON constant")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("duplicate callback JSON key")
        data[key] = value
    return data


T = TypeVar("T", bound=BaseModel)


async def _post(path: str, payload: BaseModel, accepted: tuple[type[T], ...]) -> T:
    config = office_callback_config()
    ensure_office_http_log_privacy()
    try:
        with office_http_log_scope():
            async with obtener_cliente_http().stream(
                "POST", config.base_url + path, headers=encabezado_propio(IDENTIDAD_JACOBS),
                json=payload.model_dump(mode="json"), timeout=config.timeout_seconds,
                follow_redirects=False,
            ) as response:
                if response.status_code in {409, 422, 503}:
                    raise OfficeCallbackError({409: "OFFICE_CALLBACK_CONFLICT", 422: "OFFICE_CALLBACK_REJECTED", 503: "OFFICE_CALLBACK_UNAVAILABLE"}[response.status_code])
                if response.status_code not in {200, 201, 202}:
                    raise OfficeCallbackError("OFFICE_CALLBACK_UNAVAILABLE", ambiguous=response.status_code >= 500)
                length = response.headers.get("content-length")
                if length is not None and (not length.isdecimal() or int(length) > config.max_response_bytes):
                    raise OfficeCallbackError("OFFICE_CALLBACK_RESPONSE_INVALID")
                body = bytearray()
                async for part in response.aiter_bytes():
                    body.extend(part)
                    if len(body) > config.max_response_bytes:
                        raise OfficeCallbackError("OFFICE_CALLBACK_RESPONSE_INVALID")
    except OfficeCallbackError:
        raise
    except (httpx.TimeoutException, httpx.TransportError):
        raise OfficeCallbackError("OFFICE_CALLBACK_AMBIGUOUS", ambiguous=True) from None
    except Exception:
        raise OfficeCallbackError("OFFICE_CALLBACK_UNAVAILABLE") from None
    try:
        decoded = bytes(body).decode("utf-8", errors="strict")
        data = json.loads(decoded, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
        for dto in accepted:
            try:
                return dto.model_validate(data)
            except ValidationError:
                continue
    except (UnicodeError, ValueError, TypeError):
        pass
    raise OfficeCallbackError("OFFICE_CALLBACK_RESPONSE_INVALID")


async def materialize(snapshot, phase: str):
    request = OfficeMaterializeRequest(contract_version=OFFICE_WORK_CONTRACT_VERSION,
        pipeline_id=snapshot.pipeline_id, run_epoch=snapshot.run_epoch, step_id=snapshot.step_id,
        correlation_id=snapshot.correlation_id, request_sha256=snapshot.request_sha256, phase=phase)
    return await _post(f"/internal/virtual-office/stages/{snapshot.office_stage_id}/materialize", request,
                       (OfficeMaterializedResponse, OfficeMaterializeCancelledResponse))


async def complete(snapshot, output_json: str, modelo_real: str):
    from jacobs.office_contract import office_utf8_sha256
    request = OfficeCompleteRequest(contract_version=OFFICE_WORK_CONTRACT_VERSION,
        pipeline_id=snapshot.pipeline_id, run_epoch=snapshot.run_epoch, step_id=snapshot.step_id,
        correlation_id=snapshot.correlation_id, request_sha256=snapshot.request_sha256,
        output_json=output_json, output_sha256=office_utf8_sha256(output_json), modelo_real=modelo_real)
    try:
        return await _post(f"/internal/virtual-office/stages/{snapshot.office_stage_id}/complete", request,
                           (OfficeCompleteStoredResponse, OfficeCompleteCancelledResponse))
    except OfficeCallbackError as exc:
        # Once complete has been attempted even a 4xx or malformed 2xx may have
        # committed the Office effect; reconciliation is the only truthful path.
        raise OfficeCallbackError(exc.code, ambiguous=True, effect_may_have_committed=True) from None


async def fail(snapshot, phase: str, code, model_invoked: bool, output_sha256: str | None = None):
    request = OfficeFailRequest(contract_version=OFFICE_WORK_CONTRACT_VERSION,
        pipeline_id=snapshot.pipeline_id, run_epoch=snapshot.run_epoch, step_id=snapshot.step_id,
        correlation_id=snapshot.correlation_id, request_sha256=snapshot.request_sha256,
        phase=phase, code=code.value if hasattr(code, "value") else code,
        model_invoked=model_invoked, output_sha256=output_sha256)
    return await _post(f"/internal/virtual-office/stages/{snapshot.office_stage_id}/fail", request,
                       (OfficeFailResponse, OfficeAlreadyCompletedResponse))
