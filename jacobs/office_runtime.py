"""Private runtime guards for the disabled Office V1 delivery path.

This module contains no persistence or public request surface.  It makes the
Office discriminator useful to the direct HTTP invokers while keeping Classic
calls completely outside the Office privacy behaviour.
"""
from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import httpx

from jacobs.office_contract import (
    CLASSIC_EXECUTION_CONTRACT,
    OFFICE_EXECUTION_CONTRACT,
    OFFICE_RUNTIME_POLICY,
    OfficeRejectionCode,
    OfficeRuntimePolicy,
)


class OfficeErrorPhase(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    GENERATION = "GENERATION"
    DELIVERY = "DELIVERY"


class OfficeErrorOutcome(str, Enum):
    FAILED = "FAILED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class OfficeExecutionError(RuntimeError):
    """Stable, content-free failure safe to retain in an Office event."""

    def __init__(
        self,
        code: OfficeRejectionCode,
        *,
        phase: OfficeErrorPhase = OfficeErrorPhase.GENERATION,
        outcome: OfficeErrorOutcome = OfficeErrorOutcome.FAILED,
        provider_call_made: bool = False,
        exception_classname: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.codigo = code.value
        self.phase = phase
        self.outcome = outcome
        self.provider_call_made = provider_call_made
        self.exception_classname = exception_classname
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


def stable_office_error(
    error: OfficeExecutionError, *, provider_call_made: bool | None = None,
) -> OfficeExecutionError:
    """Copy only stable fields, deliberately dropping traceback and raw frames."""
    return OfficeExecutionError(
        error.code,
        phase=error.phase,
        outcome=error.outcome,
        provider_call_made=error.provider_call_made if provider_call_made is None else provider_call_made,
        exception_classname=error.exception_classname,
        tokens_in=error.tokens_in,
        tokens_out=error.tokens_out,
    )


@dataclass(frozen=True)
class OfficeProviderResponse:
    """Validated provider data safe for Office helpers and accounting."""

    text: str
    tokens_in: int
    tokens_out: int
    truncated: bool


_MODEL_VERSION = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
_MAX_USAGE_TOKENS = 2**63 - 1


def _response_invalid(tokens_in: int = 0, tokens_out: int = 0) -> OfficeExecutionError:
    return OfficeExecutionError(
        OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID,
        tokens_in=tokens_in, tokens_out=tokens_out,
    )


def _validate_optional_model(value: object, tokens_in: int, tokens_out: int) -> None:
    if value is None:
        return
    if type(value) is not str or not _MODEL_VERSION.fullmatch(value):
        raise _response_invalid(tokens_in, tokens_out)


def _usage_pair(container: object, input_key: str, output_key: str) -> tuple[int, int]:
    """Validate both counters before rejecting either, retaining safe partial usage."""
    if container is None:
        return 0, 0
    if type(container) is not dict:
        raise _response_invalid()
    tokens_in = container.get(input_key, 0)
    tokens_out = container.get(output_key, 0)
    valid_in = type(tokens_in) is int and 0 <= tokens_in <= _MAX_USAGE_TOKENS
    valid_out = type(tokens_out) is int and 0 <= tokens_out <= _MAX_USAGE_TOKENS
    safe_in = tokens_in if valid_in else 0
    safe_out = tokens_out if valid_out else 0
    if not valid_in or not valid_out:
        raise _response_invalid(safe_in, safe_out)
    return safe_in, safe_out


def parse_office_provider_response(data: object, transport: str) -> OfficeProviderResponse:
    """Reject malformed provider JSON before it reaches any durable helper."""
    if type(data) is not dict:
        raise _response_invalid()
    if transport == "http_openai_compat":
        tokens_in, tokens_out = _usage_pair(data.get("usage"), "prompt_tokens", "completion_tokens")
        choices = data.get("choices")
        if type(choices) is not list or not choices or type(choices[0]) is not dict:
            raise _response_invalid(tokens_in, tokens_out)
        choice = choices[0]
        message = choice.get("message")
        if type(message) is not dict or type(message.get("content")) is not str:
            raise _response_invalid(tokens_in, tokens_out)
        reason = choice.get("finish_reason")
        if reason is not None and type(reason) is not str:
            raise _response_invalid(tokens_in, tokens_out)
        _validate_optional_model(data.get("model"), tokens_in, tokens_out)
        return OfficeProviderResponse(
            message["content"], tokens_in, tokens_out, reason == "length",
        )
    if transport == "ollama":
        tokens_in, tokens_out = _usage_pair(data, "prompt_eval_count", "eval_count")
        message = data.get("message")
        reason = data.get("done_reason")
        if type(message) is not dict or type(message.get("content")) is not str or (reason is not None and type(reason) is not str):
            raise _response_invalid(tokens_in, tokens_out)
        _validate_optional_model(data.get("model"), tokens_in, tokens_out)
        return OfficeProviderResponse(
            message["content"], tokens_in, tokens_out, reason == "length",
        )
    if transport == "http_gemini":
        tokens_in, tokens_out = _usage_pair(data.get("usageMetadata"), "promptTokenCount", "candidatesTokenCount")
        candidates = data.get("candidates")
        if type(candidates) is not list or not candidates or type(candidates[0]) is not dict:
            raise _response_invalid(tokens_in, tokens_out)
        candidate = candidates[0]
        content = candidate.get("content")
        parts = content.get("parts") if type(content) is dict else None
        if type(parts) is not list or not parts or any(type(part) is not dict or type(part.get("text")) is not str for part in parts):
            raise _response_invalid(tokens_in, tokens_out)
        reason = candidate.get("finishReason")
        if reason is not None and type(reason) is not str:
            raise _response_invalid(tokens_in, tokens_out)
        _validate_optional_model(data.get("modelVersion"), tokens_in, tokens_out)
        return OfficeProviderResponse(
            "".join(part["text"] for part in parts), tokens_in, tokens_out, reason == "MAX_TOKENS",
        )
    raise _response_invalid()


@dataclass(frozen=True)
class OfficeCapabilityContext:
    """Internal binding required before an Office direct call is allowed."""

    capability: str


def runtime_policy_for_pipeline(pipeline: object) -> OfficeRuntimePolicy | None:
    contract = getattr(pipeline, "execution_contract", CLASSIC_EXECUTION_CONTRACT)
    if contract == CLASSIC_EXECUTION_CONTRACT:
        return None
    if contract == OFFICE_EXECUTION_CONTRACT:
        return OFFICE_RUNTIME_POLICY
    raise OfficeExecutionError(OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED)


async def validate_office_capability_timeout(
    context: OfficeCapabilityContext | None, timeout: int,
) -> None:
    """Use the current capability catalog and its canonical fallback ceiling."""
    if context is None or type(context.capability) is not str or not context.capability:
        raise OfficeExecutionError(OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED)
    if type(timeout) is not int or timeout <= 0:
        raise OfficeExecutionError(OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED)

    # Deferred imports avoid executor -> runtime -> store import cycles.
    from jacobs import store
    from jacobs.plan import _techo_segundos

    governance = await store.get_motor_governance()
    capabilities = governance.get("capabilities") if isinstance(governance, dict) else None
    if not isinstance(capabilities, dict):
        raise OfficeExecutionError(OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED)
    ceiling, _source = _techo_segundos(capabilities, context.capability)
    if timeout > ceiling:
        raise OfficeExecutionError(OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED)


def assert_office_transport(policy: OfficeRuntimePolicy, transport: str) -> None:
    if transport not in policy.allowed_transports:
        raise OfficeExecutionError(OfficeRejectionCode.OFFICE_TRANSPORT_NOT_ALLOWED)


def map_office_exception(exc: BaseException, *, provider_call_made: bool) -> OfficeExecutionError:
    """Turn provider failures into code-only errors without retaining cause data."""
    if isinstance(exc, OfficeExecutionError):
        return exc
    outcome = OfficeErrorOutcome.FAILED
    if isinstance(exc, (httpx.PoolTimeout, httpx.ConnectTimeout, httpx.ConnectError)):
        code = OfficeRejectionCode.OFFICE_PROVIDER_UNAVAILABLE
        outcome = OfficeErrorOutcome.FAILED
    elif isinstance(exc, asyncio.TimeoutError) or isinstance(exc, httpx.TimeoutException):
        code = OfficeRejectionCode.OFFICE_PROVIDER_TIMEOUT
        outcome = OfficeErrorOutcome.OUTCOME_UNKNOWN if provider_call_made else OfficeErrorOutcome.FAILED
    elif isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RequestError)):
        code = OfficeRejectionCode.OFFICE_PROVIDER_UNAVAILABLE
    elif isinstance(exc, (ValueError, TypeError, KeyError, IndexError)):
        code = OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    else:
        code = OfficeRejectionCode.OFFICE_EXECUTION_FAILED
    return OfficeExecutionError(
        code,
        provider_call_made=False if isinstance(exc, (httpx.PoolTimeout, httpx.ConnectTimeout, httpx.ConnectError)) else provider_call_made,
        outcome=outcome,
        exception_classname=type(exc).__name__,
    )


def office_http_status_error(status_code: int) -> OfficeExecutionError:
    if 400 <= status_code < 500:
        code = OfficeRejectionCode.OFFICE_PROVIDER_REJECTED
    elif status_code >= 500:
        code = OfficeRejectionCode.OFFICE_PROVIDER_UNAVAILABLE
    else:
        code = OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    return OfficeExecutionError(code, provider_call_made=True)


_office_http_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "office_http_context", default=False,
)
_FILTER_LOGGERS = (
    "httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2",
    "httpcore.proxy", "httpcore.socks",
)
_FILTER_LOCK = threading.Lock()


class _OfficeHttpFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _office_http_context.get()


_FILTER = _OfficeHttpFilter()


def ensure_office_http_log_privacy() -> None:
    """Install a filter on every httpx/httpcore emitter audited for pinned versions."""
    try:
        versions = {
            "httpx": importlib.metadata.version("httpx"),
            "httpcore": importlib.metadata.version("httpcore"),
        }
    except Exception as exc:  # package metadata is part of the pinned audit
        raise OfficeExecutionError(
            OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED,
            exception_classname=type(exc).__name__,
        ) from None
    if versions != {"httpx": "0.28.1", "httpcore": "1.0.9"}:
        raise OfficeExecutionError(OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED)
    try:
        with _FILTER_LOCK:
            for name in _FILTER_LOGGERS:
                target = logging.getLogger(name)
                if _FILTER not in target.filters:
                    target.addFilter(_FILTER)
            if any(_FILTER not in logging.getLogger(name).filters for name in _FILTER_LOGGERS):
                raise RuntimeError("office HTTP log filter missing")
    except OfficeExecutionError:
        raise
    except Exception as exc:
        raise OfficeExecutionError(
            OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED,
            exception_classname=type(exc).__name__,
        ) from None


@contextmanager
def office_http_log_scope():
    token = _office_http_context.set(True)
    try:
        yield
    finally:
        _office_http_context.reset(token)
