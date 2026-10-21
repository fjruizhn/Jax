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
    if isinstance(exc, asyncio.TimeoutError) or isinstance(exc, httpx.TimeoutException):
        code = OfficeRejectionCode.OFFICE_PROVIDER_TIMEOUT
    elif isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RequestError)):
        code = OfficeRejectionCode.OFFICE_PROVIDER_UNAVAILABLE
    elif isinstance(exc, (ValueError, TypeError, KeyError, IndexError)):
        code = OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    else:
        code = OfficeRejectionCode.OFFICE_EXECUTION_FAILED
    return OfficeExecutionError(
        code,
        provider_call_made=provider_call_made,
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
