"""Focused no-network evidence for Office V1's direct runtime boundary."""
from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import jacobs.office_runtime as runtime
from jacobs.office_contract import OFFICE_RUNTIME_POLICY
from jacobs.office_runtime import (
    OfficeCapabilityContext,
    OfficeExecutionError,
    OfficeRejectionCode,
    ensure_office_http_log_privacy,
    office_http_log_scope,
    runtime_policy_for_pipeline,
)


class _Pipeline:
    def __init__(self, execution_contract: str):
        self.execution_contract = execution_contract


class _Facet:
    def __init__(self, transport: str):
        self.key = "office-facet"
        self.provider_id = "provider"
        self.base_url = "https://sentinel-url.invalid/v1"
        self.model = "model"
        self.credential = "sentinel-header"
        self.transport = transport
        self.persona = "persona"
        self.params = None


@pytest.mark.parametrize(
    ("transport", "payload"),
    [
        ("http_gemini", {"candidates": [{"content": {"parts": [{"text": "output"}]}}]}),
        ("http_openai_compat", {"choices": [{"message": {"content": "output"}}]}),
        ("ollama", {"message": {"content": "output"}}),
    ],
)
@pytest.mark.asyncio
async def test_office_direct_payloads_are_tool_free_and_do_not_redirect(monkeypatch, transport, payload):
    from jacobs import executor

    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: client)
    monkeypatch.setattr(executor, "record_resolved_version_safe", AsyncMock())
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    facet = _Facet(transport)

    if transport == "http_gemini":
        result = await executor._invoke_http_gemini(facet, "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY)
    elif transport == "http_openai_compat":
        result = await executor._invoke_http_openai_compat(facet, "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY)
    else:
        result = await executor._invoke_ollama(facet, "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY)
    await client.aclose()

    assert result["result"] == "output"
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert not ({"tools", "google_search", "tool_choice", "functions", "function_call"} & body.keys())
    if transport == "http_gemini":
        assert result["sources"] == [] and result["queries"] == [] and result["grounded"] is False


@pytest.mark.asyncio
async def test_office_provider_error_is_opaque(monkeypatch):
    from jacobs import executor

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b"sentinel-error-body")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: client)
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    await client.aclose()
    exc = raised.value
    assert exc.code is OfficeRejectionCode.OFFICE_PROVIDER_REJECTED
    assert str(exc) == "OFFICE_PROVIDER_REJECTED"
    assert exc.__cause__ is None and exc.__context__ is None
    assert "sentinel" not in repr(exc)


def test_policy_is_discriminator_only_and_unknown_fails_closed():
    assert runtime_policy_for_pipeline(_Pipeline("CLASSIC")) is None
    assert runtime_policy_for_pipeline(_Pipeline("OFFICE_V1")) is OFFICE_RUNTIME_POLICY
    with pytest.raises(OfficeExecutionError) as raised:
        runtime_policy_for_pipeline(_Pipeline("UNKNOWN"))
    assert raised.value.code is OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED


def test_http_log_filter_is_scoped_to_all_audited_emitters(caplog):
    ensure_office_http_log_privacy()
    targets = ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2", "httpcore.proxy", "httpcore.socks")
    with caplog.at_level(logging.DEBUG):
        with office_http_log_scope():
            for target in targets:
                logging.getLogger(target).debug("sentinel-private")
        logging.getLogger("httpx").debug("classic-visible")
    messages = [record.getMessage() for record in caplog.records]
    assert "sentinel-private" not in messages
    assert "classic-visible" in messages


@pytest.mark.asyncio
async def test_missing_office_capability_context_fails_before_transport(monkeypatch):
    from jacobs import executor

    post = AsyncMock(side_effect=AssertionError("provider must not be called"))
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: type("C", (), {"post": post})())
    step = type("Step", (), {"facet": "office-facet", "capability": "research"})()
    pipeline = type("Pipeline", (), {"user_id": "1", "tenant_id": "1", "pipeline_id": "p"})()
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._despachar_transporte_directo(
            step, pipeline, _Facet("http_openai_compat"), "secret", 5,
            runtime_policy=OFFICE_RUNTIME_POLICY, office_capability_context=None,
        )
    assert raised.value.code is OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_catalog_timeout_ceiling_is_current_and_has_no_unknown_bypass(monkeypatch):
    async def governance():
        return {"capabilities": {"research": {"max_execution_minutes": 1}}}

    monkeypatch.setattr("jacobs.store.get_motor_governance", governance)
    await runtime.validate_office_capability_timeout(OfficeCapabilityContext("research"), 60)
    with pytest.raises(OfficeExecutionError) as raised:
        await runtime.validate_office_capability_timeout(OfficeCapabilityContext("research"), 61)
    assert raised.value.code is OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED


def test_version_mismatch_denies_before_any_office_http(monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _: "not-audited")
    with pytest.raises(OfficeExecutionError) as raised:
        runtime.ensure_office_http_log_privacy()
    assert raised.value.code is OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED


def test_scope_cleanup_after_exception_and_cancellation():
    target = logging.getLogger("httpx")
    seen: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = Capture()
    previous_level = target.level
    target.setLevel(logging.DEBUG)
    target.addHandler(handler)
    try:
        with pytest.raises(RuntimeError):
            with office_http_log_scope():
                target.debug("secret")
                raise RuntimeError("cancel-like")
        target.debug("classic-after-error")
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)
    assert "secret" not in seen
    assert "classic-after-error" in seen
