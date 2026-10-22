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
    OfficeErrorPhase,
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
    resolved = AsyncMock()
    monkeypatch.setattr(executor, "record_resolved_version_safe", resolved)
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
    resolved.assert_not_awaited()
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "payload"),
    [
        ("http_gemini", {"candidates": [], "modelVersion": "sentinel-private-model"}),
        ("http_openai_compat", {"choices": [], "model": "sentinel-private-model"}),
        ("ollama", {"message": {}, "model": "sentinel-private-model"}),
    ],
)
async def test_office_rejects_malformed_response_before_metadata_helpers(monkeypatch, transport, payload):
    """Provider-controlled fields never reach catalog or usage helpers before shape validation."""
    from jacobs import executor

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolved = AsyncMock()
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: client)
    monkeypatch.setattr(executor, "record_resolved_version_safe", resolved)
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    invoke = {
        "http_gemini": executor._invoke_http_gemini,
        "http_openai_compat": executor._invoke_http_openai_compat,
        "ollama": executor._invoke_ollama,
    }[transport]

    with pytest.raises(OfficeExecutionError) as raised:
        await invoke(_Facet(transport), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY)
    await client.aclose()

    assert raised.value.code is OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    resolved.assert_not_awaited()


@pytest.mark.asyncio
async def test_office_rejects_response_metadata_that_cannot_be_safely_accounted(monkeypatch):
    """Non-integer usage cannot enter the usage writer or a durable spool."""
    from jacobs import executor

    payload = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "model": "allowed-model-v1",
        "usage": {"prompt_tokens": "sentinel-private-usage", "completion_tokens": 1},
    }
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: client)
    monkeypatch.setattr(executor, "record_resolved_version_safe", AsyncMock())
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    await client.aclose()
    assert raised.value.code is OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    assert "sentinel-private-usage" not in repr(raised.value)


@pytest.mark.asyncio
async def test_office_rejects_invalid_model_metadata_before_catalog_recording(monkeypatch):
    from jacobs import executor

    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "model": "not a model"}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    resolved = AsyncMock()
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: client)
    monkeypatch.setattr(executor, "record_resolved_version_safe", resolved)
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    await client.aclose()
    assert raised.value.code is OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    resolved.assert_not_awaited()


@pytest.mark.asyncio
async def test_office_raw_json_exception_has_no_retained_context(monkeypatch):
    """Mapping after the raw except must not retain JSONDecodeError.doc."""
    from jacobs import executor

    class BrokenResponse:
        status_code = 200

        def json(self):
            raise json.JSONDecodeError("bad", "sentinel-private-body", 0)

    class Client:
        async def post(self, *args, **kwargs):
            return BrokenResponse()

    monkeypatch.setattr(executor, "obtener_cliente_http", Client)
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    assert raised.value.code is OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert "sentinel-private-body" not in repr(raised.value)


@pytest.mark.asyncio
async def test_office_pre_post_configuration_error_does_not_claim_a_provider_call(monkeypatch):
    """A local output-limit failure happens before POST and has a known failed outcome."""
    from jacobs import executor

    post = AsyncMock(side_effect=AssertionError("must not POST"))
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: type("C", (), {"post": post})())
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(side_effect=ValueError("sentinel-config")))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    assert raised.value.provider_call_made is False
    assert raised.value.outcome.value == "FAILED"
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_office_client_getter_failure_does_not_claim_a_provider_call(monkeypatch):
    from jacobs import executor

    def broken_client():
        raise RuntimeError("sentinel-client-getter")

    monkeypatch.setattr(executor, "obtener_cliente_http", broken_client)
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    assert raised.value.provider_call_made is False
    assert raised.value.outcome.value == "FAILED"


@pytest.mark.asyncio
async def test_office_pool_timeout_before_send_is_known_not_called(monkeypatch):
    from jacobs import executor

    async def pool_timeout(*args, **kwargs):
        raise httpx.PoolTimeout("sentinel-pool")

    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: type("C", (), {"post": pool_timeout})())
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    assert raised.value.code is OfficeRejectionCode.OFFICE_PROVIDER_UNAVAILABLE
    assert raised.value.provider_call_made is False
    assert raised.value.outcome.value == "FAILED"


@pytest.mark.asyncio
async def test_office_read_timeout_after_transport_start_is_outcome_unknown(monkeypatch):
    """A read timeout may follow an accepted provider request, so it is not retry-safe."""
    from jacobs import executor

    async def timed_out(*args, **kwargs):
        raise httpx.ReadTimeout("sentinel-timeout")

    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: type("C", (), {"post": timed_out})())
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    assert raised.value.code is OfficeRejectionCode.OFFICE_PROVIDER_TIMEOUT
    assert raised.value.provider_call_made is True
    assert raised.value.outcome.value == "OUTCOME_UNKNOWN"


def test_office_connect_failure_is_a_known_non_delivery_outcome():
    error = runtime.map_office_exception(httpx.ConnectError("sentinel-connect"), provider_call_made=True)
    assert error.code is OfficeRejectionCode.OFFICE_PROVIDER_UNAVAILABLE
    assert error.provider_call_made is False
    assert error.outcome.value == "FAILED"


@pytest.mark.parametrize(
    ("transport", "payload", "expected"),
    [
        ("http_openai_compat", {"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": "bad"}}, (13, 0)),
        ("http_openai_compat", {"choices": [], "usage": {"prompt_tokens": "bad", "completion_tokens": 17}}, (0, 17)),
        ("ollama", {"message": {}, "prompt_eval_count": 14, "eval_count": "bad"}, (14, 0)),
        ("ollama", {"message": {}, "prompt_eval_count": "bad", "eval_count": 18}, (0, 18)),
        ("http_gemini", {"candidates": [], "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": "bad"}}, (15, 0)),
        ("http_gemini", {"candidates": [], "usageMetadata": {"promptTokenCount": "bad", "candidatesTokenCount": 19}}, (0, 19)),
    ],
)
def test_valid_usage_survives_a_later_or_sibling_counter_rejection(transport, payload, expected):
    with pytest.raises(OfficeExecutionError) as raised:
        runtime.parse_office_provider_response(payload, transport)
    assert raised.value.code is OfficeRejectionCode.OFFICE_PROVIDER_RESPONSE_INVALID
    assert (raised.value.tokens_in, raised.value.tokens_out) == expected


@pytest.mark.parametrize("bad", [False, -1])
def test_usage_counters_reject_bool_and_negative_values_without_losing_valid_peer(bad):
    with pytest.raises(OfficeExecutionError) as raised:
        runtime.parse_office_provider_response(
            {"choices": [], "usage": {"prompt_tokens": bad, "completion_tokens": 20}},
            "http_openai_compat",
        )
    assert (raised.value.tokens_in, raised.value.tokens_out) == (0, 20)


def test_zero_usage_is_a_valid_counter_value():
    response = runtime.parse_office_provider_response(
        {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}},
        "http_openai_compat",
    )
    assert (response.tokens_in, response.tokens_out) == (0, 0)


@pytest.mark.parametrize(
    ("transport", "payload"),
    [
        ("http_gemini", {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "MAX_TOKENS"}], "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 17}}),
        ("http_openai_compat", {"choices": [{"message": {"content": "ok"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 13, "completion_tokens": 17}}),
        ("ollama", {"message": {"content": "ok"}, "done_reason": "length", "prompt_eval_count": 13, "eval_count": 17}),
    ],
)
def test_full_provider_usage_provenance_survives_truncation(transport, payload):
    response = runtime.parse_office_provider_response(payload, transport)
    assert response.truncated and response.usage_measured
    assert (response.tokens_in, response.tokens_out) == (13, 17)


@pytest.mark.asyncio
async def test_office_assemble_is_denied_before_mechanical_assembly(monkeypatch):
    """Office stages admit only direct no-tools transports; Classic assembly remains elsewhere."""
    from jacobs import executor

    assembled = AsyncMock(side_effect=AssertionError("Office must not assemble"))
    monkeypatch.setattr(executor, "_assemble_mechanical", assembled)
    step = type("Step", (), {"capability": "assemble"})()
    pipeline = _Pipeline("OFFICE_V1")
    with pytest.raises(OfficeExecutionError) as raised:
        await executor._dispatch_step(step, pipeline)
    assert raised.value.code is OfficeRejectionCode.OFFICE_TRANSPORT_NOT_ALLOWED
    assembled.assert_not_called()


@pytest.mark.asyncio
async def test_shared_client_keeps_office_logs_scoped_while_classic_remains_visible(monkeypatch, caplog):
    """One real shared AsyncClient serves concurrent Office and Classic calls safely."""
    from jacobs import executor

    office_paused = asyncio.Event()
    allow_office = asyncio.Event()
    classic_seen = asyncio.Event()

    private = (
        "sentinel-office-url", "sentinel-office-header", "sentinel-office-prompt",
        "sentinel-office-output", "sentinel-office-errorbody",
    )
    classic_values = (
        "classic-visible-url", "classic-visible-header", "classic-visible-prompt",
        "classic-visible-output", "classic-visible-errorbody",
    )

    def emit_all(values):
        for name in runtime._FILTER_LOGGERS:
            logging.getLogger(name).debug(
                "url=%s header=%s prompt=%s output=%s",
                *values[:4], exc_info=RuntimeError(values[4]),
            )

    async def handler(request: httpx.Request) -> httpx.Response:
        if "sentinel-office-url" in str(request.url):
            office_paused.set()
            emit_all(private)
            await allow_office.wait()
        else:
            emit_all(classic_values)
            classic_seen.set()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: client)
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    monkeypatch.setattr(executor, "record_resolved_version_safe", AsyncMock())
    office = _Facet("http_openai_compat")
    office.base_url = "https://sentinel-office-url.invalid/v1"
    office.credential = "sentinel-office-header"
    classic = _Facet("http_openai_compat")
    classic.base_url = "https://classic-visible.invalid/v1"
    classic.credential = "classic-header"
    with caplog.at_level(logging.DEBUG):
        office_task = asyncio.create_task(executor._invoke_http_openai_compat(
            office, "sentinel-office-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        ))
        await office_paused.wait()
        classic_task = asyncio.create_task(executor._invoke_http_openai_compat(classic, "classic-prompt", 5))
        await classic_seen.wait()
        allow_office.set()
        await asyncio.gather(office_task, classic_task)
    await client.aclose()
    formatter = logging.Formatter()
    rendered = "\n".join(
        "\n".join((
            record.getMessage(), repr(record.args),
            formatter.formatException(record.exc_info) if record.exc_info else "",
        ))
        for record in caplog.records
    )
    assert all(value not in rendered for value in private)
    assert all(value in rendered for value in classic_values)


@pytest.mark.asyncio
async def test_next_office_call_repairs_removed_filters_on_all_emitters(monkeypatch, caplog):
    from jacobs import executor

    for name in runtime._FILTER_LOGGERS:
        target = logging.getLogger(name)
        while runtime._FILTER in target.filters:
            target.removeFilter(runtime._FILTER)

    async def handler(request: httpx.Request) -> httpx.Response:
        for name in runtime._FILTER_LOGGERS:
            logging.getLogger(name).debug("sentinel-repaired-filter")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: client)
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    with caplog.at_level(logging.DEBUG):
        await executor._invoke_http_openai_compat(
            _Facet("http_openai_compat"), "sentinel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
        )
    await client.aclose()
    assert all(runtime._FILTER in logging.getLogger(name).filters for name in runtime._FILTER_LOGGERS)
    assert "sentinel-repaired-filter" not in "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_office_cancellation_resets_the_log_scope_for_the_shared_client(monkeypatch):
    """Cancellation is re-raised and cannot suppress the next Classic HTTP log."""
    from jacobs import executor

    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_post(*args, **kwargs):
        started.set()
        await release.wait()

    monkeypatch.setattr(executor, "obtener_cliente_http", lambda: type("C", (), {"post": blocked_post})())
    monkeypatch.setattr(executor, "limite_de_salida", AsyncMock(return_value={"max_tokens": 9}))
    task = asyncio.create_task(executor._invoke_http_openai_compat(
        _Facet("http_openai_compat"), "sentinel-cancel-prompt", 5, runtime_policy=OFFICE_RUNTIME_POLICY,
    ))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    seen: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    target = logging.getLogger("httpx")
    handler = Capture()
    target.addHandler(handler)
    previous = target.level
    target.setLevel(logging.DEBUG)
    try:
        target.debug("classic-visible-after-cancel")
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)
    assert seen == ["classic-visible-after-cancel"]


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
    assert raised.value.phase is OfficeErrorPhase.GENERATION
    with pytest.raises(OfficeExecutionError) as raised:
        await runtime.validate_office_capability_timeout(
            OfficeCapabilityContext("research"), 61, phase=OfficeErrorPhase.PREFLIGHT)
    assert raised.value.phase is OfficeErrorPhase.PREFLIGHT


def test_version_mismatch_denies_before_any_office_http(monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _: "not-audited")
    with pytest.raises(OfficeExecutionError) as raised:
        runtime.ensure_office_http_log_privacy()
    assert raised.value.code is OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED


def test_filter_install_failure_fails_closed(monkeypatch):
    class BrokenLogger:
        filters: list[logging.Filter] = []

        def addFilter(self, filter):
            raise RuntimeError("sentinel-filter-install")

    original_get_logger = runtime.logging.getLogger
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: {"httpx": "0.28.1", "httpcore": "1.0.9"}[name])
    monkeypatch.setattr(runtime.logging, "getLogger", lambda name=None: original_get_logger() if name is None else BrokenLogger())
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
