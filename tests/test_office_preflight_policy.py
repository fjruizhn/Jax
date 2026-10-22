"""Pure/mocked-HTTP Office V1 preflight policy boundaries."""
from __future__ import annotations

import asyncio
from decimal import Decimal
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from jacobs.office_contract import OFFICE_RUNTIME_POLICY
from jacobs.prevuelo_reglas import Despacho, llamadas_max


def _dispatch(**overrides):
    base = dict(
        clave_salud="office-facet", via_motor=False, transporte="http_gemini",
        provider_id="provider", base_url="https://sentinel.invalid/v1", modelo="model",
        max_tokens_param="max_tokens", max_output_tokens=100, motor_max_tokens=0,
        precio_in=Decimal("1"), precio_out=Decimal("1"), tiene_herramientas=False,
        schema_con_reintento=False, persona=None,
    )
    base.update(overrides)
    return Despacho(**base)


def test_office_preflight_limits_gemini_to_one_generation_call():
    assert llamadas_max(_dispatch(), max_iteraciones=9, runtime_policy=OFFICE_RUNTIME_POLICY) == 1


def test_classic_gemini_keeps_its_historical_grounding_retry_budget():
    assert llamadas_max(_dispatch(), max_iteraciones=9) == 2


@pytest.mark.parametrize("transport", ["subprocess", "unknown", "motor"])
def test_office_probe_transport_must_be_in_direct_allowlist(transport):
    from jacobs.office_runtime import assert_office_transport, OfficeExecutionError

    with pytest.raises(OfficeExecutionError):
        assert_office_transport(OFFICE_RUNTIME_POLICY, transport)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, "provider_http_4xx"), (503, "provider_http_5xx")],
)
def test_office_probe_http_failures_have_closed_health_codes(monkeypatch, status, expected):
    """A status/body never becomes durable Office health detail."""
    from jacobs import sonda

    async def handler(request):
        return httpx.Response(status, text="private-body-and-url")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(sonda, "obtener_cliente_http", lambda: client)
            with pytest.raises(Exception) as raised:
                await sonda._post(
                    "https://private.invalid/probe", {"Authorization": "Bearer secret"},
                    {"messages": [{"content": "private prompt"}]}, 1, ["secret"],
                    runtime_policy=OFFICE_RUNTIME_POLICY,
                )
        return raised.value

    error = asyncio.run(run())
    assert sonda._office_health_detail(error) == expected
    assert "private" not in str(error)
    assert error.__cause__ is None and error.__context__ is None


@pytest.mark.parametrize(
    ("transport", "response"),
    [
        ("http_gemini", {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5}}),
        ("http_openai_compat", {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}),
        ("ollama", {"message": {"content": "ok"}, "prompt_eval_count": 3, "eval_count": 5}),
    ],
)
def test_office_probe_payloads_are_tool_free_and_use_shared_mock_client(monkeypatch, transport, response):
    from facet_resolver import ResolvedFacet
    from jacobs import sonda

    captured = []

    async def handler(request):
        captured.append(request)
        return httpx.Response(200, json=response)

    facet = ResolvedFacet(
        key="office-facet", provider_id="provider", base_url="https://private.invalid/v1",
        model="model", credential="private-key", transport=transport, persona=None, params=None,
    )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(sonda, "obtener_cliente_http", lambda: client)
            monkeypatch.setattr(sonda, "resolve_facet", AsyncMock(return_value=facet))
            monkeypatch.setattr(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock())
            monkeypatch.setattr(sonda, "record_direct_usage", AsyncMock())
            result = await sonda.sondear("office-facet", _dispatch(transporte=transport), runtime_policy=OFFICE_RUNTIME_POLICY)
        return result

    assert asyncio.run(run()).ok
    assert len(captured) == 1
    payload = json.loads(captured[0].content)
    assert not ({"tools", "google_search", "tool_choice", "functions", "function_call"} & payload.keys())


@pytest.mark.parametrize(
    ("transport", "response"),
    [
        ("http_gemini", {"modelVersion": "sentinel-model", "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": "bad"}}),
        ("http_openai_compat", {"model": "sentinel-model", "usage": {"prompt_tokens": 13, "completion_tokens": "bad"}}),
        ("ollama", {"model": "sentinel-model", "prompt_eval_count": 13, "eval_count": "bad"}),
    ],
)
def test_office_probe_rejects_raw_shape_and_usage_before_durable_helpers(monkeypatch, transport, response):
    """Known valid usage survives, but provider shape/model text never does."""
    from facet_resolver import ResolvedFacet
    from jacobs import sonda

    async def handler(request):
        return httpx.Response(200, json=response)

    facet = ResolvedFacet("office-facet", "provider", "https://sentinel.invalid/v1", "model", "key", transport, None, None)
    health, usage = AsyncMock(), AsyncMock()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(sonda, "obtener_cliente_http", lambda: client)
            monkeypatch.setattr(sonda, "resolve_facet", AsyncMock(return_value=facet))
            monkeypatch.setattr(sonda.facet_health, "registrar_evento_de_sonda", health)
            monkeypatch.setattr(sonda, "record_direct_usage", usage)
            return await sonda.sondear("office-facet", _dispatch(transporte=transport), runtime_policy=OFFICE_RUNTIME_POLICY)

    result = asyncio.run(run())
    assert (result.ok, result.detalle, result.tokens_in, result.tokens_out) == (False, "config_error", 13, 0)
    assert "sentinel" not in repr(result)
    assert "sentinel" not in repr(health.await_args) and "sentinel" not in repr(usage.await_args)


def test_office_probe_prepare_and_outer_timeout_errors_are_opaque(monkeypatch):
    from jacobs import sonda

    async def bad_prepare(*args, **kwargs):
        raise sonda._FallaDePreparacion("sentinel-config-url")

    async def run_prepare():
        monkeypatch.setattr(sonda, "_preparar", bad_prepare)
        monkeypatch.setattr(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock())
        return await sonda.sondear("office-facet", _dispatch(), runtime_policy=OFFICE_RUNTIME_POLICY)

    prepared = asyncio.run(run_prepare())
    assert prepared.detalle == "config_error" and "sentinel" not in repr(prepared)

    async def slow_prepare(*args, **kwargs):
        async def slow_call(timeout):
            await asyncio.Event().wait()
        return slow_call

    async def run_timeout():
        monkeypatch.setattr(sonda, "_preparar", slow_prepare)
        monkeypatch.setattr(sonda.prevuelo_config, "sonda_timeout_s", lambda: 0)
        monkeypatch.setattr(sonda, "MARGEN_DE_WAIT_FOR_S", 0)
        monkeypatch.setattr(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock())
        monkeypatch.setattr(sonda, "record_direct_usage", AsyncMock())
        return await sonda.sondear("office-facet", _dispatch(), runtime_policy=OFFICE_RUNTIME_POLICY)

    timed_out = asyncio.run(run_timeout())
    assert timed_out.detalle == "provider_timeout"


@pytest.mark.parametrize(
    ("transport", "response"),
    [
        ("http_gemini", {"modelVersion": {"raw": "sentinel-model"}, "candidates": [{"content": {"parts": [{"text": "private-output"}]}}], "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 17}}),
        ("http_openai_compat", {"model": {"raw": "sentinel-model"}, "choices": [{"message": {"content": "private-output"}}], "usage": {"prompt_tokens": 13, "completion_tokens": 17}}),
        ("ollama", {"model": {"raw": "sentinel-model"}, "message": {"content": "private-output"}, "prompt_eval_count": 13, "eval_count": 17}),
    ],
)
def test_office_probe_rejects_provider_model_metadata_after_preserving_safe_usage(monkeypatch, transport, response):
    from facet_resolver import ResolvedFacet
    from jacobs import sonda

    async def handler(request):
        return httpx.Response(200, json=response)

    facet = ResolvedFacet("office-facet", "provider", "https://sentinel.invalid/v1", "model", "key", transport, None, None)
    usage = AsyncMock()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(sonda, "obtener_cliente_http", lambda: client)
            monkeypatch.setattr(sonda, "resolve_facet", AsyncMock(return_value=facet))
            monkeypatch.setattr(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock())
            monkeypatch.setattr(sonda, "record_direct_usage", usage)
            return await sonda.sondear("office-facet", _dispatch(transporte=transport), runtime_policy=OFFICE_RUNTIME_POLICY)

    result = asyncio.run(run())
    assert (result.detalle, result.tokens_in, result.tokens_out) == ("config_error", 13, 17)
    assert "sentinel" not in repr(result) and "private-output" not in repr(usage.await_args)


@pytest.mark.parametrize(
    ("transport", "response"),
    [
        ("http_gemini", {"candidates": [{"content": {"parts": [{"raw": "sentinel-output"}]}}], "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 17}}),
        ("http_openai_compat", {"choices": [{"message": {"raw": "sentinel-output"}}], "usage": {"prompt_tokens": 13, "completion_tokens": 17}}),
        ("ollama", {"message": {"raw": "sentinel-output"}, "prompt_eval_count": 13, "eval_count": 17}),
    ],
)
def test_office_probe_rejects_provider_shape_after_preserving_safe_usage(monkeypatch, transport, response):
    from facet_resolver import ResolvedFacet
    from jacobs import sonda

    async def handler(request):
        return httpx.Response(200, json=response)

    facet = ResolvedFacet("office-facet", "provider", "https://sentinel.invalid/v1", "model", "key", transport, None, None)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(sonda, "obtener_cliente_http", lambda: client)
            monkeypatch.setattr(sonda, "resolve_facet", AsyncMock(return_value=facet))
            monkeypatch.setattr(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock())
            monkeypatch.setattr(sonda, "record_direct_usage", AsyncMock())
            return await sonda.sondear("office-facet", _dispatch(transporte=transport), runtime_policy=OFFICE_RUNTIME_POLICY)

    result = asyncio.run(run())
    assert (result.detalle, result.tokens_in, result.tokens_out) == ("config_error", 13, 17)
    assert "sentinel" not in repr(result)


def test_office_probe_cancellation_remains_cancellation(monkeypatch):
    from jacobs import sonda

    started = asyncio.Event()

    async def slow_prepare(*args, **kwargs):
        async def slow_call(timeout):
            started.set()
            await asyncio.Event().wait()
        return slow_call

    async def run():
        monkeypatch.setattr(sonda, "_preparar", slow_prepare)
        task = asyncio.create_task(sonda.sondear("office-facet", _dispatch(), runtime_policy=OFFICE_RUNTIME_POLICY))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_office_and_classic_probe_flights_are_partitioned_but_office_waiters_coalesce(monkeypatch):
    from jacobs import prevuelo, sonda

    async def run():
        prevuelo._sondas_en_vuelo.clear()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def fake_probe(*args, runtime_policy=None, **kwargs):
            calls.append(runtime_policy)
            if len(calls) == 2:
                started.set()
            await release.wait()
            return sonda.ResultadoSonda(True, None)

        monkeypatch.setattr(prevuelo.sonda, "sondear", fake_probe)
        d = _dispatch()
        office_one = asyncio.create_task(prevuelo._sondear_una_vez("office-facet", d, None, None, runtime_policy=OFFICE_RUNTIME_POLICY))
        office_two = asyncio.create_task(prevuelo._sondear_una_vez("office-facet", d, None, None, runtime_policy=OFFICE_RUNTIME_POLICY))
        classic = asyncio.create_task(prevuelo._sondear_una_vez("office-facet", d, None, None))
        await asyncio.wait_for(started.wait(), 1)
        assert len(calls) == 2
        release.set()
        results = await asyncio.gather(office_one, office_two, classic)
        assert all(result.ok for result in results)
        assert prevuelo._sondas_en_vuelo == {}

    asyncio.run(run())
