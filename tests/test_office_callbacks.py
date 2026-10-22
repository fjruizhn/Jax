"""Office callback configuration is fail-closed before any reservation."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from jacobs import routes
from jacobs.office_contract import OFFICE_WORK_CONTRACT_VERSION, OfficeRunRequest, office_request_digest
import uuid


def test_callback_config_requires_explicit_pair_gate(monkeypatch):
    monkeypatch.delenv("JAX_OFFICE_EXECUTION_ENABLED", raising=False)
    monkeypatch.delenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", raising=False)
    from jacobs.office_callbacks import OfficeCallbackConfigurationError, office_callback_config
    from jacobs import office_callbacks
    office_callbacks.clear_office_callback_config_cache()

    with pytest.raises(OfficeCallbackConfigurationError):
        office_callback_config()


def _request_body() -> dict:
    payload = {"contract_version": OFFICE_WORK_CONTRACT_VERSION, "office_job_id": str(uuid.uuid4()),
               "office_stage_id": str(uuid.uuid4()), "client_request_id": str(uuid.uuid4()),
               "project_id": 1, "tenant_id": 2, "usage_user_id": 3, "facet": "hipatia",
               "capability": "design", "timeout_seconds": 60, "accepted_cost_usd": "0.010000"}
    payload["request_sha256"] = office_request_digest(payload)
    return payload


def test_office_route_denies_unidentified_request_before_config_or_reservation():
    app = FastAPI(); app.include_router(routes.router)
    with patch.object(routes, "office_callback_config", side_effect=AssertionError("config")), \
         patch.object(routes.cupo, "reservar_office_v1", AsyncMock(side_effect=AssertionError("reserve"))):
        with TestClient(app) as client:
            response = client.post("/jacobs/office-v1/runs", json=_request_body())
    assert response.status_code == 403


def test_office_route_reports_full_cupo_without_dereferencing_absent_reservation():
    app = FastAPI()

    @app.middleware("http")
    async def _platform_identity(request, call_next):
        request.state.identidad_servicio = "plataforma"
        return await call_next(request)

    app.include_router(routes.router)
    with patch.object(routes, "office_callback_config"), patch.object(routes, "check_kill_switch", return_value=False), \
         patch.object(routes.cupo, "reservar_office_v1", AsyncMock(return_value=None)):
        with TestClient(app) as client:
            response = client.post("/jacobs/office-v1/runs", json=_request_body())
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "office_cupo_agotado"}}


def _platform_app() -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _platform_identity(request, call_next):
        request.state.identidad_servicio = "plataforma"
        return await call_next(request)

    app.include_router(routes.router)
    return app


def _snapshot(phase, *, pipeline_id=None):
    return SimpleNamespace(pipeline_id=pipeline_id or str(uuid.uuid4()), run_epoch=0, phase=phase,
                           step_id=str(uuid.uuid4()), correlation_id=str(uuid.uuid4()), request_sha256="a" * 64,
                           facet="hipatia", capability="design", timeout_seconds=60, user_id="3", tenant_id="2",
                           accepted_cost_usd=__import__("decimal").Decimal("1.000000"),
                           office_stage_id=str(uuid.uuid4()))


def test_ready_replay_returns_200_and_schedules_existing_runner_once():
    from jacobs.cupo import OfficeReservation
    from jacobs.office_contract import OfficePhase
    snap = _snapshot(OfficePhase.READY)
    runner = SimpleNamespace(pipeline_id=snap.pipeline_id)
    with patch.object(routes, "office_callback_config"), patch.object(routes, "check_kill_switch", return_value=False), \
         patch.object(routes.cupo, "reservar_office_v1", AsyncMock(return_value=OfficeReservation(snap.pipeline_id, False))), \
         patch.object(routes.store, "office_prepare_step", AsyncMock(return_value=False)), \
         patch.object(routes.store, "office_execution_snapshot", AsyncMock(return_value=snap)), \
         patch.object(routes.store, "office_pipeline_for_runner", AsyncMock(return_value=runner)), \
         patch.object(routes, "run_pipeline", AsyncMock()) as execute:
        with TestClient(_platform_app()) as client:
            response = client.post("/jacobs/office-v1/runs", json=_request_body())
    assert response.status_code == 200
    assert response.json()["phase"] == "READY"
    execute.assert_awaited_once_with(runner)


@pytest.mark.parametrize("phase,status", [
    ("PREFLIGHT_IN_FLIGHT", 202), ("DELIVERY_IN_FLIGHT", 202),
])
def test_inflight_replay_never_runs_probe_or_runner(phase, status):
    from jacobs.cupo import OfficeReservation
    from jacobs.office_contract import OfficePhase
    snap = _snapshot(OfficePhase(phase))
    with patch.object(routes, "office_callback_config"), patch.object(routes, "check_kill_switch", return_value=False), \
         patch.object(routes.cupo, "reservar_office_v1", AsyncMock(return_value=OfficeReservation(snap.pipeline_id, False))), \
         patch.object(routes.store, "office_prepare_step", AsyncMock(return_value=False)), \
         patch.object(routes.store, "office_execution_snapshot", AsyncMock(return_value=snap)), \
         patch.object(routes, "prevuelo", AsyncMock(side_effect=AssertionError("probe"))), \
         patch.object(routes, "run_pipeline", AsyncMock(side_effect=AssertionError("runner"))):
        with TestClient(_platform_app()) as client:
            response = client.post("/jacobs/office-v1/runs", json=_request_body())
    assert response.status_code == status


@pytest.mark.parametrize("key,value", [
    ("JAX_OFFICE_CALLBACK_BASE_URL", "ftp://example.test"),
    ("JAX_OFFICE_CALLBACK_BASE_URL", "https://user@example.test"),
    ("JAX_OFFICE_CALLBACK_BASE_URL", "https://example.test/?q=1"),
    ("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "0"),
    ("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "-1"),
    ("JAX_OFFICE_CALLBACK_BASE_URL", "http://[::1"),
    ("JAX_OFFICE_CALLBACK_BASE_URL", "https:///missing-host"),
])
def test_callback_config_rejects_unsafe_or_unbounded_values(monkeypatch, key, value):
    from jacobs import office_callbacks
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.example.test")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "1024")
    monkeypatch.setenv(key, value)
    office_callbacks.clear_office_callback_config_cache()
    with patch.object(office_callbacks, "encabezado_propio", return_value={"X": "secret"}), \
         pytest.raises(office_callbacks.OfficeCallbackConfigurationError):
        office_callbacks.office_callback_config()


def test_fail_serializes_the_registered_enum_value(monkeypatch):
    """A real DTO rejects enum objects; the wire must contain its stable value."""
    from jacobs import office_callbacks
    from jacobs.office_contract import OfficeRejectionCode, OfficeFailRequest
    captured = {}

    async def post(path, payload, accepted):
        captured["payload"] = payload
        return SimpleNamespace(outcome="FAILED")

    monkeypatch.setattr(office_callbacks, "_post", post)
    pipeline_id = str(uuid.uuid4())
    from jacobs.office_contract import deterministic_step_id, deterministic_correlation_id
    snapshot = SimpleNamespace(pipeline_id=pipeline_id, run_epoch=0,
        step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id), request_sha256="a" * 64,
        office_stage_id=str(uuid.uuid4()))
    asyncio.run(office_callbacks.fail(snapshot, "GENERATION", OfficeRejectionCode.OFFICE_EXECUTION_FAILED, False))
    assert isinstance(captured["payload"], OfficeFailRequest)
    assert captured["payload"].code is OfficeRejectionCode.OFFICE_EXECUTION_FAILED


@pytest.mark.parametrize("status", [409, 422, 503, 200])
def test_complete_any_post_attempt_failure_is_ambiguous(monkeypatch, status):
    """Complete never treats an HTTP/body acknowledgement as pre-commit proof."""
    from jacobs import office_callbacks
    pipeline_id = str(uuid.uuid4())
    from jacobs.office_contract import deterministic_step_id, deterministic_correlation_id
    snapshot = SimpleNamespace(pipeline_id=pipeline_id, run_epoch=0,
        step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id),
        request_sha256="a" * 64, office_stage_id=str(uuid.uuid4()))
    async def post(*args):
        raise office_callbacks.OfficeCallbackError("OFFICE_CALLBACK_RESPONSE_INVALID", ambiguous=status >= 500)
    monkeypatch.setattr(office_callbacks, "_post", post)
    with pytest.raises(office_callbacks.OfficeCallbackError) as caught:
        asyncio.run(office_callbacks.complete(snapshot, "output", "catalog-model"))
    assert caught.value.ambiguous and caught.value.effect_may_have_committed


@pytest.mark.parametrize("status,body", [
    (409, b"{}"), (422, b"{}"), (503, b"{}"), (200, b'{"not":"a completion dto"}'),
])
def test_complete_real_transport_errors_are_delivery_ambiguous(monkeypatch, status, body):
    """Actual callback transport: every post-attempt failure reconciles."""
    from jacobs import office_callbacks
    _callback_env(monkeypatch); office_callbacks.clear_office_callback_config_cache()
    pipeline_id = str(uuid.uuid4())
    from jacobs.office_contract import deterministic_step_id, deterministic_correlation_id
    snapshot = SimpleNamespace(pipeline_id=pipeline_id, run_epoch=0,
        step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id),
        request_sha256="a" * 64, office_stage_id=str(uuid.uuid4()))
    async def handler(request):
        return httpx.Response(status, content=body)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(office_callbacks, "obtener_cliente_http", lambda: client)
            monkeypatch.setattr(office_callbacks, "encabezado_propio", lambda identity: {"X": "secret"})
            monkeypatch.setattr(office_callbacks, "ensure_office_http_log_privacy", lambda: None)
            with pytest.raises(office_callbacks.OfficeCallbackError) as caught:
                await office_callbacks.complete(snapshot, '{"safe":true}', "catalog-model")
            return caught.value
    error = asyncio.run(run())
    assert error.ambiguous and error.effect_may_have_committed


@pytest.mark.parametrize("phase,invoked,outcome", [
    ("GENERATION", False, "FAILED"), ("GENERATION", True, "OUTCOME_UNKNOWN"),
    ("DELIVERY", True, "CANCELLED_REVOKED"),
])
def test_fail_uses_real_httpx_dto_and_preserves_phase_invocation_and_outcome(monkeypatch, phase, invoked, outcome):
    from jacobs import office_callbacks
    from jacobs.office_contract import OfficeRejectionCode
    _callback_env(monkeypatch); office_callbacks.clear_office_callback_config_cache()
    pipeline_id = str(uuid.uuid4())
    from jacobs.office_contract import deterministic_step_id, deterministic_correlation_id
    snapshot = SimpleNamespace(pipeline_id=pipeline_id, run_epoch=0,
        step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id),
        request_sha256="a" * 64, office_stage_id=str(uuid.uuid4()))
    seen = {}
    async def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"outcome": outcome, "contract_version": OFFICE_WORK_CONTRACT_VERSION})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(office_callbacks, "obtener_cliente_http", lambda: client)
            monkeypatch.setattr(office_callbacks, "encabezado_propio", lambda identity: {"X": "secret"})
            monkeypatch.setattr(office_callbacks, "ensure_office_http_log_privacy", lambda: None)
            result = await office_callbacks.fail(snapshot, phase, OfficeRejectionCode.OFFICE_EXECUTION_FAILED, invoked)
        return result
    assert asyncio.run(run()).outcome == outcome
    assert seen["phase"] == phase and seen["model_invoked"] is invoked and seen["code"] == "OFFICE_EXECUTION_FAILED"


class _FakeResponse:
    def __init__(self, status, headers, chunks):
        self.status_code, self.headers, self._chunks = status, headers, chunks
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False
    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeClient:
    def __init__(self, response): self.response = response
    def stream(self, *args, **kwargs):
        assert kwargs["follow_redirects"] is False
        return self.response


class _RaisingClient:
    def __init__(self, error): self.error = error
    def stream(self, *args, **kwargs): raise self.error


def _materialize_payload():
    from jacobs.office_contract import deterministic_step_id, deterministic_correlation_id, OfficeMaterializeRequest
    pipeline_id = str(uuid.uuid4())
    return OfficeMaterializeRequest(contract_version=OFFICE_WORK_CONTRACT_VERSION, pipeline_id=pipeline_id,
        run_epoch=0, step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id),
        request_sha256="a" * 64, phase="PREFLIGHT")


def _callback_env(monkeypatch):
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.example.test")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "1024")


@pytest.mark.parametrize("error,code,ambiguous", [
    (httpx.ConnectError("SECRET_CONNECT_SENTINEL"), "OFFICE_CALLBACK_AMBIGUOUS", True),
    (httpx.ReadTimeout("SECRET_TIMEOUT_SENTINEL"), "OFFICE_CALLBACK_AMBIGUOUS", True),
    (httpx.RemoteProtocolError("SECRET_PROTOCOL_SENTINEL"), "OFFICE_CALLBACK_AMBIGUOUS", True),
])
def test_callback_transport_errors_are_opaque_and_ambiguous(monkeypatch, error, code, ambiguous):
    from jacobs import office_callbacks
    from jacobs.office_contract import OfficeMaterializedResponse
    _callback_env(monkeypatch); office_callbacks.clear_office_callback_config_cache()
    with patch.object(office_callbacks, "encabezado_propio", return_value={"X": "secret"}), \
         patch.object(office_callbacks, "obtener_cliente_http", return_value=_RaisingClient(error)), \
         patch.object(office_callbacks, "ensure_office_http_log_privacy"):
        with pytest.raises(office_callbacks.OfficeCallbackError) as caught:
            asyncio.run(office_callbacks._post("/fixed", _materialize_payload(), (OfficeMaterializedResponse,)))
    assert caught.value.code == code and caught.value.ambiguous is ambiguous
    assert "SECRET_" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_callback_redirect_is_never_followed_or_exposed(monkeypatch):
    from jacobs import office_callbacks
    from jacobs.office_contract import OfficeMaterializedResponse
    _callback_env(monkeypatch); office_callbacks.clear_office_callback_config_cache()
    with patch.object(office_callbacks, "encabezado_propio", return_value={"X": "secret"}), \
         patch.object(office_callbacks, "obtener_cliente_http", return_value=_FakeClient(_FakeResponse(302, {"location": "https://evil.test/SECRET"}, []))), \
         patch.object(office_callbacks, "ensure_office_http_log_privacy"):
        with pytest.raises(office_callbacks.OfficeCallbackError) as caught:
            asyncio.run(office_callbacks._post("/fixed", _materialize_payload(), (OfficeMaterializedResponse,)))
    assert caught.value.code == "OFFICE_CALLBACK_UNAVAILABLE"
    assert caught.value.__cause__ is None


def test_callback_accepts_only_exact_bounded_materialization_envelope(monkeypatch):
    from jacobs import office_callbacks
    from jacobs.office_contract import OfficeMaterializedResponse, office_utf8_sha256
    _callback_env(monkeypatch); office_callbacks.clear_office_callback_config_cache()
    prompt = "exact\nUnicode: ñ"
    body = json.dumps({"outcome": "MATERIALIZED", "prompt": prompt,
                       "materialization_sha256": office_utf8_sha256(prompt), "binding_version": 1,
                       "contract_version": OFFICE_WORK_CONTRACT_VERSION}).encode("utf-8")
    with patch.object(office_callbacks, "encabezado_propio", return_value={"X": "secret"}), \
         patch.object(office_callbacks, "obtener_cliente_http", return_value=_FakeClient(_FakeResponse(200, {"content-length": str(len(body))}, [body]))), \
         patch.object(office_callbacks, "ensure_office_http_log_privacy"):
        result = asyncio.run(office_callbacks._post("/fixed", _materialize_payload(), (OfficeMaterializedResponse,)))
    assert result.prompt == prompt


@pytest.mark.parametrize("headers,chunks", [
    ({"content-length": "9999"}, [b"{}"]),
    ({}, [b"x" * 17]),
    ({}, [b'{"outcome":"MATERIALIZED","outcome":"MATERIALIZED"}']),
    ({}, [b'{"outcome":NaN}']),
    ({}, [b'{"outcome":"MATERIALIZED","prompt":"x","materialization_sha256":"' + b"0" * 64 + b'","binding_version":1,"contract_version":"office-work.1","extra":1}']),
])
def test_callback_transport_rejects_oversize_or_noncanonical_envelope(monkeypatch, headers, chunks):
    from jacobs import office_callbacks
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.example.test")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "16")
    office_callbacks.clear_office_callback_config_cache()
    response = _FakeResponse(200, headers, chunks)
    pipeline_id = str(uuid.uuid4())
    # Deterministic IDs are a real contract, so construct them from its pipeline.
    from jacobs.office_contract import deterministic_step_id, deterministic_correlation_id, OfficeMaterializeRequest, OfficeMaterializedResponse
    payload = OfficeMaterializeRequest(contract_version=OFFICE_WORK_CONTRACT_VERSION, pipeline_id=pipeline_id,
        run_epoch=0, step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id),
        request_sha256="a" * 64, phase="PREFLIGHT")
    with patch.object(office_callbacks, "encabezado_propio", return_value={"X": "secret"}), \
         patch.object(office_callbacks, "obtener_cliente_http", return_value=_FakeClient(response)), \
         patch.object(office_callbacks, "ensure_office_http_log_privacy"):
        with pytest.raises(office_callbacks.OfficeCallbackError) as error:
            asyncio.run(office_callbacks._post("/fixed", payload, (OfficeMaterializedResponse,)))
    assert error.value.code == "OFFICE_CALLBACK_RESPONSE_INVALID"


def test_callback_error_never_echoes_response_sentinel(monkeypatch):
    from jacobs import office_callbacks
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.example.test")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "1024")
    office_callbacks.clear_office_callback_config_cache()
    pipeline_id = str(uuid.uuid4())
    from jacobs.office_contract import deterministic_step_id, deterministic_correlation_id, OfficeMaterializeRequest, OfficeMaterializedResponse
    payload = OfficeMaterializeRequest(contract_version=OFFICE_WORK_CONTRACT_VERSION, pipeline_id=pipeline_id,
        run_epoch=0, step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id),
        request_sha256="a" * 64, phase="PREFLIGHT")
    with patch.object(office_callbacks, "encabezado_propio", return_value={"X": "secret"}), \
         patch.object(office_callbacks, "obtener_cliente_http", return_value=_FakeClient(_FakeResponse(500, {}, [b"SECRET_CALLBACK_SENTINEL"]))), \
         patch.object(office_callbacks, "ensure_office_http_log_privacy"):
        with pytest.raises(office_callbacks.OfficeCallbackError) as error:
            asyncio.run(office_callbacks._post("/fixed", payload, (OfficeMaterializedResponse,)))
    assert "SECRET_CALLBACK_SENTINEL" not in str(error.value)
