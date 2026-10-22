"""Real C1 MariaDB seams for the public Office submission route."""
from __future__ import annotations

import asyncio
import os
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from jacobs import cupo, executor, routes, store
from jacobs.models import Pipeline
from jacobs.office_contract import OFFICE_WORK_CONTRACT_VERSION, OfficeReceipt, OfficeRunRequest, OfficePhase, office_request_digest, office_utf8_sha256


pytestmark = pytest.mark.skipif(not os.getenv("JAX_DB_HOST"), reason="requires disposable MariaDB")


def _request() -> OfficeRunRequest:
    data = {"contract_version": OFFICE_WORK_CONTRACT_VERSION, "office_job_id": uuid.uuid4(),
            "office_stage_id": uuid.uuid4(), "client_request_id": uuid.uuid4(), "project_id": 701,
            "tenant_id": 702, "usage_user_id": 703, "facet": "hipatia", "capability": "design",
            "timeout_seconds": 60, "accepted_cost_usd": Decimal("0.010000")}
    data["request_sha256"] = office_request_digest(data)
    return OfficeRunRequest(**data)


def _app() -> FastAPI:
    app = FastAPI()
    @app.middleware("http")
    async def platform(request, call_next):
        request.state.identidad_servicio = "plataforma"
        return await call_next(request)
    app.include_router(routes.router)
    return app


async def _clean(*pipeline_ids: str) -> None:
    if pipeline_ids:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id IN (" + ",".join(["%s"] * len(pipeline_ids)) + ")", pipeline_ids)
    await store.cerrar_pool()


async def _post(app: FastAPI, request: OfficeRunRequest):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://office.test") as client:
        return await client.post("/jacobs/office-v1/runs", json=request.model_dump(mode="json"))


async def _exercise(request: OfficeRunRequest, *, precreate: bool = False, prepare: bool = False,
                    concurrent: bool = False):
    await store.init_tables()
    pipeline_ids: list[str] = []
    if precreate:
        reservation = await cupo.reservar_office_v1(request, limite=10000)
        assert reservation and reservation.created
        pipeline_ids.append(reservation.pipeline_id)
        assert await store.office_execution_snapshot(reservation.pipeline_id) is None
        if prepare:
            assert await store.office_prepare_step(reservation.pipeline_id, request)
    prompt = "exact Office prompt\nΩ"
    materialized = SimpleNamespace(outcome="MATERIALIZED", prompt=prompt,
        materialization_sha256=office_utf8_sha256(prompt), binding_version=7)
    verdict = SimpleNamespace(ok=True, costo_max_usd=Decimal("0.010000"))
    observed = []
    async def materialize(snapshot, phase):
        observed.append(("materialize", phase, snapshot.pipeline_id))
        return materialized
    async def preflight(steps, context, **kwargs):
        observed.append(("prevuelo", steps[0].input, kwargs["runtime_policy"]))
        return verdict
    try:
        with patch.object(routes, "office_callback_config"), \
             patch.object(routes, "check_kill_switch", return_value=False), \
             patch.object(routes, "materialize", AsyncMock(side_effect=materialize)) as callback, \
             patch.object(routes, "prevuelo", AsyncMock(side_effect=preflight)) as probe, \
             patch.object(routes, "run_pipeline", AsyncMock()):
            app = _app()
            if concurrent:
                responses = await asyncio.gather(_post(app, request), _post(app, request))
            else:
                responses = [await _post(app, request)]
        snapshot = await store.office_pipeline_by_client_request(request.tenant_id, str(request.client_request_id))
        assert snapshot is not None
        pipeline_ids.append(snapshot.pipeline_id)
        full = await store.office_execution_snapshot(snapshot.pipeline_id)
        assert full and full.phase is OfficePhase.READY
        assert callback.await_count == probe.await_count == 1
        assert observed[0][0] == "materialize" and observed[1][0] == "prevuelo"
        assert observed[1][1] == {"prompt": prompt}
        assert full.accepted_cost_usd == request.accepted_cost_usd
        return responses, snapshot.pipeline_id
    finally:
        await _clean(*set(pipeline_ids))


def test_office_post_fresh_reserves_prepares_and_returns_201_with_real_db():
    responses, _ = asyncio.run(_exercise(_request()))
    assert responses[0].status_code == 201


def test_office_post_replay_repairs_reserved_crash_step_then_returns_200():
    responses, _ = asyncio.run(_exercise(_request(), precreate=True))
    assert responses[0].status_code == 200


def test_office_post_replay_after_prepare_claims_one_paid_preflight():
    responses, _ = asyncio.run(_exercise(_request(), precreate=True, prepare=True))
    assert responses[0].status_code == 200


def test_two_identical_office_posts_use_one_real_reservation_and_preflight():
    responses, _ = asyncio.run(_exercise(_request(), concurrent=True))
    # The loser can observe durable PREFLIGHT_IN_FLIGHT before the winner marks
    # READY; 202 is the prescribed in-progress replay, never a second probe.
    assert sorted(response.status_code for response in responses) == [201, 202]


def test_real_route_preflight_input_stale_is_unknown_not_cancelled():
    async def run():
        request = _request()
        await store.init_tables()
        try:
            with patch.object(routes, "office_callback_config"), \
                 patch.object(routes, "check_kill_switch", return_value=False), \
                 patch.object(routes, "materialize", AsyncMock(return_value=SimpleNamespace(outcome="INPUT_STALE"))), \
                 patch.object(routes, "prevuelo", AsyncMock(side_effect=AssertionError("paid probe"))):
                response = await _post(_app(), request)
            snap = await store.office_pipeline_by_client_request(request.tenant_id, str(request.client_request_id)); assert snap
            final = await store.office_execution_snapshot(snap.pipeline_id)
            assert response.status_code == 503
            assert final and final.phase is OfficePhase.OUTCOME_UNKNOWN
            assert final.rejection_code == "OFFICE_PREFLIGHT_UNKNOWN"
        finally:
            if 'snap' in locals() and snap: await _clean(snap.pipeline_id)
            else: await store.cerrar_pool()
    asyncio.run(run())


@pytest.mark.parametrize("outcome,status", [("FAILED", 422), ("ALREADY_COMPLETED", 503)])
def test_real_route_preflight_reject_posts_truthful_fail_without_generation(monkeypatch, outcome, status):
    from jacobs import office_callbacks
    request = _request(); prompt = "preflight prompt"; seen = {}
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true"); monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.test"); monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10"); monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "1024")
    office_callbacks.clear_office_callback_config_cache()
    async def handler(http_request):
        seen.update(__import__("json").loads(http_request.content))
        body = {"outcome": outcome, "contract_version": OFFICE_WORK_CONTRACT_VERSION}
        if outcome == "ALREADY_COMPLETED":
            body["receipt"] = {"receipt_id": str(uuid.uuid4()), "effect_kind": "NOTE_CREATED", "effect_id": "note:preflight", "output_sha256": "a" * 64, "office_stage_version": 1}
        return httpx.Response(200, json=body)
    async def run():
        await store.init_tables()
        materialized = SimpleNamespace(outcome="MATERIALIZED", prompt=prompt, materialization_sha256=office_utf8_sha256(prompt), binding_version=1)
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with patch.object(office_callbacks, "obtener_cliente_http", lambda: client), patch.object(office_callbacks, "encabezado_propio", lambda i: {"X":"s"}), patch.object(office_callbacks, "ensure_office_http_log_privacy", lambda: None), patch.object(routes, "office_callback_config", office_callbacks.office_callback_config), patch.object(routes, "materialize", AsyncMock(return_value=materialized)), patch.object(routes, "prevuelo", AsyncMock(return_value=SimpleNamespace(ok=False, costo_max_usd=Decimal("9")))), patch.object(routes, "run_pipeline", AsyncMock(side_effect=AssertionError("generation"))):
                    response = await _post(_app(), request)
            snap = await store.office_pipeline_by_client_request(request.tenant_id, str(request.client_request_id)); assert snap
            assert response.status_code == status
            assert seen["phase"] == "PREFLIGHT" and seen["model_invoked"] is False
            assert (await store.office_execution_snapshot(snap.pipeline_id)).phase is (OfficePhase.FAILED if outcome == "FAILED" else OfficePhase.OUTCOME_UNKNOWN)
        finally:
            if 'snap' in locals() and snap: await _clean(snap.pipeline_id)
            else: await store.cerrar_pool()
    asyncio.run(run())


@pytest.mark.parametrize(
    "error_outcome,callback_outcome,status,phase",
    [("FAILED", "FAILED", 422, OfficePhase.FAILED),
     ("OUTCOME_UNKNOWN", "FAILED", 422, OfficePhase.FAILED),
     ("OUTCOME_UNKNOWN", "OUTCOME_UNKNOWN", 503, OfficePhase.OUTCOME_UNKNOWN)],
)
def test_real_route_preflight_execution_error_reconciles_typed_callback(monkeypatch, error_outcome, callback_outcome, status, phase):
    from jacobs import office_callbacks
    from jacobs.office_runtime import OfficeErrorOutcome, OfficeErrorPhase, OfficeExecutionError
    from jacobs.office_contract import OfficeRejectionCode
    request = _request(); seen = {}
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true"); monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.test"); monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10"); monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "1024")
    office_callbacks.clear_office_callback_config_cache()
    async def handler(http_request):
        seen.update(__import__("json").loads(http_request.content))
        return httpx.Response(200, json={"outcome": callback_outcome, "contract_version": OFFICE_WORK_CONTRACT_VERSION})
    async def run():
        await store.init_tables()
        materialized = SimpleNamespace(outcome="MATERIALIZED", prompt="p", materialization_sha256="b" * 64, binding_version=1)
        probe_error = OfficeExecutionError(OfficeRejectionCode.OFFICE_EXECUTION_FAILED,
            phase=OfficeErrorPhase.PREFLIGHT, outcome=OfficeErrorOutcome(error_outcome), provider_call_made=False)
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with patch.object(office_callbacks, "obtener_cliente_http", lambda: client), patch.object(office_callbacks, "encabezado_propio", lambda i: {"X":"s"}), patch.object(office_callbacks, "ensure_office_http_log_privacy", lambda: None), patch.object(routes, "office_callback_config", office_callbacks.office_callback_config), patch.object(routes, "materialize", AsyncMock(return_value=materialized)), patch.object(routes, "prevuelo", AsyncMock(side_effect=probe_error)) as probe, patch.object(routes, "run_pipeline", AsyncMock(side_effect=AssertionError("generation"))):
                    response = await _post(_app(), request)
            snap = await store.office_pipeline_by_client_request(request.tenant_id, str(request.client_request_id)); assert snap
            final = await store.office_execution_snapshot(snap.pipeline_id)
            assert response.status_code == status and probe.await_count == 1
            assert seen["phase"] == "PREFLIGHT" and seen["model_invoked"] is False
            assert final and final.phase is phase
        finally:
            if 'snap' in locals() and snap: await _clean(snap.pipeline_id)
            else: await store.cerrar_pool()
    asyncio.run(run())


async def _ready_pipeline():
    await store.init_tables()
    request = _request(); reserved = await cupo.reservar_office_v1(request, limite=10000)
    assert reserved and reserved.created
    pid = reserved.pipeline_id
    assert await store.office_prepare_step(pid, request)
    snapshot = await store.office_execution_snapshot(pid); assert snapshot
    assert await store.office_claim_preflight(pid, 0, snapshot.step_id)
    assert await store.office_mark_ready(pid, 0, snapshot.step_id, "b" * 64, 1)
    snapshot = await store.office_execution_snapshot(pid); assert snapshot
    pipeline = await store.office_pipeline_for_runner(pid); assert pipeline
    return pid, snapshot, pipeline


@pytest.mark.parametrize("ack", ["false", "raise"])
def test_real_receipt_commit_then_ack_loss_rereads_exact_receipt_model_and_hash(ack):
    async def run():
        pid, snapshot, pipeline = await _ready_pipeline()
        output = "exact output\nΩ"; receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="NOTE_CREATED",
            effect_id="note:ack", output_sha256=office_utf8_sha256(output), office_stage_version=1)
        original = store.office_store_receipt
        async def commit_then_loss(*args, **kwargs):
            assert await original(*args, **kwargs)
            if ack == "raise": raise RuntimeError("ack lost")
            return False
        async def dispatch(step, pipeline, prompt):
            step.modelo_real = "catalog-model"; return {"result": output}
        materialized = SimpleNamespace(outcome="MATERIALIZED", prompt="p", materialization_sha256="b" * 64, binding_version=1)
        try:
            with patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
                 patch.object(executor, "_dispatch_step", AsyncMock(side_effect=dispatch)) as provider, \
                 patch.object(executor, "office_complete", AsyncMock(return_value=SimpleNamespace(outcome="STORED", receipt=receipt))), \
                 patch.object(store, "office_store_receipt", AsyncMock(side_effect=commit_then_loss)), \
                 patch.object(executor, "office_fail", AsyncMock(side_effect=AssertionError("false fail"))):
                await executor.run_pipeline(pipeline)
            final = await store.office_execution_snapshot(pid)
            assert final and final.phase is OfficePhase.RECEIPT_STORED and final.receipt == receipt and final.modelo_real == "catalog-model"
            provider.assert_awaited_once()
            assert not await executor._office_receipt_reread(snapshot, receipt, "wrong-model")
        finally:
            await _clean(pid)
    asyncio.run(run())


@pytest.mark.parametrize("ack", ["false", "raise"])
def test_real_terminal_commit_then_ack_loss_requires_exact_phase_and_code(ack):
    async def run():
        pid, snapshot, _pipeline = await _ready_pipeline()
        original = store.office_finish_terminal
        async def commit_then_loss(*args, **kwargs):
            assert await original(*args, **kwargs)
            if ack == "raise": raise RuntimeError("ack lost")
            return False
        try:
            with patch.object(store, "office_finish_terminal", AsyncMock(side_effect=commit_then_loss)):
                assert await executor._office_terminal_with_ack(snapshot, OfficePhase.READY,
                    __import__("jacobs.office_contract", fromlist=["OfficeRejectionCode"]).OfficeRejectionCode.OFFICE_INPUT_STALE,
                    OfficePhase.FAILED)
            final = await store.office_execution_snapshot(pid)
            assert final and final.phase is OfficePhase.FAILED and final.rejection_code == "OFFICE_INPUT_STALE"
        finally:
            await _clean(pid)
    asyncio.run(run())
