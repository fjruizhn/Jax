"""The Office runner has no path through Classic persistence or dispatch."""
from __future__ import annotations

import asyncio
import json
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from jacobs import executor
from jacobs import reaper
from jacobs.models import Pipeline
from jacobs.office_contract import OfficePhase, OfficeReceipt, office_utf8_sha256


def _snapshot() -> SimpleNamespace:
    pipeline_id = str(uuid.uuid4())
    from jacobs.office_contract import deterministic_correlation_id, deterministic_step_id
    return SimpleNamespace(
        pipeline_id=pipeline_id, run_epoch=0, phase=OfficePhase.READY,
        step_id=deterministic_step_id(pipeline_id), correlation_id=deterministic_correlation_id(pipeline_id),
        request_sha256="a" * 64, facet="hipatia", capability="design", timeout_seconds=60,
        materialization_sha256="b" * 64, binding_version=2, office_stage_id=str(uuid.uuid4()),
        user_id="3", tenant_id="2", accepted_cost_usd=Decimal("1.000000"),
    )


def _pipeline(snapshot) -> Pipeline:
    return Pipeline(pipeline_id=snapshot.pipeline_id, name="office-v1", invoked_by="plataforma",
                    user_id="3", tenant_id="2", mode="autonomous", execution_contract="OFFICE_V1",
                    office_phase="READY", plan=[])


async def _canonical_dispatch(step, pipeline, prompt):
    step.modelo_real = "safe-model"
    return {"result": '{"exact":"output"}\n'}


def test_ready_office_runner_dispatches_once_and_persists_receipt_without_classic_helpers():
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel prompt")
    output = '{"exact":"output"}\n'
    receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="OFFICE_DOCUMENT",
                            effect_id="effect-1", output_sha256=office_utf8_sha256(output), office_stage_version=1)
    stored = SimpleNamespace(outcome="STORED", receipt=receipt)
    result = {"result": output}
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=True)), \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=_canonical_dispatch)) as dispatch, \
         patch.object(executor, "office_complete", AsyncMock(return_value=stored)), \
         patch.object(executor.store, "office_store_receipt", AsyncMock(return_value=True)) as save, \
         patch.object(executor, "_run_one_step", AsyncMock(side_effect=AssertionError("classic step"))), \
         patch.object(executor, "save_if_large", side_effect=AssertionError("raw artifact")), \
         patch.object(executor.store, "event_append", AsyncMock(side_effect=AssertionError("classic event"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    dispatch.assert_awaited_once()
    save.assert_awaited_once_with(snapshot.pipeline_id, 0, snapshot.step_id, receipt, "safe-model")


def test_office_binding_change_fails_before_provider_dispatch():
    snapshot = _snapshot()
    changed = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="d" * 64,
                              binding_version=2, prompt="sentinel")
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=changed)), \
         patch.object(executor.store, "office_finish_terminal", AsyncMock(return_value=True)) as terminal, \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=AssertionError("provider"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    terminal.assert_awaited_once()
    assert terminal.await_args.args[4] == "OFFICE_INPUT_STALE"


def test_execution_materialize_input_stale_fails_ready_with_exact_code():
    snapshot = _snapshot()
    stale = SimpleNamespace(outcome="INPUT_STALE")
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=stale)), \
         patch.object(executor.store, "office_finish_terminal", AsyncMock(return_value=True)) as terminal, \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=AssertionError("provider"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    assert terminal.await_args.args[3:] == (OfficePhase.READY, "OFFICE_INPUT_STALE", OfficePhase.FAILED)


def test_kill_keeps_ready_without_materialize_or_provider():
    snapshot = _snapshot()
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "check_kill_switch", return_value=True), \
         patch.object(executor, "office_materialize", AsyncMock(side_effect=AssertionError("materialize"))), \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=AssertionError("provider"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))


def test_generation_error_reports_its_original_phase_after_delivery_claim():
    from jacobs.office_runtime import OfficeErrorPhase, OfficeExecutionError
    from jacobs.office_contract import OfficeRejectionCode
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel")
    error = OfficeExecutionError(OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED,
                                 phase=OfficeErrorPhase.GENERATION)
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=True)), \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=error)), \
         patch.object(executor, "office_fail", AsyncMock(return_value=SimpleNamespace(outcome="FAILED"))) as fail, \
         patch.object(executor.store, "office_finish_terminal", AsyncMock(return_value=True)):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    assert fail.await_args.args[1] == "GENERATION"


def test_canonical_dispatch_uses_exact_office_prompt_after_admission():
    snapshot = _snapshot(); pipeline = _pipeline(snapshot)
    step = __import__("jacobs.models", fromlist=["Step"]).Step(
        step_id=snapshot.step_id, pipeline_id=snapshot.pipeline_id, step_index=0,
        facet="hipatia", capability="design", timeout_seconds=60)
    facet = SimpleNamespace(model="catalog-model", transport="http_openai_compat")
    seen = {}
    async def direct(step, pipeline, resolved, prompt, timeout, **kwargs):
        seen.update(prompt=prompt, model=step.modelo_real, capability=kwargs["office_capability_context"].capability)
        return {"result": "output"}
    with patch.object(executor, "validate_capability", AsyncMock(return_value=None)), \
         patch.object(executor, "resolve_facet", AsyncMock(return_value=facet)), \
         patch.object(executor, "_despachar_transporte_directo", AsyncMock(side_effect=direct)):
        assert asyncio.run(executor._dispatch_step(step, pipeline, "exact\\nΩ")) == {"result": "output"}
    assert seen == {"prompt": "exact\\nΩ", "model": "catalog-model", "capability": "design"}


def test_canonical_dispatch_denies_human_gated_capability_without_transport():
    from jacobs.office_contract import OfficeRejectionCode
    from jacobs.office_runtime import OfficeExecutionError
    snapshot = _snapshot(); pipeline = _pipeline(snapshot)
    step = __import__("jacobs.models", fromlist=["Step"]).Step(
        step_id=snapshot.step_id, pipeline_id=snapshot.pipeline_id, step_index=0,
        facet="hipatia", capability="design", timeout_seconds=60)
    with patch.object(executor, "validate_capability", AsyncMock(return_value="human gate required")), \
         patch.object(executor, "_despachar_transporte_directo", AsyncMock(side_effect=AssertionError("transport"))):
        try:
            asyncio.run(executor._dispatch_step(step, pipeline, "exact"))
        except OfficeExecutionError as error:
            assert error.code is OfficeRejectionCode.OFFICE_RUNTIME_POLICY_DENIED
        else:
            raise AssertionError("Office capability denial must be stable")


def test_office_runner_does_not_restart_delivery_in_flight():
    snapshot = _snapshot(); snapshot.phase = OfficePhase.DELIVERY_IN_FLIGHT
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(side_effect=AssertionError("callback"))), \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=AssertionError("provider"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))


def test_lost_complete_response_reconciles_exactly_one_receipt_without_regeneration():
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel")
    receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="OFFICE_DOCUMENT",
                            effect_id="effect-2", output_sha256=office_utf8_sha256('{"exact":"output"}\n'), office_stage_version=1)
    completed = SimpleNamespace(outcome="ALREADY_COMPLETED", receipt=receipt)
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=True)), \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=_canonical_dispatch)) as dispatch, \
         patch.object(executor, "office_complete", AsyncMock(side_effect=executor.OfficeCallbackError("OFFICE_CALLBACK_AMBIGUOUS", ambiguous=True))), \
         patch.object(executor, "office_fail", AsyncMock(return_value=completed)) as fail, \
         patch.object(executor.store, "office_store_receipt", AsyncMock(return_value=True)) as save:
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    dispatch.assert_awaited_once()
    fail.assert_awaited_once()
    save.assert_awaited_once_with(snapshot.pipeline_id, 0, snapshot.step_id, receipt, "safe-model")


def test_typed_failed_callback_overrides_ambiguous_delivery_terminal():
    snapshot = _snapshot()
    with patch.object(executor, "office_fail", AsyncMock(return_value=SimpleNamespace(outcome="FAILED"))), \
         patch.object(executor, "_office_terminal_with_ack", AsyncMock()) as terminal:
        asyncio.run(executor._office_failure(
            snapshot, __import__("jacobs.office_contract", fromlist=["OfficeRejectionCode"]).OfficeRejectionCode.OFFICE_DELIVERY_OUTCOME_UNKNOWN,
            True, "a" * 64, from_phase=OfficePhase.DELIVERY_IN_FLIGHT,
            model_real="catalog-model", fail_phase="DELIVERY", ambiguous=True))
    assert terminal.await_args.args[-1] is OfficePhase.FAILED


def test_recovery_schedules_only_ready_rows_after_pair_gate():
    first, second = _snapshot(), _snapshot()
    with patch("jacobs.office_callbacks.office_callback_config"), \
         patch.object(reaper.store, "office_ready_pipelines", AsyncMock(return_value=[_pipeline(first), _pipeline(second)])), \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as execute:
        count = asyncio.run(reaper.schedule_ready_office_pipelines(limit=2))
        asyncio.run(asyncio.sleep(0))
    assert count == 2
    assert execute.await_count == 2


def test_recovery_does_nothing_when_pair_gate_is_disabled():
    with patch("jacobs.office_callbacks.office_callback_config", side_effect=RuntimeError("disabled")), \
         patch.object(reaper.store, "office_ready_pipelines", AsyncMock(side_effect=AssertionError("query"))):
        assert asyncio.run(reaper.schedule_ready_office_pipelines()) == 0


def test_two_ready_schedules_have_one_delivery_claim_and_one_provider_dispatch():
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel")
    claims = AsyncMock(side_effect=[True, False])
    receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="OFFICE_DOCUMENT",
                            effect_id="effect-3", output_sha256=office_utf8_sha256('{"exact":"output"}\n'), office_stage_version=1)
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", claims), \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=_canonical_dispatch)) as dispatch, \
         patch.object(executor, "office_complete", AsyncMock(return_value=SimpleNamespace(outcome="STORED", receipt=receipt))), \
         patch.object(executor.store, "office_store_receipt", AsyncMock(return_value=True)):
        async def _both():
            await asyncio.gather(executor.run_pipeline(_pipeline(snapshot)), executor.run_pipeline(_pipeline(snapshot)))
        asyncio.run(_both())
    assert claims.await_count == 2
    dispatch.assert_awaited_once()


def test_claim_false_rereads_without_provider_or_fail_callback():
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel")
    reads = AsyncMock(side_effect=[snapshot, snapshot])
    with patch.object(executor.store, "office_execution_snapshot", reads), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=False)), \
         patch.object(executor, "_dispatch_step", AsyncMock(side_effect=AssertionError("provider"))), \
         patch.object(executor, "office_fail", AsyncMock(side_effect=AssertionError("fail"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    assert reads.await_count == 2


@pytest.mark.parametrize("outcome", ["FAILED", "CANCELLED_REVOKED", "OUTCOME_UNKNOWN"])
def test_runner_failure_posts_real_fail_dto_outcome(monkeypatch, outcome):
    """Runner's failure mapping reaches the real callback DTO, not a mock helper."""
    from jacobs import office_callbacks
    from jacobs.office_contract import OFFICE_WORK_CONTRACT_VERSION
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel")
    seen = {}
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.test")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "1024")
    office_callbacks.clear_office_callback_config_cache()
    async def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"outcome": outcome, "contract_version": OFFICE_WORK_CONTRACT_VERSION})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(office_callbacks, "obtener_cliente_http", lambda: client), \
                 patch.object(office_callbacks, "encabezado_propio", lambda identity: {"X": "secret"}), \
                 patch.object(office_callbacks, "ensure_office_http_log_privacy", lambda: None), \
                 patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
                 patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
                 patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=True)), \
                 patch.object(executor, "_dispatch_step", AsyncMock(side_effect=executor.OfficeExecutionError(
                     __import__("jacobs.office_contract", fromlist=["OfficeRejectionCode"]).OfficeRejectionCode.OFFICE_EXECUTION_FAILED,
                     provider_call_made=True))), \
                 patch.object(executor.store, "office_finish_terminal", AsyncMock(return_value=True)):
                await executor.run_pipeline(_pipeline(snapshot))
    asyncio.run(run())
    assert seen["phase"] == "GENERATION" and seen["model_invoked"] is True


@pytest.mark.parametrize("bad_hash", [False, True])
def test_lost_complete_reconciles_real_already_completed_only_with_exact_hash(monkeypatch, bad_hash):
    from jacobs import office_callbacks
    from jacobs.office_contract import OFFICE_WORK_CONTRACT_VERSION
    snapshot = _snapshot(); output = '{"exact":"output"}\n'
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel")
    receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="NOTE_CREATED", effect_id="note:lost",
        output_sha256=("f" * 64 if bad_hash else office_utf8_sha256(output)), office_stage_version=1)
    monkeypatch.setenv("JAX_OFFICE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("JAX_OFFICE_PAIRED_CONTRACT_VERSION", OFFICE_WORK_CONTRACT_VERSION)
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_BASE_URL", "https://paired.test")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("JAX_OFFICE_CALLBACK_MAX_RESPONSE_BYTES", "1024")
    office_callbacks.clear_office_callback_config_cache(); calls = []
    async def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/complete"):
            raise httpx.ReadTimeout("lost after effect", request=request)
        return httpx.Response(200, json={"outcome": "ALREADY_COMPLETED", "receipt": receipt.model_dump(),
                                          "contract_version": OFFICE_WORK_CONTRACT_VERSION})
    async def dispatch(step, pipeline, prompt):
        step.modelo_real = "catalog-model"; return {"result": output}
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(office_callbacks, "obtener_cliente_http", lambda: client), \
                 patch.object(office_callbacks, "encabezado_propio", lambda identity: {"X": "secret"}), \
                 patch.object(office_callbacks, "ensure_office_http_log_privacy", lambda: None), \
                 patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
                 patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
                 patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=True)), \
                 patch.object(executor, "_dispatch_step", AsyncMock(side_effect=dispatch)) as provider, \
                 patch.object(executor.store, "office_store_receipt", AsyncMock(return_value=True)) as stored, \
                 patch.object(executor.store, "office_finish_terminal", AsyncMock(return_value=True)):
                await executor.run_pipeline(_pipeline(snapshot))
        provider.assert_awaited_once()
        return stored
    stored = asyncio.run(run())
    assert calls.count(f"/internal/virtual-office/stages/{snapshot.office_stage_id}/complete") == 1
    assert calls.count(f"/internal/virtual-office/stages/{snapshot.office_stage_id}/fail") == 1
    if bad_hash:
        stored.assert_not_awaited()
    else:
        stored.assert_awaited_once_with(snapshot.pipeline_id, 0, snapshot.step_id, receipt, "catalog-model")


def test_reaper_done_callback_consumes_exception_and_logs_only_id_and_class(caplog):
    async def run():
        async def broken():
            raise RuntimeError("SENTINEL_PRIVATE_BODY")
        task = asyncio.create_task(broken())
        reaper._remember_office_task(task, "pipeline-safe-id")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    import logging
    with caplog.at_level(logging.ERROR, logger="jacobs.reaper"):
        asyncio.run(run())
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "pipeline-safe-id" in messages and "RuntimeError" in messages
    assert "SENTINEL_PRIVATE_BODY" not in messages


def test_start_reaper_loop_runs_real_ready_scheduler_each_periodic_sweep(monkeypatch):
    async def run():
        snapshot = _snapshot()
        real_sleep = asyncio.sleep
        async def stop(_seconds):
            await real_sleep(0)
            raise asyncio.CancelledError
        with patch.object(reaper, "reap_orphaned_pipelines", AsyncMock()), \
             patch.object(reaper, "check_facet_health", AsyncMock()), \
             patch.object(reaper.store, "office_ready_pipelines", AsyncMock(return_value=[_pipeline(snapshot)])), \
             patch("jacobs.office_callbacks.office_callback_config"), \
             patch("jacobs.executor.run_pipeline", AsyncMock()) as execute, \
             patch.object(reaper.asyncio, "sleep", side_effect=stop):
            with pytest.raises(asyncio.CancelledError):
                await reaper.start_reaper_loop()
        execute.assert_awaited_once()
    asyncio.run(run())
