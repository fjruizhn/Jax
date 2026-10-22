"""The Office runner has no path through Classic persistence or dispatch."""
from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from jacobs import executor
from jacobs import reaper
from jacobs.models import Pipeline
from jacobs.office_contract import OfficePhase, OfficeReceipt


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


def test_ready_office_runner_dispatches_once_and_persists_receipt_without_classic_helpers():
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel prompt")
    receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="OFFICE_DOCUMENT",
                            effect_id="effect-1", output_sha256="c" * 64, office_stage_version=1)
    stored = SimpleNamespace(outcome="STORED", receipt=receipt)
    result = {"result": '{"exact":"output"}\n', "model": "safe-model"}
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=True)), \
         patch.object(executor, "_dispatch_office_step", AsyncMock(return_value=result)) as dispatch, \
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
         patch.object(executor, "_dispatch_office_step", AsyncMock(side_effect=AssertionError("provider"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    terminal.assert_awaited_once()
    assert terminal.await_args.args[4] == "OFFICE_INPUT_STALE"


def test_office_runner_does_not_restart_delivery_in_flight():
    snapshot = _snapshot(); snapshot.phase = OfficePhase.DELIVERY_IN_FLIGHT
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(side_effect=AssertionError("callback"))), \
         patch.object(executor, "_dispatch_office_step", AsyncMock(side_effect=AssertionError("provider"))):
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))


def test_lost_complete_response_reconciles_exactly_one_receipt_without_regeneration():
    snapshot = _snapshot()
    materialized = SimpleNamespace(outcome="MATERIALIZED", materialization_sha256="b" * 64,
                                   binding_version=2, prompt="sentinel")
    receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="OFFICE_DOCUMENT",
                            effect_id="effect-2", output_sha256="c" * 64, office_stage_version=1)
    completed = SimpleNamespace(outcome="ALREADY_COMPLETED", receipt=receipt)
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", AsyncMock(return_value=True)), \
         patch.object(executor, "_dispatch_office_step", AsyncMock(return_value={"result": "output", "model": "safe-model"})) as dispatch, \
         patch.object(executor, "office_complete", AsyncMock(side_effect=executor.OfficeCallbackError("OFFICE_CALLBACK_AMBIGUOUS", ambiguous=True))), \
         patch.object(executor, "office_fail", AsyncMock(return_value=completed)) as fail, \
         patch.object(executor.store, "office_store_receipt", AsyncMock(return_value=True)) as save:
        asyncio.run(executor.run_pipeline(_pipeline(snapshot)))
    dispatch.assert_awaited_once()
    fail.assert_awaited_once()
    save.assert_awaited_once_with(snapshot.pipeline_id, 0, snapshot.step_id, receipt, "reconciled")


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
                            effect_id="effect-3", output_sha256="c" * 64, office_stage_version=1)
    with patch.object(executor.store, "office_execution_snapshot", AsyncMock(return_value=snapshot)), \
         patch.object(executor, "office_materialize", AsyncMock(return_value=materialized)), \
         patch.object(executor.store, "office_claim_delivery", claims), \
         patch.object(executor, "_dispatch_office_step", AsyncMock(return_value={"result": "output", "model": "safe-model"})) as dispatch, \
         patch.object(executor, "office_complete", AsyncMock(return_value=SimpleNamespace(outcome="STORED", receipt=receipt))), \
         patch.object(executor.store, "office_store_receipt", AsyncMock(return_value=True)):
        async def _both():
            await asyncio.gather(executor.run_pipeline(_pipeline(snapshot)), executor.run_pipeline(_pipeline(snapshot)))
        asyncio.run(_both())
    assert claims.await_count == 2
    dispatch.assert_awaited_once()
