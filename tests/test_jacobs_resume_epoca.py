"""resume y approve-step incrementan la época (spec 2026-09-17 §5.3; approve-step
por el desvío 9 del plan): un pedido doble no lanza dos ejecutores. Sin DB.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402

from jacobs import routes  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


def _interrumpido(epoca=3):
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                    status=PipelineStatus.interrupted, context={"objective": "o"}, run_epoch=epoca)


def _paso_hyde():
    return Step(step_id="s-hyde", pipeline_id="p1", step_index=0, facet="hyde",
                capability="execute", status=StepStatus.blocked_human_gate)


def test_resume_toma_la_epoca_y_lanza_con_ella():
    tomar, bg = AsyncMock(return_value=4), BackgroundTasks()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido())), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[])), \
         patch.object(routes.store, "pipeline_tomar_epoca", tomar, create=True), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False):
        r = asyncio.run(routes.resume_pipeline(
            "p1", routes.ResumeRequest(invoked_by="plataforma"), bg))
    tomar.assert_awaited_once_with("p1", 3, (PipelineStatus.interrupted,))
    assert r["run_epoch"] == 4
    assert bg.tasks[0].args[0].run_epoch == 4


def test_resume_doble_el_segundo_recibe_409_y_no_lanza():
    bg, upsert = BackgroundTasks(), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido())), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[])), \
         patch.object(routes.store, "pipeline_tomar_epoca", AsyncMock(return_value=None), create=True), \
         patch.object(routes.store, "step_upsert", upsert), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.resume_pipeline("p1", routes.ResumeRequest(invoked_by="plataforma"), bg))
    assert e.value.status_code == 409
    assert bg.tasks == []
    upsert.assert_not_awaited()


def test_approve_step_persiste_las_marcas_de_hyde_al_tomar_la_epoca():
    tomar, bg = AsyncMock(return_value=8), BackgroundTasks()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido(epoca=7))), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[_paso_hyde()])), \
         patch.object(routes.store, "pipeline_tomar_epoca", tomar, create=True), \
         patch.object(routes.store, "pipeline_update_status", AsyncMock()), \
         patch.object(routes.store, "step_upsert", AsyncMock()), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False):
        r = asyncio.run(routes.approve_step(
            "p1", routes.ApproveStepRequest(invoked_by="plataforma"), bg))
    tomar.assert_awaited_once()
    args = tomar.await_args.args
    assert args[:3] == ("p1", 7, (PipelineStatus.interrupted,))
    assert args[3]["hyde_approved_s-hyde"] is True
    assert r["run_epoch"] == 8
    assert bg.tasks[0].args[0].run_epoch == 8


def test_approve_step_doble_409_sin_tocar_pasos():
    bg, upsert = BackgroundTasks(), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido())), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[_paso_hyde()])), \
         patch.object(routes.store, "pipeline_tomar_epoca", AsyncMock(return_value=None), create=True), \
         patch.object(routes.store, "pipeline_update_status", AsyncMock()), \
         patch.object(routes.store, "step_upsert", upsert), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.approve_step("p1", routes.ApproveStepRequest(invoked_by="plataforma"), bg))
    assert e.value.status_code == 409
    assert bg.tasks == []
    upsert.assert_not_awaited()
