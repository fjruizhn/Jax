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


# ---------------------------------------------------------------------------
# F2 (ola final, Ruling R32): resume y approve-step corren el pre-vuelo sobre
# los pasos SIN ref legible ANTES de tomar la época, como continue. 422
# prevuelo_rechazado sin tomar la época ni tocar pasos; 503 si no puede
# correr; la respuesta 200 suma costo_max_usd y pasos_costo.
# ---------------------------------------------------------------------------

from decimal import Decimal  # noqa: E402

from jacobs.prevuelo_reglas import CostoPaso, Veredicto, Violacion  # noqa: E402

_REF_OK = 'inline:{"result": "listo"}'
_REF_ROTA = "artifact://jacobs/no-existe/tampoco/output.json"


def _veredicto(ok=True):
    costo = CostoPaso(paso=1, faceta="jekyll", modelo="m", llamadas_max=1, tokens_in_max=10,
                      tokens_out_max=20, usd_max=Decimal("0.1234561"), motivo="acotado")
    violaciones = () if ok else (Violacion(1, "jekyll", "faceta_caida", "timeout de sonda (20s)"),)
    return Veredicto(ok=ok, violaciones=violaciones, costo_max_usd=Decimal("0.1234561"),
                     pasos_costo=(costo,), sondeadas=("jekyll",))


def _paso(i, facet="jekyll", status=StepStatus.pending, depends_on=None):
    return Step(step_id=f"s{i}", pipeline_id="p1", step_index=i, facet=facet, capability="research",
                input={"prompt": f"paso {i}"}, depends_on=depends_on or [], status=status)


def _interrumpido_con(contexto, modo="autonomous"):
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode=modo,
                    status=PipelineStatus.interrupted, context={"objective": "o", **contexto},
                    run_epoch=3, user_id="7", tenant_id="1")


def _llamar(endpoint, pipeline, pasos, prevuelo, tomar=None, eventos=None, upsert=None):
    bg = BackgroundTasks()
    tomar = tomar or AsyncMock(return_value=4)
    eventos = eventos or AsyncMock()
    upsert = upsert or AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=pasos)), \
         patch.object(routes.store, "pipeline_tomar_epoca", tomar, create=True), \
         patch.object(routes.store, "step_upsert", upsert), \
         patch.object(routes.store, "event_append", eventos), \
         patch.object(routes, "prevuelo", prevuelo), \
         patch.object(routes, "check_kill_switch", return_value=False):
        if endpoint == "resume":
            r = asyncio.run(routes.resume_pipeline("p1", routes.ResumeRequest(invoked_by="plataforma"), bg))
        else:
            r = asyncio.run(routes.approve_step("p1", routes.ApproveStepRequest(invoked_by="plataforma"), bg))
    return r, bg


@pytest.mark.parametrize("endpoint", ["resume", "approve"])
def test_prevuelo_rechazado_da_422_sin_tomar_la_epoca_ni_tocar_pasos(endpoint):
    pasos = [_paso(0, facet="hyde", status=StepStatus.blocked_human_gate), _paso(1, depends_on=[0])]
    tomar, eventos, upsert = AsyncMock(return_value=4), AsyncMock(), AsyncMock()
    with pytest.raises(HTTPException) as e:
        _llamar(endpoint, _interrumpido_con({}), pasos, AsyncMock(return_value=_veredicto(ok=False)),
                tomar=tomar, eventos=eventos, upsert=upsert)
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "prevuelo_rechazado"
    assert e.value.detail["violaciones"][0]["regla"] == "faceta_caida"
    assert e.value.detail["costo_max_usd"] == "0.123457"
    tomar.assert_not_awaited()
    upsert.assert_not_awaited()
    tipos = [c.args[1] for c in eventos.await_args_list]
    assert tipos == ["PREVUELO_RECHAZADO"]
    assert eventos.await_args.args[2] == e.value.detail


@pytest.mark.parametrize("endpoint", ["resume", "approve"])
def test_prevuelo_que_no_puede_correr_da_503_sin_tomar_la_epoca(endpoint):
    pasos = [_paso(0, facet="hyde", status=StepStatus.blocked_human_gate), _paso(1, depends_on=[0])]
    tomar = AsyncMock(return_value=4)
    with pytest.raises(HTTPException) as e:
        _llamar(endpoint, _interrumpido_con({}), pasos,
                AsyncMock(side_effect=OSError("base caída password=hunter2")), tomar=tomar)
    assert e.value.status_code == 503
    assert e.value.detail["code"] == "prevuelo_no_disponible"
    assert "hunter2" not in e.value.detail["motivo"]
    tomar.assert_not_awaited()


@pytest.mark.parametrize("endpoint", ["resume", "approve"])
def test_respuesta_200_suma_el_costo_del_prevuelo(endpoint):
    pasos = [_paso(0, facet="hyde", status=StepStatus.blocked_human_gate), _paso(1, depends_on=[0])]
    r, bg = _llamar(endpoint, _interrumpido_con({}), pasos, AsyncMock(return_value=_veredicto()))
    assert r["costo_max_usd"] == "0.123457"
    assert r["pasos_costo"] == [_veredicto().pasos_costo[0].to_dict()]
    assert r["run_epoch"] == 4 and bg.tasks[0].args[0].run_epoch == 4


def test_resume_evalua_los_pasos_sin_ref_legible_con_contexto_sin_sus_refs():
    pasos = [_paso(0), _paso(1, depends_on=[0]), _paso(2, depends_on=[1])]
    pipeline = _interrumpido_con({"step_0_ref": _REF_OK, "step_1_ref": _REF_ROTA})
    prevuelo = AsyncMock(return_value=_veredicto())
    _llamar("resume", pipeline, pasos, prevuelo)
    prevuelo.assert_awaited_once()
    args, kwargs = prevuelo.await_args.args, prevuelo.await_args.kwargs
    assert [s.step_index for s in args[0]] == [0, 1, 2]
    assert args[1].get("step_0_ref") == _REF_OK
    assert "step_1_ref" not in args[1]
    assert kwargs["pendientes"] == {1, 2}
    assert (kwargs["user_id"], kwargs["tenant_id"]) == ("7", "1")


def test_approve_step_evalua_la_ola_completa_no_solo_el_paso_aprobado():
    pasos = [
        _paso(0),
        _paso(1, facet="hyde", status=StepStatus.blocked_human_gate, depends_on=[0]),
        _paso(2, depends_on=[1]),
        _paso(3, depends_on=[0]),
    ]
    pipeline = _interrumpido_con({"step_0_ref": _REF_OK})
    prevuelo = AsyncMock(return_value=_veredicto())
    _llamar("approve", pipeline, pasos, prevuelo)
    prevuelo.assert_awaited_once()
    assert prevuelo.await_args.kwargs["pendientes"] == {1, 2, 3}
    assert prevuelo.await_args.args[1].get("hyde_approved_s1") is True


# ---------------------------------------------------------------------------
# Pasada final R34, 4: resume y approve-step quitan del contexto las refs
# ILEGIBLES de los pasos que se rehacen (misma regla que continue). Antes el
# pre-vuelo los cobraba pero run_pipeline los daba por hechos (tenían ref en
# el contexto) y sus dependientes fallaban al leerla.
# ---------------------------------------------------------------------------

def test_resume_quita_la_ref_ilegible_y_el_paso_se_vuelve_a_correr():
    pasos = [_paso(0, status=StepStatus.completed), _paso(1, status=StepStatus.completed, depends_on=[0]),
             _paso(2, depends_on=[1])]
    pasos[1].output_ref = _REF_ROTA
    pipeline = _interrumpido_con({"step_0_ref": _REF_OK, "step_1_ref": _REF_ROTA})
    tomar, upsert = AsyncMock(return_value=4), AsyncMock()
    r, bg = _llamar("resume", pipeline, pasos, AsyncMock(return_value=_veredicto()), tomar=tomar, upsert=upsert)
    assert tomar.await_args.args[:3] == ("p1", 3, (PipelineStatus.interrupted,))
    guardado = tomar.await_args.args[3]
    assert guardado.get("step_0_ref") == _REF_OK and "step_1_ref" not in guardado
    lanzado = bg.tasks[0].args[0]
    assert "step_1_ref" not in lanzado.context and lanzado.context.get("step_0_ref") == _REF_OK
    paso_1 = lanzado.plan[1]
    assert (paso_1.status, paso_1.output_ref, paso_1.error) == (StepStatus.pending, None, None)
    assert [c.args[0].step_index for c in upsert.await_args_list] == [1]


def test_resume_sin_refs_ilegibles_no_reescribe_el_contexto():
    pasos = [_paso(0, status=StepStatus.completed), _paso(1, depends_on=[0])]
    pipeline = _interrumpido_con({"step_0_ref": _REF_OK})
    tomar, upsert = AsyncMock(return_value=4), AsyncMock()
    _llamar("resume", pipeline, pasos, AsyncMock(return_value=_veredicto()), tomar=tomar, upsert=upsert)
    tomar.assert_awaited_once_with("p1", 3, (PipelineStatus.interrupted,))
    upsert.assert_not_awaited()


def test_approve_quita_la_ref_ilegible_y_conserva_la_marca_de_hyde():
    pasos = [_paso(0, status=StepStatus.completed),
             _paso(1, facet="hyde", status=StepStatus.blocked_human_gate, depends_on=[0])]
    pasos[0].output_ref = _REF_ROTA
    pipeline = _interrumpido_con({"step_0_ref": _REF_ROTA})
    tomar, upsert = AsyncMock(return_value=4), AsyncMock()
    r, bg = _llamar("approve", pipeline, pasos, AsyncMock(return_value=_veredicto()), tomar=tomar, upsert=upsert)
    guardado = tomar.await_args.args[3]
    assert "step_0_ref" not in guardado and guardado.get("hyde_approved_s1") is True
    lanzado = bg.tasks[0].args[0]
    assert "step_0_ref" not in lanzado.context and lanzado.context.get("hyde_approved_s1") is True
    assert lanzado.plan[0].status == StepStatus.pending and lanzado.plan[0].output_ref is None
    assert sorted(c.args[0].step_index for c in upsert.await_args_list) == [0, 1]


# Pasada R37, 3: si el pedido PIERDE la carrera por la época, no toca pasos
# aunque haya refs ilegibles que rehacer (el reset va DESPUÉS de tomarla).

@pytest.mark.parametrize("endpoint", ["resume", "approve"])
def test_carrera_perdida_con_ref_ilegible_da_409_sin_tocar_pasos(endpoint):
    pasos = [_paso(0, status=StepStatus.completed),
             _paso(1, facet="hyde", status=StepStatus.blocked_human_gate, depends_on=[0])]
    pasos[0].output_ref = _REF_ROTA
    pipeline = _interrumpido_con({"step_0_ref": _REF_ROTA})
    upsert = AsyncMock()
    with pytest.raises(HTTPException) as e:
        _llamar(endpoint, pipeline, pasos, AsyncMock(return_value=_veredicto()),
                tomar=AsyncMock(return_value=None), upsert=upsert)
    assert e.value.status_code == 409 and isinstance(e.value.detail, str)
    upsert.assert_not_awaited()
    assert pasos[0].status == StepStatus.completed and pasos[0].output_ref == _REF_ROTA
