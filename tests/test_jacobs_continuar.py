"""Continuar un pipeline abortado o vencido (spec 2026-09-17 §5.1-§5.2;
desvíos 7, 10 y 11 del plan). Sin DB: store, pre-vuelo y gobernanza mockeados.
Las refs inline se leen de verdad; la ilegible es un artifact que no existe.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import continuar  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from jacobs.prevuelo_reglas import CostoPaso, Veredicto, Violacion  # noqa: E402

_NADA = object()
_REF_A = 'inline:{"result": "a"}'
_REF_B = 'inline:{"result": "b"}'
_ILEGIBLE = "artifact://jacobs/no-existe/tampoco/output.json"


def _pasos(capability_2="research", facet_1="jekyll"):
    pasos = []
    for i in range(3):
        pasos.append(Step(
            step_id=f"s{i}", pipeline_id="p1", step_index=i,
            facet=facet_1 if i == 1 else "jekyll",
            capability=capability_2 if i == 2 else "research",
            input={"prompt": f"p{i}"}, depends_on=[i - 1] if i else [],
            status=StepStatus.completed if i < 2 else StepStatus.failed,
            error=None if i < 2 else "cortado",
            started_at=1.0, finished_at=2.0,
            output_ref=_REF_A if i == 0 else (_REF_B if i == 1 else None),
        ))
    return pasos


def _abortado(status=PipelineStatus.aborted, contexto=None, epoca=2):
    return Pipeline(
        pipeline_id="p1", name="t", invoked_by="plataforma", user_id="7", tenant_id="1",
        mode="autonomous", status=status, run_epoch=epoca,
        context=contexto if contexto is not None else
        {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _REF_B},
    )


def _veredicto(ok=True, usd="0.250000"):
    costo = CostoPaso(2, "jekyll", "deepseek-v4-flash", 1, 100, 8192, Decimal(usd), "acotado")
    violaciones = () if ok else (Violacion(2, "jekyll", "faceta_caida", "timeout de sonda (20s)"),)
    return Veredicto(ok=ok, violaciones=violaciones, costo_max_usd=Decimal(usd),
                     pasos_costo=(costo,), sondeadas=())


@contextmanager
def _entorno(pipeline=_NADA, pasos=None, activos=0, veredicto=None, transaccion=3, kill=False):
    with ExitStack() as pila:
        m = {}
        m["get"] = pila.enter_context(patch.object(continuar.store, "pipeline_get", AsyncMock(
            return_value=_abortado() if pipeline is _NADA else pipeline)))
        m["pasos"] = pila.enter_context(patch.object(continuar.store, "steps_by_pipeline", AsyncMock(
            return_value=pasos if pasos is not None else _pasos())))
        m["activos"] = pila.enter_context(patch.object(
            continuar.store, "pipeline_count_active", AsyncMock(return_value=activos)))
        m["tx"] = pila.enter_context(patch.object(
            continuar.store, "continuar_transaccion", AsyncMock(return_value=transaccion)))
        m["evento"] = pila.enter_context(patch.object(continuar.store, "event_append", AsyncMock()))
        m["prevuelo"] = pila.enter_context(patch.object(
            continuar, "prevuelo", AsyncMock(return_value=veredicto or _veredicto())))
        m["capacidades"] = pila.enter_context(patch.object(
            continuar, "_validate_plan_capabilities", AsyncMock()))
        pila.enter_context(patch.object(continuar, "check_kill_switch", return_value=kill))
        yield m


def _continuar(**kw):
    return asyncio.run(continuar.continuar("p1", kw.pop("invoked_by", "plataforma"), **kw))


def _rechazo(**kw):
    with pytest.raises(continuar.ContinuarRechazado) as e:
        _continuar(**kw)
    return e.value


def test_un_abortado_se_continua_reusando_lo_que_tiene_ref():
    with _entorno() as m:
        r, p = _continuar()
        args = m["tx"].await_args.args
    assert r["pasos_reusados"] == [0, 1] and r["pasos_a_correr"] == [2]
    assert r["run_epoch"] == 3 and p.run_epoch == 3 and p.status == PipelineStatus.running
    assert args[:3] == ("p1", 2, PipelineStatus.aborted)
    assert [s.step_index for s in args[3]] == [2]


def test_un_vencido_tambien_se_continua():
    with _entorno(pipeline=_abortado(status=PipelineStatus.expired)) as m:
        r, _ = _continuar()
        assert m["tx"].await_args.args[2] == PipelineStatus.expired
    assert r["status"] == "running"


def test_los_demas_estados_dan_409_con_el_status():
    for status in (PipelineStatus.completed, PipelineStatus.running, PipelineStatus.pending,
                   PipelineStatus.interrupted, PipelineStatus.failed):
        with _entorno(pipeline=_abortado(status=status)) as m:
            e = _rechazo()
            m["tx"].assert_not_awaited()
        assert (e.status_code, e.code, e.cuerpo()["status"]) == (409, "estado_no_continuable", status.value)


def test_invocador_distinto_de_plataforma_403():
    with _entorno():
        e = _rechazo(invoked_by="ada")
    assert (e.status_code, e.code) == (403, "invocador_no_autorizado")


def test_pipeline_inexistente_404():
    with _entorno(pipeline=None):
        e = _rechazo()
    assert (e.status_code, e.code) == (404, "no_existe")


def test_kill_switch_423():
    with _entorno(kill=True) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (423, "kill_switch")


def test_ref_ilegible_se_rehace_y_sale_del_contexto():
    contexto = {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _ILEGIBLE}
    with _entorno(pipeline=_abortado(contexto=contexto)) as m:
        r, _ = _continuar()
        ctx = m["tx"].await_args.args[5]
    assert r["pasos_reusados"] == [0] and r["pasos_a_correr"] == [1, 2]
    assert "step_1_ref" not in ctx and ctx["step_0_ref"] == _REF_A


def test_los_pasos_a_correr_quedan_pendientes_limpios():
    with _entorno() as m:
        _continuar()
        paso = m["tx"].await_args.args[3][0]
    assert (paso.status, paso.error, paso.started_at, paso.finished_at, paso.output_ref) == (
        StepStatus.pending, None, None, None, None)


def test_reasignar_cambia_la_faceta_y_recalcula_el_motor():
    with _entorno() as m:
        _continuar(reasignar={"2": "kimi"})
        paso = m["tx"].await_args.args[3][0]
        evento = [c.args[2] for c in m["evento"].await_args_list if c.args[1] == "PIPELINE_CONTINUED"][0]
    assert (paso.facet, paso.motor) == ("kimi", "kimi")
    assert evento["reasignados"] == {"2": {"de": "jekyll", "a": "kimi"}}
    with _entorno() as m:
        _continuar(reasignar={"2": "thot"})
        paso = m["tx"].await_args.args[3][0]
    assert (paso.facet, paso.motor) == ("thot", None)


def test_reasignar_un_paso_reusado_da_422():
    with _entorno() as m:
        e = _rechazo(reasignar={"0": "thot"})
        m["tx"].assert_not_awaited()
        m["prevuelo"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")


def test_reasignar_a_una_faceta_desconocida_da_422():
    with _entorno() as m:
        e = _rechazo(reasignar={"2": "gpt"})
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")
    assert "gpt" in e.cuerpo()["detalle"][0]["motivo"]


def test_reasignacion_que_rompe_el_cleanroom_da_422_sin_escribir():
    pasos = _pasos(capability_2="critique", facet_1="thot")
    with _entorno(pasos=pasos) as m:
        e = _rechazo(reasignar={"2": "thot"})
        m["tx"].assert_not_awaited()
        m["evento"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")
    assert "cleanroom" in e.cuerpo()["detalle"][0]["reason"]


def test_limite_de_activos_da_429():
    with _entorno(activos=3) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (429, "limite_de_activos")


def test_prevuelo_rechazado_da_422_sin_escribir_el_pipeline():
    with _entorno(veredicto=_veredicto(ok=False)) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
        tipos = [c.args[1] for c in m["evento"].await_args_list]
    assert (e.status_code, e.code) == (422, "prevuelo_rechazado")
    assert tipos == ["PREVUELO_RECHAZADO"]


def test_costo_mayor_al_aceptado_da_409_sin_escribir():
    with _entorno() as m:
        e = _rechazo(costo_max_aceptado_usd=Decimal("0.10"))
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (409, "costo_supera_lo_aceptado")
    assert e.cuerpo()["costo_max_aceptado_usd"] == "0.100000"


def test_el_costo_cero_no_queda_pelado():
    # R18: todo monto que sale de Jacobs es texto con 6 decimales -- un
    # costo_max_usd sin evaluables (Decimal(0), sin cuantizar) no puede
    # llegar como "0" pelado ni en la respuesta ni en el evento.
    veredicto = Veredicto(ok=True, violaciones=(), costo_max_usd=Decimal(0), pasos_costo=(), sondeadas=())
    with _entorno(veredicto=veredicto) as m:
        r, _ = _continuar()
        evento = [c.args[2] for c in m["evento"].await_args_list if c.args[1] == "PIPELINE_CONTINUED"][0]
    assert r["costo_max_usd"] == "0.000000"
    assert evento["costo_max_usd"] == "0.000000"


def test_si_otro_pedido_gano_la_transaccion_da_409():
    with _entorno(transaccion=None) as m:
        e = _rechazo()
        tipos = [c.args[1] for c in m["evento"].await_args_list]
    assert (e.status_code, e.code) == (409, "estado_no_continuable")
    assert "PIPELINE_CONTINUED" not in tipos


def test_el_evento_continued_lleva_todo_el_detalle():
    with _entorno() as m:
        _continuar()
        evento = [c.args[2] for c in m["evento"].await_args_list if c.args[1] == "PIPELINE_CONTINUED"][0]
    assert evento == {"by": "plataforma", "from_status": "aborted", "run_epoch": 3,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.250000"}


def test_la_marca_de_hyde_del_paso_a_rehacer_se_quita():
    contexto = {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _REF_B,
                "hyde_approved_s0": True, "hyde_approved_s2": True}
    with _entorno(pipeline=_abortado(contexto=contexto)) as m:
        _continuar()
        ctx = m["tx"].await_args.args[5]
    assert "hyde_approved_s2" not in ctx and ctx["hyde_approved_s0"] is True


def test_el_prevuelo_solo_mira_los_pendientes():
    with _entorno() as m:
        _continuar()
        kw = m["prevuelo"].await_args.kwargs
    assert kw["pendientes"] == {2}
    assert (kw["user_id"], kw["tenant_id"]) == ("7", "1")


def test_previsualizar_no_escribe_y_explica_por_que_no():
    with _entorno() as m:
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
        m["tx"].assert_not_awaited()
        m["evento"].assert_not_awaited()
    assert r["continuable"] is True and r["motivo"] is None and r["veredicto"]["ok"] is True
    with _entorno(pipeline=_abortado(status=PipelineStatus.completed)):
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
    assert r["continuable"] is False and r["motivo"]["code"] == "estado_no_continuable"
    with _entorno(activos=3):
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
    assert r["continuable"] is False and r["motivo"]["code"] == "limite_de_activos"
