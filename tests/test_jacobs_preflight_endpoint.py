"""POST /jacobs/preflight y el pre-vuelo obligatorio al crear (spec 2026-09-17
§4.7, §6.1, §8). Sin DB: el planificador, el pre-vuelo y el store van mockeados.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from jacobs import policy, routes  # noqa: E402
from jacobs.models import PipelineCreateRequest, Step, StepSpec  # noqa: E402
from jacobs.plan import PlanRejected, PlanViolation  # noqa: E402
from jacobs.prevuelo_reglas import CostoPaso, Veredicto, Violacion  # noqa: E402


def _veredicto(ok=True, usd="0.500000"):
    costo = CostoPaso(0, "jekyll", "deepseek-v4-flash", 1, 100, 8192, Decimal(usd), "acotado")
    violaciones = () if ok else (Violacion(0, "jekyll", "credencial_ausente", "sin credencial"),)
    return Veredicto(ok=ok, violaciones=violaciones, costo_max_usd=Decimal(usd),
                     pasos_costo=(costo,), sondeadas=())


def _pasos():
    return [Step(step_index=0, facet="jekyll", capability="research", input={"prompt": "p"})]


def _spec():
    return [StepSpec(facet="jekyll", capability="research", prompt="p")]


def _spec_n(n):
    return [StepSpec(facet="jekyll", capability="research", prompt="p") for _ in range(n)]


def _parches(pila, veredicto=None, prevuelo=None, build=None):
    m = {}
    for nombre in ("pipeline_create", "step_upsert", "event_append", "pipeline_update_status"):
        m[nombre] = pila.enter_context(patch.object(routes.store, nombre, AsyncMock()))
    m["pipeline_count_active"] = pila.enter_context(
        patch.object(routes.store, "pipeline_count_active", AsyncMock(return_value=0)))
    m["build"] = pila.enter_context(patch.object(
        routes._plan_builder, "build", build or AsyncMock(return_value=_pasos())))
    m["prevuelo"] = pila.enter_context(patch.object(
        routes, "prevuelo", prevuelo or AsyncMock(return_value=veredicto or _veredicto()), create=True))
    pila.enter_context(patch.object(policy, "check_kill_switch", return_value=False))
    return m


def _crear(**cambios):
    datos = dict(name="t", objective="o", invoked_by="plataforma", mode="autonomous", steps=_spec())
    datos.update(cambios)
    return PipelineCreateRequest(**datos)


def test_preflight_devuelve_200_con_el_veredicto_aunque_rechace():
    with ExitStack() as pila:
        _parches(pila, veredicto=_veredicto(ok=False))
        r = asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert r["ok"] is False
    assert r["violaciones"][0]["regla"] == "credencial_ausente"


def test_preflight_rechaza_un_invocador_no_valido_con_403():
    """Ruling R17 (fix round 1, Task 9): el rechazo de /preflight va como
    dict {code, detalle}, a diferencia de los rechazos de texto PREEXISTENTES
    de POST /jacobs/pipeline (que no se tocan en esta rama)."""
    with ExitStack() as pila:
        m = _parches(pila)
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="Fernando", steps=_spec())))
    assert e.value.status_code == 403
    assert e.value.detail["code"] == "invocador_no_autorizado"
    assert "Fernando" in e.value.detail["detalle"]
    m["prevuelo"].assert_not_awaited()


def test_preflight_rechaza_mas_de_20_pasos_con_422():
    """Ruling R17: el tope duro de 20 pasos también va como dict
    {code: "plan_rechazado", detalle}."""
    with ExitStack() as pila:
        m = _parches(pila)
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec_n(21))))
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "plan_rechazado"
    assert "21" in e.value.detail["detalle"]
    m["build"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()


def test_preflight_no_escribe_nada_ni_con_plan_rechazado():
    with ExitStack() as pila:
        m = _parches(pila)
        asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
        for nombre in ("pipeline_create", "step_upsert", "event_append", "pipeline_update_status"):
            m[nombre].assert_not_awaited()
    rechazo = PlanRejected([PlanViolation(0, "jekyll", None, "research", "no va")])
    with ExitStack() as pila:
        m = _parches(pila, build=AsyncMock(side_effect=rechazo))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
        m["event_append"].assert_not_awaited()
    assert e.value.status_code == 422
    # Ruling R17: también dict acá, con el texto de PlanRejected en "detalle".
    assert e.value.detail["code"] == "plan_rechazado"
    assert e.value.detail["detalle"] == str(rechazo)


def test_preflight_con_la_base_caida_da_503():
    with ExitStack() as pila:
        _parches(pila, prevuelo=AsyncMock(side_effect=OSError("base caída")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert e.value.status_code == 503
    assert e.value.detail["code"] == "prevuelo_no_disponible"


def test_503_redacta_el_motivo_y_lo_recorta():
    """_prevuelo_o_503 pasa la excepción por recortar_redactado -- un secreto
    real (forma de key de Gemini) que llegara en el mensaje de una excepción
    NO puede aparecer en claro en el 503, y el motivo respeta el límite de
    300 caracteres que le pasa routes.py."""
    secreto = "AIzaSy" + "X" * 33
    with ExitStack() as pila:
        _parches(pila, prevuelo=AsyncMock(
            side_effect=RuntimeError(f"el proveedor devolvió {secreto} en el cuerpo")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert e.value.status_code == 503
    motivo = e.value.detail["motivo"]
    assert secreto not in motivo
    assert "***" in motivo
    assert len(motivo) <= 300


def test_crear_con_prevuelo_rechazado_da_422_sin_crear_y_deja_evento():
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(ok=False))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["pipeline_create"].assert_not_awaited()
        tipos = [c.args[1] for c in m["event_append"].await_args_list]
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "prevuelo_rechazado"
    assert tipos == ["PREVUELO_RECHAZADO"]


def test_evento_prevuelo_rechazado_tiene_el_mismo_contenido_que_el_422():
    """El payload que se guarda en el evento PREVUELO_RECHAZADO es EXACTAMENTE
    el mismo dict que se devuelve en el detail del 422 -- no dos cálculos que
    puedan divergir."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(ok=False))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    eventos = [c.args[2] for c in m["event_append"].await_args_list if c.args[1] == "PREVUELO_RECHAZADO"]
    assert eventos == [e.value.detail]


def test_422_prevuelo_rechazado_incluye_pipeline_id_del_evento():
    """Ruling R19: el 422 lleva pipeline_id, y es el MISMO id con el que se
    escribió el evento PREVUELO_RECHAZADO."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(ok=False))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    pipeline_id_evento = [
        c.args[0] for c in m["event_append"].await_args_list if c.args[1] == "PREVUELO_RECHAZADO"
    ][0]
    assert e.value.detail["pipeline_id"] == pipeline_id_evento


def test_prevuelo_recibe_identidad_y_objetivo_en_crear():
    """Sin esto, el evento de salud/uso de la sonda quedaría atribuido a
    nadie y el prompt armado no llevaría el objetivo real."""
    with ExitStack() as pila:
        m = _parches(pila)
        asyncio.run(routes.create_pipeline(
            _crear(objective="hacer cosas", user_id="u-1", tenant_id="t-1"), BackgroundTasks()))
    args, kwargs = m["prevuelo"].await_args
    assert args[1] == {"objective": "hacer cosas"}
    assert kwargs["user_id"] == "u-1"
    assert kwargs["tenant_id"] == "t-1"


def test_crear_con_costo_mayor_al_aceptado_da_409_sin_crear():
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(usd="0.500000"))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(
                _crear(costo_max_aceptado_usd=Decimal("0.40")), BackgroundTasks()))
        m["pipeline_create"].assert_not_awaited()
    assert e.value.status_code == 409
    assert e.value.detail["code"] == "costo_supera_lo_aceptado"
    assert e.value.detail["costo_max_usd"] == "0.500000"
    # Ruling R18: mismo grano de 6 decimales que el resto de los montos.
    assert e.value.detail["costo_max_aceptado_usd"] == "0.400000"


def test_crear_permite_cuando_el_costo_es_igual_al_aceptado():
    """El corte es `costo_max_usd > costo_max_aceptado_usd`, estricto: si
    coinciden, no es "supera lo aceptado" y la creación sigue."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(usd="0.500000"))
        r = asyncio.run(routes.create_pipeline(
            _crear(costo_max_aceptado_usd=Decimal("0.5")), BackgroundTasks()))
    m["pipeline_create"].assert_awaited_once()
    assert r["costo_max_usd"] == "0.500000"


def test_crear_costo_cero_no_queda_pelado_en_la_respuesta():
    """Ruling R18 (hallazgo propio, mismo bug que el eco del 409): el resumen
    de costo que alimenta PIPELINE_CREATED y el 200/dry_run pasa por el mismo
    formateo que Veredicto.to_dict() -- un veredicto sin costo real (p.ej.
    solo hyde/ollama) no puede devolver "0" pelado en ningún lado."""
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(usd="0"))
        r = asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        creado = [c.args[2] for c in m["event_append"].await_args_list if c.args[1] == "PIPELINE_CREATED"]
    assert r["costo_max_usd"] == "0.000000"
    assert creado[0]["costo_max_usd"] == "0.000000"


def test_crear_ok_devuelve_el_costo_y_lo_deja_en_el_evento():
    bg = BackgroundTasks()
    with ExitStack() as pila:
        m = _parches(pila)
        r = asyncio.run(routes.create_pipeline(_crear(costo_max_aceptado_usd=Decimal("1")), bg))
        creado = [c.args[2] for c in m["event_append"].await_args_list if c.args[1] == "PIPELINE_CREATED"]
    assert r["costo_max_usd"] == "0.500000" and r["pasos_costo"][0]["paso"] == 0
    assert creado[0]["costo_max_usd"] == "0.500000"
    assert len(bg.tasks) == 1


def test_dry_run_tambien_corre_el_prevuelo():
    with ExitStack() as pila:
        m = _parches(pila)
        r = asyncio.run(routes.create_pipeline(_crear(mode="dry_run"), BackgroundTasks()))
        m["prevuelo"].assert_awaited_once()
    assert r["costo_max_usd"] == "0.500000"


def test_el_prevuelo_corre_despues_de_build_y_antes_de_crear():
    orden = []

    async def build(**kw):
        orden.append("build")
        return _pasos()

    async def prevuelo(*a, **kw):
        orden.append("prevuelo")
        return _veredicto()

    with ExitStack() as pila:
        m = _parches(pila, build=AsyncMock(side_effect=build), prevuelo=AsyncMock(side_effect=prevuelo))
        m["pipeline_create"].side_effect = lambda p: orden.append("crear")
        asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    assert orden == ["build", "prevuelo", "crear"]


def test_crear_con_la_base_caida_da_503_sin_crear():
    with ExitStack() as pila:
        m = _parches(pila, prevuelo=AsyncMock(side_effect=OSError("base caída")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["pipeline_create"].assert_not_awaited()
    assert e.value.status_code == 503


def test_costo_max_aceptado_negativo_es_invalido():
    with pytest.raises(ValidationError):
        _crear(costo_max_aceptado_usd=Decimal("-1"))
