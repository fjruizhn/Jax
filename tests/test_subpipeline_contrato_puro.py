"""Contrato de sub-pipelines: la parte que NO necesita base (frente F, 2026-09-16).

Vistos en rojo contra el código de master (Task 2, Step 4 del plan):
plataforma con token, ada sin token/padre en el request, la firma de
validate_create, /plan con ada y la validación de config al arrancar.

`test_hijo_respeta_el_kill_switch_antes_de_su_primera_ola` y
`test_la_emision_no_tiene_ruta_http` son GUARDAS: ya pasan con el código viejo
y se validan por mutación (Step 7), no por rojo.

Corre con (sin base):
  cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v tests/test_subpipeline_contrato_puro.py
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from jacobs import executor, policy, routes
from jacobs import subpipelines as sp
from jacobs.models import Pipeline, PipelineCreateRequest, PipelineStatus, Step

_NO_PLANIFICAR = AsyncMock(side_effect=AssertionError("el test no debe llegar a planificar"))


def _sin_config(monkeypatch):
    monkeypatch.delenv(sp.ENV_TTL, raising=False)
    monkeypatch.delenv(sp.ENV_MAX_PROFUNDIDAD, raising=False)


def test_config_por_defecto_acotada(monkeypatch):
    _sin_config(monkeypatch)
    assert sp.config_subpipelines() == sp.ConfigSubpipelines(ttl_segundos=300, max_profundidad=3)


def test_config_lee_el_entorno(monkeypatch):
    _sin_config(monkeypatch)
    monkeypatch.setenv(sp.ENV_TTL, "90")
    monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "2")
    assert sp.config_subpipelines() == sp.ConfigSubpipelines(ttl_segundos=90, max_profundidad=2)


def test_config_no_entera_falla(monkeypatch):
    _sin_config(monkeypatch)
    monkeypatch.setenv(sp.ENV_TTL, "cinco minutos")
    with pytest.raises(ValueError, match=sp.ENV_TTL):
        sp.config_subpipelines()


def test_config_fuera_de_rango_falla(monkeypatch):
    for nombre, valor in (
        (sp.ENV_TTL, "9"), (sp.ENV_TTL, "3601"),
        (sp.ENV_MAX_PROFUNDIDAD, "0"), (sp.ENV_MAX_PROFUNDIDAD, "6"),
    ):
        _sin_config(monkeypatch)
        monkeypatch.setenv(nombre, valor)
        with pytest.raises(ValueError, match=nombre):
            sp.config_subpipelines()


def test_config_vacia_falla_en_vez_de_tomar_el_defecto(monkeypatch):
    _sin_config(monkeypatch)
    monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "")
    with pytest.raises(ValueError, match=sp.ENV_MAX_PROFUNDIDAD):
        sp.config_subpipelines()


def test_hash_token_es_sha256_hex():
    assert sp.hash_token("abc") == hashlib.sha256(b"abc").hexdigest()
    assert sp.token_ref(sp.hash_token("abc")) == hashlib.sha256(b"abc").hexdigest()[:12]


SECRETO = "SECRETO-NO-ECO"


def _forma_rechazada_sin_eco(req: PipelineCreateRequest, campo: str) -> None:
    """La forma por rol la decide validate_create (422 con `policy.reason`), no
    el validador de Pydantic: un 422 de Pydantic devuelve el cuerpo entero en
    `input`, token incluido (hallazgo I-1 de la revisión final, 2026-09-16)."""
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create(
            req.invoked_by, req.mode, req.max_steps,
            subpipeline_token=req.subpipeline_token,
            parent_pipeline_id=req.parent_pipeline_id,
        )
    assert not r.ok
    assert campo in r.reason
    assert SECRETO not in r.reason


def test_plataforma_no_puede_presentar_token_ni_padre():
    for extra in ({"subpipeline_token": SECRETO}, {"parent_pipeline_id": "p"}):
        req = PipelineCreateRequest(
            name="t", objective="o", invoked_by="plataforma", mode="dry_run", **extra)
        _forma_rechazada_sin_eco(req, "subpipeline_token")


def test_ada_exige_token_y_padre():
    for extra in ({}, {"subpipeline_token": SECRETO}, {"parent_pipeline_id": "p"}):
        req = PipelineCreateRequest(name="t", objective="o", invoked_by="ada", mode="dry_run", **extra)
        _forma_rechazada_sin_eco(req, "parent_pipeline_id")
    req = PipelineCreateRequest(
        name="t", objective="o", invoked_by="ada", mode="dry_run",
        subpipeline_token="x", parent_pipeline_id="p",
    )
    assert (req.subpipeline_token, req.parent_pipeline_id) == ("x", "p")


def test_el_422_de_forma_no_devuelve_el_token_por_http():
    """De punta a punta por HTTP: la validación ocurre antes de tocar la base
    (el conteo de activos se sustituye; kill switch apagado)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(routes.router)
    cuerpos = (
        {"invoked_by": "ada", "subpipeline_token": SECRETO},
        {"invoked_by": "plataforma", "subpipeline_token": SECRETO},
        # Otro campo inválido con el token presente: el error de Pydantic no
        # puede arrastrar el cuerpo entero.
        {"invoked_by": "ada", "subpipeline_token": SECRETO, "parent_pipeline_id": "p",
         "mode_invalido": True},
    )
    with patch.object(routes.cupo, "reservar_cupo", AsyncMock(return_value=True)), \
         patch.object(routes.cupo, "completar_reserva", AsyncMock(return_value=None)), \
         patch.object(routes.cupo, "soltar_reserva", AsyncMock(return_value=1)), \
         patch.object(policy, "check_kill_switch", return_value=False), \
         patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         TestClient(app) as cliente:
        for extra in cuerpos:
            cuerpo = {"name": "t", "objective": "o", "mode": "dry_run", **extra}
            if cuerpo.pop("mode_invalido", False):
                cuerpo["mode"] = "no-es-un-modo"
            r = cliente.post("/jacobs/pipeline", json=cuerpo)
            assert r.status_code == 422, (extra["invoked_by"], r.text)
            assert SECRETO not in r.text, (extra["invoked_by"], r.text)


def test_la_profundidad_nunca_la_pone_el_llamador():
    assert "subpipeline_depth" not in inspect.signature(policy.validate_create).parameters
    assert not hasattr(policy, "MAX_SUBPIPELINE_DEPTH")


def test_validate_create_rechaza_plataforma_con_token():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("plataforma", "dry_run", 3, subpipeline_token="x")
    assert not r.ok
    assert "subpipeline_token" in r.reason


def test_validate_create_ada_sin_padre_se_rechaza():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("ada", "dry_run", 3, subpipeline_token="x")
    assert not r.ok
    assert "parent_pipeline_id" in r.reason


def test_validate_create_ada_con_token_y_padre_pasa_solo_la_forma():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create(
            "ada", "dry_run", 3, subpipeline_token="x", parent_pipeline_id="p")
    assert r.ok, r.reason


def test_plan_no_acepta_ada():
    req = routes.PlanRequest(name="t", objective="o", invoked_by="ada", mode="dry_run")
    with patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as rechazo:
        asyncio.run(routes.plan_only(req))
    assert rechazo.value.status_code == 403


def test_motivo_de_emision():
    base = {
        "padre_status": "running", "padre_depth": 0,
        "paso_status": "running", "paso_facet": "ada", "paso_pipeline_id": "P",
    }
    casos = [
        ({"padre_status": None, "padre_depth": None}, sp.Motivo.PADRE_DESCONOCIDO),
        ({"paso_status": None, "paso_facet": None, "paso_pipeline_id": None}, sp.Motivo.PASO_DESCONOCIDO),
        ({"paso_pipeline_id": "OTRO"}, sp.Motivo.PASO_DESCONOCIDO),
        ({"padre_status": "completed"}, sp.Motivo.PADRE_INACTIVO),
        ({"paso_facet": "jekyll"}, sp.Motivo.PASO_NO_ES_ADA),
        # El paso que delegó puede estar en cualquier estado (enmienda 2026-09-16):
        # un paso de Ada terminado con el padre corriendo NO es motivo de rechazo.
        ({"paso_status": "completed"}, sp.Motivo.ESTADO_CAMBIO),
        ({"padre_depth": 3}, sp.Motivo.PROFUNDIDAD_EXCEDIDA),
        ({}, sp.Motivo.ESTADO_CAMBIO),
    ]
    for cambio, esperado in casos:
        assert sp.motivo_emision({**base, **cambio}, "P", 3) == esperado, cambio


def test_motivo_de_consumo():
    ahora = 1000.0
    base = {
        "usado_at": None, "vence_at": 2000.0, "parent_pipeline_id": "P",
        "depth_hijo": 1, "padre_status": "running", "paso_step_id": "S",
    }
    assert sp.motivo_consumo(None, "P", ahora, 3) == sp.Motivo.TOKEN_DESCONOCIDO
    casos = [
        ({"usado_at": 999.0}, sp.Motivo.TOKEN_USADO),
        ({"vence_at": 1000.0}, sp.Motivo.TOKEN_VENCIDO),
        ({"parent_pipeline_id": "OTRO"}, sp.Motivo.PADRE_NO_COINCIDE),
        ({"depth_hijo": 4}, sp.Motivo.PROFUNDIDAD_EXCEDIDA),
        ({"padre_status": "aborted"}, sp.Motivo.PADRE_INACTIVO),
        ({"padre_status": None}, sp.Motivo.PADRE_INACTIVO),
        ({"paso_step_id": None}, sp.Motivo.PASO_DESCONOCIDO),
        ({}, sp.Motivo.ESTADO_CAMBIO),
    ]
    for cambio, esperado in casos:
        assert sp.motivo_consumo({**base, **cambio}, "P", ahora, 3) == esperado, cambio


def test_hijo_respeta_el_kill_switch_antes_de_su_primera_ola():
    hijo = Pipeline(
        pipeline_id="hijo", name="h", invoked_by="ada", mode="autonomous",
        parent_pipeline_id="padre", depth=1,
        plan=[Step(pipeline_id="hijo", facet="jekyll", capability="summarize")],
    )
    eventos: list[str] = []

    async def _evento(pipeline_id, event_type, payload=None, step_id=None):
        eventos.append(event_type)

    with patch.object(executor.store, "pipeline_update_status", AsyncMock()) as estado, \
         patch.object(executor.store, "event_append", AsyncMock(side_effect=_evento)), \
         patch.object(executor.store, "step_upsert", AsyncMock()), \
         patch.object(executor, "check_kill_switch", return_value=True), \
         patch.object(executor, "_dispatch_step",
                      AsyncMock(side_effect=AssertionError("un hijo frenado no despacha"))):
        asyncio.run(executor.run_pipeline(hijo))
    assert "KILL_SWITCH_ABORTED" in eventos
    assert estado.await_args_list[-1].args[1] == PipelineStatus.aborted


def test_la_emision_no_tiene_ruta_http():
    rutas = [r.path for r in routes.router.routes]
    assert not [p for p in rutas if "token" in p or "subpipeline" in p], rutas


def test_las_manos_valida_la_config_al_arrancar():
    fuente = (Path(__file__).resolve().parents[1] / "las_manos" / "server.py").read_text(encoding="utf-8")
    inicio = fuente.index("async def _jacobs_init")
    fin = fuente.index("await jacobs_store.init_tables()", inicio)
    assert "config_subpipelines()" in fuente[inicio:fin]
