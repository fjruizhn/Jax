"""Endpoints de continuar (spec 2026-09-17 §5.1). El servicio va mockeado: lo
que se prueba acá es el mapeo HTTP y que la corrida se lance con la época nueva.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402

from jacobs import continuar, routes  # noqa: E402
from jacobs.models import Pipeline  # noqa: E402


def test_continue_ok_lanza_la_corrida_con_la_epoca_nueva():
    pipeline = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous", run_epoch=5)
    servicio = AsyncMock(return_value=({"pipeline_id": "p1", "run_epoch": 5}, pipeline))
    bg = BackgroundTasks()
    with patch.object(continuar, "continuar", servicio):
        r = asyncio.run(routes.continue_pipeline("p1", routes.ContinueRequest(
            invoked_by="plataforma", reasignar={"4": "ada"}, costo_max_aceptado_usd=Decimal("1.5")), bg))
    assert r == {"pipeline_id": "p1", "run_epoch": 5}
    servicio.assert_awaited_once_with("p1", "plataforma", reasignar={"4": "ada"}, user_id=None,
                                      tenant_id=None, costo_max_aceptado_usd=Decimal("1.5"))
    assert bg.tasks[0].func is routes.run_pipeline and bg.tasks[0].args[0].run_epoch == 5


def test_un_rechazo_del_servicio_sale_con_su_status_y_cuerpo():
    rechazo = continuar.ContinuarRechazado(409, "estado_no_continuable", {"status": "completed", "mensaje": "m"})
    bg = BackgroundTasks()
    with patch.object(continuar, "continuar", AsyncMock(side_effect=rechazo)), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.continue_pipeline("p1", routes.ContinueRequest(invoked_by="plataforma"), bg))
    assert e.value.status_code == 409
    assert e.value.detail == {"code": "estado_no_continuable", "status": "completed", "mensaje": "m"}
    assert bg.tasks == []


def test_continue_con_la_base_caida_da_503():
    with patch.object(continuar, "continuar", AsyncMock(side_effect=OSError("base caída"))), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.continue_pipeline(
            "p1", routes.ContinueRequest(invoked_by="plataforma"), BackgroundTasks()))
    assert e.value.status_code == 503 and e.value.detail["code"] == "prevuelo_no_disponible"


def test_continue_preflight_devuelve_200_y_no_lanza():
    esperado = {"continuable": False, "motivo": {"code": "limite_de_activos", "detalle": "x"},
                "pasos_a_correr": [2], "pasos_reusados": [0, 1], "veredicto": None}
    servicio = AsyncMock(return_value=esperado)
    with patch.object(continuar, "previsualizar", servicio):
        r = asyncio.run(routes.continue_preflight(
            "p1", routes.ContinuePreflightRequest(invoked_by="plataforma", reasignar={"2": "thot"})))
    assert r == esperado
    servicio.assert_awaited_once_with("p1", "plataforma", reasignar={"2": "thot"}, user_id=None, tenant_id=None)


def test_continue_preflight_de_un_pipeline_inexistente_da_404():
    rechazo = continuar.ContinuarRechazado(404, "no_existe", "Pipeline 'p1' no encontrado")
    with patch.object(continuar, "previsualizar", AsyncMock(side_effect=rechazo)), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.continue_preflight("p1", routes.ContinuePreflightRequest(invoked_by="plataforma")))
    assert e.value.status_code == 404
    assert e.value.detail == {"code": "no_existe", "detalle": "Pipeline 'p1' no encontrado"}
