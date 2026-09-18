"""
Jacobs — `invoked_by` es un ROL, no el nombre de una persona (tanda A, 2026-09-14).

Hasta hoy `VALID_INVOKERS` era {"Fernando", "jax_local", "ada"} y
`validate_resume` exigía "Fernando": un nombre de persona usado como campo de
AUTORIZACIÓN (Principio IX). Decisión de Fernando (spec 2026-09-14, §2):
`invoked_by` pasa a ser el rol "plataforma" -- "pedido de jax-platform en
nombre de un usuario autenticado"; la identidad viaja en user_id/tenant_id.

Nada sale a la red ni a la DB: `store.pipeline_get` y `_build_plan_or_reject`
se parchean en TODOS los casos, también en los que esperan un rechazo -- con el
código viejo "Fernando" pasaba el control y llegaba a la DB o al planificador.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

# Forzado, no setdefault: mismo guard que jacobs/_pipeline_identity_test.py.
from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from jacobs import policy, routes  # noqa: E402
from jacobs.models import VALID_INVOKERS, PipelineCreateRequest  # noqa: E402

_NO_PLANIFICAR = AsyncMock(side_effect=AssertionError("el test no debe llegar a planificar"))


def test_los_invocadores_validos_son_plataforma_jax_local_y_ada():
    assert VALID_INVOKERS == frozenset({"plataforma", "jax_local", "ada"})


def test_crear_rechaza_fernando_como_invocador():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("Fernando", "supervised", 3)
    assert not r.ok
    assert "Fernando" in r.reason


def test_crear_acepta_plataforma():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("plataforma", "supervised", 3)
    assert r.ok, r.reason


def test_el_request_de_creacion_rechaza_fernando_y_acepta_plataforma():
    with pytest.raises(ValidationError):
        PipelineCreateRequest(name="t", objective="o", invoked_by="Fernando", mode="supervised")
    req = PipelineCreateRequest(name="t", objective="o", invoked_by="plataforma", mode="supervised")
    assert req.invoked_by == "plataforma"


def test_validate_resume_solo_acepta_plataforma():
    assert policy.validate_resume("plataforma").ok
    for otro in ("Fernando", "jax_local", "ada", ""):
        assert not policy.validate_resume(otro).ok, otro


def test_plan_rechaza_fernando_y_deja_pasar_plataforma():
    def plan(invoked_by):
        return routes.plan_only(routes.PlanRequest(
            name="t", objective="o", invoked_by=invoked_by, mode="dry_run"))

    with patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as rechazo:
        asyncio.run(plan("Fernando"))
    assert rechazo.value.status_code == 403

    # "plataforma" pasa el control de invocador; lo frena el kill switch
    # (forzado a propósito) para que el test no llegue a planificar.
    with patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         patch.object(routes, "check_kill_switch", return_value=True), \
         pytest.raises(HTTPException) as frenado:
        asyncio.run(plan("plataforma"))
    assert frenado.value.status_code == 423


def test_reanudar_rechaza_fernando_y_deja_pasar_plataforma():
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as rechazo:
            asyncio.run(routes.resume_pipeline(
                "p-inexistente", routes.ResumeRequest(invoked_by="Fernando"), BackgroundTasks()))
        assert rechazo.value.status_code == 403

        # "plataforma" pasa la política y llega a buscar el pipeline (404).
        with pytest.raises(HTTPException) as no_existe:
            asyncio.run(routes.resume_pipeline(
                "p-inexistente", routes.ResumeRequest(invoked_by="plataforma"), BackgroundTasks()))
        assert no_existe.value.status_code == 404


def test_aprobar_paso_rechaza_fernando_y_deja_pasar_plataforma():
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as rechazo:
            asyncio.run(routes.approve_step(
                "p-inexistente", routes.ApproveStepRequest(invoked_by="Fernando"), BackgroundTasks()))
        assert rechazo.value.status_code == 403

        with pytest.raises(HTTPException) as no_existe:
            asyncio.run(routes.approve_step(
                "p-inexistente", routes.ApproveStepRequest(invoked_by="plataforma"), BackgroundTasks()))
        assert no_existe.value.status_code == 404
