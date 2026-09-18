"""Jacobs y el freno (plan 2026-09-16-frente-b-kill-switch, Task 4).

Antes: la ruta era una constante de policy.py, se miraba con Path.exists()
(suelto si el directorio es ilegible) y un step ya lanzado seguía hasta
terminar la ola: un Hyde en vuelo no se cortaba. Nada sale a la red ni a la
DB: store y el despacho se parchean.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_jacobs_interruptor.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import interruptor  # noqa: E402
from jacobs import executor, policy, routes  # noqa: E402

ES_ROOT = os.geteuid() == 0
_NO_PLANIFICAR = AsyncMock(side_effect=AssertionError("el test no debe llegar a planificar"))


def test_policy_no_guarda_una_ruta_propia():
    assert not hasattr(policy, "KILL_SWITCH_PATH")


def test_check_kill_switch_sigue_la_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_KILL_SWITCH_PATH", str(tmp_path / "PAUSE"))
    assert policy.check_kill_switch() is False
    (tmp_path / "PAUSE").write_text("")
    assert policy.check_kill_switch() is True


def test_check_kill_switch_sin_variable_falla_cerrado(monkeypatch):
    monkeypatch.delenv("JAX_KILL_SWITCH_PATH", raising=False)
    with pytest.raises(interruptor.InterruptorSinConfigurar):
        policy.check_kill_switch()


@pytest.mark.skipif(ES_ROOT, reason="root atraviesa cualquier permiso")
def test_check_kill_switch_con_directorio_ilegible_esta_puesto(tmp_path, monkeypatch):
    carpeta = tmp_path / "interruptor"
    carpeta.mkdir()
    (carpeta / "PAUSE").write_text("")
    monkeypatch.setenv("JAX_KILL_SWITCH_PATH", str(carpeta / "PAUSE"))
    carpeta.chmod(0)
    try:
        assert policy.check_kill_switch() is True
    finally:
        carpeta.chmod(0o700)


def test_validate_create_con_freno_puesto_rechaza(tmp_path, monkeypatch):
    (tmp_path / "PAUSE").write_text("")
    monkeypatch.setenv("JAX_KILL_SWITCH_PATH", str(tmp_path / "PAUSE"))
    r = policy.validate_create("plataforma", "supervised", 3)
    assert not r.ok
    assert "Kill switch" in r.reason


def test_plan_con_freno_puesto_da_423(tmp_path, monkeypatch):
    (tmp_path / "PAUSE").write_text("")
    monkeypatch.setenv("JAX_KILL_SWITCH_PATH", str(tmp_path / "PAUSE"))
    with patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         pytest.raises(HTTPException) as frenado:
        asyncio.run(routes.plan_only(routes.PlanRequest(
            name="t", objective="o", invoked_by="plataforma", mode="dry_run")))
    assert frenado.value.status_code == 423


def test_step_en_vuelo_se_corta_al_poner_el_freno(tmp_path, monkeypatch):
    ruta = tmp_path / "PAUSE"
    monkeypatch.setenv("JAX_KILL_SWITCH_PATH", str(ruta))
    cancelado = []
    fallas = []

    async def escenario():
        loop = asyncio.get_running_loop()
        en_vuelo = asyncio.Event()

        async def despacho_largo(step, pipeline):
            en_vuelo.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelado.append(True)  # run_sandboxed_claude mata el proceso acá
                raise

        async def fail_step(pipeline, step, i, error):
            fallas.append(error)

        step = SimpleNamespace(status=None, started_at=None, finished_at=None, facet="hyde",
                               capability="implementation", timeout_seconds=30, step_id="s0",
                               output_ref=None, motor=None, error=None, pipeline_id="p-freno")
        # run_epoch (spec 2026-09-17 §5.3): toda escritura del ejecutor es
        # condicional a la época del pipeline; el doble tiene que traerla.
        pipeline = SimpleNamespace(pipeline_id="p-freno", context={}, name="t", run_epoch=1)
        with patch.object(executor.store, "step_upsert", AsyncMock()), \
             patch.object(executor.store, "step_upsert_si_epoca", AsyncMock(return_value=True)), \
             patch.object(executor.store, "event_append", AsyncMock()), \
             patch.object(executor, "_dispatch_step", despacho_largo), \
             patch.object(executor, "_fail_step", fail_step):
            tarea = asyncio.create_task(executor._run_one_step(step, 0, pipeline))
            await asyncio.wait_for(en_vuelo.wait(), 2.0)
            interruptor.escribir_pausa(ruta, "{}")
            inicio = loop.time()
            completo = await asyncio.wait_for(tarea, 2.0)
            return completo, loop.time() - inicio

    completo, duracion = asyncio.run(escenario())
    assert completo is False
    assert cancelado == [True]
    assert len(fallas) == 1 and "killed_by_switch" in fallas[0]
    assert duracion < 1.0
