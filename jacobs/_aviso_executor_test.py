#!/usr/bin/env python3
"""Task 6 (2026-09-18, historial-y-arreglos-de-pipeline): `_correr_pipeline`
dispara el aviso de Telegram en las transiciones terminales, y sólo ahí.

Arnés propio (no el de tests/test_jacobs_epoca_ejecutor.py) para no competir
por el mismo archivo con las otras tareas de esta ronda que también tocan
`jacobs/executor.py` (ver aviso de concurrencia del brief). Mismo patrón de
tienda falsa, sin DB ni red.

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_aviso_executor_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from unittest.mock import AsyncMock  # noqa: E402

from jacobs import aviso, executor  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step  # noqa: E402


class _TiendaFalsa:
    def __init__(self, status="pending", epoca=0):
        self.status = status
        self.epoca = epoca
        self.contexto: dict = {}
        self.eventos: list[tuple[str, dict]] = []

    async def pipeline_update_status_si_epoca(self, pipeline_id, epoca, status,
                                                current_step_index=None, context=None, *,
                                                desde=(PipelineStatus.running,)):
        if epoca != self.epoca or self.status not in {d.value for d in desde}:
            return False
        self.status = status.value
        if context is not None:
            self.contexto = dict(context)
        return True

    async def pipeline_epoca_y_status(self, pipeline_id):
        return self.epoca, PipelineStatus(self.status)

    async def step_upsert_si_epoca(self, s, epoca):
        if epoca != self.epoca or self.status != "running":
            return False
        return True

    async def event_append(self, pipeline_id, event_type, payload=None, step_id=None):
        self.eventos.append((event_type, payload or {}))

    def tipos(self):
        return [t for t, _ in self.eventos]


def _pipeline(n_pasos=1, epoca=0, modo="autonomous"):
    pasos = [
        Step(pipeline_id="p1", step_index=i, facet="jekyll", capability="research",
             input={"prompt": f"paso {i}"}, depends_on=[i - 1] if i else [])
        for i in range(n_pasos)
    ]
    return Pipeline(pipeline_id="p1", name="mi-pipeline", invoked_by="plataforma", mode=modo,
                    plan=pasos, context={"objective": "o"}, run_epoch=epoca)


def _correr(monkeypatch, tienda, pipeline, despachar, *, kill_switch=False):
    monkeypatch.setattr(executor, "store", tienda)
    monkeypatch.setattr(executor, "check_kill_switch", lambda: kill_switch)
    monkeypatch.setattr(executor, "save_if_large", lambda pid, sid, raw, **kw: (None, raw))
    monkeypatch.setattr(executor, "_persist_step_to_repo", AsyncMock())
    monkeypatch.setattr(executor, "_dispatch_step", despachar)
    asyncio.run(executor.run_pipeline(pipeline))


def test_pipeline_completado_dispara_un_aviso(monkeypatch):
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa()

    async def despachar(step, pipeline):
        return {"success": True, "result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(1), despachar)
    assert tienda.status == "completed"
    assert len(llamadas) == 1
    assert llamadas[0] == {"pipeline_id": "p1", "nombre": "mi-pipeline", "estado": "completed"}


def test_pipeline_abortado_por_step_fallido_dispara_un_aviso(monkeypatch):
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa()

    async def despachar(step, pipeline):
        raise RuntimeError("se cayó el proveedor")

    _correr(monkeypatch, tienda, _pipeline(1), despachar)
    assert tienda.status == "aborted"
    assert len(llamadas) == 1
    assert llamadas[0]["estado"] == "aborted"


def test_dry_run_dispara_un_aviso_completed(monkeypatch):
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa(status="pending")

    async def despachar(step, pipeline):
        raise AssertionError("dry_run no debería despachar nada")

    _correr(monkeypatch, tienda, _pipeline(1, modo="dry_run"), despachar)
    assert tienda.status == "completed"
    assert len(llamadas) == 1
    assert llamadas[0]["estado"] == "completed"


def test_supervised_interrumpido_NO_dispara_aviso(monkeypatch):
    """Pausado esperando /resume no es 'terminó' -- todavía puede seguir."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa()

    async def despachar(step, pipeline):
        return {"success": True, "result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(1, modo="supervised"), despachar)
    assert tienda.status == "interrupted"
    assert llamadas == []


def test_epoca_perdida_NO_dispara_aviso(monkeypatch):
    """Una corrida que perdió la carrera (RUN_SUPERSEDED) no es la que cierra
    el pipeline -- la que ganó ya avisó (o avisará) por su cuenta. Avisar acá
    también sería el doble aviso que la regla prohíbe."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa(status="pending", epoca=1)  # otra corrida ya avanzó la época

    async def despachar(step, pipeline):
        raise AssertionError("no debería despachar nada")

    _correr(monkeypatch, tienda, _pipeline(1, epoca=0), despachar)
    assert "RUN_SUPERSEDED" in tienda.tipos()
    assert llamadas == []


def test_un_aviso_que_lanza_no_cambia_el_status_ya_escrito(monkeypatch):
    """`_disparar_aviso_fin` corre DESPUÉS de que el status terminal ya se
    escribió. Si agendarlo explota, el status queda como quedó -- no hay
    rollback, y la excepción no sube."""
    def _explota(**kw):
        raise RuntimeError("Telegram roto")

    monkeypatch.setattr(aviso, "avisar_fin_pipeline", _explota)
    tienda = _TiendaFalsa()

    async def despachar(step, pipeline):
        return {"success": True, "result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(1), despachar)  # no debe lanzar
    assert tienda.status == "completed"
    assert tienda.tipos().count("PIPELINE_COMPLETED") == 1
