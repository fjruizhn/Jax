"""Un solo ejecutor por pipeline (spec 2026-09-17 §5.3), sin DB ni red.

La tienda falsa implementa la API VIEJA (pipeline_update_status, step_upsert)
y la NUEVA (condicionales por época): así los tests corren contra el ejecutor
de hoy y fallan por comportamiento, no por un AttributeError.

Hoy cancelar no detiene nada: la escritura de fin de ola vuelve a poner
`running` y la ola siguiente se despacha.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from unittest.mock import AsyncMock  # noqa: E402

from jacobs import executor  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step  # noqa: E402


class TiendaFalsa:
    def __init__(self, status="pending", epoca=0):
        self.status = status
        self.epoca = epoca
        self.contexto: dict = {}
        self.pasos: dict[int, str] = {}
        self.eventos: list[tuple[str, dict]] = []

    async def pipeline_update_status(self, pipeline_id, status, current_step_index=None, context=None):
        self.status = status.value
        if context is not None:
            self.contexto = dict(context)

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

    async def step_upsert(self, s):
        self.pasos[s.step_index] = s.status.value

    async def step_upsert_si_epoca(self, s, epoca):
        if epoca != self.epoca or self.status != "running":
            return False
        self.pasos[s.step_index] = s.status.value
        return True

    async def event_append(self, pipeline_id, event_type, payload=None, step_id=None):
        self.eventos.append((event_type, payload or {}))

    def tipos(self):
        return [t for t, _ in self.eventos]


def _pipeline(n_pasos=2, epoca=0):
    pasos = [
        Step(pipeline_id="p1", step_index=i, facet="jekyll", capability="research",
             input={"prompt": f"paso {i}"}, depends_on=[i - 1] if i else [])
        for i in range(n_pasos)
    ]
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                    plan=pasos, context={"objective": "o"}, run_epoch=epoca)


def _correr(monkeypatch, tienda, pipeline, despachar):
    monkeypatch.setattr(executor, "store", tienda)
    monkeypatch.setattr(executor, "check_kill_switch", lambda: False)
    monkeypatch.setattr(executor, "save_if_large", lambda pid, sid, raw: (None, raw))
    monkeypatch.setattr(executor, "_persist_step_to_repo", AsyncMock())
    monkeypatch.setattr(executor, "_dispatch_step", despachar)
    asyncio.run(executor.run_pipeline(pipeline))


def test_cancelar_durante_la_ola_no_se_reescribe_running(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        tienda.status = "aborted"  # POST /cancel mientras el paso corre
        return {"success": True, "result": "hecho"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert tienda.status == "aborted"
    assert despachados == [0]
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1
    assert "PIPELINE_COMPLETED" not in tienda.tipos()


def test_continuado_por_otro_la_corrida_vieja_no_escribe_su_contexto(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        tienda.epoca = 1  # POST /continue tomó el pipeline
        return {"success": True, "result": "tarde"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert "step_0_ref" not in tienda.contexto
    assert despachados == [0]
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_un_solo_pipeline_aborted_con_los_errores(monkeypatch):
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        raise RuntimeError("se cayó el proveedor")

    _correr(monkeypatch, tienda, _pipeline(1), despachar)
    abortados = [p for t, p in tienda.eventos if t == "PIPELINE_ABORTED"]
    assert len(abortados) == 1
    assert abortados[0] == {"at_wave": 0, "failed_steps": [0],
                            "errores": {"0": "se cayó el proveedor"}}
    assert tienda.status == "aborted"


def test_arranque_con_epoca_ajena_no_despacha(monkeypatch):
    tienda, despachados = TiendaFalsa(status="pending", epoca=1), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        return {"success": True, "result": "no debería"}

    _correr(monkeypatch, tienda, _pipeline(2, epoca=0), despachar)
    assert despachados == []
    assert tienda.status == "pending"
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_relee_la_epoca_antes_de_cada_ola(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    def _cancelar_si_ya_persistio_la_ola_0():
        if "step_0_ref" in tienda.contexto and tienda.status == "running":
            tienda.status = "aborted"

    nuevo_original = tienda.pipeline_update_status_si_epoca
    viejo_original = tienda.pipeline_update_status

    async def nuevo(*a, **k):
        ok = await nuevo_original(*a, **k)
        if ok:
            _cancelar_si_ya_persistio_la_ola_0()
        return ok

    async def viejo(*a, **k):
        await viejo_original(*a, **k)
        _cancelar_si_ya_persistio_la_ola_0()

    tienda.pipeline_update_status_si_epoca = nuevo
    tienda.pipeline_update_status = viejo

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        return {"success": True, "result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert despachados == [0]
    assert tienda.status == "aborted"
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_corrida_normal_completa_y_persiste_el_contexto(monkeypatch):
    """CONTROL: pasa también con el ejecutor de hoy."""
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        return {"success": True, "result": f"r{step.step_index}"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert tienda.status == "completed"
    assert {"step_0_ref", "step_1_ref"} <= set(tienda.contexto)
    assert "RUN_SUPERSEDED" not in tienda.tipos()
    assert tienda.tipos().count("PIPELINE_COMPLETED") == 1
