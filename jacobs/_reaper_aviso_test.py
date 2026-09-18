"""IMPORTANTE 5 (revisión final 2026-09-18, tanda historial-y-arreglos-de-pipeline):
`expired` no dispara el aviso de Telegram.

EL DEFECTO. `_disparar_aviso_fin` (jacobs/executor.py) se llama en los 4
puntos donde `_correr_pipeline` gana la escritura de un status TERMINAL
(completed x2, aborted x2). El reaper (`jacobs/reaper.py::reap_orphaned_pipelines`)
también escribe un status terminal -- `expired` -- pero nunca llama al
aviso. El spec §3.3 dice "se avisa cuando termina, bien o mal" -- y los
pipelines que mueren SOLOS (huérfanos, nadie los está mirando) son justo los
que más necesitan el aviso: nadie va a notar que terminaron de otra forma.

Sin DB, sin red: mismo patrón de tienda falsa que `_aviso_executor_test.py`
(Task 6) -- acá para `reap_orphaned_pipelines`, no para `_correr_pipeline`.

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_reaper_aviso_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from jacobs import aviso, reaper
from jacobs.models import Pipeline, PipelineStatus


class _TiendaFalsaReaper:
    """Lo mínimo que `reap_orphaned_pipelines` necesita: los candidatos ya
    filtrados, una escritura condicional que siempre gana, y un registro de
    los eventos."""

    def __init__(self, candidatos):
        self._candidatos = candidatos
        self.eventos: list[tuple[str, dict]] = []
        self.escrituras: list[tuple[str, PipelineStatus]] = []

    async def candidatos_del_reaper(self, estados):
        return self._candidatos

    async def pipeline_update_status_si_epoca(
        self, pipeline_id, epoca, status, *, desde=(), sin_avance_desde=None,
    ):
        self.escrituras.append((pipeline_id, status))
        return True

    async def event_append(self, pipeline_id, event_type, payload=None, step_id=None):
        self.eventos.append((event_type, payload or {}))


def _pipeline_pending_viejo(pid="p-viejo"):
    ahora = time.time()
    return Pipeline(
        pipeline_id=pid, name="huerfano", invoked_by="plataforma", mode="autonomous",
        status=PipelineStatus.pending,
        created_at=ahora - reaper.PENDING_MAX_AGE_SECONDS - 10,
        updated_at=ahora - reaper.PENDING_MAX_AGE_SECONDS - 10,
        run_epoch=0,
    )


def test_un_pipeline_cosechado_dispara_el_aviso_de_telegram(monkeypatch):
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    pipeline = _pipeline_pending_viejo()
    tienda = _TiendaFalsaReaper([(pipeline, 0)])
    monkeypatch.setattr(reaper, "store", tienda)

    cosechados = asyncio.run(reaper.reap_orphaned_pipelines())

    assert len(cosechados) == 1
    assert len(llamadas) == 1, "el reaper cosechó un pipeline y no avisó -- el spec §3.3 exige avisar SIEMPRE al terminar"
    assert llamadas[0] == {
        "pipeline_id": "p-viejo", "nombre": "huerfano", "estado": "expired",
    }


def test_un_pipeline_NO_cosechado_no_dispara_aviso(monkeypatch):
    """Un pipeline sano (no cumple ningún umbral) no cosecha, y por lo tanto
    tampoco avisa -- el aviso está atado a la cosecha REAL, no a que el
    barrido corra."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    ahora = time.time()
    pipeline_sano = Pipeline(
        pipeline_id="p-sano", name="sano", invoked_by="plataforma", mode="autonomous",
        status=PipelineStatus.pending, created_at=ahora, updated_at=ahora, run_epoch=0,
    )
    tienda = _TiendaFalsaReaper([(pipeline_sano, 0)])
    monkeypatch.setattr(reaper, "store", tienda)

    cosechados = asyncio.run(reaper.reap_orphaned_pipelines())

    assert cosechados == []
    assert llamadas == []


def test_la_escritura_perdida_no_dispara_aviso(monkeypatch):
    """Si otra corrida ganó la carrera (pipeline_update_status_si_epoca
    devuelve False), el reaper NO cosechó -- y por lo tanto tampoco avisa:
    quien ganó la carrera avisa por su cuenta (o ya avisó)."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    pipeline = _pipeline_pending_viejo()

    class _TiendaPerdioLaCarrera(_TiendaFalsaReaper):
        async def pipeline_update_status_si_epoca(self, pipeline_id, epoca, status, *, desde=(), sin_avance_desde=None):
            self.escrituras.append((pipeline_id, status))
            return False

    tienda = _TiendaPerdioLaCarrera([(pipeline, 0)])
    monkeypatch.setattr(reaper, "store", tienda)

    cosechados = asyncio.run(reaper.reap_orphaned_pipelines())

    assert cosechados == []
    assert llamadas == []


def test_un_aviso_que_lanza_no_impide_que_la_cosecha_se_reporte(monkeypatch):
    """El aviso corre DESPUÉS de que la cosecha ya está escrita -- si
    agendarlo explota, el barrido sigue reportando lo cosechado (mismo
    criterio que `_disparar_aviso_fin` en executor.py: el status ya escrito
    nunca se revierte por un fallo de aviso)."""
    def _explota(**kw):
        raise RuntimeError("Telegram roto")

    monkeypatch.setattr(aviso, "avisar_fin_pipeline", _explota)
    pipeline = _pipeline_pending_viejo()
    tienda = _TiendaFalsaReaper([(pipeline, 0)])
    monkeypatch.setattr(reaper, "store", tienda)

    cosechados = asyncio.run(reaper.reap_orphaned_pipelines())  # no debe lanzar

    assert len(cosechados) == 1
