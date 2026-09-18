"""cancel y el reaper escriben con compare-and-set sobre lo que leyeron (ola
final F6, revisión final m7). Antes escribían `pipeline_update_status` sin
condición: un /continue o /resume que tomó la época entre la lectura y la
escritura quedaba pisado -- el pipeline recién continuado pasaba a
aborted/expired y su ejecutor nuevo moría. Ahora la escritura es
`UPDATE ... WHERE pipeline_id AND run_epoch AND status` con lo leído
(store.pipeline_update_status_si_epoca, EXPLAIN por PRIMARY en
tests/test_run_epoch_db.py). Sin DB.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, patch

# Base de tests de ESTA sesión: respeta JAX_TEST_DB_SUFIJO en vez de
# clavar el nombre (mismo override incondicional que antes).
from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from jacobs import reaper, routes  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402


def _pipeline(status=PipelineStatus.running, epoca=4, edad=0.0):
    ahora = time.time()
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                    status=status, run_epoch=epoca, created_at=ahora - edad, updated_at=ahora - edad)


def _cancelar(pipeline, cas):
    viejo, eventos = AsyncMock(), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(routes.store, "pipeline_update_status", viejo), \
         patch.object(routes.store, "pipeline_update_status_si_epoca", cas), \
         patch.object(routes.store, "event_append", eventos):
        try:
            return asyncio.run(routes.cancel_pipeline("p1")), viejo, eventos
        except HTTPException as exc:
            return exc, viejo, eventos


def test_cancel_escribe_con_la_epoca_y_el_status_leidos():
    cas = AsyncMock(return_value=True)
    r, viejo, eventos = _cancelar(_pipeline(PipelineStatus.interrupted, epoca=4), cas)
    assert r == {"pipeline_id": "p1", "status": "aborted"}
    cas.assert_awaited_once_with("p1", 4, PipelineStatus.aborted, desde=(PipelineStatus.interrupted,))
    viejo.assert_not_awaited()
    assert [c.args[1] for c in eventos.await_args_list] == ["PIPELINE_CANCELLED"]


def test_cancel_sobre_un_pipeline_que_cambio_da_409_sin_evento():
    r, viejo, eventos = _cancelar(_pipeline(PipelineStatus.running, epoca=4), AsyncMock(return_value=False))
    assert isinstance(r, HTTPException) and r.status_code == 409
    assert isinstance(r.detail, str) and "cambió" in r.detail
    viejo.assert_not_awaited()
    eventos.assert_not_awaited()


def _cosechar(candidatos, cas):
    viejo, eventos = AsyncMock(), AsyncMock()
    filas = [c if isinstance(c, tuple) else (c, 0) for c in candidatos]
    with patch.object(reaper.store, "candidatos_del_reaper", AsyncMock(return_value=filas)), \
         patch.object(reaper.store, "pipeline_update_status", viejo), \
         patch.object(reaper.store, "pipeline_update_status_si_epoca", cas), \
         patch.object(reaper.store, "event_append", eventos):
        cosechados = asyncio.run(reaper.reap_orphaned_pipelines())
    return cosechados, viejo, eventos


def test_reaper_cosecha_con_la_epoca_y_el_status_leidos():
    viejo_running = _pipeline(PipelineStatus.running, epoca=2, edad=reaper.RUNNING_STALE_SECONDS + 60)
    cas = AsyncMock(return_value=True)
    cosechados, viejo, eventos = _cosechar([viejo_running], cas)
    assert [c["pipeline_id"] for c in cosechados] == ["p1"]
    cas.assert_awaited_once()
    assert cas.await_args.args == ("p1", 2, PipelineStatus.expired)
    assert cas.await_args.kwargs["desde"] == (PipelineStatus.running,)
    viejo.assert_not_awaited()
    assert [c.args[1] for c in eventos.await_args_list] == ["REAPED"]


def test_reaper_no_pisa_un_pipeline_que_otro_tomo_despues_de_leerlo(caplog):
    viejo_running = _pipeline(PipelineStatus.running, epoca=2, edad=reaper.RUNNING_STALE_SECONDS + 60)
    with caplog.at_level("INFO", logger=reaper.logger.name):
        cosechados, viejo, eventos = _cosechar([viejo_running], AsyncMock(return_value=False))
    assert cosechados == []
    viejo.assert_not_awaited()
    eventos.assert_not_awaited()
    assert any("p1" in m and "cambió" in m for m in caplog.messages), caplog.messages


# Pasada final R34, 5: un running se cosecha sólo si SIGUE sin avance al
# escribir -- `updated_at < ahora - RUNNING_STALE_SECONDS` va en el UPDATE.

def test_reaper_de_running_pide_que_siga_sin_avance_al_escribir(monkeypatch):
    monkeypatch.setattr(reaper.time, "time", lambda: 1_000_000.0)
    p = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                 status=PipelineStatus.running, run_epoch=2, created_at=1.0,
                 updated_at=1_000_000.0 - reaper.RUNNING_STALE_SECONDS - 60)
    cas = AsyncMock(return_value=True)
    _cosechar([p], cas)
    assert cas.await_args.kwargs["sin_avance_desde"] == 1_000_000.0 - reaper.RUNNING_STALE_SECONDS


def test_reaper_usa_el_umbral_del_pipeline_en_el_corte(monkeypatch):
    """Ruling R36: con un paso en curso de 1800 s el umbral es 3600 s, y el
    corte del CAS usa ESE umbral."""
    monkeypatch.setattr(reaper.time, "time", lambda: 1_000_000.0)
    p = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                 status=PipelineStatus.running, run_epoch=2, created_at=1.0, updated_at=1_000_000.0 - 3700)
    cas = AsyncMock(return_value=True)
    cosechados, _, _ = _cosechar([(p, 1800)], cas)
    assert len(cosechados) == 1
    assert cas.await_args.kwargs["sin_avance_desde"] == 1_000_000.0 - 3600
    cas.reset_mock()
    p2 = p.model_copy(update={"updated_at": 1_000_000.0 - 1900})
    cosechados, _, _ = _cosechar([(p2, 1800)], cas)
    assert cosechados == [] and not cas.await_count
