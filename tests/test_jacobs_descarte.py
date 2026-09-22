"""Descartar pipelines detenidos (spec 2026-09-22-descartar-pipelines). Sin DB."""
from __future__ import annotations

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

import inspect  # noqa: E402

import pytest  # noqa: E402

from jacobs import descarte  # noqa: E402
from jacobs import policy  # noqa: E402
from jacobs import reaper  # noqa: E402
from jacobs.models import PipelineStatus  # noqa: E402


def test_los_estados_nuevos_existen():
    assert PipelineStatus("discarded") is PipelineStatus.discarded
    assert PipelineStatus("hidden") is PipelineStatus.hidden


def test_descartado_y_oculto_no_ocupan_cupo():
    assert PipelineStatus.discarded in policy.ESTADOS_SIN_CUPO
    assert PipelineStatus.hidden in policy.ESTADOS_SIN_CUPO
    assert PipelineStatus.discarded not in policy.ESTADOS_QUE_OCUPAN_CUPO
    assert PipelineStatus.hidden not in policy.ESTADOS_QUE_OCUPAN_CUPO


def test_el_reaper_solo_cosecha_no_terminales():
    """Baranda contra regresiones (spec §3): el conjunto no-terminal del
    reaper (`pending/running/interrupted`) no cambia con los estados nuevos."""
    fuente = inspect.getsource(reaper)
    assert "[PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted]" in fuente
    assert "PipelineStatus.discarded" not in fuente
    assert "PipelineStatus.hidden" not in fuente


# Task 2 (2026-09-22-descartar-pipelines): reglas puras de transición, sin I/O
# (jacobs/descarte.py). La escritura compare-and-set vive en
# store.pipeline_transicion_descarte (tests/test_jacobs_descarte_db.py).

def test_transiciones_permitidas():
    assert descarte.TRANSICIONES["discard"] == frozenset({PipelineStatus.aborted, PipelineStatus.expired})
    assert descarte.TRANSICIONES["recover"] == frozenset({PipelineStatus.discarded})
    assert descarte.TRANSICIONES["hide"] == frozenset({PipelineStatus.discarded})
    assert descarte.TRANSICIONES["restore"] == frozenset({PipelineStatus.hidden})


def test_recuperar_vuelve_al_estado_exacto_previo():
    assert descarte.destino_de("recover", "expired") is PipelineStatus.expired
    assert descarte.destino_de("recover", "aborted") is PipelineStatus.aborted


def test_recuperar_sin_estado_previo_valido_es_error():
    for malo in (None, "", "running", "discarded"):
        with pytest.raises(descarte.EstadoPrevioInvalido):
            descarte.destino_de("recover", malo)


def test_destinos_fijos():
    assert descarte.destino_de("discard", None) is PipelineStatus.discarded
    assert descarte.destino_de("hide", None) is PipelineStatus.hidden
    assert descarte.destino_de("restore", None) is PipelineStatus.discarded


# Fix round 1 (2026-09-22, Ruling 7, I-1): `validar_transicion` es la regla
# pura que `store.pipeline_transicion_descarte` tiene que consultar ANTES de
# tocar la base -- sin esto, un `discard` desde `running` liberaba el cupo de
# un pipeline que sigue corriendo y lo dejaba huérfano para siempre.

def test_validar_transicion_rechaza_desde_no_permitido():
    with pytest.raises(descarte.TransicionDescarteInvalida):
        descarte.validar_transicion("discard", PipelineStatus.running, PipelineStatus.discarded)


def test_validar_transicion_hide_rechaza_desde_distinto_de_discarded():
    with pytest.raises(descarte.TransicionDescarteInvalida):
        descarte.validar_transicion("hide", PipelineStatus.aborted, PipelineStatus.hidden)


def test_validar_transicion_discard_hide_restore_exigen_el_destino_fijo():
    with pytest.raises(descarte.TransicionDescarteInvalida):
        descarte.validar_transicion("discard", PipelineStatus.aborted, PipelineStatus.hidden)
    with pytest.raises(descarte.TransicionDescarteInvalida):
        descarte.validar_transicion("hide", PipelineStatus.discarded, PipelineStatus.discarded)
    with pytest.raises(descarte.TransicionDescarteInvalida):
        descarte.validar_transicion("restore", PipelineStatus.hidden, PipelineStatus.hidden)


def test_validar_transicion_recover_exige_a_en_los_previos_validos():
    with pytest.raises(descarte.TransicionDescarteInvalida):
        descarte.validar_transicion("recover", PipelineStatus.discarded, PipelineStatus.running)


def test_validar_transicion_las_transiciones_validas_no_levantan():
    descarte.validar_transicion("discard", PipelineStatus.aborted, PipelineStatus.discarded)
    descarte.validar_transicion("discard", PipelineStatus.expired, PipelineStatus.discarded)
    descarte.validar_transicion("recover", PipelineStatus.discarded, PipelineStatus.aborted)
    descarte.validar_transicion("recover", PipelineStatus.discarded, PipelineStatus.expired)
    descarte.validar_transicion("hide", PipelineStatus.discarded, PipelineStatus.hidden)
    descarte.validar_transicion("restore", PipelineStatus.hidden, PipelineStatus.discarded)


# ------------------------------------------------------------------
# Task 3 (2026-09-22-descartar-pipelines): rutas de Jacobs, eventos, y el
# rechazo de discarded/hidden en cancel y continue.
#
# Ruling 1 del controlador (2026-09-22): CUATRO rutas LITERALES
# (`/discard`, `/recover`, `/hide`, `/restore`) que llaman a la función
# común `routes.transicion_descarte`, no la ruta genérica
# `/pipeline/{id}/{accion}` que proponía el brief -- elimina de raíz el
# riesgo de orden de rutas de FastAPI (que ninguna ruta nueva futura
# pueda "colarse" antes de una ruta literal ya existente) en vez de
# apoyarse en que el código quede escrito en el orden correcto para
# siempre. La función común conserva el nombre y la firma que pide el
# brief (`transicion_descarte(pipeline_id, accion, req)`), así que los
# tests que la llaman directo no cambian.
#
# Ruling 8 del controlador: la ruta sólo atrapa
# `descarte.EstadoPrevioInvalido` -> 422. `TransicionDescarteInvalida` y
# un `ValueError` genérico NO se atrapan: con la ruta bien escrita (ya
# valida el estado y calcula `a` con `destino_de`) son inalcanzables: si
# aparecen de todos modos es un bug de contrato -> 500, no un 4xx que
# disfrazaría el error de una petición mal formada.
# ------------------------------------------------------------------

import asyncio  # noqa: E402
import time  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from jacobs import routes  # noqa: E402
from jacobs.models import Pipeline  # noqa: E402


def _p(status: PipelineStatus) -> Pipeline:
    ahora = time.time()
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                     status=status, run_epoch=3, created_at=ahora, updated_at=ahora)


def _llamar(accion, status, cas=True, previo=None):
    """Llama a la función común directo (sin HTTP), con el store simulado.
    HTTPException se captura y se devuelve como resultado -- mismo patrón
    que `_cancelar` en tests/test_jacobs_cancel_reaper_epoca.py -- para que
    el llamador pueda seguir inspeccionando los mocks después."""
    pipeline = _p(status)
    trans, eventos = AsyncMock(return_value=cas), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(routes.store, "pipeline_status_previo", AsyncMock(return_value=previo)), \
         patch.object(routes.store, "pipeline_transicion_descarte", trans), \
         patch.object(routes.store, "event_append", eventos):
        try:
            r = asyncio.run(
                routes.transicion_descarte("p1", accion, routes.DescarteRequest(user_id="u1")))
        except HTTPException as exc:
            return exc, trans, eventos
    return r, trans, eventos


def test_descartar_un_abortado():
    r, trans, eventos = _llamar("discard", PipelineStatus.aborted)
    assert r == {"pipeline_id": "p1", "status": "discarded"}
    trans.assert_awaited_once_with("p1", 3, "discard", desde=PipelineStatus.aborted,
                                   a=PipelineStatus.discarded, user_id="u1")
    eventos.assert_awaited_once_with("p1", "PIPELINE_DISCARDED",
                                     {"user_id": "u1", "desde": "aborted", "a": "discarded"})


@pytest.mark.parametrize("accion,status", [
    ("discard", PipelineStatus.running), ("discard", PipelineStatus.completed),
    ("recover", PipelineStatus.aborted), ("hide", PipelineStatus.aborted),
    ("restore", PipelineStatus.discarded),
])
def test_transicion_no_permitida_es_409(accion, status):
    r, _, eventos = _llamar(accion, status)
    assert isinstance(r, HTTPException)
    assert r.status_code == 409
    assert r.detail["code"] == "transicion_no_permitida"
    eventos.assert_not_awaited()


def test_carrera_perdida_es_409_y_no_emite_evento():
    r, trans, eventos = _llamar("discard", PipelineStatus.aborted, cas=False)
    assert isinstance(r, HTTPException)
    assert r.status_code == 409
    assert r.detail["code"] == "cambio_concurrente"
    trans.assert_awaited_once()
    # Step 4 del brief: si el evento se emitiera ANTES de saber si el CAS
    # escribió, una carrera perdida dejaría un PIPELINE_DISCARDED mintiendo
    # sobre una transición que nunca ocurrió. Esta aserción es la que hace
    # caer esa mutación.
    eventos.assert_not_awaited()


def test_recuperar_vuelve_a_expired():
    r, trans, eventos = _llamar("recover", PipelineStatus.discarded, previo="expired")
    assert r == {"pipeline_id": "p1", "status": "expired"}
    assert trans.await_args.kwargs["a"] is PipelineStatus.expired
    eventos.assert_awaited_once_with("p1", "PIPELINE_RECOVERED",
                                     {"user_id": "u1", "desde": "discarded", "a": "expired"})


def test_recuperar_con_previo_corrupto_es_422():
    r, trans, eventos = _llamar("recover", PipelineStatus.discarded, previo="running")
    assert isinstance(r, HTTPException)
    assert r.status_code == 422
    assert r.detail["code"] == "estado_previo_invalido"
    trans.assert_not_awaited()
    eventos.assert_not_awaited()


@pytest.mark.parametrize("status", [PipelineStatus.discarded, PipelineStatus.hidden])
def test_cancel_rechaza_descartados_y_ocultos(status):
    pipeline = _p(status)
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)):
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.cancel_pipeline("p1"))
    assert e.value.status_code == 409


def test_continue_no_acepta_descartados():
    from jacobs.continuar import ESTADOS_CONTINUABLES
    assert PipelineStatus.discarded not in ESTADOS_CONTINUABLES
    assert PipelineStatus.hidden not in ESTADOS_CONTINUABLES


# --- Ruling 1: cuatro rutas literales, evidencia por TestClient ---------

def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router)
    return app


@pytest.mark.parametrize("accion,desde,destino", [
    ("discard", PipelineStatus.aborted, "discarded"),
    ("recover", PipelineStatus.discarded, "aborted"),
    ("hide", PipelineStatus.discarded, "hidden"),
    ("restore", PipelineStatus.hidden, "discarded"),
])
def test_cada_ruta_literal_llega_a_su_propio_handler(accion, desde, destino):
    pipeline = _p(desde)
    trans, eventos = AsyncMock(return_value=True), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(routes.store, "pipeline_status_previo", AsyncMock(return_value="aborted")), \
         patch.object(routes.store, "pipeline_transicion_descarte", trans), \
         patch.object(routes.store, "event_append", eventos):
        with TestClient(_app()) as c:
            r = c.post(f"/jacobs/pipeline/p1/{accion}", json={"user_id": "u1"})
    assert r.status_code == 200, r.text
    assert r.json() == {"pipeline_id": "p1", "status": destino}
    trans.assert_awaited_once()
    assert trans.await_args.args[2] == accion


def test_post_cancel_sigue_llegando_a_cancel_pipeline_no_a_transicion_descarte():
    pipeline = _p(PipelineStatus.running)
    cas, eventos = AsyncMock(return_value=True), AsyncMock()
    trans_descarte = AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(routes.store, "pipeline_update_status_si_epoca", cas), \
         patch.object(routes.store, "event_append", eventos), \
         patch.object(routes.store, "pipeline_transicion_descarte", trans_descarte):
        with TestClient(_app()) as c:
            r = c.post("/jacobs/pipeline/p1/cancel")
    assert r.status_code == 200, r.text
    assert r.json() == {"pipeline_id": "p1", "status": "aborted"}
    cas.assert_awaited_once()
    trans_descarte.assert_not_awaited()


# --- Step 5: autenticación de LAS MANOS ----------------------------------

def test_plataforma_puede_llamar_las_rutas_del_descarte():
    import auth_servicio
    permiso = auth_servicio.PERMISOS[auth_servicio.IDENTIDAD_PLATAFORMA]
    for accion in ("discard", "recover", "hide", "restore"):
        assert permiso.admite_ruta("POST", f"/jacobs/pipeline/p1/{accion}")


def test_jacobs_no_puede_llamar_las_rutas_del_descarte():
    """La identidad `jacobs` (el propio proceso de LAS MANOS, que despacha
    motores y sub-pipelines) sólo admite `POST /jacobs/pipeline` EXACTO
    (crear un pipeline), por `fullmatch` -- ese patrón NO matchea
    `/jacobs/pipeline/p1/discard` ni las otras tres. Sin esto, el proceso
    que corre los pasos podría descartar/recuperar/ocultar/restaurar
    pipelines sin pasar por jax-platform, que es quien valida el ROL de
    quien pide la transición (spec §4)."""
    import auth_servicio
    permiso = auth_servicio.PERMISOS[auth_servicio.IDENTIDAD_JACOBS]
    for accion in ("discard", "recover", "hide", "restore"):
        assert not permiso.admite_ruta("POST", f"/jacobs/pipeline/p1/{accion}")
