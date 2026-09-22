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
