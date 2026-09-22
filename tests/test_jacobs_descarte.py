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
