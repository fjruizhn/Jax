#!/usr/bin/env python3
"""
Test de regresion — funciones puras nuevas (verificacion de correccion,
importancia, busqueda dual). Extraidas de jax.memory.db para poder testear
la logica de decision sin tocar DB ni red.

Uso: python -m pytest tests/test_memory_helpers.py -v
"""
from __future__ import annotations

from jax.memory.db import (
    _validate_importance,
    _should_skip_as_duplicate,
    _should_supersede,
    _merge_search_results,
)


# --- _validate_importance ---------------------------------------------

def test_importance_valid_range_passes_through():
    for v in (1, 2, 3, 4, 5):
        assert _validate_importance(v) == v


def test_importance_out_of_range_becomes_none():
    assert _validate_importance(0) is None
    assert _validate_importance(6) is None
    assert _validate_importance(99) is None


def test_importance_none_stays_none():
    assert _validate_importance(None) is None


def test_importance_non_int_becomes_none():
    assert _validate_importance("5") is None
    assert _validate_importance(3.5) is None


# --- _should_skip_as_duplicate ------------------------------------------

def test_duplicate_without_correction_skips():
    assert _should_skip_as_duplicate("duplicate", False) is True


def test_duplicate_with_correction_does_not_skip():
    # Una correccion con texto casi identico al viejo SI se inserta (para
    # que _should_supersede tenga un fact nuevo al que apuntar).
    assert _should_skip_as_duplicate("duplicate", True) is False


def test_non_duplicate_bands_never_skip():
    assert _should_skip_as_duplicate("correction_candidate", False) is False
    assert _should_skip_as_duplicate("unrelated", False) is False


# --- _should_supersede ---------------------------------------------------

def test_supersede_requires_all_conditions():
    assert _should_supersede(True, "correction_candidate", True, True) is True
    assert _should_supersede(True, "duplicate", True, True) is True  # banda 'duplicate' tambien vale


def test_supersede_false_without_candidate():
    assert _should_supersede(False, "correction_candidate", True, True) is False


def test_supersede_false_without_is_correction():
    assert _should_supersede(True, "correction_candidate", False, True) is False


def test_supersede_false_when_band_unrelated():
    assert _should_supersede(True, "unrelated", True, True) is False


def test_supersede_false_when_verification_rejects():
    # El caso real que motivo esto: la distancia sola dice 'correction_candidate'
    # pero el chequeo LLM extra determina que el candidato es el equivocado.
    assert _should_supersede(True, "correction_candidate", True, False) is False


# --- _merge_search_results ------------------------------------------------

def _row(content, created_at, distancia):
    return {"content": content, "role": "user", "created_at": created_at, "distancia": distancia}


def test_merge_deduplicates_keeping_lower_distance():
    a = [_row("hola", "t1", 0.5)]
    b = [_row("hola", "t1", 0.2)]  # mismo mensaje, mejor distancia en la otra busqueda
    merged = _merge_search_results(a, b, limit=5)
    assert len(merged) == 1
    assert merged[0]["distancia"] == 0.2


def test_merge_keeps_distinct_messages():
    a = [_row("hola", "t1", 0.3)]
    b = [_row("chau", "t2", 0.1)]
    merged = _merge_search_results(a, b, limit=5)
    assert len(merged) == 2
    assert merged[0]["content"] == "chau"  # ordenado por distancia ascendente


def test_merge_respects_limit():
    a = [_row(f"msg{i}", f"t{i}", i * 0.1) for i in range(5)]
    merged = _merge_search_results(a, [], limit=2)
    assert len(merged) == 2
    assert [r["content"] for r in merged] == ["msg0", "msg1"]
