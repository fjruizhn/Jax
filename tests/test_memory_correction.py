#!/usr/bin/env python3
"""
Test de regresion — bandas de distancia para correccion/dedup de facts.

Suite pytest. Verifica classify_fact_distance() (jax.memory.db) SIN tocar
DB ni red — es una funcion pura sobre un float.

Uso: python -m pytest tests/test_memory_correction.py -v
"""
from __future__ import annotations

from jax.memory.db import (
    classify_fact_distance,
    DUP_DISTANCE_THRESHOLD,
    CORRECTION_DISTANCE_THRESHOLD,
)


def test_duplicate_band():
    assert classify_fact_distance(0.0) == "duplicate"
    assert classify_fact_distance(0.049) == "duplicate"


def test_correction_band():
    assert classify_fact_distance(DUP_DISTANCE_THRESHOLD) == "correction_candidate"
    assert classify_fact_distance(0.24) == "correction_candidate"


def test_unrelated_band():
    assert classify_fact_distance(CORRECTION_DISTANCE_THRESHOLD) == "unrelated"
    assert classify_fact_distance(0.9) == "unrelated"
