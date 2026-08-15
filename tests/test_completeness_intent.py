#!/usr/bin/env python3
"""
Test de regresion — deteccion de preguntas de completeness (item #4).

Suite pytest. Verifica detect_completeness_intent() (jax.memory.db) SIN
tocar DB ni red — es una funcion pura sobre un string.

Uso: python -m pytest tests/test_completeness_intent.py -v
"""
from __future__ import annotations

from jax.memory.db import detect_completeness_intent


def test_detects_project_completeness():
    assert detect_completeness_intent("Que proyectos tenes activos?") == "project"


def test_detects_preference_completeness():
    assert detect_completeness_intent("Cuales son mis preferencias?") == "preference"


def test_detects_technical_completeness():
    assert detect_completeness_intent("Que decisiones tecnicas tomamos?") == "technical"


def test_normal_question_returns_none():
    assert detect_completeness_intent("Como se llama el gato de Fernando?") is None
