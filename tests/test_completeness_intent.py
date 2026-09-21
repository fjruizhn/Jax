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


def test_detects_user_completeness():
    assert detect_completeness_intent("Que sabes de mi?") == "user"


def test_detects_social_completeness():
    assert detect_completeness_intent("Quienes son mis socios?") == "social"


def test_detects_financial_completeness():
    assert detect_completeness_intent("Que sabes de mis finanzas?") == "financial"


def test_normal_question_returns_none():
    assert detect_completeness_intent("Como se llama el gato de Fernando?") is None


# ---------------------------------------------------------------------------
# Segunda ronda (2026-09-20, decision de Fernando tras el fallo real):
# "jax sabes a que me dedico?" no matcheaba ninguna categoria -- el patron
# 'user' solo cubria "que sabes de mi" en sus tres formas literales, no las
# formas naturales de preguntar por la OCUPACION/PROFESION de Fernando ni sus
# equivalentes en ingles (la app es bilingue y este detector es compartido
# entre el REPL y jax-platform). Los 4 tests de abajo son el caso real y sus
# variantes; vistos en ROJO contra el _COMPLETENESS_PATTERNS viejo antes de
# ampliarlo (ver jax/memory/db.py).
# ---------------------------------------------------------------------------

def test_el_caso_real_del_bug_jax_sabes_a_que_me_dedico():
    assert detect_completeness_intent("jax sabes a que me dedico?") == "user"


def test_a_que_me_dedico_sola_sin_el_jax_del_frente():
    assert detect_completeness_intent("a que me dedico?") == "user"


def test_dame_todo_lo_que_sepas_de_mi_trabajo():
    assert detect_completeness_intent("dame todo lo que sepas de mi trabajo") == "user"


def test_de_que_trabajo_forma_natural():
    assert detect_completeness_intent("de que trabajo?") == "user"


def test_cual_es_mi_profesion():
    assert detect_completeness_intent("cual es mi profesion?") == "user"


def test_quien_soy():
    assert detect_completeness_intent("quien soy?") == "user"


# --- Acentos y voseo: el texto real no siempre llega sin tildes -----------
# El caso real de Fernando ("a que me dedico") no tenia tilde en "que", pero
# un usuario que SI acentua bien ("¿A qué me dedico?") tiene que detectar
# igual -- normalizar solo minusculas (como hacia el detector antes) no
# alcanza. La normalizacion de acentos es deliberadamente LEXICA (quitar
# tildes de las 5 vocales), no un clasificador nuevo.

def test_a_que_me_dedico_con_tilde():
    assert detect_completeness_intent("¿A qué me dedico?") == "user"


def test_quien_soy_con_tilde():
    assert detect_completeness_intent("¿Quién soy?") == "user"


def test_voseo_que_sabes_de_mi_con_tilde():
    # "sabés" (voseo, con tilde) tiene que caer en el mismo patron que
    # "sabes" (tu) una vez normalizados los acentos.
    assert detect_completeness_intent("¿qué sabés de mí?") == "user"


# --- Ingles: el detector es compartido entre las dos apps bilingues -------

def test_ingles_whats_my_job():
    assert detect_completeness_intent("hey jax, what's my job?") == "user"


def test_ingles_who_am_i():
    assert detect_completeness_intent("who am i?") == "user"


def test_ingles_what_do_you_know_about_me():
    assert detect_completeness_intent("what do you know about me?") == "user"


def test_ingles_what_is_my_profession():
    assert detect_completeness_intent("what is my profession?") == "user"


# --- Falsos positivos: una pregunta sobre un TERCERO o un PRODUCTO no es --
# --- una pregunta sobre Fernando -------------------------------------------

def test_falso_positivo_a_que_se_dedica_un_producto():
    """El caso adversarial que da Fernando: 'se dedica' (tercera persona) no
    puede matchear el mismo patron que 'me dedico' (primera persona)."""
    assert detect_completeness_intent("¿A qué se dedica AteneaERP?") is None


def test_falso_positivo_en_que_trabaja_un_tercero():
    assert detect_completeness_intent("¿en que trabaja Maria?") is None


def test_falso_positivo_profesion_de_un_tercero():
    assert detect_completeness_intent(
        "cual es la profesion de mi socio?") is None


def test_falso_positivo_what_does_a_product_do():
    assert detect_completeness_intent("what does AteneaERP do?") is None
