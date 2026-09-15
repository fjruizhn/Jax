#!/usr/bin/env python3
"""Ruling T6-6 (2026-09-15): `redactar_secretos` -- funcion pura, sin I/O.

La key de Gemini viajaba en `?key=` en los dos caminos de hipatia
(jacobs/executor.py::_invoke_http_gemini y jax/muscles/base.py::_call_gemini)
y el texto de error del proveedor se guardaba en jacobs_steps.error y en
jacobs_events sin redactar. Mismas reglas que jax-platform/backend/redaccion.py
(la rama feat/pendientes-2026-09-22). Todas las keys de este archivo son FALSAS.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_redaccion.py
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from jax.core.redaccion import recortar_redactado, redactar_secretos, texto_de_error

_ROOT = Path(__file__).resolve().parents[1]
KEY = "AIzaFAKE-t66-0123456789abcdef"


def test_las_manos_lo_ve_por_symlink_al_mismo_archivo():
    """jacobs importa plano (`from redaccion import ...`) con las_manos/ en el
    path, como grounding_sources: una copia seria una segunda fuente de verdad."""
    link = _ROOT / "las_manos" / "redaccion.py"
    assert link.is_symlink(), "las_manos/redaccion.py debe ser symlink"
    assert link.resolve() == (_ROOT / "jax" / "core" / "redaccion.py").resolve()


@pytest.mark.parametrize("texto, esperado", [
    ('api_key="sk-FAKE-comillas-dobles" x', 'api_key="***" x'),
    ("api_key='sk-FAKE-comillas-simples' x", "api_key='***' x"),
    ('{"api_key": "sk-FAKE-json", "n": 1}', '{"api_key": "***", "n": 1}'),
    ("token = tok-FAKE-espacios fin", "token = *** fin"),
    ("Authorization: Bearer x", "Authorization: Bearer ***"),
    ("Authorization: Bearer tok-FAKE.abc_123/xyz= fin", "Authorization: Bearer *** fin"),
    ("password=hunter2-FAKE&u=1", "password=***&u=1"),
    ("secret=s3cr3t-FAKE fin", "secret=*** fin"),
    ("x-goog-api-key: AIzaFAKE-cabecera-0123456789", "x-goog-api-key: ***"),
    ('{"key": "valor-FAKE"}', '{"key": "***"}'),
    # Re-revision de la plataforma (2026-09-15): esquemas de auth SOLO tras una
    # cabecera Authorization o en un contexto authorization=; mas nombres.
    ("Authorization: Token abc", "Authorization: Token ***"),
    ("Authorization: Basic dXNlcjpGQUtF fin", "Authorization: Basic *** fin"),
    ("authorization=abc&x=1", "authorization=***&x=1"),
    ('{"Authorization": "Bearer tok-FAKE"}', '{"Authorization": "Bearer ***"}'),
    ("credential=abc", "credential=***"),
    ("credentials: abc fin", "credentials: *** fin"),
    ("private_key_id: abc", "private_key_id: ***"),
    ('{"private_key_id": "abc123-FAKE"}', '{"private_key_id": "***"}'),
])
def test_formas_de_secreto_que_se_redactan(texto, esperado):
    assert redactar_secretos(texto) == esperado


@pytest.mark.parametrize("texto", [
    "monkey=5 y turkey=3",
    "turkey=pavo",
    "Duplicate entry 'x' for key 'PRIMARY'",
    # Prosa: Basic/Bearer/Token no son un esquema fuera de Authorization.
    "the basic idea of it",
    "Basic idea: keep it simple",
    "a bearer bond and a token gesture",
])
def test_nombres_que_no_son_secretos_no_se_tocan(texto):
    assert redactar_secretos(texto) == texto


def test_query_key_se_redacta_y_el_resto_de_la_url_queda():
    out = redactar_secretos(f"for url 'https://g.example/v1beta/models?key={KEY}&x=1'")
    assert KEY not in out
    assert "?key=***&x=1" in out


def test_una_key_AIza_suelta_en_el_texto_se_redacta():
    assert redactar_secretos(f"la credencial {KEY} fue rechazada") == "la credencial *** fue rechazada"


def test_los_secretos_conocidos_se_redactan_aunque_no_tengan_forma():
    out = redactar_secretos("Bearer sk-FAKE-sin-forma rechazado",
                            secretos=["sk-FAKE-sin-forma", "", None])
    assert out == "Bearer *** rechazado"


def test_un_secreto_conocido_suelto_sin_nombre_ni_forma():
    assert redactar_secretos("eco: zz-FAKE-crudo fin", ["zz-FAKE-crudo"]) == "eco: *** fin"


def test_None_y_vacio_pasan_tal_cual():
    assert redactar_secretos(None) is None
    assert redactar_secretos("") == ""


def test_un_texto_sin_secretos_no_cambia():
    texto = "HTTPStatusError: 502 Bad Gateway en https://h.example/v1/chat?modelo=x&n=1"
    assert redactar_secretos(texto) == texto


def test_texto_de_error_de_un_HTTPStatusError_real_no_trae_la_key():
    """httpx incluye la URL con la query en str(HTTPStatusError)."""
    req = httpx.Request("POST", f"https://generativelanguage.googleapis.com/v1beta/models/m:generateContent?key={KEY}")
    resp = httpx.Response(400, request=req)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        assert KEY in str(e)          # el control: sin redactar, la key esta
        out = texto_de_error(e)
    assert KEY not in out
    assert out.startswith("HTTPStatusError: ")
    assert "key=***" in out


# --- Redactar primero, recortar despues ---------------------------------------

def test_una_key_AIza_que_cruza_el_corte_no_deja_un_pedazo():
    """Recortar primero deja `AIzaFAKE-t` (6 caracteres tras AIza): la regex de
    forma exige 10 y ya no lo reconoce -- el pedazo quedaria en claro."""
    texto = "x" * 190 + KEY + " cola"
    assert KEY[:10] in texto[:200]                       # el control: el corte cae dentro
    assert redactar_secretos(texto[:200]).endswith(KEY[:10])  # recortar primero filtra
    out = recortar_redactado(texto, 200)
    assert "AIza" not in out
    assert out == "x" * 190 + "*** cola"


def test_un_secreto_conocido_que_cruza_el_corte_no_deja_un_pedazo():
    secreto = "zz-FAKE-conocido-abcdef"
    texto = "y" * 195 + secreto
    out = recortar_redactado(texto, 200, [secreto])
    assert "zz-FA" not in out
    assert out == "y" * 195 + "***"


def test_recortar_redactado_respeta_el_limite():
    out = recortar_redactado("a" * 500, 200)
    assert out == "a" * 200


def test_recortar_redactado_None_pasa():
    assert recortar_redactado(None, 200) is None
