"""E-15 (2026-09-16): el filtro de autoetiquetas del modelo usa la cabecera de
la etiqueta CONFIGURADA (config.toml, authority_origin), no un literal.

Antes, las dos copias del filtro buscaban "⚙️ *Origen" (la etiqueta de kimi)
escrito en el código: si DeepSeek imitaba SU etiqueta (🧠), la línea falsa
llegaba al usuario pegada a la verdadera que agrega el sistema.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from jax.muscles import base  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
KIMI = "⚙️ *Origen de autoridad: Kimi K2.7 Code (Moonshot AI). Sin verificación externa.*"
PROPIO = "🧠 *Origen de autoridad: conocimiento propio del modelo (no verificado en web).*"


def test_quita_la_autoetiqueta_de_kimi():
    assert base._sin_autoetiqueta(f"hola\n{KIMI}\nchau", KIMI) == "hola\nchau"


def test_quita_la_autoetiqueta_imitada_de_otra_faceta():
    assert base._sin_autoetiqueta("respuesta\n  🧠 *Origen de autoridad: inventada*", PROPIO) == "respuesta"


def test_sin_etiqueta_configurada_no_quita_nada():
    assert base._sin_autoetiqueta(f"  a\n{KIMI}  ", "") == f"a\n{KIMI}"


def test_el_prefijo_de_kimi_ya_no_esta_escrito_en_el_codigo():
    assert "⚙️ *Origen" not in (RAIZ / "jax" / "muscles" / "base.py").read_text(encoding="utf-8")


class DeepseekLimpiaSuPropiaEtiquetaTest(unittest.IsolatedAsyncioTestCase):
    async def test_la_respuesta_sale_sin_la_etiqueta_imitada(self):
        musculo = base.HttpMuscle(
            name="jekyll", provider="deepseek", model_default="d", models_allowed=["d"], system_prompt="s",
            timeout=10, authority_origin=PROPIO, api_url="https://api.deepseek.example/chat/completions")
        cuerpo = {"model": "d", "choices": [{"message": {"content": f"respuesta\n{PROPIO}"}}]}

        async def handle(transport_self, request):
            return httpx.Response(200, json=cuerpo)

        with patch("httpx.AsyncHTTPTransport.handle_async_request", handle), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value="k")), \
             patch.object(base.HttpMuscle, "_limite_de_salida", AsyncMock(return_value={})), \
             patch.object(base, "record_resolved_version_safe", AsyncMock()):
            salida = await musculo._call_deepseek("hola", "d")
        self.assertEqual(salida, "respuesta")
