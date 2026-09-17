"""
REPL — las fuentes de Gemini del HttpMuscle son verificables, igual que en
Jacobs (jax#139, 2026-09-12).

El HttpMuscle (`jax/muscles/base.py`) armaba sus fuentes con las
redirecciones opacas de Google grounding (`vertexaisearch…/grounding-api-
redirect/…`) y una etiqueta de dominio, sin citas: nadie podía contrastarlas.
Jacobs ya las resuelve; acá se reusa el MISMO módulo, que pasa a vivir en
`jax/core/grounding_sources.py` (capa base: el REPL no puede depender de
`jacobs/`, y el proceso de LAS MANOS no puede importar `jax.*` -- le llega por
symlink en `las_manos/`, igual que `facet_resolver`).

Sin red: se parchea el POST a Gemini (con una respuesta real guardada como
fixture) y el HEAD de las redirecciones; el transporte real de httpx falla
fuerte si algo intenta salir.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

_ROOT = Path(__file__).resolve().parents[1]
_FIX = json.loads((_ROOT / "tests" / "fixtures" / "gemini_grounding_postgresql.json").read_text(encoding="utf-8"))
_Q1 = "La última versión estable de PostgreSQL es la 18.6"


class ModuloCompartidoTest(unittest.TestCase):
    def test_vive_en_jax_core(self):
        from jax.core import grounding_sources as gs
        for name in ("build_sources", "resolve_redirects", "render_sources_block"):
            assert callable(getattr(gs, name, None)), name

    def test_las_manos_lo_ve_por_symlink_al_mismo_archivo(self):
        link = _ROOT / "las_manos" / "grounding_sources.py"
        assert link.is_symlink(), "las_manos/grounding_sources.py debe ser symlink"
        assert link.resolve() == (_ROOT / "jax" / "core" / "grounding_sources.py").resolve()
        assert not (_ROOT / "jacobs" / "grounding_sources.py").exists(), "no puede quedar una copia en jacobs/"

    def test_jacobs_usa_ese_mismo_archivo(self):
        from jacobs import executor
        origen = Path(executor.build_sources.__code__.co_filename).resolve()
        assert origen == (_ROOT / "jax" / "core" / "grounding_sources.py").resolve(), origen


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class HttpMuscleGeminiTest(unittest.IsolatedAsyncioTestCase):
    async def test_fuentes_con_url_final_y_citas(self):
        from jax.muscles import base

        muscle = base.HttpMuscle(
            name="hipatia", provider="gemini", model_default="gemini-x",
            models_allowed=["gemini-x"], system_prompt="s", timeout=10,
            grounding_policy="required_web",
            api_url="https://generativelanguage.googleapis.com/v1beta",
        )
        destinos = {}

        async def fake_post(client_self, url, json=None, **kw):
            return _Resp(_FIX)

        async def fake_head(client_self, url, **kw):
            final = f"https://fuente-real.example/{len(destinos) + 1}"
            destinos[url] = final
            return httpx.Response(302, headers={"location": final}, request=httpx.Request("HEAD", url))

        boom = AsyncMock(side_effect=AssertionError("llamada de red real en un test"))
        with patch("httpx.AsyncHTTPTransport.handle_async_request", boom), \
             patch("httpx.AsyncClient.post", fake_post), \
             patch("httpx.AsyncClient.head", fake_head), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value="k")), \
             patch.object(base, "record_resolved_version_safe", AsyncMock()):
            out = await muscle._call("¿versión de PostgreSQL?", "gemini-x")

        assert "— Fuentes consultadas —" in out
        assert "grounding-api-redirect" not in out, "quedó una redirección opaca"
        assert len(destinos) == 2, destinos
        for final in destinos.values():
            assert final in out, (final, out)
        assert f'> "{_Q1}' in out, "falta la cita que respalda la fuente"
        assert "(2 fuentes)" in out, out


if __name__ == "__main__":
    unittest.main(verbosity=2)
