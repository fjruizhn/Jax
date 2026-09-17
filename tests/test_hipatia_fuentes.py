"""
Fuentes de hipatia verificables por otro modelo (2026-09-12).

Tercera E2E de la cadena (pipeline b2d87971): hipatia citó 6 fuentes, pero
todas eran redirecciones opacas de Google grounding
(`vertexaisearch.cloud.google.com/grounding-api-redirect/...`) con solo una
etiqueta de dominio y sin texto citado. La auditoría no pudo contrastarlas.
Además las fuentes ni siquiera le llegaban: `_build_context_input` pasaba
solo `result` a los pasos siguientes; las fuentes vivían aparte y solo se
escribían en el .md del repo.

Ahora cada fuente lleva la URL FINAL (siguiendo la redirección, sin bajar el
cuerpo) y los fragmentos de la respuesta que respalda (groundingSupports).
Si la redirección no se resuelve, se conserva la de Google y se marca: nunca
se inventa una URL.

La fixture es una respuesta REAL de Gemini (2026-09-12, sin la clave).
Ningún test sale a la red: el resolvedor usa httpx.MockTransport y cualquier
head/get/post real falla fuerte.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from facet_resolver import ResolvedFacet  # noqa: E402
from jacobs import executor  # noqa: E402
from jax.core import grounding_sources as gs  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402

_FIX = json.loads((Path(__file__).parent / "fixtures" / "gemini_grounding_postgresql.json").read_text(encoding="utf-8"))
_META = _FIX["candidates"][0]["groundingMetadata"]
_REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA"
_Q1 = "La última versión estable de PostgreSQL es la 18.6"
_Q2 = "Esta actualización fue publicada oficialmente el 13 de agosto de 2026"


def _no_network():
    """Bloquea el transporte REAL de httpx. El MockTransport de los tests es
    otra clase y sigue funcionando; cualquier salida a la red falla fuerte."""
    boom = AsyncMock(side_effect=AssertionError("llamada de red real en un test"))
    return [patch("httpx.AsyncHTTPTransport.handle_async_request", boom)]


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _src(url=_REDIRECT, title="postgresql.org", quotes=None):
    return {"title": title, "url": url, "quotes": quotes or []}


class BuildSourcesTest(unittest.TestCase):
    def test_cada_fuente_lleva_las_citas_que_respalda(self):
        sources = gs.build_sources(_META["groundingChunks"], _META["groundingSupports"])
        assert [s["title"] for s in sources] == ["wikipedia.org", "postgresql.org"]
        assert sources[0]["quotes"] == [_Q1, _Q2], sources[0]
        assert sources[1]["quotes"] == [_Q2], sources[1]
        assert all(s["url"].startswith("https://vertexaisearch.cloud.google.com/") for s in sources)

    def test_una_uri_repetida_es_una_sola_fuente_con_sus_citas(self):
        chunks = [{"web": {"uri": _REDIRECT, "title": "a.org"}}, {"web": {"uri": _REDIRECT, "title": "a.org"}}]
        supports = [
            {"segment": {"text": "uno"}, "groundingChunkIndices": [0]},
            {"segment": {"text": "dos"}, "groundingChunkIndices": [1]},
        ]
        sources = gs.build_sources(chunks, supports)
        assert len(sources) == 1 and sources[0]["quotes"] == ["uno", "dos"], sources


class ResolveRedirectsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._p = _no_network()
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in self._p:
            p.stop()

    async def test_302_con_location_da_la_url_final(self):
        def handler(req):
            assert req.method == "HEAD"
            return httpx.Response(302, headers={"location": "https://www.postgresql.org/about/news/"})

        async with _client(handler) as c:
            [s] = await gs.resolve_redirects([_src()], client=c)
        assert s["final_url"] == "https://www.postgresql.org/about/news/" and s["resolved"] is True

    async def test_head_sin_location_cae_a_get(self):
        def handler(req):
            if req.method == "HEAD":
                return httpx.Response(405)
            return httpx.Response(302, headers={"location": "https://es.wikipedia.org/wiki/PostgreSQL"})

        async with _client(handler) as c:
            [s] = await gs.resolve_redirects([_src()], client=c)
        assert s["final_url"] == "https://es.wikipedia.org/wiki/PostgreSQL" and s["resolved"] is True

    async def test_sin_location_se_conserva_la_de_google_y_se_marca(self):
        async with _client(lambda req: httpx.Response(404)) as c:
            [s] = await gs.resolve_redirects([_src()], client=c)
        assert s["resolved"] is False and s["final_url"] is None and s["url"] == _REDIRECT

    async def test_error_de_red_no_inventa_una_url(self):
        def handler(req):
            raise httpx.ConnectError("sin red")

        async with _client(handler) as c:
            [s] = await gs.resolve_redirects([_src()], client=c)
        assert s["resolved"] is False and s["final_url"] is None

    async def test_una_url_directa_no_se_consulta(self):
        def handler(req):
            raise AssertionError("no debía consultar una URL que no es redirección")

        async with _client(handler) as c:
            [s] = await gs.resolve_redirects([_src(url="https://www.postgresql.org/")], client=c)
        assert s["final_url"] == "https://www.postgresql.org/" and s["resolved"] is True

    async def test_las_fuentes_se_resuelven_en_paralelo(self):
        async def handler(req):
            await asyncio.sleep(0.3)
            return httpx.Response(302, headers={"location": f"https://x.org/{req.url.path[-1]}"})

        sources = [_src(url=f"{_REDIRECT}{i}") for i in range(3)]
        async with _client(handler) as c:
            t0 = time.monotonic()
            out = await gs.resolve_redirects(sources, client=c)
            elapsed = time.monotonic() - t0
        assert all(s["resolved"] for s in out)
        assert elapsed < 0.6, f"3 x 0.3 s tardó {elapsed:.2f} s: no fue concurrente"


class RenderTest(unittest.TestCase):
    def test_el_bloque_lleva_url_final_y_cita(self):
        block = gs.render_sources_block([{**_src(quotes=[_Q2]), "final_url": "https://www.postgresql.org/about/news/", "resolved": True}])
        assert "https://www.postgresql.org/about/news/" in block
        assert _Q2 in block
        assert _REDIRECT not in block, "si hay URL final, la opaca sobra"

    def test_lo_no_resuelto_se_marca(self):
        block = gs.render_sources_block([{**_src(), "final_url": None, "resolved": False}])
        assert "no resuelta" in block.lower() and _REDIRECT in block


class IntegracionExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_http_gemini_devuelve_fuentes_verificables(self):
        class _Resp:
            status_code = 200
            text = ""

            def json(self):
                return _FIX

        async def fake_post(client_self, url, **kw):
            return _Resp()

        async def fake_resolve(sources, client=None):
            for s in sources:
                s["final_url"], s["resolved"] = f"https://final.org/{s['title']}", True
            return sources

        f = ResolvedFacet(key="hipatia", provider_id="gemini", base_url="https://g/v1beta",
                          model="gemini-3.8-flash", credential="k", transport="http_gemini",
                          persona=None, params=None)
        with patch("httpx.AsyncClient.post", fake_post), \
             patch.object(executor, "resolve_redirects", fake_resolve), \
             patch.object(executor, "record_resolved_version_safe", AsyncMock()):
            out = await executor._invoke_http_gemini(f, "pregunta", timeout=5)

        assert out["sources"][1]["final_url"] == "https://final.org/postgresql.org"
        assert out["sources"][1]["quotes"] == [_Q2]

    def test_el_paso_siguiente_recibe_las_fuentes(self):
        """Antes solo viajaba `result`: la auditoría no veía ninguna fuente."""
        dep = {"result": "texto de hipatia", "sources": [
            {**_src(quotes=[_Q2]), "final_url": "https://www.postgresql.org/about/news/", "resolved": True}]}
        plan = [Step(facet="hipatia", capability="research", step_index=0),
                Step(facet="thot", capability="validate_consistency", step_index=1, depends_on=[0])]
        pipeline = Pipeline(name="t", invoked_by="t", user_id="1", tenant_id="1", mode="dry_run",
                            plan=plan, context={"step_0_ref": "inline:" + json.dumps(dep)})
        ctx = executor._build_context_input(plan[1], pipeline)
        summary = ctx["previous_outputs"][0]["summary"]
        assert "https://www.postgresql.org/about/news/" in summary and _Q2 in summary, summary

    async def test_el_md_del_repo_muestra_url_final_y_cita(self):
        with tempfile.TemporaryDirectory() as base, \
                patch.object(executor, "REPO_DOCUMENTS_DIR", Path(base) / "documents"):
            await executor._persist_step_to_repo(
                pipeline_id="abcdef12-x", pipeline_name="t", step_index=0, facet="hipatia",
                capability="research",
                raw_output={"success": True, "result": "texto", "model": "m", "sources": [
                    {**_src(quotes=[_Q2]), "final_url": "https://www.postgresql.org/about/news/", "resolved": True}]},
            )
            md = (Path(base) / "documents" / "abcdef12_00_hipatia.md").read_text(encoding="utf-8")
        assert "https://www.postgresql.org/about/news/" in md and _Q2 in md, md


if __name__ == "__main__":
    unittest.main(verbosity=2)
