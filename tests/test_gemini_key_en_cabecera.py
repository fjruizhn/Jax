#!/usr/bin/env python3
"""Ruling T6-6 (2026-09-15): la key de Gemini viaja en la cabecera
`x-goog-api-key`, nunca en la URL, en los DOS caminos de hipatia:
  - jacobs/executor.py::_invoke_http_gemini (pipelines de Jacobs)
  - jax/muscles/base.py::_call_gemini       (REPL)

Por que: httpx loguea en INFO `HTTP Request: POST <url>` en cada pedido, y
str(httpx.HTTPStatusError) incluye la URL entera. Con `?key=` la key quedaba
en el log y en el texto de error guardado (jacobs_steps.error y
jacobs_events). Ademas, todo texto de error del proveedor se redacta ANTES
de recortarlo y de guardarlo.

Ningun test sale a la red: el transporte REAL de httpx se reemplaza por un
manejador al estilo MockTransport que ve el httpx.Request que se construyo de
verdad (URL y cabeceras reales, no los argumentos de un post parcheado).
Todas las keys son FALSAS.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_gemini_key_en_cabecera.py
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado -- unset antes de correr este test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

KEY = "AIzaFAKE-t66-cabecera-0123456789"

_RESPUESTA = {
    "candidates": [{
        "content": {"parts": [{"text": "respuesta "}, {"text": "verificada"}]},
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": "https://x.test/doc", "title": "x"}}],
            "webSearchQueries": ["q1"],
        },
    }],
    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
    "modelVersion": "gemini-x-001",
}

# Un 400 real de Google devuelve la key rechazada dentro del cuerpo de error.
_CUERPO_400 = (
    '{"error": {"code": 400, "message": "API key not valid. '
    f'Please pass a valid API key. key={KEY}", "details": [{{"api_key": "{KEY}"}}]}}}}'
)


class _Transporte:
    """Reemplaza AsyncHTTPTransport.handle_async_request: guarda cada
    httpx.Request y responde como MockTransport."""

    def __init__(self, status: int = 200, json_body: dict | None = None, text: str | None = None):
        self.requests: list[httpx.Request] = []
        self.status = status
        self.json_body = json_body
        self.text = text

    def patch(self):
        cap = self

        async def handle(transport_self, request: httpx.Request) -> httpx.Response:
            cap.requests.append(request)
            if cap.text is not None:
                return httpx.Response(cap.status, text=cap.text)
            return httpx.Response(cap.status, json=cap.json_body)

        return patch("httpx.AsyncHTTPTransport.handle_async_request", handle)


def _facet():
    from facet_resolver import ResolvedFacet
    return ResolvedFacet(
        key="hipatia", provider_id="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-x", credential=KEY, transport="http_gemini",
        persona=None, params=None,
    )


def _assert_key_en_cabecera(tc: unittest.TestCase, req: httpx.Request) -> None:
    tc.assertEqual(req.headers.get("x-goog-api-key"), KEY)
    tc.assertNotIn("key=", str(req.url))
    tc.assertNotIn(KEY, str(req.url))
    tc.assertEqual(str(req.url),
                   "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent")


class JacobsGeminiCabeceraTest(unittest.IsolatedAsyncioTestCase):
    async def _invocar(self, transporte: _Transporte):
        from jacobs import executor
        with transporte.patch(), \
             patch("jacobs.executor.record_resolved_version_safe", AsyncMock()):
            return await executor._invoke_http_gemini(_facet(), "prompt", timeout=30)

    async def test_la_key_va_en_la_cabecera_y_no_en_la_url(self):
        t = _Transporte(json_body=_RESPUESTA)
        await self._invocar(t)
        self.assertEqual(len(t.requests), 1)
        _assert_key_en_cabecera(self, t.requests[0])

    async def test_el_payload_y_el_parseo_no_cambian(self):
        import json
        t = _Transporte(json_body=_RESPUESTA)
        result = await self._invocar(t)
        self.assertEqual(json.loads(t.requests[0].content), {
            "contents": [{"role": "user", "parts": [{"text": "prompt"}]}],
            "tools": [{"google_search": {}}],
        })
        self.assertEqual(result["result"], "respuesta verificada")
        self.assertTrue(result["grounded"])
        self.assertEqual(result["queries"], ["q1"])
        self.assertEqual((result["tokens_in"], result["tokens_out"]), (11, 7))
        self.assertEqual(result["sources"][0]["url"] if "url" in result["sources"][0]
                         else result["sources"][0].get("uri"), "https://x.test/doc")

    async def test_un_error_del_proveedor_no_lleva_la_key(self):
        t = _Transporte(status=400, text=_CUERPO_400)
        with self.assertRaises(RuntimeError) as ctx:
            await self._invocar(t)
        msg = str(ctx.exception)
        self.assertIn("Gemini HTTP 400", msg)
        self.assertNotIn(KEY, msg)
        self.assertNotIn(KEY[:10], msg)


class JacobsFailStepRedactaTest(unittest.IsolatedAsyncioTestCase):
    """_fail_step es donde el error de un paso se ESCRIBE (jacobs_steps.error
    y los eventos STEP_FAILED / PIPELINE_ABORTED): se redacta ahi, asi un
    llamador nuevo no puede saltarselo."""

    async def test_el_error_guardado_y_los_eventos_van_redactados(self):
        from jacobs import executor
        from jacobs.models import Pipeline, Step

        step = Step(step_index=0, facet="hipatia", capability="research", input={"prompt": "p"})
        pipeline = Pipeline(name="t", invoked_by="plataforma", mode="auto", plan=[step])
        crudo = (f"HTTPStatusError: 400 for url 'https://g.example/m:generateContent?key={KEY}'"
                 f" cuerpo {KEY}")
        with patch.object(executor.store, "step_upsert", AsyncMock()), \
             patch.object(executor.store, "event_append", AsyncMock()) as ev, \
             patch.object(executor.store, "pipeline_update_status", AsyncMock()):
            await executor._fail_step(pipeline, step, 0, crudo)

        self.assertNotIn(KEY, step.error)
        self.assertIn("key=***", step.error)
        for call in ev.await_args_list:
            self.assertNotIn(KEY, repr(call.args))


class ReplGeminiCabeceraTest(unittest.IsolatedAsyncioTestCase):
    def _muscle(self):
        from jax.muscles import base
        return base.HttpMuscle(
            name="hipatia", provider="gemini", model_default="gemini-x",
            models_allowed=["gemini-x"], system_prompt="s", timeout=10,
            grounding_policy="auto",
        )

    async def _invocar(self, transporte: _Transporte):
        from jax.muscles import base
        muscle = self._muscle()
        with transporte.patch(), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value=KEY)), \
             patch.object(base, "record_resolved_version_safe", AsyncMock()):
            return await muscle._call_gemini("hola", "gemini-x")

    async def test_la_key_va_en_la_cabecera_y_no_en_la_url(self):
        t = _Transporte(json_body=_RESPUESTA)
        await self._invocar(t)
        self.assertEqual(len(t.requests), 1)
        _assert_key_en_cabecera(self, t.requests[0])

    async def test_el_payload_y_el_parseo_no_cambian(self):
        import json
        t = _Transporte(json_body=_RESPUESTA)
        out = await self._invocar(t)
        self.assertEqual(json.loads(t.requests[0].content), {
            "system_instruction": {"parts": [{"text": "s"}]},
            "contents": [{"role": "user", "parts": [{"text": "hola"}]}],
            "tools": [{"google_search": {}}],
        })
        self.assertTrue(out.startswith("respuesta verificada"), out)
        self.assertIn("https://x.test/doc", out)

    async def test_un_error_del_proveedor_no_lleva_la_key(self):
        from jax.muscles import base
        t = _Transporte(status=400, text=_CUERPO_400)
        with self.assertRaises(base.MuscleInvocationError) as ctx:
            await self._invocar(t)
        msg = str(ctx.exception)
        self.assertIn("Gemini HTTP 400", msg)
        self.assertNotIn(KEY, msg)
        self.assertNotIn(KEY[:10], msg)


class HumanizarErrorRedactaTest(unittest.TestCase):
    """El REPL imprime el error de la faceta con humanizar_error: tambien es
    un punto donde una excepcion se vuelve texto."""

    def test_un_error_no_reconocido_sale_redactado(self):
        from jax.core.main import humanizar_error
        err = RuntimeError(f"fallo raro en https://g.example/m?key={KEY}&x=1")
        out = humanizar_error("Hipatia", err)
        self.assertNotIn(KEY, out)
        self.assertIn("key=***", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
