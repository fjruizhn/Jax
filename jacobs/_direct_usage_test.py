#!/usr/bin/env python3
# jax/jacobs/_direct_usage_test.py
"""Scope expansion (2026-08-10): los 3 transportes HTTP directos que
executor.py usa para hipatia/jekyll/thot/ada/jax_local (_invoke_http_gemini,
_invoke_http_openai_compat, _invoke_ollama) descartaban los campos de token
usage exactamente como jax-platform/backend/api/chat.py antes de su Task 2 --
mismo bug, mismo fix, distinto repo. Estos tests cubren:
  1. cada uno de los 3 invocadores extrae tokens_in/tokens_out reales del
     campo correcto de su API (verificado contra las 3 APIs reales, misma
     sesion que estos fixes).
  2. _dispatch_step llama a jacobs.usage_writer.record_direct_usage con la
     identidad del pipeline y los campos resueltos de la faceta (provider_id/
     model) tras una llamada exitosa a cualquiera de los 3 transportes.

Corre con:
  PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos .venv/bin/python jacobs/_direct_usage_test.py
"""
from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

# T4 (2026-08-22, auditoria usage_writer): mismo guard que
# las_manos/_motor_usage_writer_test.py -- fail loud si JAX_DB_NAME ya
# apunta a otra cosa, en vez de escribir en silencio contra esa DB.
_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado (¿sourceaste "
        f"/etc/jax/.env?) -- este archivo escribe filas reales a esa DB. "
        f"Unset JAX_DB_NAME antes de correr este test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus


def _fake_resolved_facet(**overrides):
    from facet_resolver import ResolvedFacet
    base = dict(
        key="jekyll", provider_id="deepseek", base_url="https://api.example.test/v1",
        model="deepseek-v4-flash", credential="k", transport="http_openai_compat",
        persona=None, params=None,
    )
    base.update(overrides)
    return ResolvedFacet(**base)


class _FakeResp:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload


class InvokeGeminiTokensTest(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_http_gemini_extrae_tokens_reales(self):
        from jacobs import executor

        payload = {
            "candidates": [{
                "content": {"parts": [{"text": "respuesta"}]},
                "groundingMetadata": {
                    "groundingChunks": [{"web": {"uri": "https://x.test", "title": "x"}}],
                    "webSearchQueries": ["q"],
                },
            }],
            "usageMetadata": {"promptTokenCount": 123, "candidatesTokenCount": 45},
        }

        async def fake_post(self, url, json=None, **kwargs):
            return _FakeResp(payload)

        f = _fake_resolved_facet(key="hipatia", transport="http_gemini", model="gemini-2.5-flash")

        with patch("httpx.AsyncClient.post", fake_post):
            result = await executor._invoke_http_gemini(f, "prompt", timeout=30)

        self.assertEqual(result["tokens_in"], 123)
        self.assertEqual(result["tokens_out"], 45)

    async def test_invoke_http_gemini_default_cero_sin_usage_metadata(self):
        from jacobs import executor

        payload = {
            "candidates": [{
                "content": {"parts": [{"text": "respuesta"}]},
                "groundingMetadata": {
                    "groundingChunks": [{"web": {"uri": "https://x.test", "title": "x"}}],
                },
            }],
        }

        async def fake_post(self, url, json=None, **kwargs):
            return _FakeResp(payload)

        f = _fake_resolved_facet(key="hipatia", transport="http_gemini", model="gemini-2.5-flash")

        with patch("httpx.AsyncClient.post", fake_post):
            result = await executor._invoke_http_gemini(f, "prompt", timeout=30)

        self.assertEqual(result["tokens_in"], 0)
        self.assertEqual(result["tokens_out"], 0)


class InvokeOpenAICompatTokensTest(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_http_openai_compat_extrae_tokens_reales(self):
        from jacobs import executor

        payload = {
            "choices": [{"message": {"content": "hola"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 80},
        }

        async def fake_post(self, url, headers=None, json=None, **kwargs):
            return _FakeResp(payload)

        f = _fake_resolved_facet()

        with patch("httpx.AsyncClient.post", fake_post):
            result = await executor._invoke_http_openai_compat(f, "prompt", timeout=30)

        self.assertEqual(result["tokens_in"], 200)
        self.assertEqual(result["tokens_out"], 80)

    async def test_invoke_http_openai_compat_default_cero_sin_usage(self):
        from jacobs import executor

        payload = {"choices": [{"message": {"content": "hola"}}]}

        async def fake_post(self, url, headers=None, json=None, **kwargs):
            return _FakeResp(payload)

        f = _fake_resolved_facet()

        with patch("httpx.AsyncClient.post", fake_post):
            result = await executor._invoke_http_openai_compat(f, "prompt", timeout=30)

        self.assertEqual(result["tokens_in"], 0)
        self.assertEqual(result["tokens_out"], 0)


class InvokeOllamaTokensTest(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_ollama_extrae_tokens_reales(self):
        from jacobs import executor

        payload = {
            "message": {"content": "hola"},
            "prompt_eval_count": 55,
            "eval_count": 30,
        }

        async def fake_post(self, url, json=None, **kwargs):
            return _FakeResp(payload)

        f = _fake_resolved_facet(
            key="jax_local", provider_id="ollama_local", base_url=None,
            model="qwen3-coder:30b", transport="ollama", credential="",
        )

        with patch("httpx.AsyncClient.post", fake_post):
            result = await executor._invoke_ollama(f, "prompt", timeout=30)

        self.assertEqual(result["tokens_in"], 55)
        self.assertEqual(result["tokens_out"], 30)


class DispatchStepUsageTest(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_step_registra_usage_para_transporte_directo(self):
        from jacobs import executor

        pipeline = Pipeline(
            pipeline_id="p1", name="t", invoked_by="Fernando", mode="supervised",
            status=PipelineStatus.pending, user_id="1", tenant_id="77",
            created_at=time.time(), updated_at=time.time(),
        )
        step = Step(
            step_id="s1", pipeline_id="p1", step_index=0, facet="jekyll",
            capability="reason", status=StepStatus.pending, trace_id="t1",
            input={"prompt": "hola"},
        )

        payload = {
            "choices": [{"message": {"content": "hola"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        async def fake_post(self, url, headers=None, json=None, **kwargs):
            return _FakeResp(payload)

        f = _fake_resolved_facet(key="jekyll", provider_id="deepseek", model="deepseek-v4-flash")

        captured = {}

        async def fake_record_direct_usage(user_id, tenant_id, facet, provider_id, model, tokens_in, tokens_out):
            captured.update(
                user_id=user_id, tenant_id=tenant_id, facet=facet,
                provider_id=provider_id, model=model,
                tokens_in=tokens_in, tokens_out=tokens_out,
            )

        with patch("jacobs.executor.resolve_facet", return_value=f), \
             patch("httpx.AsyncClient.post", fake_post), \
             patch("jacobs.executor.record_direct_usage", fake_record_direct_usage):
            result = await executor._dispatch_step(step, pipeline)

        self.assertEqual(result["result"], "hola")
        self.assertEqual(captured["user_id"], "1")
        self.assertEqual(captured["tenant_id"], "77")
        self.assertEqual(captured["facet"], "jekyll")
        self.assertEqual(captured["provider_id"], "deepseek")
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["tokens_in"], 10)
        self.assertEqual(captured["tokens_out"], 5)

    async def test_dispatch_step_no_registra_usage_para_hyde_subprocess(self):
        """Hyde (subprocess) esta deliberadamente excluido -- sin señal de
        tokens (--output-format text). record_direct_usage NO debe llamarse."""
        from jacobs import executor

        pipeline = Pipeline(
            pipeline_id="p1", name="t", invoked_by="Fernando", mode="supervised",
            status=PipelineStatus.pending, user_id="1", tenant_id="77",
            created_at=time.time(), updated_at=time.time(),
        )
        step = Step(
            step_id="s1", pipeline_id="p1", step_index=0, facet="hyde",
            capability="reason", status=StepStatus.pending, trace_id="t1",
            input={"prompt": "hola"},
        )

        f = _fake_resolved_facet(key="hyde", provider_id="anthropic", model="claude", transport="subprocess")

        called = {"n": 0}

        async def fake_record_direct_usage(*args, **kwargs):
            called["n"] += 1

        async def fake_invoke_hyde(facet, prompt, timeout):
            return {"success": True, "facet": "hyde", "model": "claude", "result": "ok"}

        with patch("jacobs.executor.resolve_facet", return_value=f), \
             patch("jacobs.executor._invoke_hyde", fake_invoke_hyde), \
             patch("jacobs.executor.record_direct_usage", fake_record_direct_usage):
            await executor._dispatch_step(step, pipeline)

        self.assertEqual(called["n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
