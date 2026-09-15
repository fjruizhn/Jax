"""
PR-K ronda 2 (I2): el Motor Registry toma NOMBRE y TOPE del límite de salida
del contrato de la fila de `model` del motor (jax/core/contrato_dispatch.py),
no de "max_tokens" fijo ni de motor.max_tokens a secas. motor.max_tokens es un
presupuesto por llamada (propósito distinto) y se respeta el MENOR.

El caso de thot: gpt-5.6-terra exige max_completion_tokens y acepta como
mucho 128000 (el HTTP 400 del 2026-08-24).

Solo se mockea el borde: la lectura de la fila, la credencial, la red y la
escritura de costo. JobStore y MotorCatalog corren de verdad.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import contrato_dispatch
from motor_registry import worker
from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus

_CAP = {
    "allowed_motors": ["thot"], "allowed_callers": ["jacobs"], "risk_level": "low",
    "sandbox_only": True, "requires_human_gate": False, "max_execution_minutes": 15,
    "max_recursion_depth": 0, "output_schema": "",
}


def _cfg(transport="http_openai_compat", provider_id="openai", model="gpt-5.6-terra", max_tokens=0):
    return {
        "motors": {"thot": {
            "enabled": True, "provider": provider_id, "transport": transport,
            "provider_id": provider_id, "api_key_env": "X", "api_url": "https://api.example/v1",
            "model": model, "max_context_tokens": 0, "sandbox_only": True,
            "default_timeout_seconds": 60, "supports_reasoning": True,
            "reasoning_default_visibility": "audit_only", "max_tokens": max_tokens,
        }},
        "capabilities": {"implementation": dict(_CAP)},
    }


def _response(content="listo"):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class LimiteDelMotorTest(unittest.IsolatedAsyncioTestCase):
    async def _limite(self, fila, **cfg):
        leer = AsyncMock(return_value=fila)
        with patch.object(contrato_dispatch, "_leer_contrato", leer):
            entry = MotorCatalog(_cfg(**cfg)).get_motor("thot")
            return await worker._limite_del_motor(entry), leer

    async def test_thot_sin_presupuesto_manda_el_nombre_y_tope_del_catalogo(self):
        (limite, origen), leer = await self._limite(("max_completion_tokens", 128000))
        self.assertEqual(limite, {"max_completion_tokens": 128000})
        self.assertEqual(origen, "catalogo")
        leer.assert_awaited_once_with("openai", "gpt-5.6-terra")

    async def test_presupuesto_del_motor_menor_gana_con_el_nombre_del_catalogo(self):
        (limite, origen), _ = await self._limite(("max_completion_tokens", 128000), max_tokens=8000)
        self.assertEqual(limite, {"max_completion_tokens": 8000})
        self.assertEqual(origen, "motor")

    async def test_presupuesto_del_motor_mayor_que_la_api_no_pasa_el_tope(self):
        (limite, origen), _ = await self._limite(("max_completion_tokens", 128000), max_tokens=131072)
        self.assertEqual(limite, {"max_completion_tokens": 128000})
        self.assertEqual(origen, "catalogo")

    async def test_motor_ollama_por_v1_manda_max_tokens_con_el_tope(self):
        (limite, _), leer = await self._limite((None, 4096), transport="ollama", provider_id="ollama",
                                               model="qwen-x")
        self.assertEqual(limite, {"max_tokens": 4096})
        leer.assert_awaited_once_with("ollama", "qwen-x")


class PayloadRealTest(unittest.IsolatedAsyncioTestCase):
    async def test_el_body_lleva_max_completion_tokens_y_no_max_tokens(self):
        cuerpos = []

        class _R:
            status_code = 200

            def json(self):
                return _response()

            def raise_for_status(self):
                return None

        async def fake_post(client_self, url, json=None, headers=None, **kw):
            cuerpos.append(json)
            return _R()

        with patch("httpx.AsyncClient.post", fake_post):
            await worker._call_http_openai_compat(
                api_url="https://api.example/v1", model="gpt-5.6-terra", api_key="k",
                messages=[{"role": "user", "content": "x"}], timeout=5,
                limite={"max_completion_tokens": 128000},
            )
        self.assertEqual(cuerpos[0]["max_completion_tokens"], 128000)
        self.assertNotIn("max_tokens", cuerpos[0])


class RunConContratoTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = JobStore(str(Path(self._tmp.name) / "jobs.jsonl"))
        self.kill = str(Path(self._tmp.name) / "PAUSE")
        for p in [
            patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")),
            patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()),
            patch("httpx.AsyncClient.post", AsyncMock(side_effect=AssertionError("red real en un test"))),
        ]:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    async def _run(self, fila, call_fn, **cfg):
        job_id = self.store.create(caller="jacobs", capability="implementation", motor="thot",
                                   trace_id="t", prompt="p", recursion_depth=0)
        with patch.object(contrato_dispatch, "_leer_contrato", AsyncMock(return_value=fila)), \
                patch.dict(worker._TRANSPORT_DISPATCH, {"http_openai_compat": call_fn}):
            await worker.run(job_id=job_id, motor="thot", capability="implementation", prompt="p",
                             context={}, store=self.store, catalog=MotorCatalog(_cfg(**cfg)),
                             kill_switch_path=self.kill)
        return self.store._index[job_id]

    async def test_run_de_thot_despacha_con_el_contrato_de_la_fila(self):
        recibidos = []

        async def call_fn(**kw):
            recibidos.append(kw["limite"])
            return _response()

        estado = await self._run(("max_completion_tokens", 128000), call_fn)
        self.assertEqual(recibidos, [{"max_completion_tokens": 128000}])
        self.assertEqual(estado["status"], JobStatus.COMPLETED.value, estado)

    async def test_sin_contrato_el_job_falla_con_el_update_y_no_despacha(self):
        call_fn = AsyncMock(side_effect=AssertionError("no debía despachar"))
        estado = await self._run((None, 128000), call_fn)
        self.assertEqual(estado["status"], JobStatus.FAILED.value, estado)
        self.assertIn("sin contrato de dispatch", estado["error"])
        self.assertIn("UPDATE model SET max_tokens_param", estado["error"])
        call_fn.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
