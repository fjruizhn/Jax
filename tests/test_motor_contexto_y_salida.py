"""
Camino de motores (kimi, jax_local) — tres defectos que dejaron sin producto
al paso "Producir" de la cadena en su E2E real (pipeline 1bb0da78,
2026-09-12). Todos anteriores a la cadena: la cadena solo los hizo visibles,
porque fue el primer plan donde un paso de motor dependía de verdad del
anterior.

1. **El contexto no llegaba al motor.** `_dispatch_step` armaba el prompt con
   las salidas de las dependencias (log: `contexto=4600 chars`) y llamaba a
   `_invoke_motor(step, pipeline, timeout)` sin pasárselo; `_invoke_motor`
   reconstruía el prompt desde `step.input`. Kimi recibió 721 caracteres:
   "produce con el plan unificado", sin el plan.
2. **El reintento pedía JSON de un schema que no existe.** `generate.v1` está
   declarado sin campos (`_KNOWN_UNIMPLEMENTED_SCHEMAS`), pero `validate()`
   exigía JSON antes de mirar eso: la primera respuesta (texto libre) se
   descartaba y el reintento le pedía a kimi "SOLO el JSON del schema
   'generate.v1'". Kimi razonó que no podía inventar un schema que no le
   dieron y devolvió `SCHEMA_NOT_PROVIDED`, marcado `completed`.
3. **La salida se recortaba a 200 caracteres** (`result_summary`) y no se
   guardaba completa en ningún lado; Jacobs pasaba esos 200 al paso
   siguiente y al repo.

Solo se mockea el borde de red. JobStore corre de verdad contra un
directorio temporal; no toca la DB.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import executor  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402
from motor_registry import worker  # noqa: E402
from motor_registry.catalog import MotorCatalog  # noqa: E402
from motor_registry.job_store import JobStore  # noqa: E402
from motor_registry.models import JobStatus  # noqa: E402

_LARGO = "Entregable completo. " * 250  # ~5.250 caracteres, muy por encima de 200

_CAP = {
    "allowed_motors": ["kimi"], "allowed_callers": ["jacobs"], "risk_level": "low",
    "sandbox_only": True, "requires_human_gate": False,
    "max_execution_minutes": 15, "max_recursion_depth": 0,
}
_CFG = {
    "motors": {"kimi": {
        "enabled": True, "provider": "kimi", "transport": "http_openai_compat",
        "provider_id": "moonshot", "api_key_env": "KIMI_API_KEY",
        "api_url": "https://api.moonshot.ai/v1/chat/completions", "model": "kimi-k3",
        "max_context_tokens": 256000, "sandbox_only": True,
        "default_timeout_seconds": 600, "supports_reasoning": True,
        "reasoning_default_visibility": "audit_only", "max_tokens": 8000,
    }},
    "capabilities": {
        "generate": {**_CAP, "output_schema": "generate.v1"},  # declarado, sin campos
        "implementation": {**_CAP, "output_schema": ""},
    },
}


def _response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


# ---------------------------------------------------------------- Jacobs

class ContextoLlegaAlMotorTest(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_step_pasa_al_motor_el_prompt_con_las_dependencias(self):
        plan = [
            Step(facet="ada", capability="reconcile", step_index=0),
            Step(facet="kimi", capability="generate", motor="kimi", step_index=1,
                 depends_on=[0], input={"prompt": "PRODUCE EL ENTREGABLE"}),
        ]
        pipeline = Pipeline(name="t", invoked_by="t", user_id="1", tenant_id="1",
                            mode="dry_run", plan=plan,
                            context={"objective": "un ERP",
                                     "step_0_ref": "inline:" + json.dumps({"result": "PLAN UNIFICADO X"})})
        seen = {}

        async def fake_invoke(step, pipeline, timeout, prompt=None):
            seen["prompt"] = prompt
            return {"success": True, "result": "ok"}

        with patch.object(executor, "validate_capability", AsyncMock(return_value=None)), \
             patch.object(executor, "_invoke_motor", fake_invoke):
            await executor._dispatch_step(plan[1], pipeline)

        assert seen.get("prompt"), "el motor no recibió el prompt armado"
        assert "PLAN UNIFICADO X" in seen["prompt"], "la salida de la dependencia no llegó al motor"
        assert "PRODUCE EL ENTREGABLE" in seen["prompt"]

    async def test_invoke_motor_manda_el_prompt_recibido_tal_cual(self):
        captured = {}

        async def fake_post(client_self, url, json=None, **kw):
            captured["payload"] = json
            return _Resp({"job_id": "j1", "status": "pending"})

        async def fake_get(client_self, url, **kw):
            return _Resp({"status": "completed", "result_summary": "ok"})

        step = Step(facet="kimi", capability="generate", motor="kimi", input={"prompt": "SOLO LA TAREA"})
        pipeline = Pipeline(name="t", invoked_by="t", user_id="1", tenant_id="1", mode="dry_run")
        armado = "REGLA...\nSalidas de las dependencias...\nTu tarea: SOLO LA TAREA"
        with patch("httpx.AsyncClient.post", fake_post), patch("httpx.AsyncClient.get", fake_get), \
             patch.object(executor, "MOTOR_POLL_INTERVAL", 0.01):
            await executor._invoke_motor(step, pipeline, timeout=5, prompt=armado)

        assert captured["payload"]["prompt"] == armado, captured["payload"]["prompt"]


class SalidaCompletaEnJacobsTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, job_payload):
        async def fake_post(client_self, url, json=None, **kw):
            return _Resp({"job_id": "j1", "status": "pending"})

        async def fake_get(client_self, url, **kw):
            return _Resp(job_payload)

        step = Step(facet="kimi", capability="generate", motor="kimi", input={"prompt": "x"})
        pipeline = Pipeline(name="t", invoked_by="t", user_id="1", tenant_id="1", mode="dry_run")
        with patch("httpx.AsyncClient.post", fake_post), patch("httpx.AsyncClient.get", fake_get), \
             patch.object(executor, "MOTOR_POLL_INTERVAL", 0.01):
            return await executor._invoke_motor(step, pipeline, timeout=5, prompt="p")

    async def test_devuelve_la_salida_completa_del_archivo_del_job(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "j1.md"
            path.write_text(_LARGO, encoding="utf-8")
            result = await self._run({"status": "completed", "result_summary": _LARGO[:200],
                                      "result_path": str(path)})
        assert result["result"] == _LARGO, len(result["result"])

    async def test_sin_archivo_cae_al_resumen(self):
        """Jobs anteriores a este arreglo no tienen result_path."""
        result = await self._run({"status": "completed", "result_summary": "resumen viejo"})
        assert result["result"] == "resumen viejo"


# ---------------------------------------------------------------- LAS MANOS

class SalidaCompletaEnWorkerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = JobStore(str(Path(self._tmp.name) / "jobs.jsonl"))
        self.catalog = MotorCatalog(_CFG)
        self._patches = [
            patch.object(worker, "resolve_credential", AsyncMock(return_value="sk-fake")),
            # PR-K ronda 2: contrato de la fila de `model` (el worker ya no manda motor.max_tokens a secas).
            patch("contrato_dispatch._leer_contrato", AsyncMock(return_value=("max_tokens", 131072))),
            patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()),
            patch("httpx.AsyncClient.post", AsyncMock(side_effect=AssertionError("llamada de red real en un test"))),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    async def _run(self, capability, call_fn):
        job_id = self.store.create(caller="jacobs", capability=capability, motor="kimi",
                                   trace_id="t", prompt="p", recursion_depth=0)
        with patch.dict(worker._TRANSPORT_DISPATCH, {"http_openai_compat": call_fn}):
            await worker.run(job_id=job_id, motor="kimi", capability=capability, prompt="p",
                             context={}, store=self.store, catalog=self.catalog,
                             kill_switch_path=str(Path(self._tmp.name) / "PAUSE"))
        return job_id, self.store._index[job_id]

    async def test_guarda_la_salida_completa_y_el_resumen_sigue_corto(self):
        async def call(**kw):
            return _response(_LARGO)

        job_id, state = await self._run("implementation", call)

        assert state["status"] == JobStatus.COMPLETED.value, state
        assert state.get("result_path"), "el job no apunta a su salida completa"
        assert Path(state["result_path"]).read_text(encoding="utf-8") == _LARGO
        assert len(state["result_summary"]) == 200
        assert self.store.get(job_id).result_path == state["result_path"], "MotorJobView no la expone"

    async def test_schema_pendiente_acepta_texto_libre_sin_reintento(self):
        """El caso del E2E: generate.v1 no tiene campos que exigir. La primera
        respuesta es el producto; pedir 'el JSON del schema' no se puede
        cumplir y la destruía."""
        calls = []

        async def call(**kw):
            calls.append(kw)
            return _response(_LARGO)

        _, state = await self._run("generate", call)

        assert len(calls) == 1, f"reintentó: {len(calls)} llamadas"
        assert state["status"] == JobStatus.COMPLETED.value, state
        assert Path(state["result_path"]).read_text(encoding="utf-8") == _LARGO


if __name__ == "__main__":
    unittest.main(verbosity=2)
