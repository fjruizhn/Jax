#!/usr/bin/env python3
"""
Motor Registry — el job expone el model_id real que usó (Task 1, ronda de
arreglo 1, 2026-09-18, historial-y-arreglos-de-pipeline).

kimi/jax_local son las facetas de la MAYORÍA de los pasos reales -- el
camino de motor no pasaba por resolve_facet(), así que Step.modelo_real
quedaba en None para casi todo el historial. worker.py conoce el model_id
real (motor_entry.model, usado para el dispatch en worker.py:171/725) pero
nunca lo escribía en el job: MotorJobView (motor_registry/models.py) no
tenía el campo, y JobStore.get() filtra cualquier clave que no esté en
MotorJobView.model_fields (job_store.py:104) -- agregar el campo al modelo
es CONDICIÓN NECESARIA, sin eso cualquier otra cosa se pierde en silencio
en ese filtro.

Este test pasa por `self.store.get(job_id)`, no por construir MotorJobView
a mano -- así ejercita el filtro real de job_store.py, no solo el modelo
Pydantic. Un test que solo mirara MotorJobView(model=...) daría verde con
el filtro intacto y el dato perdiéndose igual (la advertencia de la ronda
de arreglo).

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v las_manos/_motor_job_model_test.py

Solo mockea el límite real de red (httpx.AsyncClient.post) y la resolución
de credencial -- JobStore y MotorCatalog corren de verdad, contra un
archivo JSONL temporal. Mismo patrón que _worker_max_tokens_test.py.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore
from motor_registry import worker

_MOTOR_CFG = {
    "motors": {
        "kimi": {
            "enabled": True,
            "provider": "kimi",
            "transport": "http_openai_compat",
            "provider_id": "moonshot",
            "api_key_env": "KIMI_API_KEY",
            "api_url": "https://api.moonshot.ai/v1/chat/completions",
            "model": "kimi-k2.7-code",
            "max_context_tokens": 256000,
            "sandbox_only": True,
            "default_timeout_seconds": 600,
            "supports_reasoning": True,
            "reasoning_default_visibility": "audit_only",
        },
    },
    "capabilities": {
        "implementation": {
            "allowed_motors": ["kimi"],
            "allowed_callers": ["jacobs"],
            "risk_level": "low",
            "sandbox_only": True,
            "requires_human_gate": False,
            "max_execution_minutes": 10,
            "max_recursion_depth": 0,
            "output_schema": "",
        },
    },
}


def _fake_response(*, content="listo", finish_reason="stop", usage=None):
    payload = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    resp = AsyncMock()
    resp.json = lambda: payload
    resp.raise_for_status = lambda: None
    return resp


class WorkerJobExponeModeloTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import contrato_dispatch  # PR-K ronda 2: el worker lee el contrato de la fila de `model`
        _pc = patch.object(contrato_dispatch, "_leer_contrato", AsyncMock(return_value=("max_tokens", 131072)))
        _pc.start()
        self.addCleanup(_pc.stop)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = JobStore(str(Path(self._tmpdir.name) / "jobs.jsonl"))
        self.catalog = MotorCatalog(_MOTOR_CFG)
        self.kill_switch_path = str(Path(self._tmpdir.name) / "PAUSE")  # nunca existe

    def tearDown(self):
        self._tmpdir.cleanup()

    async def _run_job(self, fake_resp, **run_kwargs):
        job_id = self.store.create(
            caller="jacobs", capability="implementation", motor="kimi",
            trace_id="t1", prompt="prompt de prueba", recursion_depth=0,
        )
        with patch.object(worker, "resolve_credential", AsyncMock(return_value="sk-fake")), \
             patch("httpx.AsyncClient.post", AsyncMock(return_value=fake_resp)):
            await worker.run(
                job_id=job_id, motor="kimi", capability="implementation",
                prompt="prompt de prueba", context={}, store=self.store,
                catalog=self.catalog, kill_switch_path=self.kill_switch_path,
                **run_kwargs,
            )
        return job_id

    async def test_job_completado_expone_el_model_id_real(self):
        """El model_id que worker.py de verdad usó para el dispatch
        (motor_entry.model == 'kimi-k2.7-code' en este fixture) tiene que
        estar disponible en el job leído a través de JobStore.get() -- el
        MISMO camino que executor.py (jacobs) consulta vía GET /motor/job."""
        job_id = await self._run_job(_fake_response())
        job = self.store.get(job_id)
        assert job is not None, "job desapareció del índice"
        assert job.status.value == "completed", job
        assert job.model == "kimi-k2.7-code", (
            f"JobStore.get() no expuso el model_id real -- state completo: "
            f"{self.store._index.get(job_id)}"
        )

    async def test_job_fallido_antes_de_resolver_motor_no_trae_modelo_inventado(self):
        """Motor inexistente en el catálogo: el job falla ANTES de conocer
        ningún motor_entry.model -- job.model tiene que quedar None, nunca
        un valor inventado."""
        job_id = self.store.create(
            caller="jacobs", capability="implementation", motor="motor_que_no_existe",
            trace_id="t2", prompt="x", recursion_depth=0,
        )
        await worker.run(
            job_id=job_id, motor="motor_que_no_existe", capability="implementation",
            prompt="x", context={}, store=self.store, catalog=self.catalog,
            kill_switch_path=self.kill_switch_path,
        )
        job = self.store.get(job_id)
        assert job.status.value == "failed", job
        assert job.model is None, job


if __name__ == "__main__":
    unittest.main(verbosity=2)
