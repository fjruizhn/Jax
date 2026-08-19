#!/usr/bin/env python3
"""
Motor Registry — worker: max_tokens explícito + captura de finish_reason/usage.

Bug real encontrado el 2026-08-10 revisando un pipeline: un job de kimi
(motor de razonamiento, supports_reasoning=true) devolvió `content` cortado
a mitad de palabra (861 bytes en la DB, contra 20-25KB de los otros facets
del mismo pipeline). Reproducido en vivo contra la API real de Moonshot:
sin `max_tokens` en el payload, el razonamiento ya consumía más de la mitad
del completion budget con un prompt trivial (usage real:
completion_tokens=866, reasoning_tokens=467). Con un prompt largo y
demandante como el que falló, es plausible que el razonamiento se coma todo
el budget antes de escribir la respuesta final.

Dos fixes cubiertos acá:
1. `_call_kimi` manda `max_tokens` en el payload cuando el motor lo declara
   en su config (`MotorEntry.max_tokens`).
2. El job guardado captura `_finish_reason`/`_usage` — antes se descartaban
   por completo, así que un corte como el de hoy era indiagnosticable
   después del hecho (no se podía saber si fue `finish_reason='length'` u
   otra causa).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_worker_max_tokens_test.py

Solo mockea el límite real de red (httpx.AsyncClient.post) y la resolución
de credencial (evita tocar la DB real) — JobStore y MotorCatalog corren de
verdad, contra un archivo JSONL temporal.

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
            "api_key_env": "KIMI_API_KEY",
            "api_url": "https://api.moonshot.ai/v1/chat/completions",
            "model": "kimi-k2.7-code",
            "max_context_tokens": 256000,
            "sandbox_only": True,
            "default_timeout_seconds": 600,
            "supports_reasoning": True,
            "reasoning_default_visibility": "audit_only",
            "max_tokens": 8000,
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


def _fake_response(*, content, reasoning_content=None, finish_reason="stop", usage=None):
    message = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    payload = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    resp = AsyncMock()
    resp.json = lambda: payload
    resp.raise_for_status = lambda: None
    return resp


class WorkerMaxTokensTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
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
        with patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")), \
             patch("httpx.AsyncClient.post", AsyncMock(return_value=fake_resp)) as mock_post:
            await worker.run(
                job_id=job_id, motor="kimi", capability="implementation",
                prompt="prompt de prueba", context={}, store=self.store,
                catalog=self.catalog, kill_switch_path=self.kill_switch_path,
                **run_kwargs,
            )
        return job_id, mock_post

    async def test_payload_incluye_max_tokens_del_motor(self):
        job_id, mock_post = await self._run_job(_fake_response(content="respuesta completa"))
        _url, kwargs = mock_post.call_args
        sent_payload = kwargs["json"]
        assert sent_payload.get("max_tokens") == 8000, f"max_tokens no viajo en el payload: {sent_payload}"

    async def test_job_guarda_finish_reason_y_usage(self):
        job_id, _ = await self._run_job(_fake_response(
            content="respuesta completa", finish_reason="stop",
            usage={"prompt_tokens": 46, "completion_tokens": 866, "total_tokens": 912,
                   "completion_tokens_details": {"reasoning_tokens": 467}},
        ))
        state = self.store._index[job_id]
        assert state.get("_finish_reason") == "stop", state
        assert state.get("_usage", {}).get("completion_tokens") == 866, state

    async def test_run_con_identidad_llama_a_record_motor_usage(self):
        """Task 7: cuando el dispatch trae user_id/tenant_id (propagados por
        jacobs/executor.py, Task 6), worker.run debe llamar a
        record_motor_usage con provider_id/model/tokens reales -- sin tocar
        la logica de max_tokens/finish_reason/usage capturada arriba."""
        with patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()) as mock_record:
            job_id, _ = await self._run_job(
                _fake_response(
                    content="respuesta completa", finish_reason="stop",
                    usage={"prompt_tokens": 46, "completion_tokens": 866, "total_tokens": 912},
                ),
                user_id="1", tenant_id="77",
            )
        mock_record.assert_awaited_once_with(
            "1", "77", "kimi", "moonshot", "kimi-k2.7-code", 46, 866,
        )

    async def test_run_sin_identidad_pasa_none_a_record_motor_usage(self):
        """Compat con dispatches viejos / sin Jacobs: worker.run no filtra
        user_id/tenant_id ausentes antes de llamar -- el fail-soft (no
        escribir sin identidad) vive en record_motor_usage, cubierto por
        _motor_usage_writer_test.py::test_record_motor_usage_sin_identidad_no_escribe.
        Aca solo confirmamos que worker sigue completando el job igual (ver
        test_job_guarda_finish_reason_y_usage) y que pasa None tal cual,
        sin inventar un default."""
        with patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()) as mock_record:
            job_id, _ = await self._run_job(
                _fake_response(content="respuesta completa", finish_reason="stop"),
            )
        mock_record.assert_awaited_once_with(
            None, None, "kimi", "moonshot", "kimi-k2.7-code", 10, 20,
        )
        assert self.store._index[job_id]["status"] == "completed"

    async def test_finish_reason_length_queda_registrado(self):
        """El caso real que motivo el fix: si Moonshot corta por limite de
        tokens, finish_reason='length' debe quedar capturado -- antes se
        perdia por completo y el corte era indiagnosticable despues."""
        job_id, _ = await self._run_job(_fake_response(
            content="respuesta a medi", finish_reason="length",
        ))
        state = self.store._index[job_id]
        assert state.get("_finish_reason") == "length", state
        assert state["status"] == "completed"  # el job igual se marca completed -- el dato queda para diagnostico, no bloquea


if __name__ == "__main__":
    unittest.main(verbosity=2)
