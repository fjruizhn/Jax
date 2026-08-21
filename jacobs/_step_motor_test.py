#!/usr/bin/env python3
"""step.motor desacoplado de step.facet (R4). Cuando el spec trae 'motor'
explicito, viaja tal cual al Motor Registry. Cuando no, se manda motor=None
-- activa MotorPolicy._resolve_motor(), que ya existia y nunca corria
porque _invoke_motor siempre mandaba motor=step.facet (executor.py:519
antes de este fix).

Corre desde /home/fruiz/jax con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python -m unittest jacobs._step_motor_test

NO como `python jacobs/_step_motor_test.py` directo -- eso pone
jacobs/ al frente de sys.path y produce una identidad de clase Step
duplicada entre este archivo y jacobs.executor/jacobs.store, causando
fallos falsos (AttributeError/payload incorrecto) aunque el código real
sea correcto. Verificado 2026-08-19: -m unittest da 4/4 verde, el
invocado directo falla por esto, no por un bug real.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import unittest
import uuid
from unittest.mock import AsyncMock, patch

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

from jacobs import store
from jacobs.executor import _invoke_motor
from jacobs.models import Pipeline, Step, StepStatus


def _pipeline():
    return Pipeline(name="test", invoked_by="test", user_id="1", tenant_id="1", mode="dry_run")


class StepMotorTest(unittest.IsolatedAsyncioTestCase):
    async def test_motor_explicito_viaja_tal_cual(self):
        step = Step(facet="kimi", capability="implementation", motor="ada")
        pipeline = _pipeline()

        class _DispatchResp:
            status_code = 200
            def json(self): return {"job_id": "j1", "status": "pending"}
            def raise_for_status(self): pass

        fake_poll = AsyncMock()
        fake_poll.return_value.status_code = 200
        fake_poll.return_value.json = lambda: {"status": "completed", "result_summary": "ok"}
        fake_poll.return_value.raise_for_status = lambda: None

        captured = {}

        async def fake_post(self, url, json=None, **kw):
            captured["payload"] = json
            return _DispatchResp()

        with patch("httpx.AsyncClient.post", fake_post), \
             patch("httpx.AsyncClient.get", fake_poll):
            await _invoke_motor(step, pipeline, timeout=5)

        assert captured["payload"]["motor"] == "ada", captured["payload"]

    async def test_motor_ausente_manda_none_para_activar_resolver(self):
        step = Step(facet="kimi", capability="implementation", motor=None)
        pipeline = _pipeline()
        fake_dispatch_json = {"job_id": "j2", "status": "pending"}
        captured = {}

        class _Resp:
            status_code = 200
            def json(self): return fake_dispatch_json
            def raise_for_status(self): pass

        class _RespDone:
            status_code = 200
            def json(self): return {"status": "completed", "result_summary": "ok"}
            def raise_for_status(self): pass

        async def fake_post(self, url, json=None, **kw):
            captured["payload"] = json
            return _Resp()

        async def fake_get(self, url, **kw):
            return _RespDone()

        with patch("httpx.AsyncClient.post", fake_post), \
             patch("httpx.AsyncClient.get", fake_get):
            await _invoke_motor(step, pipeline, timeout=5)

        assert captured["payload"]["motor"] is None, captured["payload"]


class StepMotorPersistenceTest(unittest.IsolatedAsyncioTestCase):
    """step.motor debe sobrevivir el round-trip por jacobs_steps -- si no, un
    pin explícito (ej. motor="ada") revierte a None en silencio cada vez que
    un pipeline pasa por resume_pipeline/approve_step (routes.py), que
    reconstruyen pipeline.plan entero desde steps_by_pipeline() antes de
    re-despachar."""

    async def asyncSetUp(self):
        await store.init_tables()

    async def test_motor_sobrevive_upsert_y_reload(self):
        pid = str(uuid.uuid4())
        step = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=0,
            facet="kimi", motor="ada", capability="implementation",
            status=StepStatus.pending, trace_id=str(uuid.uuid4()),
        )
        await store.step_upsert(step)
        reloaded = await store.steps_by_pipeline(pid)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0].motor, "ada")

    async def test_motor_none_sobrevive_upsert_y_reload(self):
        pid = str(uuid.uuid4())
        step = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=0,
            facet="kimi", motor=None, capability="implementation",
            status=StepStatus.pending, trace_id=str(uuid.uuid4()),
        )
        await store.step_upsert(step)
        reloaded = await store.steps_by_pipeline(pid)
        self.assertEqual(len(reloaded), 1)
        self.assertIsNone(reloaded[0].motor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
