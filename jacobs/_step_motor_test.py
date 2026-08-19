#!/usr/bin/env python3
"""step.motor desacoplado de step.facet (R4). Cuando el spec trae 'motor'
explicito, viaja tal cual al Motor Registry. Cuando no, se manda motor=None
-- activa MotorPolicy._resolve_motor(), que ya existia y nunca corria
porque _invoke_motor siempre mandaba motor=step.facet (executor.py:519
antes de este fix).

Corre desde /home/fruiz/jax con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python jacobs/_step_motor_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from jacobs.executor import _invoke_motor
from jacobs.models import Pipeline, Step


def _pipeline():
    return Pipeline(name="test", invoked_by="test", user_id="1", tenant_id="1", mode="dry_run")


class StepMotorTest(unittest.IsolatedAsyncioTestCase):
    async def test_motor_explicito_viaja_tal_cual(self):
        step = Step(facet="kimi", capability="implementation", motor="ada")
        pipeline = _pipeline()
        fake_dispatch = AsyncMock()
        fake_dispatch.return_value.status_code = 200
        fake_dispatch.return_value.json = lambda: {"job_id": "j1", "status": "pending"}
        fake_dispatch.return_value.raise_for_status = lambda: None
        fake_poll = AsyncMock()
        fake_poll.return_value.status_code = 200
        fake_poll.return_value.json = lambda: {"status": "completed", "result_summary": "ok"}
        fake_poll.return_value.raise_for_status = lambda: None

        captured = {}

        async def fake_post(self, url, json=None, **kw):
            captured["payload"] = json
            return fake_dispatch.return_value

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
