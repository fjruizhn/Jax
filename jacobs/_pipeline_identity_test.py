#!/usr/bin/env python3
# jax/jacobs/_pipeline_identity_test.py
"""Pipeline.user_id/tenant_id: identidad real para atribuir costo de Kimi
(via motor_registry) — antes jacobs_pipelines solo tenia invoked_by (label
humano, no una FK a un usuario real). Corre con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python jacobs/_pipeline_identity_test.py
"""
from __future__ import annotations

import os
import time
import unittest
import uuid
from unittest.mock import patch

os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs import store
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus


class PipelineIdentityTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await store.init_tables()

    async def test_pipeline_create_y_lectura_conservan_user_id_tenant_id(self):
        pid = str(uuid.uuid4())
        p = Pipeline(
            pipeline_id=pid, name="test", invoked_by="Fernando", mode="supervised",
            status=PipelineStatus.pending, user_id="1", tenant_id="test-tenant",
            created_at=time.time(), updated_at=time.time(),
        )
        await store.pipeline_create(p)
        loaded = await store.pipeline_get(pid)  # confirmado: nombre real, ver store.py:116
        self.assertEqual(loaded.user_id, "1")
        self.assertEqual(loaded.tenant_id, "test-tenant")


class ExecutorMotorPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_motor_incluye_identidad_del_pipeline(self):
        from jacobs import executor

        pipeline = Pipeline(
            pipeline_id="p1", name="t", invoked_by="Fernando", mode="supervised",
            status=PipelineStatus.pending, user_id="1", tenant_id="test-tenant",
            created_at=time.time(), updated_at=time.time(),
        )
        step = Step(
            step_id="s1", pipeline_id="p1", step_index=0, facet="kimi",
            capability="implementation", status=StepStatus.pending,
            trace_id="t1", input={"prompt": "hola"},
        )

        captured = {}

        class _FakeResp:
            status_code = 202
            def json(self):
                return {"job_id": "j1", "status": "running"}
            def raise_for_status(self):
                pass

        async def fake_post(self, url, json=None, **kwargs):
            captured["json"] = json
            return _FakeResp()

        with patch("httpx.AsyncClient.post", fake_post):
            try:
                await executor._invoke_motor(step, pipeline, timeout=30)
            except Exception:
                pass  # el resto del polling puede fallar en este test acotado; solo interesa el payload inicial

        self.assertEqual(captured["json"].get("user_id"), "1")
        self.assertEqual(captured["json"].get("tenant_id"), "test-tenant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
