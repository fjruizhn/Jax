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

os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs import store
from jacobs.models import Pipeline, PipelineStatus


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
