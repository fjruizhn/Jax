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
from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import store
from jacobs.executor import _invoke_motor
from jacobs.models import Pipeline, Step, StepStatus
from policy.execution_control.errors import GovernedExecutionRequiredError


def _pipeline():
    return Pipeline(name="test", invoked_by="test", user_id="1", tenant_id="1", mode="dry_run")


class StepMotorTest(unittest.IsolatedAsyncioTestCase):
    async def test_motor_explicito_viaja_tal_cual(self):
        step = Step(facet="kimi", capability="implementation", motor="ada")
        pipeline = _pipeline()

        with patch("jacobs.executor.obtener_cliente_http", side_effect=AssertionError("no HTTP")) as client, \
             self.assertRaises(GovernedExecutionRequiredError):
            await _invoke_motor(step, pipeline, timeout=5)
        client.assert_not_called()

    async def test_motor_ausente_manda_none_para_activar_resolver(self):
        step = Step(facet="kimi", capability="implementation", motor=None)
        pipeline = _pipeline()
        with patch("jacobs.executor.obtener_cliente_http", side_effect=AssertionError("no HTTP")) as client, \
             self.assertRaises(GovernedExecutionRequiredError):
            await _invoke_motor(step, pipeline, timeout=5)
        client.assert_not_called()


class StepMotorPersistenceTest(unittest.IsolatedAsyncioTestCase):
    """step.motor debe sobrevivir el round-trip por jacobs_steps -- si no, un
    pin explícito (ej. motor="ada") revierte a None en silencio cada vez que
    un pipeline pasa por resume_pipeline/approve_step (routes.py), que
    reconstruyen pipeline.plan entero desde steps_by_pipeline() antes de
    re-despachar."""

    async def asyncSetUp(self):
        await store.init_tables()
        self.pids = []

    async def asyncTearDown(self):
        """O5 (re-review de la ronda 3, 2026-09-17): antes cada corrida dejaba
        una fila en jacobs_steps de jax_memory_test para siempre (medido: 116
        pasos huérfanos con este patrón). Se borran los pids PROPIOS y se
        verifica que no quede ninguno."""
        conn = await store.conexion_dedicada()
        try:
            async with conn.cursor() as cur:
                for pid in self.pids:
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
                marcas = ",".join(["%s"] * len(self.pids))
                await cur.execute(
                    f"SELECT COUNT(*) FROM jacobs_steps WHERE pipeline_id IN ({marcas})", tuple(self.pids))
                restantes = (await cur.fetchone())[0]
        finally:
            conn.close()
        await store.cerrar_pool()
        assert restantes == 0, f"el test dejó {restantes} pasos en jacobs_steps"

    def _pid(self) -> str:
        pid = str(uuid.uuid4())
        self.pids.append(pid)
        return pid

    async def test_motor_sobrevive_upsert_y_reload(self):
        pid = self._pid()
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
        pid = self._pid()
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
