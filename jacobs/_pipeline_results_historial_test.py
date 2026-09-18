#!/usr/bin/env python3
"""Task 9 (2026-09-18, historial-y-arreglos-de-pipeline): el detalle de un
pipeline en la Mesa necesita ver, por paso, el prompt exacto, el modelo real
y de qué pasos dependía -- no solo faceta/capability/status/result.

Hoy routes.py:get_pipeline_results arma el dict a mano con step_index, facet,
capability, name, status, result, sources, duration_seconds, error. Le faltan
tres campos que YA existen en el modelo Step y sobreviven el round-trip por
jacobs_steps (ver jacobs/_modelo_real_test.py): input.get("prompt"),
modelo_real y depends_on. Sin esto el usuario no puede ver qué se le pidió
exactamente a cada faceta ni qué modelo respondió -- que es justo lo que pidió
Fernando ("un lugar... donde se pueda leer el prompt que generaste").

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_pipeline_results_historial_test.py

Necesita JAX_DB_HOST/JAX_DB_PORT en el entorno (de /etc/jax/.env, NUNCA
JAX_DB_NAME) y corre contra jax_memory_test / jax_memory_test_<sufijo>
(base_de_test.py / conftest.py de la raíz) -- mismo guard que
_modelo_real_test.py.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import time
import unittest
import uuid

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import routes, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


class PipelineResultsHistorialTest(unittest.IsolatedAsyncioTestCase):
    """get_pipeline_results tiene que exponer prompt, modelo_real y
    depends_on por paso -- los tres campos que el historial de la Mesa
    necesita mostrar y que hoy quedan afuera de la lista blanca del dict."""

    async def asyncSetUp(self):
        await store.init_tables()
        self.pids: list[str] = []

    async def asyncTearDown(self):
        conn = await store.conexion_dedicada()
        try:
            async with conn.cursor() as cur:
                for pid in self.pids:
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                marcas = ",".join(["%s"] * len(self.pids)) or "NULL"
                await cur.execute(
                    f"SELECT COUNT(*) FROM jacobs_steps WHERE pipeline_id IN ({marcas})",
                    tuple(self.pids),
                )
                restantes = (await cur.fetchone())[0]
        finally:
            conn.close()
        await store.cerrar_pool()
        assert restantes == 0, f"el test dejó {restantes} pasos en jacobs_steps"

    def _pid(self) -> str:
        pid = str(uuid.uuid4())
        self.pids.append(pid)
        return pid

    async def test_results_trae_prompt_modelo_real_y_depends_on(self):
        pid = self._pid()
        ahora = time.time()

        paso_0 = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=0,
            facet="thot", capability="analysis",
            input={"prompt": "resumí este documento"},
            status=StepStatus.completed, trace_id=str(uuid.uuid4()),
            modelo_real="glm-4.6", depends_on=[],
            started_at=ahora, finished_at=ahora + 1.5,
        )
        paso_1 = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=1,
            facet="kimi", capability="implementation",
            input={"prompt": "implementá lo que dijo el paso 0"},
            status=StepStatus.completed, trace_id=str(uuid.uuid4()),
            modelo_real="kimi-k2.7-code", depends_on=[0],
            started_at=ahora + 1.5, finished_at=ahora + 3.0,
        )

        pipeline = Pipeline(
            pipeline_id=pid, name="t-historial", invoked_by="plataforma",
            mode="autonomous", status=PipelineStatus.completed,
            plan=[paso_0, paso_1], created_at=ahora, updated_at=ahora + 3.0,
        )
        await store.pipeline_create(pipeline)
        await store.step_upsert(paso_0)
        await store.step_upsert(paso_1)

        respuesta = await routes.get_pipeline_results(pid)
        pasos = respuesta["steps"]
        self.assertEqual(len(pasos), 2)

        p0, p1 = pasos
        self.assertEqual(p0["prompt"], "resumí este documento")
        self.assertEqual(p0["modelo_real"], "glm-4.6")
        self.assertEqual(p0["depends_on"], [])

        self.assertEqual(p1["prompt"], "implementá lo que dijo el paso 0")
        self.assertEqual(p1["modelo_real"], "kimi-k2.7-code")
        self.assertEqual(p1["depends_on"], [0])

    async def test_prompt_ausente_no_revienta(self):
        """Un paso sin 'prompt' en input (p.ej. uno que solo lleva 'name')
        devuelve prompt=None, no KeyError."""
        pid = self._pid()
        paso = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=0,
            facet="thot", capability="analysis", input={"name": "sin prompt"},
            status=StepStatus.pending, trace_id=str(uuid.uuid4()),
        )
        pipeline = Pipeline(
            pipeline_id=pid, name="t-sin-prompt", invoked_by="plataforma",
            mode="autonomous", status=PipelineStatus.running, plan=[paso],
        )
        await store.pipeline_create(pipeline)
        await store.step_upsert(paso)

        respuesta = await routes.get_pipeline_results(pid)
        self.assertIsNone(respuesta["steps"][0]["prompt"])
        self.assertIsNone(respuesta["steps"][0]["modelo_real"])
        self.assertEqual(respuesta["steps"][0]["depends_on"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
