#!/usr/bin/env python3
"""Task 1 (2026-09-18, historial-y-arreglos-de-pipeline): el paso guarda qué
modelo lo ejecutó de verdad.

Hoy Step no tiene dónde guardar qué modelo corrió. resolve_facet() lo sabe
(f.model) y lo tira. Por eso el informe de b8f80733 decía "Modelo desconocido".
Saber que actuó jax_local no dice qué modelo fue: el binding cambia, y un
historial que no lo fija miente con el tiempo.

Ampliación de alcance (mensaje del coordinador, 2026-09-18): modelo_real
también tiene que sobrevivir el viaje completo por jacobs_steps -- routes.py
(get_pipeline_results) lee de store.steps_by_pipeline(), NO del JSON de
jacobs_pipelines.plan. Un campo que solo vive en el objeto Step en memoria
da verde en un test de memoria y se pierde en producción.

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_modelo_real_test.py

Los tests que tocan DB real (DespachoEscribeModeloTest y
ModeloRealPersistenceTest) necesitan JAX_DB_HOST/JAX_DB_PORT en el entorno
(sourceados de /etc/jax/.env, NUNCA su JAX_DB_NAME -- exigir_base_de_test()
lo rechaza si no es una base de test) y corren contra jax_memory_test o
jax_memory_test_<sufijo> (base_de_test.py / conftest.py de la raíz).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import time
import unittest
import uuid
from unittest.mock import patch

# T4 (2026-08-22, auditoria usage_writer): mismo guard que
# las_manos/_motor_usage_writer_test.py -- fail loud si JAX_DB_NAME ya
# apunta a otra cosa, en vez de escribir en silencio contra esa DB.
from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import executor, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


def test_step_tiene_donde_guardar_el_modelo_real():
    """Sin este campo, el historial solo puede mostrar la faceta, y la faceta
    no dice qué modelo corrió: el binding cambia."""
    paso = Step(step_id="s1", pipeline_id="p1", step_index=0,
                facet="thot", capability="analysis", input={"prompt": "x"})
    assert hasattr(paso, "modelo_real")
    assert paso.modelo_real is None


class DespachoEscribeModeloTest(unittest.IsolatedAsyncioTestCase):
    """El modelo se conoce en resolve_facet() y hoy se tira.

    _dispatch_step valida la capability contra la DB real ANTES de resolver
    la faceta (validate_capability -> store.get_motor_governance(), y para
    facets HTTP directos como 'thot' también MotorCatalog.from_db() /
    check_capability_admission) -- por eso este test necesita DB, no solo
    resolve_facet/el transporte mockeados. 'analysis' es una capability real
    (verificado contra jax_memory_test 2026-09-18: 'text_generation' del
    brief original NO existe en la tabla `capability`, el brief tenía un
    nombre inventado -- corregido acá)."""

    async def test_el_despacho_escribe_el_modelo_resuelto(self):
        class FacetaFalsa:
            # 'http_openai_compat' (con el prefijo 'http_'), no 'openai_compat'
            # -- el brief original traía el literal equivocado: executor.py
            # solo reconoce 'http_gemini'/'http_openai_compat'/'ollama'/
            # 'subprocess'; con el nombre viejo caía al ValueError
            # "Transporte desconocido" y el test fallaba por una razón que no
            # era la que se quería medir.
            transport = "http_openai_compat"
            model = "glm-4.6"
            persona = ""
            base_url = "http://x"
            provider_id = 1
            key = "thot"
            credential = "x"
            params = {}

        async def resolve_falso(_clave):
            return FacetaFalsa()

        async def invoke_falso(*_a, **_k):
            return {"text": "ok"}

        paso = Step(step_id=str(uuid.uuid4()), pipeline_id="p1", step_index=0,
                    facet="thot", capability="analysis", input={"prompt": "x"})
        pipeline = Pipeline(pipeline_id="p1", name="n", invoked_by="test",
                             mode="autonomous", status=PipelineStatus.running,
                             plan=[paso])

        # _invoke_http_openai_compat, no _invoke_openai_compat -- mismo motivo
        # que el transport de arriba: ese nombre no existe en executor.py.
        with patch.object(executor, "resolve_facet", resolve_falso), \
             patch.object(executor, "_invoke_http_openai_compat", invoke_falso):
            await executor._dispatch_step(paso, pipeline)

        self.assertEqual(paso.modelo_real, "glm-4.6")


class ModeloRealPersistenceTest(unittest.IsolatedAsyncioTestCase):
    """modelo_real tiene que sobrevivir el round-trip por jacobs_steps -- si
    no, get_pipeline_results (routes.py:1096, que lee steps_by_pipeline() y
    NO el JSON de jacobs_pipelines.plan) nunca lo ve, y cualquier camino que
    reconstruye pipeline.plan desde la DB (resume_pipeline, approve_step) lo
    pierde en silencio. Mismo patrón que jacobs/_step_motor_test.py::
    StepMotorPersistenceTest para step.motor."""

    async def asyncSetUp(self):
        await store.init_tables()
        self.pids: list[str] = []

    async def asyncTearDown(self):
        """O5 (re-review ronda 3, 2026-09-17): borrar solo los pids PROPIOS y
        verificar que no queda ninguno -- un test que no limpia deja pasos
        huérfanos en jacobs_steps para siempre."""
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

    async def test_modelo_real_sobrevive_step_upsert_y_reload(self):
        """El escritor de creación (step_upsert, jacobs/routes.py y
        jacobs/_arnes_ada.py) tiene que llevar la columna."""
        pid = self._pid()
        step = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=0,
            facet="thot", capability="analysis", modelo_real="glm-4.6",
            status=StepStatus.pending, trace_id=str(uuid.uuid4()),
        )
        await store.step_upsert(step)
        reloaded = await store.steps_by_pipeline(pid)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0].modelo_real, "glm-4.6")

    async def test_modelo_real_none_sobrevive_step_upsert_y_reload(self):
        pid = self._pid()
        step = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=0,
            facet="thot", capability="analysis", modelo_real=None,
            status=StepStatus.pending, trace_id=str(uuid.uuid4()),
        )
        await store.step_upsert(step)
        reloaded = await store.steps_by_pipeline(pid)
        self.assertEqual(len(reloaded), 1)
        self.assertIsNone(reloaded[0].modelo_real)

    async def test_modelo_real_sobrevive_step_upsert_si_epoca(self):
        """El escritor del ejecutor (step_upsert_si_epoca, el que corre en
        CADA despacho real -- executor.py:1062 y otros) tiene que llevar la
        columna. Sin esto, el valor que _dispatch_step acaba de escribir en
        memoria se pierde en la primera escritura condicional a la época."""
        pid = self._pid()
        step = Step(
            step_id=str(uuid.uuid4()), pipeline_id=pid, step_index=0,
            facet="thot", capability="analysis",
            status=StepStatus.pending, trace_id=str(uuid.uuid4()),
        )
        ahora = time.time()
        pipeline = Pipeline(
            pipeline_id=pid, name="t-modelo-real", invoked_by="plataforma",
            mode="autonomous", status=PipelineStatus.running, plan=[step],
            created_at=ahora, updated_at=ahora, run_epoch=0,
        )
        await store.pipeline_create(pipeline)
        await store.step_upsert(step)

        step.modelo_real = "glm-4.6"
        step.status = StepStatus.completed
        escrito = await store.step_upsert_si_epoca(step, 0)
        self.assertTrue(escrito)

        reloaded = await store.steps_by_pipeline(pid)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0].modelo_real, "glm-4.6")


if __name__ == "__main__":
    unittest.main(verbosity=2)
