"""Descartar pipelines: columnas e índices contra la base de TEST (spec
2026-09-22-descartar-pipelines §3). `store.get_pool`/`store.init_schema`
del brief no existen en este repo: los nombres reales son
`store.conexion()`/`store.init_tables()`, los mismos que usa
`tests/test_jacobs_reaper_cas_db.py`.

Fix round 1 (2026-09-22, revisión del coordinador): "strengthen the reaper
guard" -- `tests/test_jacobs_descarte.py::test_el_reaper_solo_cosecha_no_terminales`
solo mira el texto fuente de `jacobs/reaper.py` (grep de la lista de estados
no-terminales). Acá se agrega la prueba de comportamiento real: un pipeline
`discarded` y uno `hidden`, los dos con `updated_at`/`created_at` MUY viejos
(muy por encima de cualquier umbral del reaper), sobreviven a un barrido
REAL de `reaper.reap_orphaned_pipelines()` contra la base de TEST -- no un
mock del barrido, la función que corre en producción."""
from __future__ import annotations

import time
import unittest
import uuid

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import reaper, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402


class DescarteColumnasEIndicesDBTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)

    async def _columnas(self) -> dict[str, str]:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW COLUMNS FROM jacobs_pipelines")
                return {fila[0]: fila[1] for fila in await cur.fetchall()}

    async def test_columnas_del_descarte_existen(self):
        await store.init_tables()
        cols = await self._columnas()
        self.assertEqual(cols["status_previo"].lower(), "varchar(20)")
        self.assertEqual(cols["descartado_por"].lower(), "varchar(50)")
        self.assertEqual(cols["descartado_at"].lower(), "double")

    async def test_indices_del_descarte_existen(self):
        await store.init_tables()
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW INDEX FROM jacobs_pipelines")
                filas = await cur.fetchall()
        por_indice: dict[str, list[tuple[int, str]]] = {}
        for f in filas:
            por_indice.setdefault(f[2], []).append((f[3], f[4]))
        cols = {k: [c for _, c in sorted(v)] for k, v in por_indice.items()}
        self.assertEqual(
            cols["idx_pipelines_descartados"],
            ["user_id", "tenant_id", "status", "descartado_at"],
        )
        self.assertEqual(cols["idx_pipelines_ocultos"], ["status", "descartado_at"])


class ReaperNoTocaDescartadosNiOcultosDBTest(unittest.IsolatedAsyncioTestCase):
    """Comportamiento real (no grep): el reaper NO cosecha discarded/hidden,
    ni siquiera con updated_at/created_at extremadamente stale."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        self._pids: list[str] = []

    async def asyncTearDown(self):
        for pid in self._pids:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                await conn.commit()

    async def _crear_stale(self, status: PipelineStatus) -> str:
        pid = str(uuid.uuid4())
        # Muy por encima de PENDING_MAX_AGE_SECONDS (300s), RUNNING_STALE_SECONDS
        # (1800s) e INTERRUPTED_NO_OWNER_MAX_AGE_SECONDS (600s) -- si el reaper
        # clasificara mal a discarded/hidden como "candidato", esta antigüedad
        # los cosecharía con cualquiera de los tres umbrales.
        viejo = time.time() - 10 * reaper.RUNNING_STALE_SECONDS
        await store.init_tables()
        await store.pipeline_create(Pipeline(
            pipeline_id=pid, name="t-reaper-descarte", invoked_by="plataforma",
            mode="autonomous", status=status, created_at=viejo, updated_at=viejo,
            run_epoch=0,
        ))
        self._pids.append(pid)
        return pid

    async def test_el_reaper_no_toca_un_discarded_stale(self):
        pid = await self._crear_stale(PipelineStatus.discarded)
        cosechados = await reaper.reap_orphaned_pipelines()
        self.assertNotIn(pid, [c["pipeline_id"] for c in cosechados])
        fila = await store.pipeline_get(pid)
        self.assertEqual(fila.status, PipelineStatus.discarded)

    async def test_el_reaper_no_toca_un_hidden_stale(self):
        pid = await self._crear_stale(PipelineStatus.hidden)
        cosechados = await reaper.reap_orphaned_pipelines()
        self.assertNotIn(pid, [c["pipeline_id"] for c in cosechados])
        fila = await store.pipeline_get(pid)
        self.assertEqual(fila.status, PipelineStatus.hidden)
