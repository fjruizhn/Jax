"""Contrato de sub-pipelines contra una MariaDB REAL (frente F, 2026-09-16):
esquema, emisión, consumo atómico, carreras y plan de la consulta.

Las tablas de Jacobs las crea ESTE repo (store.init_tables), así que el job de
CI no necesita el esquema de jax-platform.

Seguridad: la barrera de base vive en jacobs/_arnes_ada.py, que se importa
PRIMERO: con JAX_DB_NAME apuntando a otra base que no sea jax_memory_test, el
módulo entero levanta RuntimeError.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v jacobs/_subpipeline_contrato_io_test.py"
"""
from __future__ import annotations

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import patch  # noqa: E402

import aiomysql  # noqa: E402

from jacobs import store  # noqa: E402
from jacobs import subpipelines as sp  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402

COLUMNAS_TOKENS = [
    "token_hash", "parent_pipeline_id", "parent_step", "depth_hijo",
    "emitido_at", "vence_at", "usado_at", "hijo_pipeline_id",
]


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class _ConBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await store.init_tables()
        freno = patch("jacobs.policy.check_kill_switch", return_value=False)
        freno.start()
        self.addCleanup(freno.stop)
        entorno = patch.dict(os.environ)
        entorno.start()
        self.addCleanup(entorno.stop)
        os.environ.pop(sp.ENV_TTL, None)
        os.environ.pop(sp.ENV_MAX_PROFUNDIDAD, None)
        self.padres: list[str] = []

    async def asyncTearDown(self):
        for pid in self.padres:
            await ada.cerrar(pid)

    async def padre(self, **kw) -> tuple[str, str]:
        pid, paso = await ada.padre_en_ejecucion(**kw)
        self.padres.append(pid)
        return pid, paso

    async def columnas(self, tabla: str) -> list[str]:
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
                    (tabla,),
                )
                return [r[0] for r in await cur.fetchall()]
        finally:
            conn.close()


class EsquemaTest(_ConBase):
    async def test_init_tables_crea_la_tabla_de_tokens_sin_columna_para_el_token(self):
        # Se BORRA y se recrea: la jax_memory_test local puede tenerla de una
        # corrida anterior, y entonces el test pasaría sin init_tables().
        await ada.ejecutar("DROP TABLE IF EXISTS jacobs_subpipeline_tokens")
        await store.init_tables()
        self.assertEqual(await self.columnas("jacobs_subpipeline_tokens"), COLUMNAS_TOKENS)

    async def test_tokens_tienen_pk_por_hash_e_indice_por_padre(self):
        sql = (
            "SELECT GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS cols "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() "
            "AND TABLE_NAME='jacobs_subpipeline_tokens' AND INDEX_NAME=%s"
        )
        self.assertEqual((await ada.una_fila(sql, ("PRIMARY",)))["cols"], "token_hash")
        self.assertEqual(
            (await ada.una_fila(sql, ("idx_subpipeline_tokens_padre",)))["cols"],
            "parent_pipeline_id",
        )

    async def test_jacobs_pipelines_gana_parent_y_depth_aun_si_la_tabla_ya_existia(self):
        await ada.ejecutar(
            "ALTER TABLE jacobs_pipelines DROP COLUMN IF EXISTS parent_pipeline_id, "
            "DROP COLUMN IF EXISTS depth"
        )
        await store.init_tables()
        columnas = await self.columnas("jacobs_pipelines")
        self.assertIn("parent_pipeline_id", columnas)
        self.assertIn("depth", columnas)

    async def test_init_tables_es_idempotente_con_el_contrato(self):
        await store.init_tables()
        await store.init_tables()
        self.assertEqual(await self.columnas("jacobs_subpipeline_tokens"), COLUMNAS_TOKENS)
        self.assertEqual((await self.columnas("jacobs_pipelines")).count("depth"), 1)

    async def test_pipeline_create_conserva_parent_y_depth(self):
        pid = str(uuid.uuid4())
        ahora = time.time()
        await store.pipeline_create(Pipeline(
            pipeline_id=pid, name="hijo", invoked_by="ada", mode="dry_run",
            status=PipelineStatus.completed, parent_pipeline_id="padre-x", depth=1,
            created_at=ahora, updated_at=ahora,
        ))
        leido = await store.pipeline_get(pid)
        self.assertEqual((leido.parent_pipeline_id, leido.depth), ("padre-x", 1))


if __name__ == "__main__":
    unittest.main()
