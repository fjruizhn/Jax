"""Descartar pipelines: columnas e índices contra la base de TEST (spec
2026-09-22-descartar-pipelines §3). `store.get_pool`/`store.init_schema`
del brief no existen en este repo: los nombres reales son
`store.conexion()`/`store.init_tables()`, los mismos que usa
`tests/test_jacobs_reaper_cas_db.py`."""
from __future__ import annotations

import unittest

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import store  # noqa: E402


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
