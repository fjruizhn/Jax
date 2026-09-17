"""E-03 (2026-09-16): get_motor_governance() trae las facetas ACTIVAS de la
tabla `facet`, la misma foto que capabilities y motors (una sola consulta por
build). Contra MariaDB real: el esquema lo crean las migraciones de
jax-platform en el job jacobs-gobernanza-db.

`facet` es un catálogo (7 filas en producción, 2026-09-16): la consulta por
status sin índice es un recorrido de 7 filas, declarado en DEUDA.md.
"""
from __future__ import annotations

import os
import unittest

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import store  # noqa: E402

_CLAVE = "zz_test_facet_deshabilitada"


class FacetasDeGobernanzaDBTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)

    async def asyncTearDown(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM facet WHERE `key`=%s", (_CLAVE,))
            await conn.commit()

    async def test_las_facetas_son_las_activas_de_la_tabla(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT `key` FROM facet WHERE status = 'active'")
                esperadas = frozenset(k for (k,) in await cur.fetchall())
        gobernanza = await store.get_motor_governance()
        self.assertEqual(gobernanza["facets"], esperadas)
        self.assertIn("hipatia", gobernanza["facets"])

    async def test_una_faceta_deshabilitada_no_entra(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO facet (`key`, display_name, transport, status) VALUES (%s, 'test', 'ollama', 'disabled')",
                    (_CLAVE,))
            await conn.commit()
        gobernanza = await store.get_motor_governance()
        self.assertNotIn(_CLAVE, gobernanza["facets"])
