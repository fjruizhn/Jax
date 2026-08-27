#!/usr/bin/env python3
"""check_facet_admission() -- gobernanza de nivel FACET para callers que
NO tienen concepto de capability (Mesa web -- ver docs/superpowers/specs/
2026-08-27-http-facets-motor-policy-governance-design.md, seccion 3, la
correccion sobre por que esto NO reusa check_capability_admission()).

Corre contra la DB real de desarrollo, mismo criterio que
_catalog_from_db_test.py -- sin mock de aiomysql. Requiere que la Task 3
(migracion de facet.allowed_callers) ya haya corrido contra esta DB.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/motor_registry/_facet_policy_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_DB_NAME", os.environ.get("JAX_DB_NAME", "jax_memory"))

from motor_registry.facet_policy import check_facet_admission


class CheckFacetAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_caller_autorizado_pasa(self):
        # hipatia se siembra con allowed_callers incluyendo "jacobs" (Task 3).
        allowed, reason = await check_facet_admission("jacobs", "hipatia")
        self.assertTrue(allowed, reason)

    async def test_caller_no_autorizado_rechaza(self):
        allowed, reason = await check_facet_admission("caller_fantasma", "hipatia")
        self.assertFalse(allowed)
        self.assertIn("no autorizado", reason)

    async def test_facet_sin_allowed_callers_configurado_rechaza(self):
        # jax_local/kimi/hyde quedan NULL a propósito (fuera de alcance,
        # spec seccion 3) -- NULL debe denegar, no "lista vacia = todos".
        allowed, reason = await check_facet_admission("jacobs", "jax_local")
        self.assertFalse(allowed)
        self.assertIn("no configurado", reason)

    async def test_facet_inexistente_rechaza(self):
        allowed, reason = await check_facet_admission("jacobs", "no_existe")
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
