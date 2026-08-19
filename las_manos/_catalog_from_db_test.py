#!/usr/bin/env python3
"""MotorCatalog.from_db() — lee motor/capability/capability_motor de la DB
compartida (jax_memory) en vez de config.toml (R4 — motor desacoplado de
faceta). Corre contra la DB real de desarrollo -- mismo criterio que
credential_resolver.py, sin mock de DB (aiomysql no tiene un modo in-memory
liviano establecido en este repo).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_catalog_from_db_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.catalog import MotorCatalog


class CatalogFromDbTest(unittest.IsolatedAsyncioTestCase):
    async def test_from_db_carga_kimi_y_ada(self):
        catalog = await MotorCatalog.from_db()
        kimi = catalog.get_motor("kimi")
        assert kimi is not None, "kimi no cargó desde DB"
        assert kimi.transport == "http_openai_compat", kimi.transport
        assert kimi.provider_id == "moonshot", kimi.provider_id
        assert kimi.model == "kimi-k3", kimi.model  # model reusa el campo existente (no model_id nuevo)
        assert kimi.max_tokens == 8000, kimi.max_tokens
        assert kimi.enabled is True

    async def test_from_db_carga_capability_con_allowed_motors_en_orden(self):
        catalog = await MotorCatalog.from_db()
        cap = catalog.get_capability("generate")
        assert cap is not None, "capability 'generate' no cargó desde DB"
        assert cap.allowed_motors == ["kimi", "ada"], cap.allowed_motors

    async def test_from_db_capability_sin_thot_no_incluye_motor_inexistente(self):
        catalog = await MotorCatalog.from_db()
        cap = catalog.get_capability("critique")
        assert cap is not None
        assert "thot" not in cap.allowed_motors, cap.allowed_motors
        assert cap.allowed_motors == ["ada"], cap.allowed_motors


if __name__ == "__main__":
    unittest.main(verbosity=2)
