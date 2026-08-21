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
        # Task 4 agregó jax_local a 'generate' (priority 2, detrás de kimi/ada).
        assert cap.allowed_motors == ["kimi", "ada", "jax_local"], cap.allowed_motors

    async def test_from_db_capability_critique_incluye_thot_en_orden(self):
        catalog = await MotorCatalog.from_db()
        cap = catalog.get_capability("critique")
        assert cap is not None
        # Task 8 agregó thot a 'critique' (priority 0, delante de ada).
        assert "thot" in cap.allowed_motors, cap.allowed_motors
        assert cap.allowed_motors == ["thot", "ada"], cap.allowed_motors

    async def test_from_db_has_tool_access_jax_local_true_kimi_false(self):
        """T1 (diagnóstico pipeline 19ad2c42-cdf): has_tool_access vivía solo
        como `if motor == "jax_local"` en worker.py:488, sin fuente
        consultable. Ahora es columna en `motor` -- este test confirma que
        MotorCatalog la lee (no la vuelve a hardcodear en otro lado)."""
        catalog = await MotorCatalog.from_db()
        jax_local = catalog.get_motor("jax_local")
        assert jax_local is not None
        assert jax_local.has_tool_access is True, jax_local
        kimi = catalog.get_motor("kimi")
        assert kimi is not None
        # kimi tiene filas en capability_motor para file_write/file_read
        # (ronda 7) pero NO recibe el catálogo de tools (worker.py:488,
        # GAP2 Fase1, posterior) -- esta es exactamente la contradicción
        # diagnosticada en T2 de la sesión anterior, y has_tool_access=False
        # es la fuente de verdad que la hace explícita.
        assert kimi.has_tool_access is False, kimi


if __name__ == "__main__":
    unittest.main(verbosity=2)
