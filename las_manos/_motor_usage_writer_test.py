#!/usr/bin/env python3
# jax/las_manos/_motor_usage_writer_test.py
"""motor_registry escribe axioma_usage directo (misma DB jax_memory, las_manos
ya se conecta ahi via credential_resolver). Costo: mismo lookup contra
`model` que usa jax-platform, espejado aca (mismo criterio que
credential_resolver.py/model_catalog.py -- repos independientes, sin
paquete compartido). Corre con:
  PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python _motor_usage_writer_test.py

Nota (verificado 2026-08-10 contra el schema real): tenant_id/user_id en
`axioma_usage` son INT(11) y la DB corre con STRICT_TRANS_TABLES -- un
tenant_id no numerico como "test-tenant" hace que el INSERT falle con
"Incorrect integer value" (probado a mano contra jax_memory_test). Por eso
este test usa un tenant_id numerico ("77"), igual que en produccion (ver
jacobs/models.py Pipeline.tenant_id, siempre el id real como string).
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from motor_registry import usage_writer


async def _seed_priced_model(provider_id, model_id, price_in, price_out):
    import aiomysql
    conn = await aiomysql.connect(
        host=os.getenv("JAX_DB_HOST", "localhost"), port=int(os.getenv("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory_test"), autocommit=True,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO model (provider_id, model_id, status, source, source_checked_at, "
                "price_input_per_1m_usd, price_output_per_1m_usd) "
                "VALUES (%s, %s, 'available', 'manual', NOW(), %s, %s) "
                "ON DUPLICATE KEY UPDATE price_input_per_1m_usd=%s, price_output_per_1m_usd=%s",
                (provider_id, model_id, price_in, price_out, price_in, price_out),
            )
    finally:
        conn.close()


async def _fetch_last_usage_row():
    import aiomysql
    conn = await aiomysql.connect(
        host=os.getenv("JAX_DB_HOST", "localhost"), port=int(os.getenv("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory_test"), autocommit=True,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT tokens_in, tokens_out, cost_usd, model, facet FROM axioma_usage ORDER BY id DESC LIMIT 1"
            )
            return await cur.fetchone()
    finally:
        conn.close()


class MotorUsageWriterTest(unittest.IsolatedAsyncioTestCase):
    async def test_record_motor_usage_calcula_costo_real(self):
        await _seed_priced_model("moonshot", "kimi-k2.7-code", 0.95, 4.00)
        await usage_writer.record_motor_usage(
            "1", "77", "kimi", "moonshot", "kimi-k2.7-code", 1000, 500,
        )
        row = await _fetch_last_usage_row()
        tokens_in, tokens_out, cost_usd, model, facet = row
        self.assertEqual(tokens_in, 1000)
        self.assertEqual(tokens_out, 500)
        expected = (1000 * 0.95 + 500 * 4.00) / 1_000_000
        self.assertAlmostEqual(float(cost_usd), expected, places=9)

    async def test_record_motor_usage_sin_identidad_no_escribe(self):
        row_before = await _fetch_last_usage_row()
        await usage_writer.record_motor_usage(None, None, "kimi", "moonshot", "kimi-k2.7-code", 100, 50)
        row_after = await _fetch_last_usage_row()
        self.assertEqual(row_before, row_after)  # fail-soft: sin identidad, no escribe nada


if __name__ == "__main__":
    unittest.main(verbosity=2)
