#!/usr/bin/env python3
# jax/jacobs/_usage_writer_test.py
"""jacobs/usage_writer.py escribe axioma_usage directo (misma DB jax_memory),
mismo patron que las_manos/_motor_usage_writer_test.py -- espejado aca porque
jacobs no importa motor_registry (no esta en su sys.path standalone). Corre
con:
  PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos .venv/bin/python jacobs/_usage_writer_test.py

Nota (mismo hallazgo que _motor_usage_writer_test.py, 2026-08-10): tenant_id/
user_id en `axioma_usage` son INT(11) bajo STRICT_TRANS_TABLES -- este test
usa un tenant_id numerico ("77"), igual que en produccion.
"""
from __future__ import annotations

import os
import unittest

# T4 (2026-08-22, auditoria usage_writer): mismo guard que
# las_manos/_motor_usage_writer_test.py -- setdefault() no pisa un
# JAX_DB_NAME ya exportado, y ese silencio ya escribió una fila real de
# prueba en axioma_usage esta sesión. Fail loud en vez de fail silent.
_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado (¿sourceaste "
        f"/etc/jax/.env?) -- este archivo escribe filas reales a esa DB. "
        f"Unset JAX_DB_NAME antes de correr este test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs import usage_writer


async def _seed_priced_model(provider_id, model_id, price_in, price_out):
    import aiomysql
    _host = os.environ.get("JAX_DB_HOST")
    _port = os.environ.get("JAX_DB_PORT")
    if not _host or not _port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    conn = await aiomysql.connect(
        host=_host, port=int(_port),
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
    _host = os.environ.get("JAX_DB_HOST")
    _port = os.environ.get("JAX_DB_PORT")
    if not _host or not _port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    conn = await aiomysql.connect(
        host=_host, port=int(_port),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory_test"), autocommit=True,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT tokens_in, tokens_out, cost_usd, model, facet, request_type "
                "FROM axioma_usage ORDER BY id DESC LIMIT 1"
            )
            return await cur.fetchone()
    finally:
        conn.close()


class DirectUsageWriterTest(unittest.IsolatedAsyncioTestCase):
    async def test_record_direct_usage_calcula_costo_real_y_marca_request_type_pipeline(self):
        await _seed_priced_model("deepseek", "deepseek-v4-flash", 0.30, 1.20)
        await usage_writer.record_direct_usage(
            "1", "77", "jekyll", "deepseek", "deepseek-v4-flash", 1000, 500,
        )
        row = await _fetch_last_usage_row()
        tokens_in, tokens_out, cost_usd, model, facet, request_type = row
        self.assertEqual(tokens_in, 1000)
        self.assertEqual(tokens_out, 500)
        expected = (1000 * 0.30 + 500 * 1.20) / 1_000_000
        self.assertAlmostEqual(float(cost_usd), expected, places=9)
        self.assertEqual(request_type, "pipeline")

    async def test_record_direct_usage_sin_identidad_escribe_con_null_y_loguea(self):
        """T1.c (2026-08-22, auditoria usage_writer): mismo bug que
        motor_registry/usage_writer.py -- antes retornaba en silencio."""
        with self.assertLogs("jacobs.usage_writer", level="WARNING") as cm:
            await usage_writer.record_direct_usage(None, None, "jekyll", "deepseek", "deepseek-v4-flash", 100, 50)
        assert any("sin identidad" in m for m in cm.output), cm.output
        row = await _fetch_last_usage_row()
        tokens_in, tokens_out, cost_usd, model, facet, request_type = row
        self.assertEqual(tokens_in, 100)
        self.assertEqual(facet, "jekyll")


if __name__ == "__main__":
    unittest.main(verbosity=2)
