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

# T4 (2026-08-22, auditoria usage_writer): setdefault() no pisa un
# JAX_DB_NAME ya exportado -- si alguien sourcea /etc/jax/.env (JAX_DB_NAME=
# jax_memory, prod) ANTES de correr este archivo, este test escribe filas
# reales contra la DB real en silencio. Pasó de verdad esta sesión: la fila
# huérfana tenant_id=77/tokens 1000-500 en axioma_usage es exactamente este
# test corrido así. Fail loud en vez de fail silent.
_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado (¿sourceaste "
        f"/etc/jax/.env?) -- este archivo escribe filas reales a esa DB. "
        f"Unset JAX_DB_NAME antes de correr este test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from motor_registry import usage_writer


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
                "SELECT tokens_in, tokens_out, cost_usd, model, facet, status, job_id, tenant_id, user_id "
                "FROM axioma_usage ORDER BY id DESC LIMIT 1"
            )
            return await cur.fetchone()
    finally:
        conn.close()


class MotorUsageWriterTest(unittest.IsolatedAsyncioTestCase):
    async def test_record_motor_usage_calcula_costo_real(self):
        await _seed_priced_model("moonshot", "kimi-k2.7-code", 0.95, 4.00)
        await usage_writer.record_motor_usage(
            "1", "77", "kimi", "moonshot", "kimi-k2.7-code", 1000, 500,
            job_id="job-1", status="completed",
        )
        row = await _fetch_last_usage_row()
        tokens_in, tokens_out, cost_usd, model, facet, status, job_id, tenant_id, user_id = row
        self.assertEqual(tokens_in, 1000)
        self.assertEqual(tokens_out, 500)
        expected = (1000 * 0.95 + 500 * 4.00) / 1_000_000
        self.assertAlmostEqual(float(cost_usd), expected, places=9)
        self.assertEqual(status, "completed")
        self.assertEqual(job_id, "job-1")

    async def test_record_motor_usage_registra_desenlace_failed(self):
        """T1.b (2026-08-22, auditoria usage_writer): un job fallido gastó
        tokens igual de reales -- la fila debe existir y decir 'failed', no
        quedar indistinguible de un éxito ni desaparecer."""
        await usage_writer.record_motor_usage(
            "1", "77", "kimi", "moonshot", "kimi-k2.7-code", 300, 120,
            job_id="job-2", status="failed",
        )
        row = await _fetch_last_usage_row()
        self.assertEqual(row[5], "failed")
        self.assertEqual(row[6], "job-2")

    async def test_record_motor_usage_sin_identidad_escribe_con_null_y_loguea(self):
        """T1.c: antes esto retornaba en silencio (fail-open puro) -- un
        dispatch sin identidad sigue gastando dinero real. Ahora escribe con
        tenant_id/user_id NULL (distinguible, filtrable) y loguea WARNING,
        nunca en silencio."""
        with self.assertLogs("motor_registry.usage_writer", level="WARNING") as cm:
            await usage_writer.record_motor_usage(
                None, None, "kimi", "moonshot", "kimi-k2.7-code", 100, 50,
                job_id="job-3", status="completed",
            )
        assert any("sin identidad" in m for m in cm.output), cm.output
        row = await _fetch_last_usage_row()
        self.assertEqual(row[6], "job-3")  # la fila SÍ se escribió
        self.assertIsNone(row[7])  # tenant_id
        self.assertIsNone(row[8])  # user_id

    async def test_record_motor_usage_reintenta_y_escala_a_error_si_agota_intentos(self):
        """T1.d: el except que traga pasa a reintentar (2 intentos) y, si
        agota, escala a logger.error (no solo warning) -- máxima visibilidad
        posible desde este módulo, ver justificación en el código sobre por
        qué no jacobs_events (sin pipeline_id en este scope)."""
        import unittest.mock as mock
        with mock.patch("motor_registry.usage_writer.aiomysql.connect", side_effect=RuntimeError("DB caída")):
            with self.assertLogs("motor_registry.usage_writer", level="ERROR") as cm:
                await usage_writer.record_motor_usage(
                    "1", "77", "kimi", "moonshot", "kimi-k2.7-code", 100, 50,
                    job_id="job-4", status="completed",
                )
        assert any("job-4" in m and "RuntimeError" in m for m in cm.output), cm.output


if __name__ == "__main__":
    unittest.main(verbosity=2)
