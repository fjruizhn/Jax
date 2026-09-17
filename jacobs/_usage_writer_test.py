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
import uuid

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

from jacobs import store, usage_writer

try:
    from db_connect_config import db_connect_timeout_seconds
except ImportError:
    from jax.core.db_connect_config import db_connect_timeout_seconds


# R46 (R38 fix round 3, 2026-09-17): este archivo no depende de filas ajenas
# ni deja las suyas. Antes leía "la última fila" de axioma_usage (cualquier
# escritura concurrente de otra sesión la cambiaba), nunca borraba lo que
# escribía y pisaba el precio de un modelo compartido. Ahora cada test:
# - siembra un modelo PROPIO `r46-<uuid>` (nunca toca una fila de `model` ajena),
# - lee sólo las filas de axioma_usage con ese modelo y con id mayor al máximo
#   anterior (rango por PRIMARY, sin recorrer la tabla),
# - borra sus filas de axioma_usage y su modelo al terminar.
async def _conexion():
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
    return await aiomysql.connect(
        host=_host, port=int(_port),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory_test"), autocommit=True,
        connect_timeout=db_connect_timeout_seconds(),
    )


async def _ejecutar(sql, params=(), filas=False):
    conn = await _conexion()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchall() if filas else cur.rowcount
    finally:
        conn.close()


class _ModeloPropio:
    """Un modelo con precio sembrado para ESTE test y la marca de sus filas."""

    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        self.model_id = f"r46-{uuid.uuid4().hex[:12]}"
        self.id_previo = 0

    async def sembrar(self, price_in=None, price_out=None):
        (maximo,), = await _ejecutar("SELECT COALESCE(MAX(id), 0) FROM axioma_usage", filas=True)
        self.id_previo = int(maximo)
        await _ejecutar(
            "INSERT INTO model (provider_id, model_id, status, source, source_checked_at, "
            "price_input_per_1m_usd, price_output_per_1m_usd) "
            "VALUES (%s, %s, 'available', 'manual', NOW(), %s, %s)",
            (self.provider_id, self.model_id, price_in, price_out),
        )

    async def filas_de_uso(self, columnas: str):
        return await _ejecutar(
            f"SELECT {columnas} FROM axioma_usage WHERE id > %s AND model = %s ORDER BY id",
            (self.id_previo, self.model_id), filas=True,
        )

    async def limpiar(self):
        await _ejecutar("DELETE FROM axioma_usage WHERE id > %s AND model = %s", (self.id_previo, self.model_id))
        await _ejecutar("DELETE FROM model WHERE provider_id = %s AND model_id = %s", (self.provider_id, self.model_id))
        restantes = await _ejecutar(
            "SELECT (SELECT COUNT(*) FROM axioma_usage WHERE id > %s AND model = %s), "
            "(SELECT COUNT(*) FROM model WHERE provider_id = %s AND model_id = %s)",
            (self.id_previo, self.model_id, self.provider_id, self.model_id), filas=True,
        )
        assert restantes[0] == (0, 0), f"el test dejó filas propias: {restantes}"


class DirectUsageWriterTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.modelo = _ModeloPropio("deepseek")

    async def asyncTearDown(self):
        await self.modelo.limpiar()
        await store.cerrar_pool()  # usage_writer escribe por el pool del store (R38)

    async def test_record_direct_usage_calcula_costo_real_y_marca_request_type_pipeline(self):
        await self.modelo.sembrar(0.30, 1.20)
        await usage_writer.record_direct_usage(
            "1", "77", "jekyll", "deepseek", self.modelo.model_id, 1000, 500,
        )
        filas = await self.modelo.filas_de_uso("tokens_in, tokens_out, cost_usd, model, facet, request_type")
        self.assertEqual(len(filas), 1, filas)
        tokens_in, tokens_out, cost_usd, model, facet, request_type = filas[0]
        self.assertEqual(tokens_in, 1000)
        self.assertEqual(tokens_out, 500)
        expected = (1000 * 0.30 + 500 * 1.20) / 1_000_000
        self.assertAlmostEqual(float(cost_usd), expected, places=9)
        self.assertEqual(request_type, "pipeline")

    async def test_record_direct_usage_sin_identidad_escribe_con_null_y_loguea(self):
        """T1.c (2026-08-22, auditoria usage_writer): mismo bug que
        motor_registry/usage_writer.py -- antes retornaba en silencio."""
        await self.modelo.sembrar()
        with self.assertLogs("jacobs.usage_writer", level="WARNING") as cm:
            await usage_writer.record_direct_usage(None, None, "jekyll", "deepseek", self.modelo.model_id, 100, 50)
        assert any("sin identidad" in m for m in cm.output), cm.output
        filas = await self.modelo.filas_de_uso("tokens_in, tokens_out, facet, tenant_id, user_id")
        self.assertEqual(len(filas), 1, filas)
        tokens_in, tokens_out, facet, tenant_id, user_id = filas[0]
        self.assertEqual(tokens_in, 100)
        self.assertEqual(facet, "jekyll")
        self.assertIsNone(tenant_id)
        self.assertIsNone(user_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
