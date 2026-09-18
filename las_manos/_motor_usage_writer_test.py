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
import uuid

# T4 (2026-08-22, auditoria usage_writer): setdefault() no pisa un
# JAX_DB_NAME ya exportado -- si alguien sourcea /etc/jax/.env (JAX_DB_NAME=
# jax_memory, prod) ANTES de correr este archivo, este test escribe filas
# reales contra la DB real en silencio. Pasó de verdad esta sesión: la fila
# huérfana tenant_id=77/tokens 1000-500 en axioma_usage es exactamente este
# test corrido así. Fail loud en vez de fail silent.
from base_de_test import (  # noqa: E402
    exigir_base_de_test,
    nombre_base_de_test,
)

exigir_base_de_test()

# Respaldo de uso aislado (Task 7, 2026-09-15). El conftest.py de la raiz ya lo
# hace para toda la suite, pero este archivo esta escrito para correrse SOLO
# (`python <archivo>`, ver el docstring de arriba) y ahi pytest no carga ningun
# conftest. Sin esto, un fallo de la DB encola en /srv/jax-data/usage-spool --
# el directorio REAL del que jax-platform drena e inserta en axioma_usage.
# setdefault, no asignacion: bajo pytest el conftest ya gano y no se lo pisa.
import tempfile  # noqa: E402
os.environ.setdefault(
    "JAX_USAGE_SPOOL_DIR", tempfile.mkdtemp(prefix="jax-test-respaldo-uso-"))

from motor_registry import usage_writer

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
        db=os.getenv("JAX_DB_NAME", nombre_base_de_test()), autocommit=True,
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


_COLUMNAS = "tokens_in, tokens_out, cost_usd, model, facet, status, job_id, tenant_id, user_id"


class MotorUsageWriterTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.modelo = _ModeloPropio("moonshot")
        self.job_id = str(uuid.uuid4())  # único: antes eran literales job-1..4

    async def asyncTearDown(self):
        await self.modelo.limpiar()

    async def test_record_motor_usage_calcula_costo_real(self):
        await self.modelo.sembrar(0.95, 4.00)
        await usage_writer.record_motor_usage(
            "1", "77", "kimi", "moonshot", self.modelo.model_id, 1000, 500,
            job_id=self.job_id, status="completed",
        )
        filas = await self.modelo.filas_de_uso(_COLUMNAS)
        self.assertEqual(len(filas), 1, filas)
        tokens_in, tokens_out, cost_usd, model, facet, status, job_id, tenant_id, user_id = filas[0]
        self.assertEqual(tokens_in, 1000)
        self.assertEqual(tokens_out, 500)
        expected = (1000 * 0.95 + 500 * 4.00) / 1_000_000
        self.assertAlmostEqual(float(cost_usd), expected, places=9)
        self.assertEqual(status, "completed")
        self.assertEqual(job_id, self.job_id)

    async def test_record_motor_usage_registra_desenlace_failed(self):
        """T1.b (2026-08-22, auditoria usage_writer): un job fallido gastó
        tokens igual de reales -- la fila debe existir y decir 'failed', no
        quedar indistinguible de un éxito ni desaparecer."""
        await self.modelo.sembrar()
        await usage_writer.record_motor_usage(
            "1", "77", "kimi", "moonshot", self.modelo.model_id, 300, 120,
            job_id=self.job_id, status="failed",
        )
        filas = await self.modelo.filas_de_uso(_COLUMNAS)
        self.assertEqual(len(filas), 1, filas)
        self.assertEqual(filas[0][5], "failed")
        self.assertEqual(filas[0][6], self.job_id)

    async def test_record_motor_usage_sin_identidad_escribe_con_null_y_loguea(self):
        """T1.c: antes esto retornaba en silencio (fail-open puro) -- un
        dispatch sin identidad sigue gastando dinero real. Ahora escribe con
        tenant_id/user_id NULL (distinguible, filtrable) y loguea WARNING,
        nunca en silencio."""
        await self.modelo.sembrar()
        with self.assertLogs("motor_registry.usage_writer", level="WARNING") as cm:
            await usage_writer.record_motor_usage(
                None, None, "kimi", "moonshot", self.modelo.model_id, 100, 50,
                job_id=self.job_id, status="completed",
            )
        assert any("sin identidad" in m for m in cm.output), cm.output
        filas = await self.modelo.filas_de_uso(_COLUMNAS)
        self.assertEqual(len(filas), 1, filas)  # la fila SÍ se escribió
        self.assertEqual(filas[0][6], self.job_id)
        self.assertIsNone(filas[0][7])  # tenant_id
        self.assertIsNone(filas[0][8])  # user_id

    async def test_record_motor_usage_reintenta_y_escala_a_error_si_agota_intentos(self):
        """T1.d: el except que traga pasa a reintentar (2 intentos) y, si
        agota, escala a logger.error (no solo warning) -- máxima visibilidad
        posible desde este módulo, ver justificación en el código sobre por
        qué no jacobs_events (sin pipeline_id en este scope)."""
        # Desde la cola durable (T7, 2026-09-15) agotar los intentos ENCOLA y
        # loguea INFO; el ERROR es solo cuando tampoco se puede encolar. Este
        # test quedo afirmando lo de antes y fallaba ya en eb72e78 (visto en la
        # ronda del pool, 2026-09-17): se simula tambien la cola sin lugar. Y la
        # base caida se simula en el POOL, que es por donde conecta el escritor.
        import contextlib
        import unittest.mock as mock

        pedidos = []

        def pool_caido(desechable=False):
            pedidos.append(1)

            @contextlib.asynccontextmanager
            async def _ctx():
                raise RuntimeError("DB caída")
                yield  # pragma: no cover

            return _ctx()

        async def cola_sin_lugar(_fila):
            return None

        with mock.patch("jacobs.store.conexion", pool_caido), \
                mock.patch("motor_registry.usage_writer.encolar_uso", cola_sin_lugar), \
                mock.patch("motor_registry.usage_writer.asyncio.sleep", mock.AsyncMock()):
            with self.assertLogs("motor_registry.usage_writer", level="ERROR") as cm:
                await usage_writer.record_motor_usage(
                    "1", "77", "kimi", "moonshot", self.modelo.model_id, 100, 50,
                    job_id=self.job_id, status="completed",
                )
        self.assertEqual(len(pedidos), usage_writer._WRITE_MAX_ATTEMPTS)
        # Merge 2026-09-17: el job_id del test es un uuid propio (no el literal
        # "job-4" de antes), para no pisar filas de otra corrida.
        assert any(self.job_id in m and "RuntimeError" in m for m in cm.output), cm.output


if __name__ == "__main__":
    unittest.main(verbosity=2)
