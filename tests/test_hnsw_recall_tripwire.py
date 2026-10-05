#!/usr/bin/env python3
"""El aviso de que hay que volver a medir el recall del indice vectorial.

POR QUE EXISTE. Desde jax#128 la busqueda semantica usa el indice HNSW, que es
APROXIMADO: `mhnsw_ef_search=400` da 93,3 % de recall@5 **medido sobre 1.607
filas**. Ese numero no es una propiedad del sistema, es una propiedad de ESE
tamano: el recall de un grafo HNSW se degrada al crecer la tabla, y lo hace en
silencio -- ninguna consulta falla, simplemente empiezan a faltar recuerdos.

DEUDA.md decia "hay que volver a medirlo cuando `messages` crezca un orden de
magnitud". Una condicion asi no la vigila nadie: no tiene fecha, no tiene dueno
y no hay nada que la mire. Desde 2026-10-05 lo ejecuta vector-health y falla la unidad.

La unidad de vector-health sale con error (no bloquea la extraccion ni escribe nada). Lo que
hay que hacer cuando aparezca esta en el propio mensaje.
"""
from __future__ import annotations

import logging
import unittest
from unittest.mock import AsyncMock, patch

from jax.memory import embedding_worker, recall_tripwire as tripwire


class TripwireRecallTest(unittest.IsolatedAsyncioTestCase):
    async def test_avisa_cuando_messages_crecio_un_orden_de_magnitud(self):
        contar = AsyncMock(return_value=tripwire.FILAS_AL_MEDIR_RECALL * 10)

        with self.assertLogs(tripwire.logger, level="WARNING") as capturado:
            self.assertTrue(await tripwire.recall_requiere_remedicion(contar))

        mensaje = "\n".join(capturado.output)
        self.assertIn("recall", mensaje.lower())
        self.assertIn("JAX_MEMORY_HNSW_EF_SEARCH", mensaje,
                      "el aviso tiene que decir QUE se toca, no solo que algo pasa")

    async def test_no_avisa_mientras_la_tabla_sigue_en_el_mismo_orden(self):
        contar = AsyncMock(return_value=tripwire.FILAS_AL_MEDIR_RECALL * 3)

        with self.assertNoLogs(tripwire.logger, level="WARNING"):
            self.assertFalse(await tripwire.recall_requiere_remedicion(contar))

    async def test_un_fallo_al_contar_no_se_traga(self):
        # CAMBIO 2026-10-05: el tripwire ahora vive en vector-health, que es el chequeo de salud.
        # Un chequeo que no puede mirar no puede dar verde: el error sube y la unidad falla.
        contar = AsyncMock(side_effect=RuntimeError("base caida"))
        with self.assertRaises(RuntimeError):
            await tripwire.recall_requiere_remedicion(contar)
        with self.assertRaises(RuntimeError):
            await tripwire.recall_requiere_remedicion(AsyncMock(return_value=None))


class _PoolFalso:
    """aiomysql.create_pool falso: `faltan` generaciones sin vector y `filas` en messages."""
    def __init__(self, faltan, filas):
        self.faltan, self.filas = faltan, filas
    def acquire(self):
        pool = self
        class Cur:
            sql = ""
            async def execute(self, sql, args=None): self.sql = sql
            async def fetchone(self): return (pool.filas,) if "FROM messages" in self.sql else (pool.faltan,)
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
        class Conn:
            def cursor(self): return Cur()
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
        return Conn()
    def close(self): pass
    async def wait_closed(self): pass


class VectorHealthLlamaAlTripwireTest(unittest.IsolatedAsyncioTestCase):
    """Recableado (2026-10-05): el tripwire dejo de ser codigo muerto; vector-health lo ejecuta."""

    async def _correr(self, faltan, filas):
        pool = _PoolFalso(faltan, filas)
        with patch.dict("os.environ", {"JAX_DB_HOST": "h", "JAX_DB_PORT": "1"}), \
             patch.object(embedding_worker, "resolver_identidad",
                          AsyncMock(return_value=embedding_worker.EmbeddingSpaceIdentity("b9-v1", "ollama", "m", "d", 2, "unit", "cosine"))), \
             patch.object(embedding_worker, "cerrar_cliente_http", AsyncMock()), \
             patch.object(embedding_worker.aiomysql, "create_pool", AsyncMock(return_value=pool)):
            return await embedding_worker.run_b9_vector_health()

    async def test_sano_sale_cero(self):
        self.assertEqual(await self._correr(0, tripwire.FILAS_AL_MEDIR_RECALL), 0)

    async def test_si_hay_que_remedir_el_recall_sale_distinto_de_cero(self):
        self.assertEqual(await self._correr(0, tripwire.FILAS_AL_MEDIR_RECALL * 10), 1)

    async def test_vectores_faltantes_siguen_dando_rojo(self):
        self.assertEqual(await self._correr(3, 10), 1)


if __name__ == "__main__":
    unittest.main()
