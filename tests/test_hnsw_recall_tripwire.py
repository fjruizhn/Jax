#!/usr/bin/env python3
"""El aviso de que hay que volver a medir el recall del indice vectorial.

POR QUE EXISTE. Desde jax#128 la busqueda semantica usa el indice HNSW, que es
APROXIMADO: `mhnsw_ef_search=400` da 93,3 % de recall@5 **medido sobre 1.607
filas**. Ese numero no es una propiedad del sistema, es una propiedad de ESE
tamano: el recall de un grafo HNSW se degrada al crecer la tabla, y lo hace en
silencio -- ninguna consulta falla, simplemente empiezan a faltar recuerdos.

DEUDA.md decia "hay que volver a medirlo cuando `messages` crezca un orden de
magnitud". Una condicion asi no la vigila nadie: no tiene fecha, no tiene dueno
y no hay nada que la mire. Esto la convierte en un WARNING en el journal del
worker, que corre cada 20 minutos.

No falla nada ni bloquea: avisa. Lo que hay que hacer cuando aparezca esta en el
propio mensaje.
"""
from __future__ import annotations

import logging
import unittest
from unittest.mock import AsyncMock

from jax.memory import worker


class TripwireRecallTest(unittest.IsolatedAsyncioTestCase):
    async def test_avisa_cuando_messages_crecio_un_orden_de_magnitud(self):
        db = AsyncMock()
        db.contar_filas = AsyncMock(return_value=worker.FILAS_AL_MEDIR_RECALL * 10)

        with self.assertLogs(worker.logger, level="WARNING") as capturado:
            await worker._avisar_si_hay_que_remedir_recall(db)

        mensaje = "\n".join(capturado.output)
        self.assertIn("recall", mensaje.lower())
        self.assertIn("JAX_MEMORY_HNSW_EF_SEARCH", mensaje,
                      "el aviso tiene que decir QUE se toca, no solo que algo pasa")

    async def test_no_avisa_mientras_la_tabla_sigue_en_el_mismo_orden(self):
        db = AsyncMock()
        db.contar_filas = AsyncMock(return_value=worker.FILAS_AL_MEDIR_RECALL * 3)

        with self.assertNoLogs(worker.logger, level="WARNING"):
            await worker._avisar_si_hay_que_remedir_recall(db)

    async def test_un_fallo_al_contar_no_tumba_la_corrida_del_worker(self):
        # El tripwire es una cortesia: si la consulta falla, el worker tiene que
        # seguir procesando conversaciones. Un aviso que rompe lo que vigila es
        # peor que no tenerlo.
        db = AsyncMock()
        db.contar_filas = AsyncMock(side_effect=RuntimeError("base caida"))

        with self.assertLogs(worker.logger, level="ERROR"):
            await worker._avisar_si_hay_que_remedir_recall(db)


if __name__ == "__main__":
    unittest.main()
