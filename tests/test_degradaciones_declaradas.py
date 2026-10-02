#!/usr/bin/env python3
"""Cuando JAX funciona a medias, tiene que DECIRLO.

Las cuatro decisiones que Fernando resolvió el 2026-09-16, cerrando la
auditoría P10. Ninguna era un error de programación: las cuatro eran sitios
donde el sistema seguía andando en un estado degradado **sin declararlo**, que
es la forma callada de la certeza fabricada (Principios V y VIII).

  1. `search_similar_messages` devolvía `[]` ante un fallo, indistinguible de
     «no hay nada parecido»: la Mesa respondía sin memoria creyendo que no
     había memoria.
  2. `run_task` escribía el error en el archivo de resultado pero el proceso
     salía con código 0: cualquier automatización daba la tarea por buena.
  3. Si `facet_binding` no se podía leer, el REPL seguía con el `config.toml`
     —— la fuente stale conocida, y saltándose la gobernanza del modelo ——
     avisando una sola vez, al arranque de una sesión de horas.
  4. `Router._classify` caía a la faceta por defecto sin una línea de log: con
     el clasificador roto, el 100 % del ruteo degradaba para siempre en
     silencio.

Y `MemoryDB.health_check()`, que no lo llamaba nadie, ahora lo llama el REPL.

T16 (2026-10-02): el REPL y `jax --task` se retiraron; se fueron con ellos las
comprobaciones sobre el codigo de jax/core/main.py (puntos 2 y 3 y el consumidor
de health_check). Quedan el Router (4) y la memoria (1).
"""
from __future__ import annotations

import ast
import asyncio
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from jax.core.router import Router  # noqa: E402
from jax.memory import db as dbmod  # noqa: E402


class RouterTest(unittest.TestCase):
    def _router_con_clasificador_roto(self) -> Router:
        clasificador = mock.MagicMock()
        clasificador.invoke = mock.AsyncMock(side_effect=OSError("ollama caido"))
        return Router(classifier=clasificador)

    def test_un_clasificador_caido_deja_rastro(self):
        r = self._router_con_clasificador_roto()
        with self.assertLogs("jax.router", level=logging.WARNING) as capturado:
            self.assertIsNone(asyncio.run(r._classify("hola")))
        self.assertIn("clasificador del router caido", "\n".join(capturado.output))

    def test_se_cuentan_los_fallos(self):
        r = self._router_con_clasificador_roto()
        for _ in range(3):
            asyncio.run(r._classify("hola"))
        self.assertEqual(r._fallos_clasificador, 3)

    def test_no_inunda_el_log(self):
        """Un clasificador en bucle no puede llenar el log: un log inundado se
        deja de leer, que es otra forma de callar."""
        r = self._router_con_clasificador_roto()
        with self.assertLogs("jax.router", level=logging.WARNING) as capturado:
            for _ in range(10):
                asyncio.run(r._classify("hola"))
        self.assertEqual(len(capturado.output), 1, "aviso repetido en cada turno")

    def test_el_router_sigue_sin_lanzar(self):
        """El contrato de siempre no cambia: el router NUNCA lanza."""
        r = self._router_con_clasificador_roto()
        self.assertIsNone(asyncio.run(r._classify("hola")))


class MemoriaTest(unittest.TestCase):
    def test_una_busqueda_fallida_devuelve_None_no_lista_vacia(self):
        m = dbmod.MemoryDB()
        m.pool = mock.MagicMock()
        m.pool.acquire = mock.MagicMock(side_effect=OSError("la base no responde"))
        m.get_embedding = mock.AsyncMock(return_value=[0.1, 0.2])
        resultado = asyncio.run(m.search_similar_messages("hola"))
        self.assertIsNone(
            resultado,
            "un fallo de búsqueda se presentó como «no hay nada parecido»")


if __name__ == "__main__":
    unittest.main()
