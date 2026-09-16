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


class _Tripwire(unittest.TestCase):
    """Comprobaciones sobre el código fuente de `jax/core/main.py`.

    Es un módulo de arranque interactivo (voz, muscles, REPL) que no se puede
    instanciar en un test sin montar medio JAX. Se verifica sobre el AST, que es
    el mismo estilo que ya usan los tripwires de este repo: lo que se afirma es
    una propiedad del código, y si alguien la revierte, esto se pone rojo.
    """

    @classmethod
    def setUpClass(cls):
        cls.fuente = (RAIZ / "jax" / "core" / "main.py").read_text(encoding="utf-8")
        cls.arbol = ast.parse(cls.fuente)

    def test_run_task_declara_si_la_tarea_salio_bien(self):
        fn = next((n for n in ast.walk(self.arbol)
                   if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_task"), None)
        self.assertIsNotNone(fn, "run_task desapareció")
        self.assertIsNotNone(fn.returns, "run_task ya no declara tipo de retorno")
        self.assertEqual(ast.unparse(fn.returns), "bool")
        devoluciones = {ast.unparse(n.value) for n in ast.walk(fn)
                        if isinstance(n, ast.Return) and n.value is not None}
        self.assertIn("True", devoluciones)
        self.assertIn("False", devoluciones, "run_task ya no señala el fallo")

    def test_una_tarea_fallida_sale_con_codigo_distinto_de_cero(self):
        self.assertIn("if not asyncio.run(run_task(", self.fuente)
        self.assertIn("sys.exit(1)", self.fuente)

    def test_el_aviso_de_degradacion_viaja_con_el_prompt(self):
        """Un aviso solo al arranque se pierde en una sesión de horas."""
        self.assertIn('_MARCA_DEGRADADO = ""', self.fuente)
        self.assertIn('_MARCA_DEGRADADO = "[sin gobernanza] "', self.fuente)
        self.assertIn('input(f"\\n{_MARCA_DEGRADADO}> ")', self.fuente)

    def test_el_turno_declara_cuando_responde_sin_memoria(self):
        self.assertIn("if similares is None:", self.fuente)
        self.assertIn("memoria no disponible", self.fuente)

    def test_health_check_tiene_consumidor(self):
        """Existía y no lo llamaba nadie: un control sin consumidor."""
        self.assertIn("await db.health_check()", self.fuente)
        self.assertIn("MEMORIA DEGRADADA", self.fuente)


if __name__ == "__main__":
    unittest.main()
