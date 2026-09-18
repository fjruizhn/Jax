#!/usr/bin/env python3
"""Ruling T6-6 (2026-09-15): el indice de dueño de jacobs_pipelines.

jax-platform lista los pipelines de un dueño con
`WHERE user_id=%s AND tenant_id=%s ORDER BY created_at`: el indice compuesto
(user_id, tenant_id, created_at) cubre el filtro y el orden. Se crea en
init_tables(), que corre en CADA arranque de los procesos que usan Jacobs.

Re-revision de la plataforma (2026-09-15): ese DDL no puede colgar el
arranque ni dejar a Jacobs detras de un metadata lock:
  - `ALGORITHM=INPLACE, LOCK=NONE` explicitos: si MariaDB no puede hacerlo en
    linea, FALLA en vez de caer en silencio a COPY (que bloquea escrituras).
  - `lock_wait_timeout` acotado a 30 s SOLO para ese DDL, y restaurado
    despues: el default de MariaDB es 86400 s, un dia colgado en el arranque
    si una transaccion larga tiene la tabla.
  - Si vence la espera (ER_LOCK_WAIT_TIMEOUT, 1205): ERROR en el log y el
    arranque sigue; el proximo arranque lo reintenta (el chequeo de
    information_schema ve que falta). Cualquier otro error SUBE.

Pruebas puras: un cursor falso, sin DB. La existencia real del indice en una
base vacia la prueba jacobs/_store_indexes_test.py (job facet-health-io).

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_store_indice_duenio.py
"""
from __future__ import annotations

import os
import unittest

import aiomysql

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from jacobs import store  # noqa: E402

_ENTRADA = ("jacobs_pipelines", "idx_jacobs_pipelines_duenio")


class _CursorFalso:
    """Registra cada execute; el DDL puede fallar con el error que se pida."""

    def __init__(self, previo: int = 86400, error_ddl: Exception | None = None):
        self.ejecutados: list[tuple[str, tuple | None]] = []
        self.previo = previo
        self.error_ddl = error_ddl
        self._ultimo = None

    async def execute(self, sql, args=None):
        self.ejecutados.append((sql, args))
        if sql.startswith("CREATE INDEX") and self.error_ddl is not None:
            raise self.error_ddl
        self._ultimo = sql

    async def fetchone(self):
        if self._ultimo and "lock_wait_timeout" in self._ultimo and self._ultimo.startswith("SELECT"):
            return (self.previo,)
        return (0,)


def _ddl_de_duenio() -> str:
    for tabla, indice, ddl, _acotado in store._INDICES:
        if (tabla, indice) == _ENTRADA:
            return ddl
    raise AssertionError(f"{_ENTRADA} no esta en store._INDICES")


class FormaDelDDLTest(unittest.TestCase):
    def test_el_indice_de_duenio_esta_en_la_lista_idempotente(self):
        entradas = {(t, i): acotado for t, i, _d, acotado in store._INDICES}
        self.assertIn(_ENTRADA, entradas)
        self.assertTrue(entradas[_ENTRADA], "el DDL de dueño va con la espera acotada")

    def test_columnas_en_orden_y_en_linea_sin_caer_a_copy(self):
        ddl = _ddl_de_duenio()
        self.assertEqual(
            ddl,
            "CREATE INDEX idx_jacobs_pipelines_duenio ON jacobs_pipelines "
            "(user_id, tenant_id, created_at) ALGORITHM=INPLACE LOCK=NONE",
        )


class EsperaAcotadaTest(unittest.IsolatedAsyncioTestCase):
    async def test_acota_la_espera_solo_para_el_ddl_y_la_restaura(self):
        cur = _CursorFalso(previo=86400)
        creado = await store._crear_indice_acotado(cur, *_ENTRADA, _ddl_de_duenio())
        self.assertTrue(creado)
        sqls = [s for s, _a in cur.ejecutados]
        i_set = sqls.index("SET SESSION lock_wait_timeout=%s")
        i_ddl = next(i for i, s in enumerate(sqls) if s.startswith("CREATE INDEX"))
        self.assertLess(i_set, i_ddl)
        self.assertEqual(cur.ejecutados[i_set][1], (30,))
        # Lo ultimo que corre es la restauracion al valor previo de la sesion.
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (86400,)))

    async def test_vencida_la_espera_deja_ERROR_sigue_y_restaura(self):
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1205, "Lock wait timeout exceeded; try restarting transaction"))
        with self.assertLogs("jacobs.store", level="ERROR") as logs:
            creado = await store._crear_indice_acotado(cur, *_ENTRADA, _ddl_de_duenio())
        self.assertFalse(creado)
        texto = "\n".join(logs.output)
        self.assertIn("idx_jacobs_pipelines_duenio", texto)
        self.assertIn("proximo arranque", texto)
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))

    async def test_otro_error_sube_y_restaura(self):
        """ALGORITHM=INPLACE no soportado (1846) no es una espera: es un DDL
        que no puede correr en linea, y eso no se traga."""
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1846, "LOCK=NONE is not supported"))
        with self.assertRaises(aiomysql.OperationalError):
            await store._crear_indice_acotado(cur, *_ENTRADA, _ddl_de_duenio())
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
