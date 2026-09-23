#!/usr/bin/env python3
"""Ruling T6-6 (2026-09-15): el indice de dueño de jacobs_pipelines.

jax-platform lista los pipelines de un dueño con
`WHERE user_id=%s AND tenant_id=%s ORDER BY created_at`: el indice compuesto
(user_id, tenant_id, created_at) cubre el filtro y el orden. Se crea en
init_tables(), que corre en CADA arranque de LAS MANOS -- el unico proceso de
produccion que lo llama (verificado 2026-09-22 contra el codigo; jax-platform
y el Ejecutor NO lo hacen) -- y ademas de scripts de este repo o de la suite
de tests que abren la misma base.

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


# --- Pendiente 631, parte B2 (2026-09-23): indices RETIRADOS ----------------


def _columnas_del_ddl(ddl: str) -> tuple[str, ...]:
    """Las columnas de un `CREATE INDEX ... ON tabla (a, b, c) ...`."""
    dentro = ddl[ddl.index("(") + 1:ddl.index(")")]
    return tuple(c.strip() for c in dentro.split(","))


class FormaDeLosRetiradosTest(unittest.TestCase):
    def test_ningun_nombre_esta_en_las_dos_listas(self):
        creados = {(t, i) for t, i, _d, _a in store._INDICES}
        retirados = {(t, r) for t, r, _rep, _p in store._INDICES_RETIRADOS}
        self.assertEqual(creados & retirados, set(),
                         "un indice no puede crearse y retirarse en el mismo init_tables()")

    def test_idx_pipelines_status_esta_retirado_y_no_se_crea(self):
        self.assertIn(
            ("jacobs_pipelines", "idx_pipelines_status", "idx_pipelines_ocultos", ("status",)),
            store._INDICES_RETIRADOS)
        self.assertNotIn("idx_pipelines_status", {i for _t, i, _d, _a in store._INDICES})

    def test_cada_reemplazo_se_crea_en_la_misma_tabla_con_el_prefijo_declarado(self):
        ddl_de = {(t, i): d for t, i, d, _a in store._INDICES}
        for tabla, retirado, reemplazo, prefijo in store._INDICES_RETIRADOS:
            self.assertIn((tabla, reemplazo), ddl_de,
                          f"{retirado}: su reemplazo {reemplazo} no esta en _INDICES de {tabla}")
            self.assertTrue(prefijo, f"{retirado}: prefijo vacio no protege nada")
            columnas = _columnas_del_ddl(ddl_de[(tabla, reemplazo)])
            self.assertEqual(columnas[: len(prefijo)], prefijo,
                             f"{reemplazo} no empieza por {prefijo}: la precondicion nunca pasaria")

    def test_el_drop_es_en_linea_y_sin_caer_a_copy(self):
        self.assertEqual(store._ALGORITMO_DROP_INDEX, "ALGORITHM=NOCOPY, LOCK=NONE")


class _CursorGuionado:
    """Contesta segun la consulta: si el retirado existe, las columnas del
    reemplazo, y el error que se pida en el ALTER."""

    def __init__(self, existe=1, columnas=(("status", "NO"), ("descartado_at", "NO")),
                 error_alter: Exception | None = None, previo: int = 86400):
        self.existe = existe
        self.columnas = list(columnas)
        self.error_alter = error_alter
        self.previo = previo
        self.ejecutados: list[tuple[str, tuple | None]] = []
        self._ultimo = ""

    async def execute(self, sql, args=None):
        self.ejecutados.append((sql, args))
        self._ultimo = sql
        if sql.startswith("ALTER TABLE") and self.error_alter is not None:
            raise self.error_alter

    async def fetchone(self):
        if "lock_wait_timeout" in self._ultimo:
            return (self.previo,)
        return (self.existe,)

    async def fetchall(self):
        return self.columnas

    def alters(self):
        return [s for s, _a in self.ejecutados if s.startswith("ALTER TABLE")]


class RetirarIndicesTest(unittest.IsolatedAsyncioTestCase):
    async def test_si_no_existe_no_hace_nada(self):
        cur = _CursorGuionado(existe=0)
        await store._retirar_indices(cur)
        self.assertEqual(cur.alters(), [])
        self.assertEqual(len(cur.ejecutados), 1, "solo el chequeo de existencia")

    async def test_con_reemplazo_valido_borra_en_linea_con_espera_acotada_y_restaura(self):
        cur = _CursorGuionado(previo=77)
        await store._retirar_indices(cur)
        self.assertEqual(cur.alters(), [
            "ALTER TABLE jacobs_pipelines DROP INDEX idx_pipelines_status, "
            "ALGORITHM=NOCOPY, LOCK=NONE"])
        sqls = [s for s, _a in cur.ejecutados]
        i_set = sqls.index("SET SESSION lock_wait_timeout=%s")
        self.assertLess(i_set, sqls.index(cur.alters()[0]))
        self.assertEqual(cur.ejecutados[i_set][1], (30,))
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (77,)))

    async def _conserva(self, columnas, motivo):
        cur = _CursorGuionado(columnas=columnas)
        with self.assertLogs("jacobs.store", level="ERROR") as logs:
            await store._retirar_indices(cur)
        self.assertEqual(cur.alters(), [], "sin reemplazo valido NO se borra")
        texto = "\n".join(logs.output)
        self.assertIn("idx_pipelines_status", texto)
        self.assertIn(motivo, texto)

    async def test_sin_reemplazo_conserva_y_deja_ERROR(self):
        await self._conserva([], "no existe")

    async def test_reemplazo_ignored_conserva_y_deja_ERROR(self):
        await self._conserva([("status", "YES"), ("descartado_at", "YES")], "IGNORED")

    async def test_reemplazo_con_otro_prefijo_conserva_y_deja_ERROR(self):
        await self._conserva([("descartado_at", "NO"), ("status", "NO")], "no empiezan por")

    async def test_1205_deja_ERROR_sigue_y_restaura(self):
        cur = _CursorGuionado(previo=50, error_alter=aiomysql.OperationalError(
            1205, "Lock wait timeout exceeded; try restarting transaction"))
        with self.assertLogs("jacobs.store", level="ERROR") as logs:
            await store._retirar_indices(cur)
        self.assertIn("proximo arranque", "\n".join(logs.output))
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))

    async def test_1091_otro_proceso_lo_borro_es_exito(self):
        cur = _CursorGuionado(previo=50, error_alter=aiomysql.OperationalError(
            1091, "Can't DROP INDEX `idx_pipelines_status`; check that it exists"))
        await store._retirar_indices(cur)  # no sube
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))

    async def test_otro_error_sube_y_restaura(self):
        """NOCOPY/LOCK=NONE no soportado no es una espera: no se traga."""
        cur = _CursorGuionado(previo=50, error_alter=aiomysql.OperationalError(
            1846, "ALGORITHM=NOCOPY is not supported"))
        with self.assertRaises(aiomysql.OperationalError):
            await store._retirar_indices(cur)
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
