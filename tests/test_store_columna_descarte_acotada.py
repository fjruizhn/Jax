#!/usr/bin/env python3
"""Fix round 1 de Task 1 (descartar-pipelines, 2026-09-22, revisión del
coordinador): las tres columnas nuevas de `jacobs_pipelines`
(status_previo/descartado_por/descartado_at) son un CONTRATO -- Task 2 las
escribe en la MISMA transacción que la transición de estado -- no una
aceleración como un índice. `_agregar_columna_acotada` (jacobs/store.py) las
agrega con `lock_wait_timeout` acotado (mismo mecanismo que
`_crear_indice_acotado`), pero a diferencia de un índice FALLA CERRADO si la
espera del metadata lock vence (1205): el arranque se aborta en vez de
seguir sin la columna.

Antes de este fix, las tres nuevas ALTER TABLE del bloque de columnas
compartían el `for col, ddl in [...]` sin acotar: con el `lock_wait_timeout`
default de MariaDB (86400 s), una transacción larga sobre `jacobs_pipelines`
en el primer arranque tras el deploy dejaba a Jacobs colgado hasta 24 h, en
silencio -- cada SELECT/UPDATE nuevo se encolaba detrás del ALTER.

Pruebas puras: un cursor falso, sin DB. La existencia real de las columnas
en una base vacía la prueba `tests/test_jacobs_descarte_db.py`.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_store_columna_descarte_acotada.py -v
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

import aiomysql

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from jacobs import store  # noqa: E402

_TABLA = "jacobs_pipelines"
_COLUMNA = "status_previo"
_DDL = ("ALTER TABLE jacobs_pipelines ADD COLUMN "
        "status_previo VARCHAR(20) NULL, ALGORITHM=INSTANT")

# Las tres columnas CONTRATO del descarte, tal como aparecen literalmente en
# el `for col, ddl, acotado in [...]` de init_tables() -- ninguna vive en una
# lista de módulo (a diferencia de `_INDICES`), así que la forma se vigila
# sobre el texto fuente, igual criterio que la baranda del reaper
# (tests/test_jacobs_descarte.py::test_el_reaper_solo_cosecha_no_terminales).
_COLUMNAS_CONTRATO = ("status_previo", "descartado_por", "descartado_at")

# Columnas viejas: su comportamiento NO debe cambiar (siguen sin acotar).
_COLUMNAS_VIEJAS_SIN_ACOTAR = (
    "user_id", "tenant_id", "owner_ack_at", "run_epoch",
    "parent_pipeline_id", "depth", "costo_max_aceptado_usd", "devoluciones",
)


class _CursorFalso:
    """Registra cada execute; el ALTER puede fallar con el error que se
    pida, y la reconsulta de information_schema.COLUMNS devuelve lo que se
    pida (para simular la carrera entre dos procesos que arrancan a la
    vez)."""

    def __init__(self, previo: int = 86400, error_ddl: Exception | None = None,
                 existe_tras_el_error: bool = False):
        self.ejecutados: list[tuple[str, tuple | None]] = []
        self.previo = previo
        self.error_ddl = error_ddl
        self.existe_tras_el_error = existe_tras_el_error
        self._ultimo = None

    async def execute(self, sql, args=None):
        self.ejecutados.append((sql, args))
        self._ultimo = sql
        if sql.startswith("ALTER TABLE") and self.error_ddl is not None:
            raise self.error_ddl

    async def fetchone(self):
        if self._ultimo and self._ultimo.startswith("SELECT @@SESSION.lock_wait_timeout"):
            return (self.previo,)
        if self._ultimo and self._ultimo.startswith("SELECT COUNT(*) FROM information_schema.COLUMNS"):
            return (1 if self.existe_tras_el_error else 0,)
        return (0,)


class EsperaAcotadaFallaCerradoTest(unittest.IsolatedAsyncioTestCase):
    async def test_acota_la_espera_solo_para_el_ddl_y_la_restaura(self):
        cur = _CursorFalso(previo=86400)
        await store._agregar_columna_acotada(cur, _TABLA, _COLUMNA, _DDL)
        sqls = [s for s, _a in cur.ejecutados]
        i_set = sqls.index("SET SESSION lock_wait_timeout=%s")
        i_ddl = next(i for i, s in enumerate(sqls) if s.startswith("ALTER TABLE"))
        self.assertLess(i_set, i_ddl)
        self.assertEqual(cur.ejecutados[i_set][1], (30,))
        # Lo ultimo que corre es la restauracion al valor previo de la sesion.
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (86400,)))

    async def test_vencida_la_espera_sin_que_otro_la_haya_creado_FALLA_CERRADO(self):
        """El caso central del fix: 1205 y la columna sigue sin existir --
        `init_tables()` tiene que ABORTAR, no seguir sin la columna."""
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1205, "Lock wait timeout exceeded; try restarting transaction"),
            existe_tras_el_error=False)
        with self.assertRaises(RuntimeError) as ctx:
            await store._agregar_columna_acotada(cur, _TABLA, _COLUMNA, _DDL)
        self.assertIn(_TABLA, str(ctx.exception))
        self.assertIn(_COLUMNA, str(ctx.exception))
        self.assertIn("CONTRATO", str(ctx.exception))
        # La sesion queda restaurada aunque haya fallado.
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))

    async def test_vencida_la_espera_pero_otro_proceso_ya_la_creo_no_es_fallo(self):
        """Dos procesos de Jacobs (LAS MANOS, jax-platform, el Ejecutor)
        llaman a init_tables() al arrancar: el que pierde la carrera del MDL
        puede encontrar la columna ya creada por el que ganó. Eso NO es un
        fallo."""
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1205, "Lock wait timeout exceeded; try restarting transaction"),
            existe_tras_el_error=True)
        with self.assertLogs("jacobs.store", level="WARNING") as logs:
            await store._agregar_columna_acotada(cur, _TABLA, _COLUMNA, _DDL)
        texto = "\n".join(logs.output)
        self.assertIn(_COLUMNA, texto)
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))

    async def test_otro_error_sube_directo_sin_reconsultar_y_restaura(self):
        """ALGORITHM=INSTANT no soportado (1845/1846) no es una espera: es un
        DDL que no puede correr como se declaró, y eso no se traga ni se
        reconsulta information_schema por las dudas."""
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1846, "LOCK=NONE is not supported"))
        with self.assertRaises(aiomysql.OperationalError):
            await store._agregar_columna_acotada(cur, _TABLA, _COLUMNA, _DDL)
        sqls = [s for s, _a in cur.ejecutados]
        self.assertNotIn(
            "SELECT COUNT(*) FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
            sqls,
        )
        self.assertEqual(cur.ejecutados[-1], ("SET SESSION lock_wait_timeout=%s", (50,)))


def _tuplas_del_loop_de_columnas() -> list[tuple[str, str, bool]]:
    """Parsea el AST de `init_tables()` y evalúa el `for col, ddl, acotado in
    [...]` de `jacobs_pipelines` -- no depende de la indentación ni del
    formato del comentario, a diferencia de buscar substrings a mano."""
    fuente = textwrap.dedent(inspect.getsource(store.init_tables))
    arbol = ast.parse(fuente)
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.For) and isinstance(nodo.target, ast.Tuple):
            nombres = [n.id for n in nodo.target.elts if isinstance(n, ast.Name)]
            if nombres == ["col", "ddl", "acotado"]:
                return [ast.literal_eval(elt) for elt in nodo.iter.elts]
    raise AssertionError(
        "no se encontró 'for col, ddl, acotado in [...]' en init_tables() -- "
        "¿se renombraron las variables del loop?"
    )


class FormaDelLoopDeColumnasTest(unittest.TestCase):
    """Sobre el AST de init_tables(): las tres columnas del descarte van
    acotadas (acotado=True), las viejas NO cambiaron (acotado=False).
    Mutación (pedida por el coordinador): sacar el True de cualquiera de las
    tres columnas del descarte tiene que poner este test en rojo."""

    def test_las_tres_columnas_del_descarte_van_acotadas(self):
        por_columna = {col: acotado for col, _ddl, acotado in _tuplas_del_loop_de_columnas()}
        for col in _COLUMNAS_CONTRATO:
            self.assertIn(col, por_columna, f"falta la entrada de {col} en el loop")
            self.assertTrue(por_columna[col], f"{col} no está acotada (acotado=False)")

    def test_las_columnas_viejas_no_se_acotaron(self):
        por_columna = {col: acotado for col, _ddl, acotado in _tuplas_del_loop_de_columnas()}
        for col in _COLUMNAS_VIEJAS_SIN_ACOTAR:
            self.assertIn(col, por_columna, f"falta la entrada de {col} en el loop")
            self.assertFalse(
                por_columna[col],
                f"{col} se acotó sin pedirlo -- cambia su comportamiento (era sin bound)",
            )

    def test_la_lista_tiene_exactamente_las_columnas_conocidas(self):
        """Nadie agregó una columna nueva a este loop sin decidir si va
        acotada -- partición exhaustiva, mismo criterio que
        tests/test_creacion_sin_candado_global.py con PipelineStatus."""
        columnas = {col for col, _ddl, _a in _tuplas_del_loop_de_columnas()}
        esperadas = set(_COLUMNAS_CONTRATO) | set(_COLUMNAS_VIEJAS_SIN_ACOTAR)
        self.assertEqual(columnas, esperadas)


if __name__ == "__main__":
    unittest.main(verbosity=2)
