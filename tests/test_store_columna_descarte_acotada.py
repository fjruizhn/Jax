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

Task 1-bis (2026-09-22, Ruling 18) suma una cuarta CONTRATO al mismo
mecanismo: `visible`, GENERATED VIRTUAL a partir de `status` -- nadie la
escribe (la calcula MariaDB por fila), pero jax-platform va a leerla para su
listado principal, así que el mismo criterio de "algo de afuera depende de
que exista" aplica igual.

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

# Las columnas CONTRATO del descarte, tal como aparecen literalmente en el
# `for col, ddl, acotado in [...]` de init_tables() -- ninguna vive en una
# lista de módulo (a diferencia de `_INDICES`), así que la forma se vigila
# sobre el texto fuente, igual criterio que la baranda del reaper
# (tests/test_jacobs_descarte.py::test_el_reaper_solo_cosecha_no_terminales).
# `visible` se suma acá (Task 1-bis, 2026-09-22, Ruling 18) -- no la ESCRIBE
# ninguna transición (es GENERATED, MariaDB la calcula de `status`), pero
# jax-platform va a LEERLA para su listado principal: mismo criterio de
# CONTRATO que las otras tres (algo de afuera depende de que exista), así
# que también va acotada y fail-closed.
_COLUMNAS_CONTRATO = ("status_previo", "descartado_por", "descartado_at", "visible")

# Columnas viejas: su comportamiento NO debe cambiar (siguen sin acotar).
_COLUMNAS_VIEJAS_SIN_ACOTAR = (
    "user_id", "tenant_id", "owner_ack_at", "run_epoch",
    "parent_pipeline_id", "depth", "costo_max_aceptado_usd", "devoluciones",
)


class _CursorFalso:
    """Registra cada execute; el ALTER puede fallar con el error que se
    pida, y la reconsulta de information_schema.COLUMNS devuelve lo que se
    pida (para simular la carrera entre dos procesos que arrancan a la
    vez).

    `expresion_generada` (fix round 3, 2026-09-22): lo que devuelve la
    reconsulta de `GENERATION_EXPRESSION` cuando `_agregar_columna_acotada`
    pierde la carrera del 1205 para `columna="visible"` y llama a
    `store._verificar_expresion_visible`. Default: la expresión esperada
    (`store._EXPRESION_VISIBLE`) -- el caso feliz, "otro proceso ganó Y
    creó lo mismo que yo hubiera creado"."""

    def __init__(self, previo: int = 86400, error_ddl: Exception | None = None,
                 existe_tras_el_error: bool = False,
                 expresion_generada: str | None = "__default__"):
        self.ejecutados: list[tuple[str, tuple | None]] = []
        self.previo = previo
        self.error_ddl = error_ddl
        self.existe_tras_el_error = existe_tras_el_error
        self.expresion_generada = expresion_generada
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
        if self._ultimo and self._ultimo.startswith("SELECT GENERATION_EXPRESSION"):
            expr = self.expresion_generada
            return (store._EXPRESION_VISIBLE if expr == "__default__" else expr,)
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


class CarreraDeVisibleLlamaAlDriftCheckTest(unittest.IsolatedAsyncioTestCase):
    """Fix round 3 (revisión del coordinador, 2026-09-22): "otro proceso la
    creó primero" (1205 + existe_tras_el_error=True) es correcto SÓLO si
    creó la MISMA columna -- para `visible` (GENERADA) eso incluye la
    expresión. `_agregar_columna_acotada` ahora llama a
    `store._verificar_expresion_visible` antes de devolver en esa rama,
    pero SÓLO para `columna="visible"` -- las otras tres columnas CONTRATO
    (status_previo/descartado_por/descartado_at) no son generadas y no
    tienen que pagar esta consulta extra (verificado con `_COLUMNA` de
    arriba, "status_previo", en `EsperaAcotadaFallaCerradoTest` de arriba:
    ese cursor falso nunca ve un `SELECT GENERATION_EXPRESSION`)."""

    _DDL_VISIBLE = ("ALTER TABLE jacobs_pipelines ADD COLUMN "
                     "visible TINYINT(1) GENERATED ALWAYS AS "
                     "(status NOT IN ('discarded','hidden') "
                     "AND owner_ack_at IS NOT NULL) VIRTUAL, "
                     "ALGORITHM=INSTANT")

    async def test_otro_proceso_creo_la_misma_expresion_no_es_fallo(self):
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1205, "Lock wait timeout exceeded; try restarting transaction"),
            existe_tras_el_error=True)  # expresion_generada default: la esperada
        await store._agregar_columna_acotada(cur, _TABLA, "visible", self._DDL_VISIBLE)
        sqls = [s for s, _a in cur.ejecutados]
        self.assertTrue(any(s.startswith("SELECT GENERATION_EXPRESSION") for s in sqls))

    async def test_otro_proceso_creo_una_expresion_distinta_SI_es_fallo(self):
        """El caso que este fix round cierra: sin la llamada al drift
        check, esta rama devolvía en silencio aunque la columna que "ganó
        la carrera" tuviera la expresión VIEJA."""
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1205, "Lock wait timeout exceeded; try restarting transaction"),
            existe_tras_el_error=True,
            expresion_generada="`status` not in ('discarded','hidden')")  # vieja, sin ack
        with self.assertRaises(RuntimeError) as ctx:
            await store._agregar_columna_acotada(cur, _TABLA, "visible", self._DDL_VISIBLE)
        self.assertIn("expresión DISTINTA", str(ctx.exception))

    async def test_columnas_no_generadas_no_pagan_esta_consulta_extra(self):
        """Control: `status_previo` (CONTRATO pero no generada) pierde la
        misma carrera y NO dispara `SELECT GENERATION_EXPRESSION` -- el
        chequeo es específico de `visible`."""
        cur = _CursorFalso(previo=50, error_ddl=aiomysql.OperationalError(
            1205, "Lock wait timeout exceeded; try restarting transaction"),
            existe_tras_el_error=True)
        await store._agregar_columna_acotada(cur, _TABLA, _COLUMNA, _DDL)
        sqls = [s for s, _a in cur.ejecutados]
        self.assertFalse(any(s.startswith("SELECT GENERATION_EXPRESSION") for s in sqls))


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
    """Sobre el AST de init_tables(): las columnas CONTRATO del descarte
    (incluida `visible`, Task 1-bis) van acotadas (acotado=True), las viejas
    NO cambiaron (acotado=False). Mutación (pedida por el coordinador): sacar
    el True de cualquiera de las columnas CONTRATO tiene que poner este test
    en rojo."""

    def test_las_columnas_contrato_del_descarte_van_acotadas(self):
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

    def test_visible_es_la_ultima_columna_del_loop(self):
        """MAJOR-1 (fix round 1, Ruling 19b, 2026-09-22): la TRAMPA del
        INSTANT. Medido contra MariaDB 12.3.3 real
        (jacobs/_subpipeline_contrato_io_test.py, Task 1-bis): en cuanto
        `jacobs_pipelines` tiene un ÍNDICE sobre `visible` (una columna
        VIRTUAL -- idx_pipelines_visibles, Ruling 18), CUALQUIER `ADD
        COLUMN` de OTRA columna sobre esa tabla, aunque pida
        `ALGORITHM=INSTANT` explícito, puede rechazarse con
        `1845 ALGORITHM=INSTANT is not supported` -- y ni siquiera
        `ALGORITHM=INPLACE` alcanza con `LOCK=NONE`
        (`1846 ... online rebuild with indexed virtual columns`, pide
        `LOCK=SHARED`).

        CI NO VE ESTO: el contenedor efímero de cada job arranca de una
        base VACÍA, así que TODAS las columnas del loop se agregan a una
        tabla que todavía no tiene el índice -- el 1845/1846 sólo aparece
        en una base que YA pasó por este deploy antes (exactamente el
        estado de producción una vez que Ruling 18 esté desplegado). Por
        eso este test es MECÁNICO, no una corrida contra DB: mientras
        `visible` sea la ÚLTIMA tupla del loop, ninguna columna futura
        queda expuesta al `ADD COLUMN` posterior a un índice sobre columna
        generada. El día que haga falta una columna nueva DESPUÉS de
        `visible`, este test cae -- y CADA columna nueva se diseña
        APARTE, con su propia evidencia contra una base que YA tiene
        `visible` indexada (ver CanarioTrampaInstantDBTest en
        tests/test_jacobs_descarte_db.py para el canario que vigila que
        la trampa siga vigente)."""
        tuplas = _tuplas_del_loop_de_columnas()
        ultima = tuplas[-1][0]
        self.assertEqual(
            ultima, "visible",
            f"`visible` dejó de ser la última columna del loop (ahora lo es "
            f"{ultima!r}) -- eso expone la columna nueva al 1845/1846 en "
            f"producción, invisible en CI (base fresca sin el índice "
            f"todavía). Diseñala con su propia evidencia contra una base "
            f"que YA tiene idx_pipelines_visibles."
        )

    def test_la_ddl_de_visible_usa_la_expresion_esperada(self):
        """MINOR-A (fix round 2, 2026-09-22): `store._EXPRESION_VISIBLE` (la
        fuente que usa `_verificar_expresion_visible` para el chequeo de
        drift) y el texto DENTRO del DDL de la tupla `visible` del loop son
        DOS copias escritas a mano -- no se arman con un f-string a propósito
        (`ast.literal_eval` no acepta interpolación, ver el comentario de
        `_EXPRESION_VISIBLE` en store.py). Esta es la baranda MECÁNICA que
        reemplaza esa garantía: si alguien edita una copia y no la otra, cae
        acá, normalizado (mismo criterio que la comparación en runtime, así
        que un cambio de mayúsculas o espacios entre las dos copias NO hace
        caer este test por las razones equivocadas)."""
        (ddl_de_visible,) = [ddl for col, ddl, _a in _tuplas_del_loop_de_columnas() if col == "visible"]
        self.assertIn(
            store._normalizar_expresion_generada(store._EXPRESION_VISIBLE),
            store._normalizar_expresion_generada(ddl_de_visible),
            "el DDL de la tupla 'visible' no contiene (normalizado) el texto "
            "de store._EXPRESION_VISIBLE -- las dos copias se desincronizaron",
        )


class NormalizarExpresionGeneradaTest(unittest.TestCase):
    """MINOR-A (fix round 2, 2026-09-22): pura, sin DB. Casos tomados de lo
    que MariaDB 12.3.3 REAL devuelve en `GENERATION_EXPRESSION` (medido
    antes de escribir este test, no supuesto) más variantes de cómo
    alguien podría escribir el MISMO DDL a mano."""

    def test_forma_real_que_devuelve_mariadb(self):
        # SHOW/information_schema tal cual se midió contra jax_memory_test:
        # backticks en los identificadores, palabras clave en minúscula,
        # los literales de string sin tocar.
        devuelta = "`status` not in ('discarded','hidden') and `owner_ack_at` is not null"
        self.assertEqual(
            store._normalizar_expresion_generada(devuelta),
            store._normalizar_expresion_generada(store._EXPRESION_VISIBLE),
        )

    def test_no_da_falsa_alarma_por_mayusculas_ni_espacios_del_ddl_fuente(self):
        """El DDL fuente (como lo escribe un humano, ANTES de que MariaDB lo
        reescriba) puede venir con mayúsculas distintas o espaciado
        distinto del que MariaDB termina guardando -- normalizar tiene que
        absorber eso, no sólo lo que ya viene canónico de information_schema."""
        variantes = [
            "status NOT IN ('discarded','hidden') AND owner_ack_at IS NOT NULL",
            "status   not in  ('discarded','hidden')    and owner_ack_at is not null",
            "STATUS NOT IN ('discarded','hidden') AND OWNER_ACK_AT IS NOT NULL",
            "\n  status NOT IN ('discarded','hidden')\n  AND owner_ack_at IS NOT NULL  \n",
        ]
        esperada = store._normalizar_expresion_generada(store._EXPRESION_VISIBLE)
        for variante in variantes:
            with self.subTest(variante=variante):
                self.assertEqual(store._normalizar_expresion_generada(variante), esperada)

    def test_una_expresion_realmente_distinta_no_normaliza_igual(self):
        """Control negativo: el normalizador absorbe FORMA, no CONTENIDO --
        la expresión vieja (sin el AND de ack, Ruling 19a) tiene que seguir
        distinguiéndose de la esperada después de normalizar."""
        vieja = "`status` not in ('discarded','hidden')"
        self.assertNotEqual(
            store._normalizar_expresion_generada(vieja),
            store._normalizar_expresion_generada(store._EXPRESION_VISIBLE),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
