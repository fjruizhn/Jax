"""Pool de conexiones de Jacobs contra una MariaDB REAL (2026-09-17).

POR QUE EXISTE. La Task 8 del frente F midio la creacion de pipelines bajo carga
y a 10 VUs ya fallaba entre el 22 y el 38 % de las peticiones, en los tres
escenarios, incluido el que no toca el contrato. La causa medida:
`jacobs/store.py::get_conn()` abria una conexion `aiomysql` NUEVA por llamada
(y una creacion en dry_run hace 5-6 llamadas). A ~1.000 req/s eso son miles de
conexiones por segundo, cada una deja un socket en TIME_WAIT y el rango de
puertos efimeros del host se agota: `Lost connection ... system error 11`.

Politica 2 de LAS CUATRO DEL RENDIMIENTO: los pools de DB se comparten, nunca se
rehacen por request.

Que prueba:
  - el defecto: N llamadas concurrentes NO abren N conexiones (visto en rojo
    contra eb72e78: 20 de 20);
  - que la sesion que entrega el pool es identica a la de antes (autocommit,
    aislamiento, charset, sql_mode, zona horaria);
  - que ningun camino de error filtra una conexion ni devuelve al pool una
    sesion sucia (excepcion, cancelacion a mitad de consulta, autocommit
    apagado, conexion desechable);
  - que esperar un hueco tiene limite (fail-closed, no espera infinita) y que
    descartar una conexion despierta a quien espera;
  - la validacion del tamano y el ciclo de vida (un pool por loop, cierre).

Seguridad: la barrera de base vive en jacobs/_arnes_ada.py, que se importa
PRIMERO.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v jacobs/_store_pool_test.py"
"""
from __future__ import annotations

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import ast  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import aiomysql  # noqa: E402
from aiomysql.connection import Connection  # noqa: E402

from jacobs import store  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ENV_TAMANIO = "JAX_JACOBS_DB_POOL_SIZE"


class _ContadorDeConexiones:
    """Cuenta los handshakes REALES (`Connection._connect`), que es por donde
    pasan tanto `aiomysql.connect()` como el pool al crecer. Contar en el
    cliente y no con `SHOW STATUS LIKE 'Connections'`: otros frentes usan la
    misma MariaDB en paralelo y el contador del servidor es de todos."""

    def __init__(self):
        self.n = 0
        original = Connection._connect
        contador = self

        async def _connect(conn_self, *a, **kw):
            contador.n += 1
            return await original(conn_self, *a, **kw)

        self._patch = patch.object(Connection, "_connect", _connect)

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


def _crudo(vigilada):
    """La conexion real detras de un envoltorio, AUN invalidado: solo para que
    los tests miren si el socket se cerro o volvio al pool despues del bloque.
    Codigo de servicio nunca: ahi el acceso posterior lanza ConexionInvalidada."""
    return vigilada._ConexionVigilada__crudo


async def _a_lo_sumo(esperable, segundos: float = 15):
    """Toda espera de un test con tareas va acotada: un defecto (o una mutacion)
    tiene que dar un FAILED, no colgar la suite -- visto al mutar el semaforo."""
    return await asyncio.wait_for(esperable, timeout=segundos)


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class _ConBase(unittest.IsolatedAsyncioTestCase):
    TAMANIO = "5"

    async def asyncSetUp(self):
        entorno = patch.dict(os.environ, {ENV_TAMANIO: self.TAMANIO})
        entorno.start()
        self.addCleanup(entorno.stop)
        # Cada test tiene su loop (IsolatedAsyncioTestCase): se cierra el pool
        # de ESTE loop al terminar, para no dejar sockets colgados del loop muerto.
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()


class DefectoTest(_ConBase):
    async def test_llamadas_concurrentes_no_abren_una_conexion_cada_una(self):
        """El defecto medido en la Task 8. Contra eb72e78: 20 conexiones para
        20 llamadas. Con el pool: a lo sumo el tamano del pool."""
        with _ContadorDeConexiones() as c:
            await asyncio.gather(*[store.pipeline_count_active() for _ in range(20)])
        self.assertLessEqual(
            c.n, int(self.TAMANIO),
            f"{c.n} conexiones nuevas para 20 llamadas concurrentes: sin pool",
        )

    async def test_llamadas_en_serie_reusan_la_conexion(self):
        await store.pipeline_count_active()  # calienta el pool de este loop
        with _ContadorDeConexiones() as c:
            for _ in range(30):
                await store.pipeline_count_active()
        self.assertEqual(c.n, 0, f"{c.n} conexiones nuevas en 30 llamadas en serie")

    async def test_mas_llamadas_que_conexiones_esperan_y_terminan_todas(self):
        """Con el pool lleno se espera un hueco, no se falla."""
        resultados = await asyncio.gather(
            *[ada.una_fila("SELECT SLEEP(0.05) AS s") for _ in range(4 * int(self.TAMANIO))]
        )
        self.assertEqual(len(resultados), 4 * int(self.TAMANIO))
        pool = await store.obtener_pool()
        self.assertLessEqual(pool.size, int(self.TAMANIO))


class SesionIdenticaTest(_ConBase):
    VARIABLES = (
        "@@SESSION.autocommit", "@@SESSION.transaction_isolation",
        "@@SESSION.character_set_client", "@@SESSION.character_set_connection",
        "@@SESSION.character_set_results", "@@SESSION.collation_connection",
        "@@SESSION.sql_mode", "@@SESSION.time_zone", "DATABASE()",
    )

    async def _variables(self, conn) -> tuple:
        async with conn.cursor() as cur:
            await cur.execute("SELECT " + ", ".join(self.VARIABLES))
            return await cur.fetchone()

    async def test_la_sesion_del_pool_es_la_de_una_conexion_directa(self):
        """Lo que antes daba `get_conn()`: `aiomysql.connect(**_db_cfg())`."""
        directa = await aiomysql.connect(
            **store._db_cfg(), connect_timeout=store.db_connect_timeout_seconds()
        )
        try:
            esperado = await self._variables(directa)
        finally:
            directa.close()
        async with store.conexion() as conn:
            self.assertEqual(await self._variables(conn), esperado)
            self.assertTrue(conn.get_autocommit())
        self.assertEqual(esperado[0], 1, "autocommit tiene que seguir encendido")


class SinFugasTest(_ConBase):
    async def _estado(self):
        pool = await store.obtener_pool()
        return pool, len(pool._used)

    async def test_excepcion_en_el_cuerpo_descarta_la_conexion(self):
        pool, _ = await self._estado()
        with self.assertRaises(ZeroDivisionError):
            async with store.conexion() as conn:
                1 / 0
        self.assertTrue(_crudo(conn).closed, "una conexion con estado desconocido volvio al pool")
        self.assertNotIn(_crudo(conn), pool._free)
        self.assertEqual(len(pool._used), 0, "conexion filtrada: sigue marcada en uso")

    async def test_error_de_sql_en_una_funcion_del_store_no_filtra(self):
        pool, _ = await self._estado()
        for _ in range(3 * int(self.TAMANIO)):
            with self.assertRaises(aiomysql.ProgrammingError):
                await ada.una_fila("SELECT * FROM tabla_que_no_existe_pool_test")
        self.assertEqual(len(pool._used), 0)
        self.assertLessEqual(pool.size, int(self.TAMANIO))
        self.assertEqual((await ada.una_fila("SELECT 1 AS uno"))["uno"], 1)

    async def test_cancelacion_a_mitad_de_consulta_descarta_la_conexion(self):
        pool, _ = await self._estado()
        capturada: list = []

        async def larga():
            async with store.conexion() as conn:
                capturada.append(conn)
                async with conn.cursor() as cur:
                    await cur.execute("SELECT SLEEP(5)")

        tarea = asyncio.create_task(larga())
        await asyncio.sleep(0.5)
        tarea.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await _a_lo_sumo(tarea)
        self.assertTrue(_crudo(capturada[0]).closed, "el socket a mitad de respuesta volvio al pool")
        self.assertEqual(len(pool._used), 0)
        self.assertEqual((await ada.una_fila("SELECT 1 AS uno"))["uno"], 1)

    async def test_autocommit_apagado_no_vuelve_al_pool(self):
        pool, _ = await self._estado()
        async with store.conexion() as conn:
            await conn.autocommit(False)
        self.assertTrue(_crudo(conn).closed)
        async with store.conexion() as otra:
            async with otra.cursor() as cur:
                await cur.execute("SELECT @@SESSION.autocommit")
                self.assertEqual((await cur.fetchone())[0], 1)

    async def test_transaccion_abierta_no_vuelve_al_pool(self):
        async with store.conexion() as conn:
            await conn.begin()
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        self.assertTrue(_crudo(conn).closed)

    async def test_transaccion_abierta_se_descarta_aunque_aiomysql_no_lo_haga(self):
        """aiomysql 0.3.2 cierra en release() una conexion con transaccion
        abierta; la garantia no puede depender de eso (ronda de revision
        2026-09-17). Se simula un aiomysql que no mira la transaccion."""
        original = aiomysql.Pool.release

        def release_sin_mirar_transaccion(pool_self, conn):
            with patch.object(type(conn), "get_transaction_status", return_value=False):
                return original(pool_self, conn)

        with patch.object(aiomysql.Pool, "release", release_sin_mirar_transaccion):
            async with store.conexion() as conn:
                await conn.begin()
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
        pool = await store.obtener_pool()
        self.assertTrue(_crudo(conn).closed, "una sesion con transaccion abierta volvio al pool")
        self.assertNotIn(_crudo(conn), pool._free)

    async def test_si_aiomysql_falla_al_entregar_el_permiso_vuelve(self):
        """Sin devolver el permiso cuando `pool.acquire()` explota, cada falla
        de la base se comeria un lugar del semaforo para siempre."""
        n = int(self.TAMANIO) + 2
        with patch.object(aiomysql.Pool, "acquire", side_effect=OSError("la base no esta")):
            for _ in range(n):
                with self.assertRaises(OSError):
                    async with store.conexion():
                        pass
        with patch.dict(os.environ, {"JAX_DB_CONNECT_TIMEOUT_SECONDS": "1"}):
            self.assertEqual((await ada.una_fila("SELECT 1 AS uno"))["uno"], 1)

    async def test_desechable_se_cierra_aunque_este_limpia(self):
        """Para quien cambia variables de sesion (el DDL acotado de init_tables)."""
        pool, _ = await self._estado()
        async with store.conexion(desechable=True) as conn:
            pass
        self.assertTrue(_crudo(conn).closed)
        self.assertNotIn(_crudo(conn), pool._free)
        self.assertEqual(len(pool._used), 0)

    async def test_conexion_limpia_vuelve_al_pool(self):
        """Control del control: sin esto, cerrar SIEMPRE pasaria los de arriba."""
        async with store.conexion() as conn:
            pass
        pool = await store.obtener_pool()
        self.assertFalse(_crudo(conn).closed)
        self.assertIn(_crudo(conn), pool._free)


_TABLA_INEXISTENTE = "tabla_que_no_existe_pool_test"


class ErrorDelServidorTest(_ConBase):
    """Un error que MANDA EL SERVIDOR no ensucia el socket (2026-09-17).

    POR QUE EXISTE. El job `subpipeline-contrato-db` de 13b7759 fallo con "20
    conexiones para 20 autorizaciones": su MariaDB recien creada no tiene la
    tabla `facet`, la consulta da ProgrammingError 1146 y `conexion()` cerraba
    la conexion ante CUALQUIER excepcion -- un handshake nuevo por pedido
    fallido. En local la tabla existe y el test pasaba por el camino feliz.
    Es el mismo agotamiento de puertos que el pool vino a cerrar, disparado por
    una rafaga de errores (tabla faltante, clave duplicada, deadlock). El error
    del servidor llega como un paquete completo: el protocolo queda en limite
    de paquete y la sesion se sigue validando con _sesion_reutilizable."""

    async def _falla_en_el_servidor(self):
        with self.assertRaises(aiomysql.ProgrammingError) as ctx:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SELECT * FROM {_TABLA_INEXISTENTE}")
        self.assertEqual(ctx.exception.args[0], 1146)
        return conn

    async def test_errores_de_sql_concurrentes_no_abren_una_conexion_cada_una(self):
        await store.obtener_pool()
        with _ContadorDeConexiones() as c:
            resultados = await _a_lo_sumo(asyncio.gather(
                *[self._falla_en_el_servidor() for _ in range(20)],
                return_exceptions=True,
            ))
        self.assertEqual([r for r in resultados if isinstance(r, BaseException)], [])
        self.assertLessEqual(c.n, int(self.TAMANIO), f"{c.n} conexiones para 20 errores de SQL")

    async def test_error_del_servidor_devuelve_la_conexion_sana(self):
        conn = await self._falla_en_el_servidor()
        pool = await store.obtener_pool()
        self.assertFalse(_crudo(conn).closed, "un error del servidor descarto una conexion sana")
        self.assertIn(_crudo(conn), pool._free)
        self.assertEqual(len(pool._used), 0)
        async with store.conexion() as otra:
            self.assertIs(_crudo(otra), _crudo(conn))
            async with otra.cursor() as cur:
                await cur.execute("SELECT 1, @@SESSION.autocommit")
                self.assertEqual(await cur.fetchone(), (1, 1))

    async def test_error_del_servidor_con_transaccion_abierta_se_descarta(self):
        with self.assertRaises(aiomysql.ProgrammingError):
            async with store.conexion() as conn:
                await conn.begin()
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    await cur.execute(f"SELECT * FROM {_TABLA_INEXISTENTE}")
        self.assertTrue(_crudo(conn).closed, "una transaccion abierta volvio al pool tras un error")

    async def test_error_del_cliente_descarta_aunque_parezca_de_sql(self):
        """Perdida de conexion (2013), un error sin codigo del servidor o uno
        del servidor que no esta en la lista de los que dejan el socket sano:
        estado desconocido, se descarta. Fail-closed ante la duda."""
        for exc in (
            aiomysql.OperationalError(2013, "Lost connection to MySQL server during query"),
            aiomysql.ProgrammingError("Cursor closed"),
            aiomysql.ProgrammingError(2014, "Commands out of sync"),
            aiomysql.InterfaceError(0, ""),
            aiomysql.OperationalError(1927, "Connection was killed"),
            aiomysql.InternalError(1105, "Unknown error"),
        ):
            with self.subTest(exc=repr(exc)):
                with self.assertRaises(type(exc)):
                    async with store.conexion() as conn:
                        raise exc
                self.assertTrue(_crudo(conn).closed, f"{exc!r} devolvio la conexion al pool")


async def _consulta(conn, sql: str):
    async with conn.cursor() as cur:
        await cur.execute(sql)
        return await cur.fetchall()


async def _conexion_directa():
    return await aiomysql.connect(
        **store._db_cfg(), connect_timeout=store.db_connect_timeout_seconds()
    )


class ErrorAtribuidoTest(_ConBase):
    """Atribucion POR CONSTRUCCION (ruling 2026-09-17 sobre b97e0fb): solo el
    error que el cursor vigilado marco, por identidad, devuelve la conexion."""

    async def _error_de(self, conn) -> BaseException:
        try:
            await _consulta(conn, f"SELECT * FROM {_TABLA_INEXISTENTE}")
        except aiomysql.ProgrammingError as e:
            return e
        self.fail("la consulta a una tabla inexistente no fallo")

    async def _error_de_y_relanza(self, conn):
        await _consulta(conn, f"SELECT * FROM {_TABLA_INEXISTENTE}")

    async def test_error_de_otra_conexion_propagado_en_el_bloque_descarta(self):
        otra = await _conexion_directa()
        try:
            ajeno = await self._error_de(otra)
        finally:
            otra.close()
        with self.assertRaises(aiomysql.ProgrammingError):
            async with store.conexion() as conn:
                raise ajeno
        self.assertTrue(_crudo(conn).closed, "un error de OTRA conexion devolvio esta al pool")

    async def test_error_de_otra_conexion_del_pool_descarta(self):
        """Dos bloques del pool: el error marcado en uno no vale en el otro."""
        with self.assertRaises(aiomysql.ProgrammingError):
            async with store.conexion() as conn:
                async with store.conexion() as vecina:
                    ajeno = await self._error_de(vecina)
                raise ajeno
        self.assertTrue(_crudo(conn).closed)
        self.assertFalse(_crudo(vecina).closed)

    async def test_error_relanzado_como_otro_descarta(self):
        with self.assertRaises(aiomysql.ProgrammingError):
            async with store.conexion() as conn:
                try:
                    await self._error_de_y_relanza(conn)
                except aiomysql.ProgrammingError as e:
                    raise aiomysql.ProgrammingError(*e.args) from None
        self.assertTrue(_crudo(conn).closed)

    async def test_error_ajeno_tras_uno_marcado_descarta(self):
        """El marcado se atrapo; lo que sale del bloque es otra cosa."""
        with self.assertRaises(RuntimeError):
            async with store.conexion() as conn:
                await self._error_de(conn)
                raise RuntimeError("otra cosa")
        self.assertTrue(_crudo(conn).closed)

    async def test_gather_sin_hermana_viva_la_devuelve(self):
        with self.assertRaises(aiomysql.ProgrammingError):
            async with store.conexion() as conn:
                await asyncio.gather(asyncio.sleep(0), self._error_de_y_relanza(conn))
        self.assertFalse(_crudo(conn).closed)
        self.assertIn(_crudo(conn), (await store.obtener_pool())._free)

    async def test_cursor_sin_buffer_abierto_descarta(self):
        """Resultados sin leer que el envoltorio SI conoce: un SSCursor abierto."""
        with self.assertRaises(aiomysql.ProgrammingError):
            async with store.conexion() as conn:
                e = await self._error_de(conn)
                cur = await conn.cursor(aiomysql.SSCursor)
                await cur.execute("SELECT seq FROM seq_1_to_10000")
                await cur.fetchone()
                raise e
        self.assertTrue(_crudo(conn).closed, "volvio al pool con un cursor sin buffer a medio leer")

    async def test_cursor_sin_buffer_cerrado_la_devuelve(self):
        """Control del de arriba."""
        with self.assertRaises(aiomysql.ProgrammingError):
            async with store.conexion() as conn:
                async with conn.cursor(aiomysql.SSCursor) as cur:
                    await cur.execute("SELECT seq FROM seq_1_to_10")
                    await cur.fetchall()
                await self._error_de_y_relanza(conn)
        self.assertFalse(_crudo(conn).closed)

    async def test_error_real_del_servidor_fuera_de_la_lista_descarta(self):
        """El servidor responde 1927 (conexion matada) por el cursor vigilado.
        SIGNAL lo emite sin matar nada: solo la lista blanca decide."""
        with self.assertRaises(aiomysql.OperationalError) as ctx:
            async with store.conexion() as conn:
                await _consulta(conn, "BEGIN NOT ATOMIC SIGNAL SQLSTATE '70100' "
                                      "SET MYSQL_ERRNO = 1927, MESSAGE_TEXT = 'simulado'; END")
        self.assertEqual(ctx.exception.args[0], 1927)
        self.assertIsNone(conn.ultimo_error_sano)
        self.assertTrue(_crudo(conn).closed, "un 1927 del servidor devolvio la conexion al pool")


class EnvoltorioInvalidadoTest(_ConBase):
    """Revision de eca432b: una tarea hermana que conserva el envoltorio y lo
    usa DESPUES de que la conexion volvio al pool la estaria usando para otro
    pedido. Al salir de `conexion()` el envoltorio queda invalidado."""

    TAMANIO = "1"  # un solo socket: el siguiente pedido recibe el mismo

    async def _com_select(self, conn) -> int:
        filas = await _consulta(conn, "SHOW SESSION STATUS LIKE 'Com_select'")
        return int(filas[0][1])

    async def test_hermana_que_despierta_despues_no_usa_la_conexion_de_otro(self):
        evento = asyncio.Event()

        async def hermana(c):
            await evento.wait()
            return await _consulta(c, "SELECT 'sql de la hermana'")

        async with store.conexion() as conn:
            crudo = conn.crudo
            tarea = asyncio.create_task(hermana(conn))
            await asyncio.sleep(0)

        async with store.conexion() as otra:
            self.assertIs(_crudo(otra), crudo, "el pool de tamano 1 no reentrego el mismo socket")
            antes = await self._com_select(otra)
            evento.set()
            resultado = (await _a_lo_sumo(asyncio.gather(tarea, return_exceptions=True)))[0]
            despues = await self._com_select(otra)
        # SHOW STATUS no cuenta en Com_select: solo un SELECT de la hermana lo mueve.
        self.assertEqual(despues - antes, 0, "el SQL de la hermana llego a la sesion del otro pedido")
        self.assertIsInstance(resultado, store.ConexionInvalidada,
                              f"la hermana uso la conexion de otro pedido: {resultado!r}")

    async def test_todo_acceso_posterior_falla_sin_tocar_la_conexion(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur_abierto:
                await cur_abierto.execute("SELECT 1")
            cur_vivo = await conn.cursor()
        accesos = (
            lambda: conn.crudo,
            lambda: conn.closed,
            lambda: conn.cursor(),
            lambda: conn.commit,
            lambda: cur_vivo.rowcount,
            lambda: cur_vivo.nextset,
        )
        for acceso in accesos:
            with self.subTest(acceso=acceso):
                with self.assertRaises(store.ConexionInvalidada):
                    acceso()
        for llamada in (lambda: cur_vivo.execute("SELECT 1"), lambda: cur_vivo.fetchall(),
                        lambda: cur_vivo.close()):
            with self.subTest(llamada=llamada):
                with self.assertRaises(store.ConexionInvalidada):
                    await llamada()
        async with store.conexion() as otra:
            self.assertEqual(await _consulta(otra, "SELECT 1"), ((1,),))

    async def test_salida_normal_con_llamada_hermana_en_vuelo_cierra(self):
        """La hermana ya esta leyendo cuando el bloque termina sin error: el
        socket esta a mitad de una respuesta y no puede volver al pool."""
        async with store.conexion() as conn:
            tarea = asyncio.create_task(_consulta(conn, "SELECT SLEEP(0.5)"))
            await asyncio.sleep(0.1)
        cerrada = _crudo(conn).closed
        await _a_lo_sumo(asyncio.gather(tarea, return_exceptions=True))
        self.assertTrue(cerrada, "volvio al pool con una llamada propia en vuelo")
        async with store.conexion() as otra:
            self.assertEqual(await _consulta(otra, "SELECT 1"), ((1,),))

    async def test_se_invalida_tambien_al_salir_con_error(self):
        with self.assertRaises(RuntimeError):
            async with store.conexion() as conn:
                raise RuntimeError("x")
        with self.assertRaises(store.ConexionInvalidada):
            conn.cursor()


class EnvoltorioPuroTest(unittest.IsolatedAsyncioTestCase):
    """Puro, sin base: la marca y la decision de ConexionVigilada."""

    class _CursorFalso:
        def __init__(self, error=None, espera=None):
            self.error, self.espera, self.closed = error, espera, False

        async def execute(self, *a, **k):
            if self.espera is not None:
                await self.espera.wait()
            if self.error is not None:
                raise self.error

        async def close(self):
            self.closed = True

    def _vigilada(self):
        return store.ConexionVigilada(object())

    async def test_marca_y_reusa_solo_ese_objeto(self):
        c = self._vigilada()
        e = aiomysql.ProgrammingError(1146, "x")
        with self.assertRaises(aiomysql.ProgrammingError):
            await store.CursorVigilado(c, self._CursorFalso(e)).execute("q")
        self.assertIs(c.ultimo_error_sano, e)
        self.assertTrue(c.reutilizable_tras(e))
        self.assertFalse(c.reutilizable_tras(aiomysql.ProgrammingError(*e.args)))
        self.assertFalse(c.reutilizable_tras(RuntimeError()))

    async def test_sin_marca_no_reusa(self):
        c = self._vigilada()
        self.assertFalse(c.reutilizable_tras(aiomysql.ProgrammingError(1146, "x")))

    async def test_error_fuera_de_la_lista_no_marca(self):
        c = self._vigilada()
        e = aiomysql.OperationalError(2013, "lost")
        with self.assertRaises(aiomysql.OperationalError):
            await store.CursorVigilado(c, self._CursorFalso(e)).execute("q")
        self.assertIsNone(c.ultimo_error_sano)
        self.assertFalse(c.reutilizable_tras(e))

    async def test_llamada_hermana_en_vuelo_no_reusa(self):
        c = self._vigilada()
        e = aiomysql.ProgrammingError(1146, "x")
        evento = asyncio.Event()
        hermana = asyncio.create_task(store.CursorVigilado(c, self._CursorFalso(espera=evento)).execute("q"))
        await asyncio.sleep(0)
        with self.assertRaises(aiomysql.ProgrammingError):
            await store.CursorVigilado(c, self._CursorFalso(e)).execute("q")
        self.assertFalse(c.reutilizable_tras(e), "reuso con otra llamada propia en vuelo")
        evento.set()
        await hermana
        self.assertTrue(c.reutilizable_tras(e))

    async def test_cada_metodo_vigilado_marca(self):
        e = aiomysql.IntegrityError(1062, "dup")
        for metodo in ("execute", "executemany", "fetchone", "fetchmany", "fetchall"):
            with self.subTest(metodo=metodo):
                c = self._vigilada()
                crudo = self._CursorFalso()

                async def lanza(*a, **k):
                    raise e

                setattr(crudo, metodo, lanza)
                cur = store.CursorVigilado(c, crudo)
                args = ("q", []) if metodo == "executemany" else (("q",) if metodo == "execute" else ())
                with self.assertRaises(aiomysql.IntegrityError):
                    await getattr(cur, metodo)(*args)
                self.assertIs(c.ultimo_error_sano, e)
                self.assertEqual(c._en_vuelo, 0)


_TABLA_LOCKS = "jacobs_pool_test_locks"


class ErroresDeLockRealesTest(_ConBase):
    """1205 y 1213 REALES del servidor, con dos conexiones compitiendo."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.duena = await _conexion_directa()
        self.addAsyncCleanup(self._limpiar)
        await _consulta(self.duena, f"DROP TABLE IF EXISTS {_TABLA_LOCKS}")
        await _consulta(
            self.duena,
            f"CREATE TABLE {_TABLA_LOCKS} (id INT PRIMARY KEY, v INT NOT NULL) ENGINE=InnoDB",
        )
        await _consulta(self.duena, f"INSERT INTO {_TABLA_LOCKS} VALUES (1, 0), (2, 0)")

    async def _limpiar(self):
        if not self.duena.closed:
            try:
                await self.duena.rollback()
            finally:
                self.duena.close()
        otra = await _conexion_directa()
        try:
            await _consulta(otra, f"DROP TABLE IF EXISTS {_TABLA_LOCKS}")
        finally:
            otra.close()

    async def _bloquear_fila_1(self):
        await self.duena.begin()
        await _consulta(self.duena, f"UPDATE {_TABLA_LOCKS} SET v = v + 1 WHERE id = 1")

    _ESPERA_CORTA = f"SET STATEMENT innodb_lock_wait_timeout=1 FOR UPDATE {_TABLA_LOCKS} SET v = v + 1 WHERE id = 1"

    async def test_1205_real_con_transaccion_abierta_descarta(self):
        await self._bloquear_fila_1()
        with self.assertRaises(aiomysql.OperationalError) as ctx:
            async with store.conexion() as conn:
                await conn.begin()
                await _consulta(conn, f"UPDATE {_TABLA_LOCKS} SET v = v + 1 WHERE id = 2")
                await _consulta(conn, self._ESPERA_CORTA)
        self.assertEqual(ctx.exception.args[0], 1205)
        self.assertTrue(_crudo(conn).closed, "volvio al pool con la transaccion abierta tras un 1205")

    async def test_1205_real_en_autocommit_la_devuelve(self):
        """El 1205 de la lista blanca, de punta a punta: sin transaccion la
        conexion esta sana y vuelve."""
        await self._bloquear_fila_1()
        with self.assertRaises(aiomysql.OperationalError) as ctx:
            async with store.conexion() as conn:
                await _consulta(conn, self._ESPERA_CORTA)
        self.assertEqual(ctx.exception.args[0], 1205)
        self.assertFalse(_crudo(conn).closed, "un 1205 sin transaccion descarto una conexion sana")
        async with store.conexion() as otra:
            self.assertIs(_crudo(otra), _crudo(conn))
            self.assertEqual(await _consulta(otra, "SELECT 1"), ((1,),))

    async def test_1213_real_con_transaccion_descarta(self):
        # La duena pesa mas (muchas filas modificadas): InnoDB elige como
        # victima del deadlock a la transaccion mas liviana, la del pool.
        await self._bloquear_fila_1()
        await _consulta(
            self.duena,
            f"INSERT INTO {_TABLA_LOCKS} SELECT seq, 0 FROM seq_100_to_300",
        )
        espera = None
        with self.assertRaises(aiomysql.OperationalError) as ctx:
            async with store.conexion() as conn:
                await conn.begin()
                await _consulta(conn, f"UPDATE {_TABLA_LOCKS} SET v = v + 1 WHERE id = 2")
                espera = asyncio.create_task(_consulta(
                    self.duena, f"UPDATE {_TABLA_LOCKS} SET v = v + 1 WHERE id = 2"))
                await asyncio.sleep(0.3)
                await _a_lo_sumo(_consulta(conn, f"UPDATE {_TABLA_LOCKS} SET v = v + 1 WHERE id = 1"))
        self.assertEqual(ctx.exception.args[0], 1213)
        self.assertTrue(_crudo(conn).closed, "volvio al pool tras un deadlock con transaccion")
        await _a_lo_sumo(espera)


class ClasificacionDeErrorTest(unittest.TestCase):
    """Puro, sin base: `_codigo_del_servidor_sano`."""

    def test_positivos_uno_por_clase_de_la_lista(self):
        for exc in (
            aiomysql.ProgrammingError(1146, "no such table"),
            aiomysql.IntegrityError(1062, "duplicate entry"),
            aiomysql.DataError(1406, "data too long"),
            aiomysql.NotSupportedError(1235, "not supported yet"),
            aiomysql.OperationalError(1205, "lock wait timeout"),
            aiomysql.OperationalError(1213, "deadlock"),
        ):
            with self.subTest(exc=repr(exc)):
                self.assertTrue(store._codigo_del_servidor_sano(exc))

    def test_negativos(self):
        for exc in (
            aiomysql.ProgrammingError(999, "x"),
            aiomysql.ProgrammingError(1999, "x"),
            aiomysql.ProgrammingError(2000, "x"),
            aiomysql.ProgrammingError(2014, "commands out of sync"),
            aiomysql.ProgrammingError(2999, "x"),
            aiomysql.ProgrammingError(3000, "x"),
            aiomysql.ProgrammingError(True, "x"),
            aiomysql.ProgrammingError("Cursor closed"),
            aiomysql.ProgrammingError(),
            aiomysql.ProgrammingError(1062, "codigo de IntegrityError en otra clase"),
            aiomysql.InternalError(1105, "unknown"),
            aiomysql.InternalError(1146, "x"),
            aiomysql.OperationalError(2013, "lost connection"),
            aiomysql.OperationalError(1205.0, "codigo no entero"),
            aiomysql.OperationalError(1146, "x"),
            aiomysql.InterfaceError(0, ""),
            ValueError(1146),
        ):
            with self.subTest(exc=repr(exc)):
                self.assertFalse(store._codigo_del_servidor_sano(exc))

    def test_1927_pymysql_lo_mapea_a_operational_y_no_se_reusa(self):
        import struct

        from pymysql import err

        with self.assertRaises(err.OperationalError) as ctx:
            err.raise_mysql_exception(b"\xff" + struct.pack("<h", 1927) + b"#70100Connection was killed")
        self.assertIs(type(ctx.exception), err.OperationalError)
        self.assertEqual(ctx.exception.args[0], 1927)
        self.assertFalse(store._codigo_del_servidor_sano(ctx.exception))

    def test_los_positivos_salen_de_la_tabla_de_pymysql(self):
        """Fija el supuesto: pymysql elige la clase POR el codigo."""
        import struct

        from pymysql import err

        for codigo, clase in ((1146, err.ProgrammingError), (1062, err.IntegrityError),
                              (1406, err.DataError), (1235, err.NotSupportedError),
                              (1205, err.OperationalError), (1213, err.OperationalError)):
            with self.subTest(codigo=codigo):
                with self.assertRaises(clase) as ctx:
                    err.raise_mysql_exception(b"\xff" + struct.pack("<h", codigo) + b"#HY000x")
                self.assertIs(type(ctx.exception), clase)
                self.assertTrue(store._codigo_del_servidor_sano(ctx.exception))


class EsperaAcotadaTest(_ConBase):
    TAMANIO = "1"

    async def test_pool_lleno_falla_con_limite_y_no_espera_para_siempre(self):
        with patch.dict(os.environ, {"JAX_DB_CONNECT_TIMEOUT_SECONDS": "1"}):
            async with store.conexion():
                t0 = time.monotonic()
                async def segunda():
                    async with store.conexion():
                        pass

                # wait_for de 5 s: sin el limite del store, el test FALLA en vez
                # de colgarse (visto con la mutacion que lo quita). El mensaje
                # distingue quien corto.
                with self.assertRaisesRegex(TimeoutError, "sin conexion libre"):
                    await asyncio.wait_for(segunda(), timeout=5)
                self.assertLess(time.monotonic() - t0, 3.0)
        pool = await store.obtener_pool()
        self.assertEqual(len(pool._used), 0)

    async def test_descartar_una_conexion_despierta_al_que_espera(self):
        """aiomysql 0.3.2 no despierta a los que esperan cuando la conexion
        devuelta esta cerrada: con el pool lleno, el que espera se quedaria
        hasta el limite aunque ya hay hueco."""
        with patch.dict(os.environ, {"JAX_DB_CONNECT_TIMEOUT_SECONDS": "5"}):
            liberar = asyncio.Event()

            async def dueño():
                with self.assertRaises(RuntimeError):
                    async with store.conexion():
                        await liberar.wait()
                        raise RuntimeError("se descarta")

            tarea = asyncio.create_task(dueño())
            await asyncio.sleep(0.2)

            async def esperador():
                async with store.conexion() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT 1")
                        return await cur.fetchone()

            espera = asyncio.create_task(esperador())
            await asyncio.sleep(0.2)
            t0 = time.monotonic()
            liberar.set()
            await _a_lo_sumo(tarea)
            self.assertEqual(await asyncio.wait_for(espera, timeout=10), (1,))
            self.assertLess(time.monotonic() - t0, 2.0, "el que esperaba no se desperto")


class AvisoPerdidoTest(_ConBase):
    """Ronda de revision 2026-09-17: con el pool lleno, que se CANCELE a quien
    libera o a quien espera no puede dejar a los demas esperando hasta el limite
    con una conexion libre (TimeoutError falso)."""

    TAMANIO = "1"

    async def _tomar_con_timeout(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                return await cur.fetchone()

    async def test_cancelar_al_que_libera_no_pierde_el_aviso(self):
        with patch.dict(os.environ, {"JAX_DB_CONNECT_TIMEOUT_SECONDS": "3"}):
            await self._tomar_con_timeout()  # calienta: la conexion ya existe
            listo = asyncio.Event()
            soltar = asyncio.Event()

            async def dueño():
                async with store.conexion():
                    listo.set()
                    await soltar.wait()
                # Aca, ya fuera del cuerpo, se avisa a los que esperan.
                await asyncio.sleep(3600)

            a = asyncio.create_task(dueño())
            await _a_lo_sumo(listo.wait())
            espera = asyncio.create_task(self._tomar_con_timeout())
            await asyncio.sleep(0.2)
            soltar.set()
            # Un solo turno del loop: el dueño sale del cuerpo y queda
            # esperando el aviso; se lo cancela JUSTO ahi.
            await asyncio.sleep(0)
            a.cancel()
            t0 = time.monotonic()
            self.assertEqual(await _a_lo_sumo(espera), (1,))
            self.assertLess(time.monotonic() - t0, 1.5, "el aviso se perdio con la cancelacion")
            with self.assertRaises(asyncio.CancelledError):
                await _a_lo_sumo(a)

    async def test_cancelar_a_un_esperador_no_bloquea_al_siguiente(self):
        with patch.dict(os.environ, {"JAX_DB_CONNECT_TIMEOUT_SECONDS": "3"}):
            listo = asyncio.Event()
            soltar = asyncio.Event()

            async def dueño():
                async with store.conexion():
                    listo.set()
                    await soltar.wait()

            a = asyncio.create_task(dueño())
            await _a_lo_sumo(listo.wait())
            cancelado = asyncio.create_task(self._tomar_con_timeout())
            await asyncio.sleep(0.1)
            siguiente = asyncio.create_task(self._tomar_con_timeout())
            await asyncio.sleep(0.1)
            cancelado.cancel()
            soltar.set()
            t0 = time.monotonic()
            self.assertEqual(await _a_lo_sumo(siguiente), (1,))
            self.assertLess(time.monotonic() - t0, 1.5)
            await _a_lo_sumo(a)
            pool = await store.obtener_pool()
            self.assertEqual(len(pool._used), 0)


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class EnFrioTest(unittest.IsolatedAsyncioTestCase):
    """SIN calentar en setUp: la carrera solo existe cuando no hay pool."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)

    async def test_pedidos_concurrentes_en_frio_crean_un_solo_pool(self):
        creados = []
        original = aiomysql.create_pool

        async def contar(*a, **kw):
            p = await original(*a, **kw)
            creados.append(p)
            return p

        with patch.object(store.aiomysql, "create_pool", contar):
            pools = await asyncio.gather(*[store.obtener_pool() for _ in range(20)])
        extra = [p for p in creados if p is not pools[0]]
        for p in extra:  # no dejar sockets de los pools de la carrera
            p.close()
            await p.wait_closed()
        self.assertEqual(len(creados), 1, f"{len(creados)} pools creados por 20 pedidos en frio")
        self.assertEqual(len({id(p) for p in pools}), 1)


class CicloDeVidaTest(_ConBase):
    async def test_cerrar_pool_cierra_y_el_siguiente_pedido_crea_otro(self):
        p1 = await store.obtener_pool()
        await store.cerrar_pool()
        self.assertTrue(p1.closed)
        p2 = await store.obtener_pool()
        self.assertIsNot(p1, p2)
        self.assertEqual((await ada.una_fila("SELECT 1 AS uno"))["uno"], 1)

    async def test_el_tamano_sale_de_la_variable(self):
        pool = await store.obtener_pool()
        self.assertEqual(pool.maxsize, int(self.TAMANIO))


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class UnPoolPorLoopTest(unittest.TestCase):
    def test_dos_loops_no_comparten_pool(self):
        """El pool de aiomysql queda atado al loop que lo creo (misma trampa
        que resolvio jax-platform en backend/db/connection.py)."""
        pools = []

        async def uso():
            fila = await ada.una_fila("SELECT 1 AS uno")
            pools.append(await store.obtener_pool())
            await store.cerrar_pool()
            return fila["uno"]

        self.assertEqual(asyncio.run(uso()), 1)
        self.assertEqual(asyncio.run(uso()), 1)
        self.assertIsNot(pools[0], pools[1])


    def test_el_fin_del_loop_cierra_el_pool_aunque_nadie_llame_a_cerrar(self):
        """Red para quien crea un loop y se olvida de cerrar_pool(): sin ella la
        suite dejaba 29 conexiones que el recolector cerraba con el loop muerto."""
        pools = []

        async def uso_sin_cerrar():
            await ada.una_fila("SELECT 1 AS uno")
            pool = await store.obtener_pool()
            pools.append((pool, list(pool._free)))

        asyncio.run(uso_sin_cerrar())
        pool, libres = pools[0]
        self.assertTrue(pool.closed, "el pool sobrevivio a su loop")
        self.assertTrue(libres and all(c.closed for c in libres))


class CierreAcotadoTest(_ConBase):
    async def test_una_conexion_colgada_no_cuelga_el_apagado(self):
        with patch.dict(os.environ, {"JAX_DB_CONNECT_TIMEOUT_SECONDS": "1"}):
            capturada: list = []
            entro = asyncio.Event()

            async def colgada():
                async with store.conexion() as conn:
                    capturada.append(conn)
                    entro.set()
                    await asyncio.sleep(30)

            tarea = asyncio.create_task(colgada())
            await _a_lo_sumo(entro.wait())
            pool = await store.obtener_pool()
            t0 = time.monotonic()
            with self.assertLogs("jacobs.store", level="ERROR"):
                await store.cerrar_pool()
            self.assertLess(time.monotonic() - t0, 3.0)
            self.assertTrue(pool.closed)
            self.assertTrue(_crudo(capturada[0]).closed)
            tarea.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await _a_lo_sumo(tarea)


class DevolucionPorConstruccionTest(_ConBase):
    """El cuelgue del apagado (hallazgo del 2026-09-17).

    `conexion()` decidia el destino de la conexion (sesion limpia? desechable?)
    y recien despues llamaba `pool.release(conn)`, las dos cosas en el MISMO
    `finally`. Una excepcion inesperada en el tramo de la decision -- la
    destapo un AttributeError de un doble de test -- saltaba el release: la
    conexion quedaba marcada en `_used` del pool de aiomysql y
    `pool.wait_closed()` la esperaba PARA SIEMPRE. Consecuencia real: el
    apagado del servicio no termina nunca.

    Dos garantias, dos capas:
      1. `conexion()`: la devolucion vive en un `finally` propio, afuera del
         tramo que puede explotar. Pase lo que pase, la conexion vuelve.
      2. `_cerrar()`: `wait_closed()` va acotado y con `terminate()` detras. Si
         una conexion queda atrapada igual -- por un defecto nuestro o de
         aiomysql --, el apagado TERMINA, con ERROR en el log.

    Toda espera de estos tests esta acotada: un cuelgue tiene que dar FAILED
    con un mensaje claro, no colgar la suite entera."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        # La limpieza del padre (`cerrar_pool`) NO esta acotada: contra el
        # codigo viejo estos tests dejan una conexion atrapada y esa limpieza
        # colgaria la suite entera. Los cleanups corren LIFO: este, agregado
        # despues, corre ANTES y deja el pool cortado a la fuerza.
        self.addAsyncCleanup(self._cortar_a_la_fuerza)

    async def _cortar_a_la_fuerza(self):
        estado = store._pools.get(asyncio.get_running_loop())
        if estado is not None and estado.pool is not None:
            estado.pool.terminate()

    async def _apagado_acotado(self, pool, segundos: float = 10):
        """`cerrar_pool()` con limite duro. Si vence, corta a la fuerza (para no
        dejar la limpieza colgada) y FALLA con el motivo."""
        try:
            await asyncio.wait_for(store.cerrar_pool(), timeout=segundos)
        except TimeoutError:
            pool.terminate()
            self.fail(
                f"cerrar_pool() no termino en {segundos} s: una conexion quedo atrapada en "
                "_used y wait_closed() la espera para siempre -- el apagado del servicio "
                "se cuelga."
            )

    async def test_error_al_decidir_el_destino_devuelve_la_conexion(self):
        """Contra el codigo viejo: el AttributeError sube sin release y la
        conexion queda en `_used`."""
        pool = await store.obtener_pool()
        with patch.object(store, "_sesion_reutilizable",
                          side_effect=AttributeError("doble sin get_autocommit")):
            with self.assertLogs("jacobs.store", level="ERROR"):
                async with store.conexion() as conn:
                    pass
        self.assertEqual(
            len(pool._used), 0,
            "la conexion quedo marcada en uso: pool.wait_closed() la espera para siempre",
        )
        self.assertTrue(
            _crudo(conn).closed,
            "fail-closed: sin poder decidir si la sesion sirve, la conexion no se reusa",
        )
        # Y el pool sigue sirviendo: el permiso del semaforo tambien volvio.
        self.assertEqual((await ada.una_fila("SELECT 1 AS uno"))["uno"], 1)

    async def test_error_al_decidir_no_cuelga_el_apagado(self):
        """El cuelgue, de punta a punta: la excepcion en el tramo de la decision
        y despues el apagado. Contra el codigo viejo, `cerrar_pool()` no vuelve."""
        pool = await store.obtener_pool()
        with patch.object(store, "_sesion_reutilizable",
                          side_effect=AttributeError("doble sin get_autocommit")):
            # Contra el codigo viejo la excepcion sube; con el arreglo, no. Lo
            # que se prueba aca es el apagado, no por donde sale el error.
            with contextlib.suppress(AttributeError):
                async with store.conexion():
                    pass
        await self._apagado_acotado(pool)
        self.assertTrue(pool.closed)

    async def test_si_release_falla_la_conexion_no_queda_en_uso(self):
        """`pool.release()` tambien puede explotar (mira la transaccion de la
        conexion). Ni asi se pierde el permiso ni sube un error ajeno."""
        pool = await store.obtener_pool()
        with patch.object(aiomysql.Pool, "release", side_effect=RuntimeError("release roto")):
            with self.assertLogs("jacobs.store", level="ERROR"):
                async with store.conexion() as conn:
                    pass
        self.assertTrue(_crudo(conn).closed, "fail-closed: si no vuelve al pool, se cierra")
        self.assertEqual((await ada.una_fila("SELECT 1 AS uno"))["uno"], 1,
                         "el permiso del semaforo no volvio")

    async def test_el_apagado_termina_aunque_una_conexion_quede_en_uso(self):
        """Capa 2, independiente de `conexion()`: si una conexion queda marcada
        en `_used` pase lo que pase, el apagado igual TERMINA (acotado +
        terminate), con ERROR en el log. Se simula un aiomysql cuyo `release()`
        no saca la conexion de `_used`."""
        with patch.dict(os.environ, {"JAX_DB_CONNECT_TIMEOUT_SECONDS": "1"}):
            pool = await store.obtener_pool()
            with patch.object(aiomysql.Pool, "release", lambda self_, conn: None):
                async with store.conexion() as conn:
                    pass
            self.assertIn(_crudo(conn), pool._used, "el arnes no dejo la conexion atrapada")
            t0 = time.monotonic()
            with self.assertLogs("jacobs.store", level="ERROR"):
                await self._apagado_acotado(pool)
            self.assertLess(time.monotonic() - t0, 6.0, "el apagado tardo mas que los limites")
            self.assertTrue(pool.closed)


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class RegistroDeLoopsTest(unittest.TestCase):
    def test_n_loops_terminados_no_dejan_entradas_en_el_registro(self):
        """El Pool retiene su loop: un WeakKeyDictionary nunca soltaba la
        entrada. El fin del loop la quita."""
        antes = len(store._pools)

        async def uso_sin_cerrar():
            await ada.una_fila("SELECT 1 AS uno")

        for _ in range(5):
            asyncio.run(uso_sin_cerrar())
        self.assertEqual(len(store._pools), antes, "entradas de loops muertos en el registro")


class VersionDeAiomysqlTest(unittest.TestCase):
    """Puro. El pool depende de comportamiento de aiomysql revisado a mano
    (release() que saca de _used antes de devolver, _acquire que solo espera en
    cond.wait si el pool esta lleno, connect_timeout solo del socket). Otra
    version puede cambiarlo sin que ningun test de logica lo note."""

    def test_la_version_instalada_es_la_revisada(self):
        self.assertEqual(
            aiomysql.__version__, store.AIOMYSQL_REVISADO,
            f"aiomysql {aiomysql.__version__} instalado, revisado {store.AIOMYSQL_REVISADO}: "
            "jacobs/store.py (pool por loop con semaforo propio) asume el "
            "comportamiento de release/_acquire/connect_timeout de la version "
            "revisada. Releer aiomysql/pool.py y connection.py antes de subirla.",
        )

    def test_requirements_fija_la_version_revisada(self):
        pins = [l.strip() for l in (RAIZ / "requirements.txt").read_text().splitlines()
                if l.strip().lower().startswith("aiomysql")]
        self.assertEqual(pins, [f"aiomysql=={store.AIOMYSQL_REVISADO}"])

    def test_pymysql_instalado_y_fijado_es_el_revisado(self):
        """La lista blanca de errores reutilizables lee la tabla `error_map` de
        PyMySQL (que clase elige para cada codigo): otra version puede moverla."""
        from importlib.metadata import version

        # `pymysql.__version__` es la compatibilidad con mysqlclient (2.2.x), no
        # la version de la distribucion.
        self.assertEqual(version("PyMySQL"), store.PYMYSQL_REVISADO)
        pins = [l.strip() for l in (RAIZ / "requirements.txt").read_text().splitlines()
                if l.strip().lower().startswith("pymysql")]
        self.assertEqual(pins, [f"PyMySQL=={store.PYMYSQL_REVISADO}"])


class MigradosAlPoolTest(unittest.TestCase):
    """Puro: los modulos del camino de pedidos usan store.conexion(), no una
    conexion suelta (ronda de revision 2026-09-17)."""

    MIGRADOS = (
        "las_manos/motor_registry/facet_policy.py",
        "jacobs/usage_writer.py",
        "las_manos/motor_registry/usage_writer.py",
    )

    def test_sin_aiomysql_connect(self):
        for rel in self.MIGRADOS:
            arbol = ast.parse((RAIZ / rel).read_text())
            llamadas = [
                n.lineno for n in ast.walk(arbol)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "connect"
            ]
            self.assertEqual(llamadas, [], f"{rel} abre conexiones sueltas en {llamadas}")
            self.assertIn("conexion()", (RAIZ / rel).read_text(), rel)


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class DedicadasVivasTest(unittest.IsolatedAsyncioTestCase):
    """El tope del pool NO es el tope de conexiones del proceso (2026-09-17).

    La carga del 2026-09-17 vio hasta 13 conexiones del proceso contra
    `JAX_DB_POOL_MAX=10`. `conexion_dedicada()` abre FUERA del pool a
    proposito, asi que el total es pool + dedicadas vivas. Este contador es lo
    que permite medirlo en vez de razonarlo.
    """

    async def test_abrir_suma_y_cerrar_resta(self):
        antes = store.dedicadas_vivas()
        conn = await store.conexion_dedicada()
        try:
            self.assertEqual(store.dedicadas_vivas(), antes + 1)
        finally:
            conn.close()
        self.assertEqual(store.dedicadas_vivas(), antes)

    async def test_cerrar_dos_veces_resta_una_sola_vez(self):
        """Sin idempotencia el contador se iria a negativo y taparia una fuga."""
        antes = store.dedicadas_vivas()
        conn = await store.conexion_dedicada()
        conn.close()
        conn.close()
        self.assertEqual(store.dedicadas_vivas(), antes)

    async def test_la_que_nadie_cierra_queda_contada(self):
        """Una dedicada sin cerrar TIENE que verse: es el caso que interesa."""
        antes = store.dedicadas_vivas()
        conn = await store.conexion_dedicada()
        self.assertEqual(store.dedicadas_vivas(), antes + 1)
        conn.close()  # limpieza del test, ya medido

    async def test_el_pool_no_cuenta_como_dedicada(self):
        """Si el pool sumara acá, el número no distinguiría una cosa de la otra."""
        await store.obtener_pool()
        antes = store.dedicadas_vivas()
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
            self.assertEqual(store.dedicadas_vivas(), antes)
        self.assertEqual(store.dedicadas_vivas(), antes)

    async def test_found_rows_tambien_cuenta(self):
        """Las escrituras condicionales son DOS de los tres llamadores: si su
        rama no contara, el excedente medido quedaría sin explicar."""
        antes = store.dedicadas_vivas()
        conn = await store.conexion_dedicada(found_rows=True)
        try:
            self.assertEqual(store.dedicadas_vivas(), antes + 1)
        finally:
            conn.close()
        self.assertEqual(store.dedicadas_vivas(), antes)


class ExcepcionAlPoolTest(unittest.TestCase):
    """Puro: la lista de llamadores de `conexion_dedicada()` EN CODIGO DE
    SERVICIO no crece sola (2026-09-17).

    `conexion_dedicada()` es la unica excepcion al pool: abre una conexion
    fuera de el y quien la pide la cierra. Su docstring decia "tiene
    exactamente DOS llamadores" y era falso -- tests y scripts de medicion
    tambien la usan --, asi que la garantia no la sostenia nadie. Este guard
    la sostiene: enumera por AST los llamadores en `jacobs/`, `las_manos/`,
    `jax/` y `tools/` (sin tests ni `scripts/`, que son medicion, no camino
    de pedidos) y falla si aparece uno nuevo.
    """

    # Cada entrada es un llamador VIVO en codigo de servicio, con su razon.
    LLAMADORES = {
        # (candado_de_activos salio de esta lista el 2026-09-17, con el GET_LOCK
        # del cupo: era la UNICA excepcion al pool que existia por ESPERAR. El
        # cupo ya no se espera -- se decide dentro de la misma sentencia que
        # escribe --, asi que crear dejo de abrir una conexion dedicada.)
        # Escrituras CONDICIONALES por epoca: necesitan CLIENT.FOUND_ROWS, que
        # se negocia en el handshake y no se enciende por sesion.
        "jacobs/store.py::_ejecutar_condicional",
        "jacobs/store.py::continuar_transaccion",
    }

    CARPETAS = ("jacobs", "las_manos", "jax", "tools")

    @staticmethod
    def _es_de_servicio(f: Path) -> bool:
        if any(p in (".venv", "__pycache__", "tests", "scripts") for p in f.parts):
            return False
        return not (f.name.endswith("_test.py") or f.name.startswith("test_")
                    or f.name == "conftest.py")

    def _hallados(self) -> set[str]:
        hallados = set()
        for carpeta in self.CARPETAS:
            raiz = RAIZ / carpeta
            if not raiz.is_dir():
                continue
            for f in sorted(raiz.rglob("*.py")):
                if not self._es_de_servicio(f):
                    continue
                arbol = ast.parse(f.read_text(errors="ignore"))
                # Por AST, no por texto: un comentario puede nombrarla; el
                # codigo no. Se atribuye cada uso a la funcion que lo contiene.
                for nodo in ast.walk(arbol):
                    if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    for hijo in ast.walk(nodo):
                        nombre = getattr(hijo, "attr", None) or getattr(hijo, "id", None)
                        if (isinstance(hijo, (ast.Name, ast.Attribute))
                                and nombre == "conexion_dedicada"):
                            hallados.add(f"{f.relative_to(RAIZ)}::{nodo.name}")
        return hallados

    def test_la_excepcion_al_pool_no_se_amplia_sola(self):
        hallados = self._hallados()
        nuevos = sorted(hallados - self.LLAMADORES)
        self.assertEqual(
            nuevos, [],
            "La excepcion al pool NO se amplia sin una decision: "
            f"llamador(es) nuevo(s) de conexion_dedicada() en codigo de servicio: {nuevos}. "
            "Toda conexion del camino de pedidos va por store.conexion() / "
            "conexion_del_pool(); conexion_dedicada() solo cubre CLIENT.FOUND_ROWS y el "
            "GET_LOCK que se espera. Si hace falta uno mas, se decide, se justifica en el "
            "docstring de conexion_dedicada() y se agrega a LLAMADORES con su razon.",
        )

    def test_la_lista_no_tiene_llamadores_muertos(self):
        """Un guard con entradas que ya no existen deja de vigilar en silencio."""
        muertos = sorted(self.LLAMADORES - self._hallados())
        self.assertEqual(muertos, [], f"LLAMADORES nombra lo que ya no existe: {muertos}")

    def test_el_docstring_no_promete_una_garantia_que_no_cumple(self):
        """El docstring decia "exactamente DOS llamadores" contando solo el
        codigo de servicio, y tests y scripts tambien la usan: un comentario
        que afirma una garantia que el codigo no cumple es deuda."""
        doc = store.conexion_dedicada.__doc__ or ""
        self.assertNotIn("exactamente DOS llamadores", doc)
        self.assertIn("CODIGO DE SERVICIO", doc)


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class AdmisionDeFacetUsaElPoolTest(_ConBase):
    async def test_autorizaciones_concurrentes_no_abren_una_conexion_cada_una(self):
        from motor_registry.facet_policy import check_facet_admission

        await store.obtener_pool()
        with _ContadorDeConexiones() as c:
            # return_exceptions: en el job de CI sin la tabla `facet` la
            # consulta falla, pero la conexion ya se pidio -- lo que se cuenta.
            await asyncio.gather(
                *[check_facet_admission("jacobs", "hipatia") for _ in range(20)],
                return_exceptions=True,
            )
        self.assertLessEqual(c.n, int(self.TAMANIO), f"{c.n} conexiones para 20 autorizaciones")

    async def test_autorizaciones_sin_tabla_facet_no_abren_una_conexion_cada_una(self):
        """La condicion EXACTA del job de CI de 13b7759 (base sin `facet`), fijada
        aca para que no dependa de que la base local tenga o no la tabla: la
        consulta se redirige a una tabla inexistente y el SERVIDOR responde 1146."""
        from aiomysql.cursors import Cursor
        from motor_registry.facet_policy import check_facet_admission

        original = Cursor.execute

        async def execute_sin_facet(cur_self, query, args=None):
            return await original(cur_self, query.replace("FROM facet ", f"FROM {_TABLA_INEXISTENTE} "), args)

        await store.obtener_pool()
        with patch.object(Cursor, "execute", execute_sin_facet), _ContadorDeConexiones() as c:
            resultados = await asyncio.gather(
                *[check_facet_admission("jacobs", "hipatia") for _ in range(20)],
                return_exceptions=True,
            )
        codigos = {r.args[0] for r in resultados if isinstance(r, aiomysql.ProgrammingError)}
        self.assertEqual(codigos, {1146}, f"no se reprodujo la base sin `facet`: {resultados[:3]}")
        self.assertLessEqual(c.n, int(self.TAMANIO), f"{c.n} conexiones para 20 autorizaciones")


class TamanioValidadoTest(unittest.TestCase):
    """Puro: sin base."""

    def test_default_sin_variable(self):
        with patch.dict(os.environ):
            os.environ.pop(ENV_TAMANIO, None)
            self.assertEqual(store.tamanio_pool(), store.TAMANIO_POOL_POR_DEFECTO)

    def test_valores_validos(self):
        for v in ("1", str(store.TAMANIO_POOL_MAXIMO)):
            with patch.dict(os.environ, {ENV_TAMANIO: v}):
                self.assertEqual(store.tamanio_pool(), int(v))

    def test_invalidos_fallan_cerrado(self):
        for v in ("", "0", "-1", "abc", "2.5", str(store.TAMANIO_POOL_MAXIMO + 1)):
            with patch.dict(os.environ, {ENV_TAMANIO: v}):
                with self.assertRaises(RuntimeError, msg=repr(v)):
                    store.tamanio_pool()

    def test_el_maximo_deja_lugar_a_los_demas_procesos(self):
        """max_connections medido en la MariaDB de hall9000 el 2026-09-17: 151.
        jax-platform abre hasta 10 por proceso; el resto del ecosistema y una
        consola de emergencia necesitan hueco."""
        self.assertLessEqual(store.TAMANIO_POOL_MAXIMO, 50)
        self.assertLessEqual(store.TAMANIO_POOL_POR_DEFECTO, store.TAMANIO_POOL_MAXIMO)


class ServidorTest(unittest.TestCase):
    """Sin importar server.py (en CI no importa: ver
    tests/test_health_de_las_manos_puede_fallar.py): se lee su AST."""

    def _handlers(self, evento: str) -> list[ast.AsyncFunctionDef]:
        arbol = ast.parse((RAIZ / "las_manos" / "server.py").read_text())
        out = []
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.AsyncFunctionDef):
                for d in nodo.decorator_list:
                    if (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "on_event"
                            and d.args and getattr(d.args[0], "value", None) == evento):
                        out.append(nodo)
        return out

    def test_shutdown_cierra_el_pool(self):
        cuerpos = [ast.unparse(h) for h in self._handlers("shutdown")]
        self.assertTrue(
            any("jacobs_store.cerrar_pool()" in c for c in cuerpos),
            "LAS MANOS no cierra el pool de Jacobs al apagarse",
        )

    def test_el_tamano_se_valida_al_arrancar(self):
        cuerpos = [ast.unparse(h) for h in self._handlers("startup")]
        self.assertTrue(any("jacobs_store.tamanio_pool()" in c for c in cuerpos))

    def test_nadie_abre_conexiones_sueltas_con_get_conn(self):
        """`get_conn()` desaparece: dejarla seria dejar la puerta del defecto."""
        self.assertFalse(hasattr(store, "get_conn"))
        hallazgos = []
        for carpeta in ("jacobs", "las_manos", "loadtest", "tests"):
            for f in (RAIZ / carpeta).rglob("*.py"):
                if ".venv" in f.parts or f.name == "_store_pool_test.py":
                    continue
                # Por AST, no por texto: los comentarios que cuentan la
                # historia pueden nombrarla; el codigo no.
                for nodo in ast.walk(ast.parse(f.read_text(errors="ignore"))):
                    nombre = getattr(nodo, "attr", None) or getattr(nodo, "id", None)
                    if isinstance(nodo, (ast.Name, ast.Attribute)) and nombre == "get_conn":
                        hallazgos.append(f"{f.relative_to(RAIZ)}:{nodo.lineno}")
        self.assertEqual(hallazgos, [])


if __name__ == "__main__":
    unittest.main()
