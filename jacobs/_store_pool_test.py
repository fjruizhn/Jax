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
        self.assertTrue(conn.closed, "una conexion con estado desconocido volvio al pool")
        self.assertNotIn(conn, pool._free)
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
            await tarea
        self.assertTrue(capturada[0].closed, "el socket a mitad de respuesta volvio al pool")
        self.assertEqual(len(pool._used), 0)
        self.assertEqual((await ada.una_fila("SELECT 1 AS uno"))["uno"], 1)

    async def test_autocommit_apagado_no_vuelve_al_pool(self):
        pool, _ = await self._estado()
        async with store.conexion() as conn:
            await conn.autocommit(False)
        self.assertTrue(conn.closed)
        async with store.conexion() as otra:
            async with otra.cursor() as cur:
                await cur.execute("SELECT @@SESSION.autocommit")
                self.assertEqual((await cur.fetchone())[0], 1)

    async def test_transaccion_abierta_no_vuelve_al_pool(self):
        async with store.conexion() as conn:
            await conn.begin()
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        self.assertTrue(conn.closed)

    async def test_desechable_se_cierra_aunque_este_limpia(self):
        """Para quien cambia variables de sesion (el DDL acotado de init_tables)."""
        pool, _ = await self._estado()
        async with store.conexion(desechable=True) as conn:
            pass
        self.assertTrue(conn.closed)
        self.assertNotIn(conn, pool._free)
        self.assertEqual(len(pool._used), 0)

    async def test_conexion_limpia_vuelve_al_pool(self):
        """Control del control: sin esto, cerrar SIEMPRE pasaria los de arriba."""
        async with store.conexion() as conn:
            pass
        pool = await store.obtener_pool()
        self.assertFalse(conn.closed)
        self.assertIn(conn, pool._free)


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
            await tarea
            self.assertEqual(await asyncio.wait_for(espera, timeout=10), (1,))
            self.assertLess(time.monotonic() - t0, 2.0, "el que esperaba no se desperto")


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
            await entro.wait()
            pool = await store.obtener_pool()
            t0 = time.monotonic()
            with self.assertLogs("jacobs.store", level="ERROR"):
                await store.cerrar_pool()
            self.assertLess(time.monotonic() - t0, 3.0)
            self.assertTrue(pool.closed)
            self.assertTrue(capturada[0].closed)
            tarea.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tarea


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
