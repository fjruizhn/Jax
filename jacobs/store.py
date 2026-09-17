"""
Jacobs — Almacén de pipelines en MariaDB.

Base: jax_memory. Tablas: jacobs_pipelines, jacobs_steps.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator

import aiomysql
from pymysql import err as _pymysql_err
from pymysql.constants import CLIENT

from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus

try:
    # LAS MANOS produccion (cwd=las_manos, uvicorn) y jobs con PYTHONPATH
    # incluyendo las_manos/: bare, resuelve a las_manos/db_connect_config.py
    # (symlink) o directo si jacobs corre con las_manos en su propio path.
    from db_connect_config import db_connect_timeout_seconds
except ImportError:
    # CI sin PYTHONPATH propio (p.ej. facet-health-io) y REPL: cwd=raiz del
    # repo, solo el paquete jax.core es importable.
    from jax.core.db_connect_config import db_connect_timeout_seconds


def _db_cfg() -> dict:
    host = os.environ.get("JAX_DB_HOST")
    port = os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    return {
        "host":     host,
        "port":     int(port),
        "user":     os.getenv("JAX_DB_USER", ""),
        "password": os.getenv("JAX_DB_PASSWORD", ""),
        "db":       os.getenv("JAX_DB_NAME", "jax_memory"),
        "charset":  "utf8mb4",
        "autocommit": True,
    }


# --- Pool de conexiones ------------------------------------------------------
# POR QUE (2026-09-17). Hasta eb72e78 cada consulta abria una conexion
# `aiomysql` NUEVA, cerrada al terminar. Una creacion en dry_run hace
# 5-6 llamadas; la Task 8 del frente F midio bajo carga entre 22 y 38 % de fallas
# a 10 VUs en los tres escenarios (incluido el que no toca el contrato): cada
# conexion deja un socket en TIME_WAIT, el rango de puertos efimeros del host se
# agota y MariaDB ni se entera (`Lost connection ... system error 11`). Politica
# 2 de LAS CUATRO DEL RENDIMIENTO: el pool se comparte, no se rehace por request.
#
# UN POOL POR EVENT LOOP, no un global unico -- misma trampa que resolvio
# jax-platform (backend/db/connection.py): el pool de aiomysql queda atado al
# loop que lo creo. En LAS MANOS hay un solo loop y una sola entrada; en los
# tests cada IsolatedAsyncioTestCase trae su loop.
#
# REGISTRO: dict comun, NO WeakKeyDictionary. El Pool guarda una referencia
# fuerte a su loop, asi que una entrada debil nunca se soltaba (revision
# 2026-09-17: 5 `asyncio.run` dejaban 5 entradas vivas). La entrada la quita el
# fin del loop (_guardian) o `cerrar_pool()`; como red, un pedido nuevo purga las
# de loops ya cerrados.
#
# SEMAFORO PROPIO. Cada `conexion()` toma un permiso (hay `maxsize`) ANTES de
# pedirle la conexion a aiomysql, y lo devuelve despues del release. Con eso el
# `_acquire` de aiomysql nunca llega a `cond.wait()`: la espera por hueco es del
# semaforo, que se libera sincronico y no pierde avisos. Antes se dependia de
# `pool._wakeup()` (API privada) para despertar tras un descarte, y cancelar a
# quien liberaba mientras esperaba ese aviso dejaba a los demas colgados hasta
# el limite con una conexion libre (TimeoutError falso, probado).
#
# CICLO DE VIDA. Se crea perezosamente en el primer `conexion()` (en LAS MANOS,
# init_tables() del startup) y se cierra en el shutdown de las_manos/server.py
# con `cerrar_pool()`. Quien cree un loop propio (scripts, tests) lo cierra el;
# si no, lo cierra el fin del loop (ver _guardian_del_pool).
ENV_TAMANIO_POOL = "JAX_JACOBS_DB_POOL_SIZE"

# Version de aiomysql cuyo pool se reviso a mano para este modulo
# (aiomysql/pool.py: release() saca de _used antes de devolver y cierra una
# conexion con transaccion abierta; _acquire solo espera en cond.wait si
# size >= maxsize; connection.py: connect_timeout acota solo el socket). La fija
# requirements.txt y la vigila jacobs/_store_pool_test.py::VersionDeAiomysqlTest.
AIOMYSQL_REVISADO = "0.3.2"
# _codigo_del_servidor_sano lee la tabla error_map de PyMySQL: se fija junto
# con aiomysql y lo vigila VersionDeAiomysqlTest.
PYMYSQL_REVISADO = "1.2.0"

# Medido 2026-09-17 (loadtest/jacobs_subpipelines.js, app aislada sobre
# jax_memory_test, pool-report.md del frente F): ver la tabla de tamanos en el
# reporte. Un solo proceso de LAS MANOS con un solo loop.
TAMANIO_POOL_POR_DEFECTO = 10

# max_connections de la MariaDB de hall9000 = 151 (SHOW GLOBAL VARIABLES,
# 2026-09-17; Max_used_connections=95 historico). jax-platform abre hasta 10 por
# proceso y hay mas clientes (workers de memoria, jobs, la consola de
# emergencia): mas de un tercio del servidor para Jacobs solo seria quitarselo a
# los demas.
TAMANIO_POOL_MAXIMO = 50

# Una conexion ociosa mas vieja que esto se recicla al pedirla. wait_timeout del
# servidor = 28800 s (medido el mismo dia): una hora queda muy por debajo, asi
# que el pool nunca entrega una conexion que MariaDB ya corto por inactividad.
_RECICLAR_SEGUNDOS = 3600


class _PoolDelLoop:
    """Lo que un loop tiene: el candado de creacion, y una vez creado, el pool,
    su semaforo y el guardian que lo cierra al terminar el loop."""

    def __init__(self) -> None:
        self.candado = asyncio.Lock()
        self.pool: aiomysql.Pool | None = None
        self.permisos: asyncio.Semaphore | None = None
        self.guardian: Any = None


# ------------------------------------------------------------------
#  MERGE 2026-09-17 (master <- feat/prevuelo-y-continuar)
#  Las dos ramas escribieron un pool para este módulo el mismo día. Queda el
#  de master (frente F): tiene la higiene de sesión que la otra no tenía
#  (ConexionVigilada, _sesion_reutilizable, _codigo_del_servidor_sano,
#  pool_recycle) y su medición de carga. De la rama de pre-vuelo se conservan,
#  portadas sobre él: la espera de turno SIN PLAZO de los trabajos de fondo
#  (R38 fix round 1), `conexion_del_pool` como nombre alterno de `conexion()`,
#  `_conexion_o_pool` (conexión prestada o del pool), `transaccion` +
#  `EstadoDeTransaccion` (R41: commit incierto) y las conexiones DEDICADAS con
#  CLIENT.FOUND_ROWS de las escrituras condicionales por época
#  (`conexion_dedicada`, ver su docstring). `JAX_DB_POOL_MAX` sigue
#  dimensionando el pool, igual que `JAX_JACOBS_DB_POOL_SIZE`.
# ------------------------------------------------------------------

# --- Pool de conexiones del store de Jacobs (Task 15b y R38, 2026-09-17, LAS CUATRO #2) ---
# Medido (task-15b-report.md): cada pre-vuelo abría 2 conexiones nuevas
# (MotorCatalog.from_db y prevuelo_catalogo.leer_catalogo), ~0,15 ms de CPU
# del event loop cada una más el handshake en la base. Ahora el pre-vuelo
# toma UNA conexión de este pool y la pasa a los dos lectores.
#
# Ruling R38 (2026-09-17): el pool es del STORE, no sólo del pre-vuelo. Medido
# a c=50 sostenido contra /jacobs/preflight: 0 %, 13 %, 49 % y 61 % de errores
# (OperationalError 2013) por get_motor_governance() abriendo una conexión por
# pedido fuera del pool; el inventario (r38-report.md) encontró lo mismo en
# crear, continue, resume y approve-step. Ahora van por este pool las lecturas
# y las escrituras SIN condición de esos caminos (pipeline_get,
# steps_by_pipeline, pipeline_count_active, get_motor_governance,
# pipeline_create, step_upsert, pipeline_update_status, event_append, y --
# desde el fix round 1 -- los dos escritores de la sonda del pre-vuelo,
# facet_health.registrar_evento_de_sonda y usage_writer.record_direct_usage):
# son
# autocommit, no miran el conteo de filas y no dejan estado de sesión, así que
# la conexión que vuelve al pool vuelve igual a como salió. Quedan DEDICADAS
# a propósito: GET_LOCK (candado_de_activos, el candado vive en la sesión) y
# las escrituras condicionales con CLIENT.FOUND_ROWS (ver abajo), además de
# init_tables (cambia lock_wait_timeout de la sesión). Ninguna de estas
# funciones pide una segunda conexión del pool mientras tiene una: con el pool
# lleno, eso sería esperar a sí misma.
#
# - Un pool por proceso y por event loop: se crea perezosamente en el primer
#   pedido y queda atado al loop que lo creó (aiomysql.Pool guarda el loop).
#   Usarlo desde otro loop VIVO se niega con RuntimeError -- fallaría más tarde
#   con un error críptico. LAS MANOS lo cierra en su shutdown y el CLI
#   (tools/jacobs_relaunch.py) antes de salir.
# - R38: el pool se cierra también CUANDO SE APAGA SU LOOP. Los tests y los
#   scripts corren un asyncio.run por llamada al store; sin esto el loop
#   siguiente chocaba con el RuntimeError de arriba, y soltar el pool viejo sin
#   cerrarlo deja que el recolector intente cerrar sus sockets sobre un loop
#   muerto (PytestUnraisableExceptionWarning, medido: +31 en gobernanza-db). Un
#   generador asíncrono guardián, arrancado en el loop del pool, lo cierra en
#   su `finally`: asyncio.run (y uvicorn, que corre sobre asyncio.Runner)
#   llaman loop.shutdown_asyncgens() antes de cerrar el loop. Un loop que se
#   cerró SIN ese paso (loop.close() a mano) deja un pool que no puede volver
#   a usarse nunca: se suelta y se crea otro.
# - minsize=0: crear el pool no toca la base; la primera conexión se abre al
#   pedirla, así una base caída falla en el pedido (503), no en un pool roto
#   guardado.
# - SIN CLIENT.FOUND_ROWS, por la semántica de las escrituras CONDICIONALES:
#   pipeline_tomar_epoca, pipeline_update_status_si_epoca, step_upsert_si_epoca
#   y continuar_transaccion necesitan contar filas ENCONTRADAS (un UPDATE que
#   escribe los mismos valores cuenta 0 sin el flag y la época se daría por
#   perdida), así que siguen con conexion_dedicada(found_rows=True). Lo que
#   va por el pool (lecturas y escrituras sin condición) no mira el conteo, y
#   meterle el flag al pool cambiaría en silencio el conteo de cualquier UPDATE
#   que se agregue después.
# - Pedir una conexión espera a lo sumo JAX_DB_CONNECT_TIMEOUT_SECONDS: con el
#   pool lleno de conexiones colgadas, un pedido no espera para siempre.
# - El TURNO lo da un asyncio.Semaphore propio del tamaño del pool, no la
#   condición interna de aiomysql (fix round 1 de la revisión, 2026-09-17):
#   aiomysql 0.3.2 `Pool.release()` no despierta a quien espera si la conexión
#   devuelta ya está cerrada (sólo agenda `_wakeup()` en la rama
#   `not conn.closed`), así que tras una conexión rota el siguiente esperaba el
#   timeout entero y daba 503. Con el semáforo, nunca hay más pedidos adentro
#   del pool que conexiones posibles: `pool.acquire()` encuentra una libre o
#   abre otra, sin esperar la condición de aiomysql, y soltar el semáforo
#   despierta siempre al siguiente.
# - Una conexión que sale del bloque con una excepción se CIERRA y no vuelve:
#   puede tener filas sin leer o el socket roto. Las que la base cortó en
#   reposo (wait_timeout) las descarta el propio Pool de aiomysql al pedirlas
#   (EOF en el lector).


_pools: dict[asyncio.AbstractEventLoop, _PoolDelLoop] = {}


DB_POOL_MAX = "JAX_DB_POOL_MAX"


def db_pool_max() -> int:
    """Conexiones máximas del pool del store (JAX_DB_POOL_MAX, default 10).

    Ruling R38 (2026-09-17). Se lee al CREAR el pool: un cambio vale después
    de reiniciar el proceso (o de cerrar_pool()). Un valor inválido lanza con
    el nombre de la variable (mismo patrón que jacobs/prevuelo_config.py), no
    cae a un default.

    Derivación del default. Demanda PICO de conexiones a la vez en LAS MANOS:
    al arrancar una ola, cada paso escribe su STEP_STARTED a la vez --
    MAX_PARALLEL_PIPELINES (3) x MAX_STEPS_PER_PIPELINE (20, una ola puede
    tenerlos todos) = 60 -- más los pedidos HTTP en curso (a c=50, 50 más) y
    el reaper (1): ~111. NO se dimensiona al pico: la MariaDB es compartida
    con producción, max_connections=151 y Max_used_connections=96 medido el
    2026-09-17 (SHOW GLOBAL STATUS), así que un pool de 111 la agotaría. Cada
    uso del pool es una consulta (~1 ms en esta base; perfil de la Task 15b:
    execute p95 1,2 ms a 50 trabajadores); con 10 conexiones, el pico de 60
    escrituras de una ola se drena en ~6 ms de cola. El perfil de la 15b
    midió que más conexiones no bajan el p95 del pre-vuelo (5 -> 30,3 ms,
    10 -> 31,1, 25 -> 38,4 a c=25): el límite es la CPU del event loop. 10
    (el doble de lo que alcanzaba al pre-vuelo solo) ocupa 10 de las 55
    libres. La cola no mata a los trabajos de fondo: el ejecutor, el reaper y
    los jobs del Motor Registry (R38 fix round 3, N1) esperan turno sin plazo
    (espera_de_turno_sin_plazo); los pedidos HTTP
    esperan a lo sumo JAX_DB_CONNECT_TIMEOUT_SECONDS y dan 503."""
    crudo = os.environ.get(DB_POOL_MAX)
    if crudo is None:
        return TAMANIO_POOL_POR_DEFECTO
    return _tamanio_valido(DB_POOL_MAX, crudo)


def _tamanio_valido(nombre: str, crudo: str) -> int:
    """Valida el valor de una de las dos variables que dimensionan el pool.
    Fail-closed: un typo no se convierte en un pool de 0 o de 5000, y el error
    NOMBRA la variable que estaba mal puesta."""
    try:
        valor = int(crudo)
    except ValueError:
        valor = None
    if valor is None or not 1 <= valor <= TAMANIO_POOL_MAXIMO:
        raise RuntimeError(
            f"{nombre}={crudo!r} invalido -- tiene que ser un entero entre 1 "
            f"y {TAMANIO_POOL_MAXIMO} (max_connections de la MariaDB es 151 y la "
            "comparten todos los servicios)."
        )
    return valor


def tamanio_pool() -> int:
    """Tamano del pool. Lee `JAX_JACOBS_DB_POOL_SIZE` (frente F) o, si no esta,
    `JAX_DB_POOL_MAX` (Task 15b / R38): las dos ramas nombraron la misma
    perilla distinto el mismo dia y el merge conserva los dos nombres en vez de
    romper en silencio el despliegue de cualquiera de las dos. Ninguna de las
    dos -> TAMANIO_POOL_POR_DEFECTO. Presente pero vacia, no entera o fuera de
    [1, TAMANIO_POOL_MAXIMO] -> RuntimeError nombrandola."""
    for nombre in (ENV_TAMANIO_POOL, DB_POOL_MAX):
        crudo = os.environ.get(nombre)
        if crudo is not None:
            return _tamanio_valido(nombre, crudo)
    return TAMANIO_POOL_POR_DEFECTO


async def _estado_del_loop() -> _PoolDelLoop:
    """El estado de ESTE loop con el pool ya creado. Doble chequeo bajo un
    candado por loop: 20 pedidos concurrentes en frio crean UN pool, no 20
    (revision 2026-09-17, visto en rojo)."""
    loop = asyncio.get_running_loop()
    estado = _pools.get(loop)
    if estado is None:
        for viejo in [l for l in _pools if l.is_closed()]:
            del _pools[viejo]
        estado = _pools[loop] = _PoolDelLoop()
    if estado.pool is not None and not estado.pool.closed:
        return estado
    async with estado.candado:
        if estado.pool is not None and not estado.pool.closed:
            return estado
        cfg = _db_cfg()
        maximo = tamanio_pool()
        # Merge 2026-09-17: CREAR el pool tambien esta acotado. minsize=1 abre
        # una conexion al crear, y `connect_timeout` solo acota el socket: una
        # base que ACEPTA y no responde el handshake colgaba la creacion sin
        # limite (visto en rojo con la base falsa de
        # tests/test_prevuelo_pool.py::test_conexion_colgada_al_abrir_da_503_
        # acotado). Con el timeout, una base colgada es un 503, no un pedido
        # que no vuelve nunca: fail-closed.
        async with asyncio.timeout(db_connect_timeout_seconds()):
            pool = await aiomysql.create_pool(
                minsize=1,
                maxsize=maximo,
                pool_recycle=_RECICLAR_SEGUNDOS,
                # connect_timeout explicito: sin esto aiomysql espera sin limite si
                # la DB se cuelga (Tarea 2b, tanda A, 2026-09-14).
                connect_timeout=db_connect_timeout_seconds(),
                **cfg,
            )
        guardian = _guardian_del_pool(loop, estado, pool)
        await guardian.__anext__()
        estado.pool, estado.permisos, estado.guardian = pool, asyncio.Semaphore(maximo), guardian
        return estado


async def obtener_pool() -> aiomysql.Pool:
    """El pool de ESTE loop; lo crea si no hay o si el que habia se cerro."""
    return (await _estado_del_loop()).pool


async def _guardian_del_pool(loop, estado: _PoolDelLoop, pool: aiomysql.Pool):
    """Ata el cierre del pool al fin de SU loop.

    Un generador asincrono vivo queda registrado en el loop, y
    `asyncio.run()`/`asyncio.Runner` (y por lo tanto uvicorn y cada
    IsolatedAsyncioTestCase) llaman `loop.shutdown_asyncgens()` antes de cerrar
    el loop: eso corre este `finally`. Sin esto, quien crea un loop y no llama a
    `cerrar_pool()` deja sockets abiertos que el recolector cierra con el loop ya
    muerto (medido en la suite: 29 `Exception ignored ... Event loop is closed`).
    `cerrar_pool()` explicito sigue siendo el camino declarado; este es la red."""
    try:
        yield
    finally:
        if _pools.get(loop) is estado:
            del _pools[loop]
        await _cerrar(pool, estado.permisos)


async def _cerrar(pool: aiomysql.Pool, permisos: asyncio.Semaphore | None) -> None:
    """Toma los `maxsize` permisos con limite -- cuando los tiene, nadie esta
    usando una conexion ni puede empezar a pedirla -- y cierra. Si vence el
    limite (una consulta colgada no puede colgar el apagado), corta a la fuerza
    (terminate) y deja ERROR en el log."""
    if pool.closed:
        return
    limite = db_connect_timeout_seconds()
    tomados = 0
    forzar = False
    if permisos is not None:
        try:
            async with asyncio.timeout(limite):
                while tomados < pool.maxsize:
                    await permisos.acquire()
                    tomados += 1
        except TimeoutError:
            forzar = True
            logger.error(
                "jacobs.store: %d conexion(es) seguian en uso %d s despues de pedir el "
                "cierre del pool; se cortan a la fuerza.", pool.maxsize - tomados, limite,
            )
    pool.close()
    if forzar:
        pool.terminate()
    await pool.wait_closed()


async def cerrar_pool() -> None:
    """Cierra el pool DE ESTE loop (esperar el de otro loop volveria a cruzar
    loops). Idempotente."""
    estado = _pools.pop(asyncio.get_running_loop(), None)
    if estado is None or estado.pool is None:
        return
    await estado.guardian.aclose()               # corre el finally: _cerrar(...)
    await _cerrar(estado.pool, estado.permisos)  # por si el guardian ya habia terminado


def _sesion_reutilizable(conn: aiomysql.Connection) -> bool:
    """Solo vuelve al pool una sesion igual a la que el pool entrego: abierta,
    autocommit encendido y sin transaccion en curso. No se delega en aiomysql
    (0.3.2 solo mira la transaccion; un `autocommit(False)` sin consulta pasaria).

    CONTRATO DE `conexion()`: esto NO ve el resto del estado de sesion. Quien lo
    cambia -- `SET SESSION ...`, `GET_LOCK()`, `LOCK TABLES`, tablas `TEMPORARY`,
    `USE otra_base`, variables de usuario `@x` de las que otro dependa -- pide
    `conexion(desechable=True)`, y la conexion se cierra al salir en vez de
    volver al pool con ese estado.

    CASO CIEGO: un procedimiento almacenado (`CALL`) con autocommit=1 puede
    dejar cualquiera de esos estados (SET SESSION, GET_LOCK, LOCK TABLES, una
    TEMPORARY, @variables) sin que el cliente lo vea: el OK final trae
    autocommit y sin transaccion, y esto da verde. Quien llama un procedimiento
    que toca estado de sesion pide `desechable=True`."""
    return (not conn.closed) and conn.get_autocommit() and not conn.get_transaction_status()


# Una conexion cuyo cuerpo termino con excepcion vuelve al pool SOLO si esa
# excepcion es, por IDENTIDAD, un error del servidor que el propio envoltorio
# vio salir de una llamada suya: atribucion por construccion (ruling del
# 2026-09-17). Sin tracebacks, pilas de tareas ni internals de asyncio/aiomysql.
#
# POR QUE. Cerrar ante cualquier error abre un handshake por pedido fallido.
# Medido el 2026-09-17 (k6, 25 VUs, 60 s, /motor/authorize-facet con la
# consulta forzada a 1146, cerrando siempre): 1040 rps, 12 548 TIME-WAIT hacia
# la base y 3072 conexiones perdidas (2013) en la app: el agotamiento de
# puertos que el pool vino a cerrar, disparado por una rafaga de errores (tabla
# faltante, clave duplicada, lock). Tabla en pool-report.md del frente F.
#
# LISTA BLANCA (_codigo_del_servidor_sano): pymysql mapea un codigo desconocido
# a OperationalError, y ahi caen tambien los que SI cortan la sesion (1927
# conexion matada, 1053 apagado). Ante la duda se cierra.
_CLASES_DE_ERROR_DEL_SERVIDOR = (
    aiomysql.ProgrammingError,
    aiomysql.IntegrityError,
    aiomysql.DataError,
    aiomysql.NotSupportedError,
)
# 1205 (lock wait timeout) no esta en la tabla de pymysql: le llega como
# OperationalError por defecto. 1213 (deadlock) si esta, como OperationalError.
_OPERACIONALES_SANOS = frozenset({1205, 1213})


def _codigo_del_servidor_sano(e: BaseException) -> bool:
    """Puro. Programming/Integrity/Data/NotSupported solo si pymysql eligio esa
    clase POR el codigo del servidor (`error_map[codigo] is type(e)`): un
    ProgrammingError sin codigo ("Cursor closed") o con uno inventado no entra.
    OperationalError solo 1205 y 1213. La tabla de pymysql no tiene codigos del
    CLIENTE (2000-2999), asi que 2013 o 2014 nunca entran."""
    if not isinstance(e, aiomysql.MySQLError) or not e.args:
        return False
    codigo = e.args[0]
    if type(codigo) is not int:  # 1205.0 == 1205 en un frozenset
        return False
    if type(e) is aiomysql.OperationalError:
        return codigo in _OPERACIONALES_SANOS
    return type(e) in _CLASES_DE_ERROR_DEL_SERVIDOR and _pymysql_err.error_map.get(codigo) is type(e)


class ConexionInvalidada(RuntimeError):
    """Uso de un envoltorio despues de salir de `conexion()`. Para entonces la
    conexion real ya volvio al pool (y puede ser de otro pedido) o se cerro: se
    falla sin tocarla."""


_MENSAJE_INVALIDADA = (
    "jacobs.store: esta conexion ya salio de `async with conexion()`; el socket "
    "puede ser de otro pedido. Pedi otra con conexion()."
)


class CursorVigilado:
    """El cursor de `ConexionVigilada.cursor()`. Sus llamadas de uso normal
    (`execute`, `executemany`, `fetchone`, `fetchmany`, `fetchall`) cuentan como
    en vuelo mientras corren y, si lanzan un error de la lista blanca, lo marcan
    en la conexion ANTES de relanzarlo. Lo demas se delega al cursor de aiomysql
    sin marcar: un error por ahi cierra la conexion."""

    def __init__(self, conexion: "ConexionVigilada", crudo: aiomysql.Cursor):
        self._conexion = conexion
        self.__crudo = crudo
        self._sin_buffer = isinstance(crudo, aiomysql.SSCursor)
        if self._sin_buffer:
            conexion._sin_buffer_abiertos += 1

    @property
    def _crudo(self) -> aiomysql.Cursor:
        self._conexion._exigir_vigente()
        return self.__crudo

    async def _vigilar(self, metodo, *args, **kwargs):
        c = self._conexion
        c._en_vuelo += 1
        try:
            return await metodo(*args, **kwargs)
        except aiomysql.MySQLError as e:
            if _codigo_del_servidor_sano(e):
                c.ultimo_error_sano = e
            raise
        finally:
            c._en_vuelo -= 1

    async def execute(self, *args, **kwargs):
        return await self._vigilar(self._crudo.execute, *args, **kwargs)

    async def executemany(self, *args, **kwargs):
        return await self._vigilar(self._crudo.executemany, *args, **kwargs)

    async def fetchone(self):
        return await self._vigilar(self._crudo.fetchone)

    async def fetchmany(self, *args, **kwargs):
        return await self._vigilar(self._crudo.fetchmany, *args, **kwargs)

    async def fetchall(self):
        return await self._vigilar(self._crudo.fetchall)

    async def close(self):
        if self._sin_buffer:
            self._sin_buffer = False
            self._conexion._sin_buffer_abiertos -= 1
        await self._crudo.close()

    def __getattr__(self, nombre):
        return getattr(self._crudo, nombre)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


class _CursorPendiente:
    """`conn.cursor(...)` se usa con `async with` o con `await`, como en aiomysql."""

    def __init__(self, conexion: "ConexionVigilada", args, kwargs):
        self._conexion, self._args, self._kwargs = conexion, args, kwargs
        self._cursor: CursorVigilado | None = None

    async def _abrir(self) -> CursorVigilado:
        crudo = await self._conexion.crudo.cursor(*self._args, **self._kwargs)
        return CursorVigilado(self._conexion, crudo)

    def __await__(self):
        return self._abrir().__await__()

    async def __aenter__(self) -> CursorVigilado:
        self._cursor = await self._abrir()
        return self._cursor

    async def __aexit__(self, *exc):
        await self._cursor.close()


class ConexionVigilada:
    """Lo que entrega `conexion()`. Delega en la conexion de aiomysql (`crudo`)
    salvo `cursor()`, que devuelve un CursorVigilado.

    Vale SOLO dentro del `async with`: al salir, `conexion()` la invalida antes
    de devolver o cerrar el socket, y todo acceso posterior -- `crudo`,
    `cursor()`, lo que pasa por `__getattr__` y los cursores que abrio -- lanza
    ConexionInvalidada. Una tarea hermana que se quedo con el envoltorio no
    puede mandar SQL por un socket que ya es de otro pedido."""

    def __init__(self, crudo: aiomysql.Connection):
        self.__crudo = crudo
        self.__vigente = True
        self.ultimo_error_sano: BaseException | None = None
        self._en_vuelo = 0
        self._sin_buffer_abiertos = 0

    def _exigir_vigente(self) -> None:
        if not self.__vigente:
            raise ConexionInvalidada(_MENSAJE_INVALIDADA)

    def _invalidar(self) -> None:
        self.__vigente = False

    @property
    def crudo(self) -> aiomysql.Connection:
        self._exigir_vigente()
        return self.__crudo

    def cursor(self, *args, **kwargs) -> _CursorPendiente:
        self._exigir_vigente()
        return _CursorPendiente(self, args, kwargs)

    def __getattr__(self, nombre):
        return getattr(self.crudo, nombre)

    def reutilizable_tras(self, e: BaseException) -> bool:
        """Tras `e`, vuelve al pool solo si `e` ES el error que marco una llamada
        propia, no queda ninguna llamada propia en vuelo y no hay un cursor sin
        buffer abierto (resultados sin leer). La sesion la mira despues
        _sesion_reutilizable."""
        return (
            self.ultimo_error_sano is not None
            and e is self.ultimo_error_sano
            and self._en_vuelo == 0
            and self._sin_buffer_abiertos == 0
        )


# --- Espera de turno de los trabajos de fondo (R38, fix round 1, 2a) ---------
# Revisión de 1d84e82: con el pool compartido, una escritura del ejecutor que
# esperaba turno más de JAX_DB_CONNECT_TIMEOUT_SECONDS con la base SANA
# lanzaba TimeoutError. `event_append(STEP_STARTED)` está fuera del try del
# paso: salía del gather, mataba run_pipeline y el paso quedaba en running.
# Decisión: el ejecutor, el reaper y (fix round 3, N1) los jobs del Motor
# Registry (trabajos de fondo, sin nadie esperando la
# respuesta) esperan turno SIN plazo; abrir la conexión sigue acotado por
# connect_timeout, así que una base caída falla igual (fail-closed). Los
# pedidos HTTP mantienen la espera acotada y responden 503. Se descartaron:
# un cupo aparte para el ejecutor (más conexiones sobre una MariaDB
# compartida, y el cupo propio también se llena en una ola de 20 pasos) y
# tratar el vencimiento como fallo del paso (un paso fallaría por cola con la
# base sana).
# Es una ContextVar y no un argumento: run_pipeline la pone una vez y la
# heredan las tareas del gather de cada ola (asyncio copia el contexto al
# crearlas) y todas las escrituras de store/usage_writer que hacen, sin
# pasar un parámetro por ~20 llamadas. Límite: una consulta que la base deja
# COLGADA retiene su turno sin plazo (igual que antes del pool, que no
# acotaba consultas).
_turno_sin_plazo: ContextVar[bool] = ContextVar("jacobs_turno_sin_plazo", default=False)


def turno_sin_plazo() -> bool:
    return _turno_sin_plazo.get()


@contextmanager
def espera_de_turno_sin_plazo() -> Iterator[None]:
    """Dentro del bloque (y en las tareas que se creen en él), pedir una
    conexión del pool espera turno sin plazo. Para trabajos de fondo."""
    marca = _turno_sin_plazo.set(True)
    try:
        yield
    finally:
        _turno_sin_plazo.reset(marca)


@contextlib.asynccontextmanager
async def conexion(desechable: bool = False):
    """Una conexion del pool, devuelta al salir.

    Se DESCARTA (se cierra, el pool abre otra cuando haga falta) si el cuerpo
    termino con excepcion o cancelacion -- el socket puede haber quedado a mitad
    de una respuesta --, si la sesion quedo sucia, o si `desechable=True` (ver el
    contrato en _sesion_reutilizable). La excepcion a esa regla es un error del
    servidor de la lista blanca que el propio envoltorio marco al salir de una
    llamada suya (ConexionVigilada.reutilizable_tras): vuelve al pool si la
    sesion sigue limpia.

    Entrega una ConexionVigilada, no la conexion de aiomysql: `cursor()` vigila
    las llamadas, el resto se delega (`crudo` es la conexion real).

    Esperar un hueco tiene limite: `JAX_DB_CONNECT_TIMEOUT_SECONDS`, el mismo
    que acota abrir el socket. Con el pool lleno mas alla de eso, TimeoutError:
    fail-closed, no una espera infinita (aiomysql espera sin limite)."""
    estado = await _estado_del_loop()
    pool, permisos = estado.pool, estado.permisos
    limite = db_connect_timeout_seconds()
    if _turno_sin_plazo.get():
        # Trabajo de fondo (ejecutor, reaper, jobs del Motor Registry): espera
        # la cola sin plazo -- ver el bloque de arriba. Abrir la conexion sigue
        # acotado, asi que una base caida falla igual (fail-closed).
        await permisos.acquire()
    else:
        try:
            await asyncio.wait_for(permisos.acquire(), timeout=limite)
        except TimeoutError as e:
            raise TimeoutError(
                f"jacobs.store: sin conexion libre en el pool tras {limite} s "
                f"(tamano {pool.maxsize}, en uso {pool.size - pool.freesize})"
            ) from e
    try:
        # Con el permiso tomado hay hueco: esto no espera en cond.wait. El
        # limite acota lo unico que puede tardar, abrir una conexion nueva.
        conn = await asyncio.wait_for(pool.acquire(), timeout=limite)
    except BaseException:
        permisos.release()
        raise
    vigilada = ConexionVigilada(conn)
    limpia = False
    try:
        yield vigilada
        # Una llamada propia todavia en vuelo (tarea hermana): socket a mitad.
        limpia = vigilada._sin_buffer_abiertos == 0 and vigilada._en_vuelo == 0
    except BaseException as e:
        limpia = vigilada.reutilizable_tras(e)
        raise
    finally:
        # Antes de devolver o cerrar: quien se quedo con el envoltorio (una
        # tarea hermana) ya no llega al socket.
        vigilada._invalidar()
        try:
            if desechable or not limpia or not _sesion_reutilizable(conn):
                conn.close()
            # release() saca la conexion de `_used` antes de devolver. El aviso
            # que agenda es para el cond.wait de aiomysql, al que con el
            # semaforo nadie llega: no se espera.
            pool.release(conn)
        finally:
            permisos.release()


# `conexion_del_pool()` es el MISMO objeto que `conexion()`: las dos ramas le
# pusieron nombre distinto a lo mismo y los dos nombres quedan en uso en el
# arbol. Tambien evita el sombreado en las funciones del store que reciben un
# parametro llamado `conexion` (pipeline_create, pipeline_update_status, ...):
# ahi se usa `_conexion_o_pool(conexion)`.
conexion_del_pool = conexion


async def conexion_dedicada(found_rows: bool = False) -> aiomysql.Connection:
    """Una conexion DEDICADA (fuera del pool), que quien la pide cierra.

    Es la UNICA excepcion al pool. En CODIGO DE SERVICIO (jacobs/, las_manos/,
    jax/, tools/) la piden exactamente TRES funciones, por las dos razones que
    siguen, cada una fuera del alcance del pool. Fuera del codigo de servicio
    tambien la usan TESTS (tests/, jacobs/*_test.py) y SCRIPTS DE MEDICION
    (scripts/perfil_prevuelo.py, scripts/medir_min_output_tokens.py), que
    necesitan una conexion propia y la cierran ellos; eso no es camino de
    pedidos y no cuenta para esta garantia. La lista de servicio la vigila
    `jacobs/_store_pool_test.py::ExcepcionAlPoolTest`: un llamador nuevo ahi
    pone el guard en rojo, porque la excepcion al pool no se amplia sin una
    decision.

    1. `found_rows=True` -- las escrituras CONDICIONALES por epoca. La piden
       `_ejecutar_condicional` (por donde pasa `pipeline_tomar_epoca`, que no
       la llama directo) y `continuar_transaccion`. Por defecto MariaDB devuelve de un UPDATE las
       filas CAMBIADAS, no las que cumplen el WHERE: una escritura condicional
       que reescribe los mismos valores devolveria 0 y el ejecutor creeria
       haber PERDIDO la epoca -- y dejaria de escribir. CLIENT.FOUND_ROWS se
       negocia en el handshake, no se enciende por sesion, y ponerselo al pool
       cambiaria en silencio el conteo de filas de cualquier UPDATE que se
       agregue despues.
    2. `candado_de_activos()` -- el GET_LOCK del cupo. Podria pedir
       `conexion(desechable=True)`, pero el candado se ESPERA (hasta
       JAX_PREVUELO_CANDADO_TIMEOUT_S): varios `crear`/`continue` a la vez se
       quedarian con las conexiones del pool mientras esperan el candado y
       dejarian sin conexiones al resto del servicio.

    Cualquier otro uso va por `conexion()` / `conexion_del_pool()`; una sesion
    con estado propio que NO espera (SET SESSION, temporales) pide
    `conexion(desechable=True)`. Reemplaza a `get_conn()`, que era de uso
    general: ese nombre no vuelve (jacobs/_store_pool_test.py lo vigila)."""
    extra = {"client_flag": CLIENT.FOUND_ROWS} if found_rows else {}
    return await aiomysql.connect(
        **_db_cfg(), connect_timeout=db_connect_timeout_seconds(), **extra,
    )


@asynccontextmanager
async def _conexion_o_pool(conexion: aiomysql.Connection | None) -> AsyncIterator[aiomysql.Connection]:
    """La conexión prestada (p. ej. la del candado, dentro de una transacción)
    sin tocarla, o una del pool."""
    if conexion is not None:
        yield conexion
        return
    async with conexion_del_pool() as conn:
        yield conn


@dataclass
class EstadoDeTransaccion:
    """Hasta dónde llegó una transacción (Ruling R41). `enviando_commit`
    queda en True desde que se manda el COMMIT; `confirmada`, cuando el
    servidor respondió. Si falla entre las dos, el resultado es INCIERTO: el
    servidor pudo haber confirmado antes de que se cortara la respuesta."""

    enviando_commit: bool = False
    confirmada: bool = False

    @property
    def incierta(self) -> bool:
        return self.enviando_commit and not self.confirmada


@asynccontextmanager
async def transaccion(
    conn: aiomysql.Connection, estado: EstadoDeTransaccion | None = None,
) -> AsyncIterator[aiomysql.Connection]:
    """Una transacción sobre `conn` (R38, fix round 1, 2b: crear escribe
    pipeline, pasos y eventos en UNA, sobre la conexión del candado).

    Si el bloque falla o lo cancela un timeout, la conexión se CIERRA en vez
    de mandar ROLLBACK: tras una consulta cancelada el protocolo queda en un
    estado desconocido y un ROLLBACK por la red podría colgarse de nuevo.
    Cerrar la sesión hace que el servidor descarte la transacción sin
    confirmar (y suelte el GET_LOCK de esa sesión).

    Límite declarado (R41): si la conexión se corta o vence DURANTE el COMMIT,
    desde acá no se puede saber si el servidor confirmó. `estado.incierta`
    queda en True para que quien llama lo diga (crear responde 503 con
    `detalle`: el pipeline puede existir)."""
    estado = estado if estado is not None else EstadoDeTransaccion()
    await conn.begin()
    try:
        yield conn
        estado.enviando_commit = True
        await conn.commit()
        estado.confirmada = True
    except BaseException:
        conn.close()
        raise


# Hijo de "jacobs": LAS MANOS le pone handler INFO a ese logger al arrancar
# (server.py::_jacobs_init), asi un ERROR de aca llega al journal.
logger = logging.getLogger("jacobs.store")

# --- Indices de las columnas por las que se FILTRA ---------------------------
# (tabla, indice, DDL, acotado). Ver el comentario del loop en init_tables().
#
# `acotado`: el DDL corre con lock_wait_timeout corto (_crear_indice_acotado).
# Los cuatro primeros ya existen en produccion y no se tocan (el chequeo de
# information_schema los salta); el de dueño es nuevo y va acotado.
#
# idx_jacobs_pipelines_duenio (Ruling T6-6, 2026-09-15): jax-platform lista
# los pipelines de un dueño con WHERE user_id AND tenant_id ORDER BY
# created_at. ALGORITHM=INPLACE LOCK=NONE explicitos (re-revision de la
# plataforma): si MariaDB no puede crearlo en linea, FALLA con error en vez de
# caer en silencio a COPY, que bloquea las escrituras de Jacobs mientras copia.
_INDICES: list[tuple[str, str, str, bool]] = [
    ("jacobs_events", "idx_events_pipeline",
     "CREATE INDEX idx_events_pipeline ON jacobs_events (pipeline_id)", False),
    ("jacobs_steps", "idx_steps_pipeline",
     "CREATE INDEX idx_steps_pipeline ON jacobs_steps (pipeline_id)", False),
    ("jacobs_steps", "idx_steps_status",
     "CREATE INDEX idx_steps_status ON jacobs_steps (status)", False),
    ("jacobs_pipelines", "idx_pipelines_status",
     "CREATE INDEX idx_pipelines_status ON jacobs_pipelines (status)", False),
    ("jacobs_pipelines", "idx_jacobs_pipelines_duenio",
     "CREATE INDEX idx_jacobs_pipelines_duenio ON jacobs_pipelines "
     "(user_id, tenant_id, created_at) ALGORITHM=INPLACE LOCK=NONE", True),
    # idx_events_pipeline_tipo (2026-09-17, Ruling R20, LAS CUATRO #1): la
    # Mesa (jax-platform backend/api/pipelines.py::sql_eventos_de_causa) lee
    # la causa de aborto de hasta 50 pipelines con
    # WHERE pipeline_id IN (...) AND event_type IN (5 tipos). Medido por el
    # plan P en jax_memory_test (50 pipelines, 11.050 eventos): con solo
    # idx_events_pipeline (pipeline_id), EXPLAIN range examina las 11.050
    # filas de esos pipelines para devolver 1.050 (8,1 ms; carga c=25 p95
    # 147 ms). idx_events_pipeline NO se borra -- otras consultas filtran
    # solo por pipeline_id (events_by_pipeline) y ese acceso les sigue
    # sirviendo igual.
    ("jacobs_events", "idx_events_pipeline_tipo",
     "CREATE INDEX idx_events_pipeline_tipo ON jacobs_events "
     "(pipeline_id, event_type) ALGORITHM=INPLACE LOCK=NONE", True),
]

# Espera maxima por el metadata lock de un DDL acotado. El default de MariaDB
# (lock_wait_timeout) es 86400 s: una transaccion larga sobre la tabla dejaria
# el arranque colgado un dia entero, sin error.
#
# Costo de la espera (review de 05c028b): mientras el DDL espera su metadata
# lock EXCLUSIVO (hasta estos 30 s), ese pedido queda en la cola del MDL y las
# lecturas y escrituras NUEVAS sobre jacobs_pipelines se encolan detras de el.
# Por eso la espera es corta: 30 s de Jacobs detenido como peor caso, no un dia.
# Si vence, el indice no se crea (ERROR en el log); la red de seguridad es el
# test de EXPLAIN de la plataforma en CI, que falla si la consulta de dueño no
# usa este indice.
_LOCK_WAIT_DDL_SEGUNDOS = 30
_ER_LOCK_WAIT_TIMEOUT = 1205


async def _crear_indice_acotado(cur, tabla: str, indice: str, ddl: str) -> bool:
    """Corre `ddl` con lock_wait_timeout de 30 s en ESTA sesion y restaura el
    valor previo pase lo que pase. True si lo creo.

    Decision (re-revision de la plataforma, 2026-09-15): si la espera vence
    (1205), ERROR en el log con el indice y el motivo, y el arranque SIGUE: el
    proximo arranque lo reintenta, porque el chequeo de information_schema ve
    que falta. No es un salto silencioso. Por que no fallar el arranque: el
    indice es de rendimiento, no un contrato (sin el, la consulta de dueño es
    un scan -- 60 filas hoy); init_tables() corre en el arranque de LAS MANOS,
    y tumbar LAS MANOS entero porque otra transaccion tiene la tabla cambiaria
    una consulta lenta por el sistema caido. Cualquier OTRO error (INPLACE o
    LOCK=NONE no soportados, sintaxis) SUBE: no es una espera, es un DDL que
    no puede correr como se declaro."""
    await cur.execute("SELECT @@SESSION.lock_wait_timeout")
    (previo,) = await cur.fetchone()
    await cur.execute("SET SESSION lock_wait_timeout=%s", (_LOCK_WAIT_DDL_SEGUNDOS,))
    try:
        await cur.execute(ddl)
        return True
    except aiomysql.OperationalError as e:  # fail-soft: el indice solo acelera; la consulta de dueño sigue correcta como scan; se reintenta en el proximo arranque
        if not (e.args and e.args[0] == _ER_LOCK_WAIT_TIMEOUT):
            raise
        logger.error(
            "init_tables: no se creo %s en %s -- otra transaccion tiene la tabla y "
            "vencio la espera de %d s (%s). El arranque sigue SIN el indice (la "
            "consulta de dueño hace scan); se reintenta en el proximo arranque.",
            indice, tabla, _LOCK_WAIT_DDL_SEGUNDOS, e,
        )
        return False
    finally:
        await cur.execute("SET SESSION lock_wait_timeout=%s", (int(previo),))


async def init_tables() -> None:
    """Crea las tablas si no existen. Llamar al arrancar."""
    # desechable: _crear_indice_acotado cambia lock_wait_timeout de la SESION.
    # Lo restaura en su finally, pero una sesion tocada no vuelve al pool.
    async with conexion(desechable=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS jacobs_pipelines (
                    pipeline_id        VARCHAR(36) PRIMARY KEY,
                    name               TEXT NOT NULL,
                    invoked_by         VARCHAR(50) NOT NULL,
                    mode               VARCHAR(20) NOT NULL,
                    status             VARCHAR(20) NOT NULL,
                    plan               JSON,
                    current_step_index INT DEFAULT 0,
                    max_steps          INT DEFAULT 20,
                    context_refs       JSON,
                    created_at         DOUBLE NOT NULL,
                    updated_at         DOUBLE NOT NULL
                )
            """)
            for col, ddl in [
                ("user_id", "ALTER TABLE jacobs_pipelines ADD COLUMN user_id VARCHAR(50) NULL"),
                ("tenant_id", "ALTER TABLE jacobs_pipelines ADD COLUMN tenant_id VARCHAR(50) NULL"),
                # Ronda 5 (2026-08-20, T1): reemplaza el owner file de
                # filesystem -- ver Pipeline.owner_ack_at en models.py.
                ("owner_ack_at", "ALTER TABLE jacobs_pipelines ADD COLUMN owner_ack_at DOUBLE NULL"),
                # 2026-09-17 (spec prevuelo-y-continuar §5.3): época de corrida.
                ("run_epoch", "ALTER TABLE jacobs_pipelines ADD COLUMN run_epoch INT NOT NULL DEFAULT 0"),
                # Frente F (2026-09-16): de quién es hijo un pipeline de Ada y a
                # qué profundidad. ALGORITHM=INSTANT explícito: si MariaDB no
                # puede agregarla sin copiar la tabla, FALLA en vez de bloquear
                # las escrituras de Jacobs mientras copia.
                ("parent_pipeline_id", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "parent_pipeline_id VARCHAR(36) NULL, ALGORITHM=INSTANT"),
                ("depth", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "depth INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT"),
            ]:
                await cur.execute(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' AND COLUMN_NAME=%s",
                    (col,),
                )
                (exists,) = await cur.fetchone()
                if not exists:
                    await cur.execute(ddl)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS jacobs_steps (
                    step_id          VARCHAR(36) PRIMARY KEY,
                    pipeline_id      VARCHAR(36) NOT NULL,
                    step_index       INT NOT NULL,
                    facet            VARCHAR(30) NOT NULL,
                    capability       VARCHAR(50) NOT NULL,
                    input_ref        TEXT,
                    output_ref       TEXT,
                    status           VARCHAR(20) NOT NULL,
                    timeout_seconds  INT DEFAULT 300,
                    retries_allowed  INT DEFAULT 0,
                    skip_on_fail     BOOLEAN DEFAULT FALSE,
                    trace_id         VARCHAR(36),
                    started_at       DOUBLE,
                    finished_at      DOUBLE,
                    error            TEXT
                )
            """)
            for col, ddl in [
                ("motor", "ALTER TABLE jacobs_steps ADD COLUMN motor VARCHAR(30) NULL"),
                # depends_on existía en jax_memory (prod) desde antes -- agregado
                # por fuera de esta lista de migración (ALTER manual, sin
                # registrar acá), así que nunca se propagó a una DB nueva
                # (jax_memory_test incluida). Confirmado con SHOW CREATE TABLE
                # contra jax_memory 2026-08-24 -- DDL exacto, mismo collation.
                ("depends_on", "ALTER TABLE jacobs_steps ADD COLUMN depends_on LONGTEXT "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL "
                    "CHECK (json_valid(depends_on))"),
            ]:
                await cur.execute(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_steps' AND COLUMN_NAME=%s",
                    (col,),
                )
                (exists,) = await cur.fetchone()
                if not exists:
                    await cur.execute(ddl)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS jacobs_events (
                    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                    pipeline_id VARCHAR(36) NOT NULL,
                    step_id     VARCHAR(36),
                    event_type  VARCHAR(50) NOT NULL,
                    payload     JSON,
                    ts          DOUBLE NOT NULL
                )
            """)
            # Frente F (2026-09-16): contrato de sub-pipelines. Se guarda SOLO
            # el sha256 del token. Tabla nueva -> el índice va en el CREATE (no
            # hay filas que migrar). Tiempos en DOUBLE epoch como el resto de
            # Jacobs: inmunes a la zona horaria de la sesión.
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS jacobs_subpipeline_tokens (
                    token_hash         CHAR(64)    NOT NULL PRIMARY KEY,
                    parent_pipeline_id VARCHAR(36) NOT NULL,
                    parent_step        VARCHAR(36) NOT NULL,
                    depth_hijo         INT         NOT NULL,
                    emitido_at         DOUBLE      NOT NULL,
                    vence_at           DOUBLE      NOT NULL,
                    usado_at           DOUBLE      NULL,
                    hijo_pipeline_id   VARCHAR(36) NULL,
                    INDEX idx_subpipeline_tokens_padre (parent_pipeline_id)
                ) ENGINE=InnoDB
            """)
            # Human gate de LAS MANOS (2026-09-17): mismo contrato que los tokens
            # de sub-pipelines. SOLO el sha256; un solo uso; sin ruta HTTP de
            # emisión (la emite las_manos/emitir_token_gate.py con la credencial
            # de la base). Tabla nueva -> la PK es el único índice que usa el
            # camino caliente (consumo por token_hash).
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS las_manos_human_gate_tokens (
                    token_hash   CHAR(64)     NOT NULL PRIMARY KEY,
                    emitido_por  VARCHAR(64)  NOT NULL,
                    emitido_at   DOUBLE       NOT NULL,
                    vence_at     DOUBLE       NOT NULL,
                    usado_at     DOUBLE       NULL,
                    usado_en     VARCHAR(128) NULL
                ) ENGINE=InnoDB
            """)
            # --- Indices de las columnas por las que se FILTRA ---------------
            # Las tres tablas nacieron con la PK y nada mas, y el codigo las
            # consulta por pipeline_id y por status. Con 611 filas el scan no
            # se nota; el plan es un scan igual y crece lineal (politica 1 de
            # LAS CUATRO DEL RENDIMIENTO: se juzga el plan, no el reloj de hoy).
            #
            # Van aca y no en un `ALTER TABLE` a mano por la misma razon que la
            # columna `depends_on`, que existia en produccion y en ninguna base
            # nueva: lo que no pasa por init_tables() no llega a un dev nuevo,
            # ni a CI, ni a un restore de desastre.
            #
            # `CREATE INDEX` no acepta IF NOT EXISTS en MariaDB, asi que se
            # chequea information_schema primero -- mismo patron idempotente
            # que las columnas de arriba. init_tables() corre en CADA arranque
            # de los tres procesos: si esto no fuera idempotente, el segundo
            # arranque romperia en produccion.
            #
            # La lista vive en _INDICES (arriba) para que su forma se pruebe
            # sin DB (tests/test_store_indice_duenio.py).
            for tabla, indice, ddl, acotado in _INDICES:
                await cur.execute(
                    "SELECT COUNT(*) FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s",
                    (tabla, indice),
                )
                (existe,) = await cur.fetchone()
                if existe:
                    continue
                if acotado:
                    await _crear_indice_acotado(cur, tabla, indice, ddl)
                else:
                    await cur.execute(ddl)


# ----------------------------------------------------------------
#  Pipeline CRUD
# ----------------------------------------------------------------

async def pipeline_create(p: Pipeline, conexion: aiomysql.Connection | None = None) -> None:
    async with _conexion_o_pool(conexion) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO jacobs_pipelines
                    (pipeline_id, name, invoked_by, mode, status,
                     plan, current_step_index, max_steps, context_refs,
                     created_at, updated_at, user_id, tenant_id, run_epoch,
                     parent_pipeline_id, depth)
                VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s, %s, %s,%s)
                """,
                (
                    p.pipeline_id, p.name, p.invoked_by, p.mode, p.status.value,
                    json.dumps([s.model_dump() for s in p.plan], ensure_ascii=False),
                    p.current_step_index, p.max_steps,
                    json.dumps(p.context, ensure_ascii=False),
                    p.created_at, p.updated_at,
                    p.user_id, p.tenant_id, p.run_epoch,
                    p.parent_pipeline_id, p.depth,
                ),
            )


async def pipeline_get(pipeline_id: str) -> Pipeline | None:
    async with conexion_del_pool() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM jacobs_pipelines WHERE pipeline_id=%s", (pipeline_id,)
            )
            row = await cur.fetchone()
    if not row:
        return None
    return _row_to_pipeline(row)


async def pipeline_update_status(
    pipeline_id: str,
    status: PipelineStatus,
    current_step_index: int | None = None,
    context: dict | None = None,
    conexion: aiomysql.Connection | None = None,
) -> None:
    now = time.time()
    async with _conexion_o_pool(conexion) as conn:
        async with conn.cursor() as cur:
            if current_step_index is not None and context is not None:
                await cur.execute(
                    """UPDATE jacobs_pipelines
                       SET status=%s, current_step_index=%s, context_refs=%s, updated_at=%s
                       WHERE pipeline_id=%s""",
                    (
                        status.value, current_step_index,
                        json.dumps(context, ensure_ascii=False),
                        now, pipeline_id,
                    ),
                )
            elif current_step_index is not None:
                await cur.execute(
                    """UPDATE jacobs_pipelines
                       SET status=%s, current_step_index=%s, updated_at=%s
                       WHERE pipeline_id=%s""",
                    (status.value, current_step_index, now, pipeline_id),
                )
            else:
                await cur.execute(
                    "UPDATE jacobs_pipelines SET status=%s, updated_at=%s WHERE pipeline_id=%s",
                    (status.value, now, pipeline_id),
                )


def _sql_candidatos_del_reaper(n_estados: int) -> str:
    """Ruling R36 (2026-09-17): el barrido del reaper trae, en la MISMA
    consulta, el mayor timeout_seconds de los pasos EN CURSO (status
    'running' en jacobs_steps, el valor que el ejecutor aplica con
    asyncio.wait_for) de cada pipeline candidato. EXPLAIN en
    tests/test_jacobs_reaper_cas_db.py: range por idx_pipelines_status y la
    subconsulta ref por idx_steps_pipeline (a lo sumo 20 pasos por plan)."""
    estados = ",".join(["%s"] * n_estados)
    return (
        "SELECT p.*, (SELECT MAX(s.timeout_seconds) FROM jacobs_steps s "
        "WHERE s.pipeline_id = p.pipeline_id AND s.status = 'running') AS max_timeout_en_curso "
        f"FROM jacobs_pipelines p WHERE p.status IN ({estados})"
    )


async def candidatos_del_reaper(statuses: list[PipelineStatus]) -> list[tuple[Pipeline, int]]:
    """Usado por jacobs/reaper.py: los pipelines en los status dados, cada uno
    con el mayor timeout_seconds de sus pasos en curso (0 si no tiene). No
    filtra por antigüedad: eso es criterio del reaper."""
    if not statuses:
        return []
    async with conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(_sql_candidatos_del_reaper(len(statuses)), tuple(s.value for s in statuses))
            rows = await cur.fetchall()
    salida = []
    for row in rows:
        maximo = row.pop("max_timeout_en_curso", None)
        salida.append((_row_to_pipeline(row), int(maximo or 0)))
    return salida


_SQL_CONTAR_ACTIVOS = "SELECT COUNT(*) FROM jacobs_pipelines WHERE status IN ('pending','running')"


async def pipeline_count_active(conexion: aiomysql.Connection | None = None) -> int:
    """Pipelines activos. Con `conexion` (la del candado de activos, F3) lee por
    ella y no la cierra; sin ella lee por una del pool (R38)."""
    if conexion is not None:
        return await _contar_activos(conexion)
    async with conexion_del_pool() as conn:
        return await _contar_activos(conn)


async def _contar_activos(conn: aiomysql.Connection) -> int:
    async with conn.cursor() as cur:
        await cur.execute(_SQL_CONTAR_ACTIVOS)
        row = await cur.fetchone()
        return int(row[0]) if row else 0


# ----------------------------------------------------------------
#  Candado del cupo de activos entre procesos (ola final F3, Ruling R31)
# ----------------------------------------------------------------
# MAX_PARALLEL_PIPELINES se contaba bajo un asyncio.Lock de PROCESO
# (jacobs/candado.py): el CLI de continuar (tools/jacobs_relaunch.py) corre en
# otro proceso y cada uno contaba y escribía sin ver la reserva del otro. Ahora
# crear y continuar recuentan y escriben dentro de un candado con nombre del
# SERVIDOR MariaDB (GET_LOCK), tomado en UNA conexión dedicada que vive todo el
# bloque. El asyncio.Lock sigue por fuera: dentro del proceso evita pedir
# conexiones de más.
#
# - El nombre lleva la base: los candados con nombre son del servidor, no de la
#   base, y jax_memory_test comparte servidor con producción.
# - GET_LOCK devuelve 1 (tomado), 0 (venció) o NULL (error): lo que no sea 1,
#   o una conexión que no abre, es CandadoNoDisponible -> quien llama falla
#   cerrado (503 prevuelo_no_disponible). Nunca se cuenta sin candado.
# - Se suelta SIEMPRE: RELEASE_LOCK en finally y, pase lo que pase, la conexión
#   se cierra -- cerrar la sesión libera el candado en el servidor aunque
#   RELEASE_LOCK haya fallado.
# EXPLAIN en tests/test_jacobs_candado_activos_db.py.
NOMBRE_CANDADO_DE_ACTIVOS = "jacobs_crear_o_continuar"
_SQL_TOMAR_CANDADO = "SELECT GET_LOCK(%s, %s)"
_SQL_SOLTAR_CANDADO = "SELECT RELEASE_LOCK(%s)"


class CandadoNoDisponible(RuntimeError):
    """No se obtuvo el candado del cupo de activos: venció, la base lo negó o
    no hubo conexión. Falla cerrado: sin candado no se crea ni se continúa."""


def nombre_del_candado_de_activos() -> str:
    return f"{NOMBRE_CANDADO_DE_ACTIVOS}:{_db_cfg()['db']}"


@asynccontextmanager
async def candado_de_activos() -> AsyncIterator[aiomysql.Connection]:
    """Toma el candado del cupo de activos y entrega su conexión (para
    recontar por ella). Ver el bloque de comentarios de arriba."""
    from jacobs.prevuelo_config import candado_timeout_s

    timeout = candado_timeout_s()
    nombre = nombre_del_candado_de_activos()
    try:
        conn = await conexion_dedicada()
    except Exception as exc:  # fail-closed: se relanza como CandadoNoDisponible; sin conexión no hay candado y sin candado no se cuenta
        raise CandadoNoDisponible(
            f"no se pudo abrir la conexión del candado '{nombre}': {type(exc).__name__}: {exc}"
        ) from exc
    try:
        try:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_TOMAR_CANDADO, (nombre, timeout))
                fila = await cur.fetchone()
        except Exception as exc:  # fail-closed: se relanza como CandadoNoDisponible
            raise CandadoNoDisponible(
                f"GET_LOCK('{nombre}') falló: {type(exc).__name__}: {exc}"
            ) from exc
        resultado = fila[0] if fila else None
        if resultado != 1:
            motivo = "venció" if resultado == 0 else "devolvió NULL"
            raise CandadoNoDisponible(
                f"GET_LOCK('{nombre}', {timeout}) {motivo}: otro proceso tiene el cupo de "
                f"pipelines activos tomado o la base no lo concedió"
            )
        try:
            yield conn
        finally:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(_SQL_SOLTAR_CANDADO, (nombre,))
            except Exception as exc:  # fail-soft: cerrar la conexión (finally de abajo) termina la sesión y el servidor suelta el candado igual; se deja WARNING
                logger.warning("RELEASE_LOCK('%s') falló (%s); se cierra la conexión, que lo suelta",
                               nombre, type(exc).__name__)
    finally:
        conn.close()


# ----------------------------------------------------------------
#  Época de corrida (spec 2026-09-17 §5.3)
# ----------------------------------------------------------------
# Un solo ejecutor por pipeline. `cancel`, el kill switch y el reaper cambian
# el STATUS; `resume`, `approve-step` y `continue` INCREMENTAN la época. El
# ejecutor escribe sólo si el pipeline sigue en SU época y `running`: si no,
# perdió, registra RUN_SUPERSEDED una vez y termina sin escribir más.
# Todas van por clave primaria (EXPLAIN en tests/test_run_epoch_db.py).

_SQL_EPOCA_Y_STATUS = "SELECT run_epoch, status FROM jacobs_pipelines WHERE pipeline_id=%s"

_SQL_STEP_SI_EPOCA = (
    "UPDATE jacobs_steps s JOIN jacobs_pipelines p ON p.pipeline_id = s.pipeline_id "
    "SET s.status=%s, s.facet=%s, s.motor=%s, s.output_ref=%s, s.timeout_seconds=%s, "
    "    s.started_at=%s, s.finished_at=%s, s.error=%s "
    "WHERE s.step_id=%s AND p.pipeline_id=%s AND p.run_epoch=%s AND p.status='running'"
)


def _sql_update_si_epoca(con_indice: bool, con_contexto: bool, n_desde: int,
                         con_corte: bool = False) -> str:
    sets = ["status=%s", "updated_at=%s"]
    if con_indice:
        sets.append("current_step_index=%s")
    if con_contexto:
        sets.append("context_refs=%s")
    desde = ",".join(["%s"] * n_desde)
    # con_corte (pasada final R34): el reaper exige que la fila SIGA sin avance
    # al escribir; un avance entre su lectura y esta escritura la saca.
    corte = " AND updated_at < %s" if con_corte else ""
    return (
        f"UPDATE jacobs_pipelines SET {', '.join(sets)} "
        f"WHERE pipeline_id=%s AND run_epoch=%s AND status IN ({desde}){corte}"
    )


def _sql_tomar_epoca(con_contexto: bool, n_desde: int) -> str:
    extra = ", context_refs=%s" if con_contexto else ""
    desde = ",".join(["%s"] * n_desde)
    return (
        f"UPDATE jacobs_pipelines SET run_epoch=run_epoch+1, updated_at=%s{extra} "
        f"WHERE pipeline_id=%s AND run_epoch=%s AND status IN ({desde})"
    )


async def _ejecutar_condicional(sql: str, params: tuple | list) -> int:
    conn = await conexion_dedicada(found_rows=True)
    try:
        async with conn.cursor() as cur:
            return await cur.execute(sql, params)
    finally:
        conn.close()


async def pipeline_epoca_y_status(pipeline_id: str) -> tuple[int, PipelineStatus] | None:
    # Merge 2026-09-17: es un SELECT sin condición de filas afectadas -- va por
    # el pool, no por una conexión dedicada (la dedicada queda para
    # CLIENT.FOUND_ROWS y para el GET_LOCK; ver conexion_dedicada).
    async with conexion_del_pool() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_SQL_EPOCA_Y_STATUS, (pipeline_id,))
            fila = await cur.fetchone()
    if not fila:
        return None
    return int(fila[0]), PipelineStatus(fila[1])


async def pipeline_update_status_si_epoca(
    pipeline_id: str,
    epoca: int,
    status: PipelineStatus,
    current_step_index: int | None = None,
    context: dict | None = None,
    *,
    desde: tuple[PipelineStatus, ...] = (PipelineStatus.running,),
    sin_avance_desde: float | None = None,
) -> bool:
    """True si escribió: el pipeline estaba en `epoca` y en uno de `desde` (y,
    con `sin_avance_desde`, su updated_at sigue anterior a ese instante --
    lo usa el reaper, pasada final R34)."""
    if not desde:
        raise ValueError(
            "pipeline_update_status_si_epoca: 'desde' no puede estar vacío -- "
            "'status IN ()' es SQL inválido, es un error de contrato del llamador."
        )
    params: list = [status.value, time.time()]
    if current_step_index is not None:
        params.append(current_step_index)
    if context is not None:
        params.append(json.dumps(context, ensure_ascii=False))
    params += [pipeline_id, epoca, *(d.value for d in desde)]
    if sin_avance_desde is not None:
        params.append(sin_avance_desde)
    sql = _sql_update_si_epoca(current_step_index is not None, context is not None, len(desde),
                               con_corte=sin_avance_desde is not None)
    return await _ejecutar_condicional(sql, params) == 1


async def step_upsert_si_epoca(s: Step, epoca: int) -> bool:
    """Escritura de un paso YA EXISTENTE desde el ejecutor. True si escribió."""
    params = (
        s.status.value, s.facet, s.motor, s.output_ref, s.timeout_seconds,
        s.started_at, s.finished_at, s.error,
        s.step_id, s.pipeline_id, epoca,
    )
    return await _ejecutar_condicional(_SQL_STEP_SI_EPOCA, params) == 1


async def pipeline_tomar_epoca(
    pipeline_id: str,
    epoca_leida: int,
    desde: tuple[PipelineStatus, ...],
    context: dict | None = None,
) -> int | None:
    """Incrementa la época si nadie la tomó desde que se leyó. Devuelve la
    nueva, o None si otro pedido ganó (doble resume, doble approve)."""
    if not desde:
        raise ValueError(
            "pipeline_tomar_epoca: 'desde' no puede estar vacío -- "
            "'status IN ()' es SQL inválido, es un error de contrato del llamador."
        )
    params: list = [time.time()]
    if context is not None:
        params.append(json.dumps(context, ensure_ascii=False))
    params += [pipeline_id, epoca_leida, *(d.value for d in desde)]
    filas = await _ejecutar_condicional(_sql_tomar_epoca(context is not None, len(desde)), params)
    return epoca_leida + 1 if filas == 1 else None


_SQL_BLOQUEAR_PIPELINE = "SELECT run_epoch, status FROM jacobs_pipelines WHERE pipeline_id=%s FOR UPDATE"
_SQL_PASO_A_CORRER = (
    "UPDATE jacobs_steps SET facet=%s, motor=%s, status='pending', output_ref=NULL, "
    "started_at=NULL, finished_at=NULL, error=NULL WHERE step_id=%s AND pipeline_id=%s"
)
_SQL_PIPELINE_CONTINUAR = (
    "UPDATE jacobs_pipelines SET status='running', run_epoch=run_epoch+1, plan=%s, "
    "context_refs=%s, current_step_index=%s, updated_at=%s "
    "WHERE pipeline_id=%s AND run_epoch=%s"
)


_SQL_EVENTO_CONTINUED = (
    "INSERT INTO jacobs_events (pipeline_id, step_id, event_type, payload, ts) "
    "VALUES (%s,%s,%s,%s,%s)"
)


async def continuar_transaccion(
    pipeline_id: str,
    epoca_leida: int,
    status_leido: PipelineStatus,
    pasos_a_correr: list[Step],
    plan: list[Step],
    context: dict,
    current_step_index: int,
    evento_payload: dict | None,
    estado: EstadoDeTransaccion | None = None,
) -> int | None:
    """Escrituras de continue en UNA transacción (spec 2026-09-17 §5.2 regla
    10): bloquea la fila del pipeline, confirma que nadie la cambió desde el
    análisis (misma época y mismo status), resetea los pasos a correr, reescribe
    plan y contexto, pone running e incrementa la época, y -- si `evento_payload`
    no es None -- inserta el evento PIPELINE_CONTINUED con el MISMO cursor,
    antes del commit (Ruling R22: regla 10 lo exige dentro de la transacción,
    no después). `evento_payload` es OBLIGATORIO (Principio IX / revisión
    fix round 2): un default silencioso dejaría que un llamador se saltara el
    evento de auditoría sin que se note en el sitio de la llamada -- el
    llamador tiene que decidir explícitamente None si de verdad no quiere
    evento (ningún camino de producción lo hace: continuar.py siempre arma un
    payload real). Devuelve la época nueva, o None si otro pedido ganó
    (época/status ya no coinciden, o -- cinturón, Ruling R23 -- el UPDATE
    final no tocó la fila que el SELECT...FOR UPDATE acababa de ver). Un error
    a mitad hace ROLLBACK: nada cambia, ni los pasos, ni el pipeline, ni el
    evento.

    `estado` (EstadoDeTransaccion, re-revisión final): mismo mecanismo que
    crear (R41). Si la conexión se corta o el plazo vence DURANTE el COMMIT,
    desde acá no se puede saber si el servidor confirmó; queda
    `estado.incierta` para que quien llama lo diga en su 503."""
    estado = estado if estado is not None else EstadoDeTransaccion()
    conn = await conexion_dedicada(found_rows=True)
    try:
        await conn.begin()
        try:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_BLOQUEAR_PIPELINE, (pipeline_id,))
                fila = await cur.fetchone()
                if fila is None or int(fila[0]) != epoca_leida or fila[1] != status_leido.value:
                    await conn.rollback()
                    return None
                for paso in pasos_a_correr:
                    await cur.execute(_SQL_PASO_A_CORRER, (paso.facet, paso.motor, paso.step_id, pipeline_id))
                filas_pipeline = await cur.execute(_SQL_PIPELINE_CONTINUAR, (
                    json.dumps([s.model_dump() for s in plan], ensure_ascii=False),
                    json.dumps(context, ensure_ascii=False),
                    current_step_index, time.time(), pipeline_id, epoca_leida,
                ))
                # R23: el SELECT...FOR UPDATE ya lo garantiza (misma fila,
                # bajo lock, época/status verificados arriba) -- esto es el
                # cinturón explícito del requisito (a), no una rama que se
                # espere alcanzar en producción.
                if filas_pipeline != 1:
                    await conn.rollback()
                    return None
                if evento_payload is not None:
                    await cur.execute(_SQL_EVENTO_CONTINUED, (
                        pipeline_id, None, "PIPELINE_CONTINUED",
                        json.dumps(evento_payload, ensure_ascii=False), time.time(),
                    ))
            estado.enviando_commit = True
            await conn.commit()
            estado.confirmada = True
        except BaseException:
            await conn.rollback()
            raise
    finally:
        conn.close()
    return epoca_leida + 1


def _row_to_pipeline(row: dict) -> Pipeline:
    plan_raw = row.get("plan") or "[]"
    plan_data = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
    steps = [Step(**s) for s in plan_data]

    ctx_raw = row.get("context_refs") or "{}"
    ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw

    return Pipeline(
        pipeline_id=row["pipeline_id"],
        name=row["name"],
        invoked_by=row["invoked_by"],
        # .get() (M1, 2026-08-10): jacobs_relaunch.py llama pipeline_get() sin
        # garantizar init_tables() primero -- contra una DB no migrada
        # (columnas user_id/tenant_id ausentes) esto degrada a None en vez de
        # KeyError.
        user_id=row.get("user_id"),
        tenant_id=row.get("tenant_id"),
        owner_ack_at=row.get("owner_ack_at"),
        run_epoch=int(row.get("run_epoch") or 0),
        parent_pipeline_id=row.get("parent_pipeline_id"),
        depth=int(row.get("depth") or 0),
        mode=row["mode"],
        status=PipelineStatus(row["status"]),
        plan=steps,
        current_step_index=row["current_step_index"],
        max_steps=row["max_steps"],
        context=ctx,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ----------------------------------------------------------------
#  Step CRUD
# ----------------------------------------------------------------

async def step_upsert(s: Step, conexion: aiomysql.Connection | None = None) -> None:
    async with _conexion_o_pool(conexion) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO jacobs_steps
                    (step_id, pipeline_id, step_index, facet, motor, capability,
                     input_ref, output_ref, status, timeout_seconds,
                     retries_allowed, skip_on_fail, trace_id,
                     started_at, finished_at, error, depends_on)
                VALUES (%s,%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s, %s,%s,%s, %s)
                ON DUPLICATE KEY UPDATE
                    status=VALUES(status),
                    facet=VALUES(facet),
                    motor=VALUES(motor),
                    output_ref=VALUES(output_ref),
                    timeout_seconds=VALUES(timeout_seconds),
                    started_at=VALUES(started_at),
                    finished_at=VALUES(finished_at),
                    error=VALUES(error),
                    depends_on=VALUES(depends_on)
                """,
                (
                    s.step_id, s.pipeline_id, s.step_index, s.facet, s.motor, s.capability,
                    json.dumps(s.input, ensure_ascii=False), s.output_ref,
                    s.status.value, s.timeout_seconds,
                    s.retries_allowed, s.skip_on_fail, s.trace_id,
                    s.started_at, s.finished_at, s.error,
                    json.dumps(s.depends_on, ensure_ascii=False),
                ),
            )


async def steps_by_pipeline(pipeline_id: str) -> list[Step]:
    async with conexion_del_pool() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM jacobs_steps WHERE pipeline_id=%s ORDER BY step_index",
                (pipeline_id,),
            )
            rows = await cur.fetchall()
    result = []
    for row in rows:
        input_raw = row.get("input_ref") or "{}"
        try:
            input_data = json.loads(input_raw)
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        deps_raw = row.get("depends_on")
        try:
            depends_on = json.loads(deps_raw) if deps_raw else []
        except (json.JSONDecodeError, TypeError):
            depends_on = []
        result.append(Step(
            step_id=row["step_id"],
            pipeline_id=row["pipeline_id"],
            step_index=row["step_index"],
            facet=row["facet"],
            motor=row.get("motor"),
            capability=row["capability"],
            input=input_data,
            output_ref=row["output_ref"],
            status=StepStatus(row["status"]),
            timeout_seconds=row["timeout_seconds"],
            retries_allowed=row["retries_allowed"],
            skip_on_fail=bool(row["skip_on_fail"]),
            trace_id=row["trace_id"] or "",
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error=row["error"],
            depends_on=depends_on,
        ))
    return result


# ----------------------------------------------------------------
#  Motor governance (T2, 2026-08-21)
# ----------------------------------------------------------------

async def get_motor_governance() -> dict[str, dict]:
    """Vista completa de motor/capability/capability_motor -- MISMA tabla
    que /api/motors/capabilities (jax-platform) y MotorCatalog.from_db()
    (las_manos), no una copia.

    Bloque 3 (2026-08-21): extendida de "solo allowed_capabilities +
    has_tool_access por motor" a la fila `capability` COMPLETA, keyed por
    capability -- antes devolvía una vista parcial que PlanBuilder.build()
    (jacobs/plan.py::_validate_plan_capabilities) consultaba para rechazar
    un plan ANTES de persistirlo, pero le faltaba allowed_callers y el
    resto de columnas de gobernanza. Sustituir NIVEL B de
    executor.py::validate_capability() (antes: las_manos/config.toml) por
    esta función tal cual, sin extenderla, habría eliminado el chequeo de
    allowed_callers en silencio -- el mismo patrón de dos fuentes que
    divergen que esta consolidación existe para cerrar. Ahora plan-time
    (_validate_plan_capabilities) y dispatch-time (validate_capability)
    miran exactamente la misma estructura, una sola query.

    Devuelve:
      {"capabilities": {capability_key: {allowed_motors: [motor_key, ...]
       (orden = capability_motor.priority ASC), allowed_callers: [...],
       risk_level, sandbox_only, requires_human_gate, max_execution_minutes,
       max_recursion_depth, output_schema, fallback_motor, fallback_mode,
       forbidden_paths, auditor_motor}},
       "motors": {motor_key: has_tool_access (bool)},
       "facets": frozenset de facet.key con status='active'}

    Costo medido en vivo (2026-08-21, DB real) con 3 SELECTs: 0.00024s de
    ejecución total en el servidor (motor: 4 filas, capability: ~17,
    capability_motor: ~26) -- insignificante para llamar en cada dispatch,
    no solo en plan-build. El 4º SELECT (facet, 7 filas, E-17) se agregó
    después y NO está medido.

    Ruling R38 (2026-09-17): por el pool del store, no por una conexión propia
    -- era la conexión por pedido que tiraba /jacobs/preflight a c=50. Sin
    caché: cada llamada sigue leyendo las tablas (misma foto que antes);
    un error de la base se propaga igual (fail-closed)."""
    async with conexion_del_pool() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT `key`, has_tool_access FROM motor")
            motors: dict[str, bool] = {key: bool(has_tools) for key, has_tools in await cur.fetchall()}

            await cur.execute(
                "SELECT `key`, risk_level, sandbox_only, requires_human_gate, "
                "max_execution_minutes, max_recursion_depth, output_schema, "
                "fallback_motor, fallback_mode, allowed_callers, forbidden_paths, "
                "auditor_motor FROM capability"
            )
            capabilities: dict[str, dict] = {}
            for (key, risk_level, sandbox_only, gate, max_exec, max_rec, schema,
                 fallback_motor, fallback_mode, callers, forbidden, auditor_motor) in await cur.fetchall():
                capabilities[key] = {
                    "allowed_motors": [],
                    "allowed_callers": json.loads(callers) if callers else [],
                    "risk_level": risk_level,
                    "sandbox_only": bool(sandbox_only),
                    "requires_human_gate": bool(gate),
                    "max_execution_minutes": max_exec,
                    "max_recursion_depth": max_rec,
                    "output_schema": schema or "",
                    "fallback_motor": fallback_motor,
                    "fallback_mode": fallback_mode or "manual_only",
                    "forbidden_paths": json.loads(forbidden) if forbidden else [],
                    "auditor_motor": auditor_motor,
                }

            await cur.execute(
                "SELECT capability_key, motor_key FROM capability_motor "
                "ORDER BY capability_key, priority ASC"
            )
            for capability_key, motor_key in await cur.fetchall():
                # setdefault cubre una fila de capability_motor para una
                # capability sin fila propia en `capability` -- no debería
                # pasar (FK), pero no asumir consistencia entre SELECTs no
                # transaccionales.
                capabilities.setdefault(capability_key, {
                    "allowed_motors": [], "allowed_callers": [], "risk_level": "high",
                    "sandbox_only": True, "requires_human_gate": True,
                    "max_execution_minutes": 5, "max_recursion_depth": 0,
                    "output_schema": "", "fallback_motor": None,
                    "fallback_mode": "manual_only", "forbidden_paths": [],
                    "auditor_motor": None,
                })
                capabilities[capability_key]["allowed_motors"].append(motor_key)

            # E-03 (2026-09-16): el vocabulario de facetas del planner es la
            # tabla `facet`, no una lista fija. Mismas reglas que resolve_facet
            # (status='active'): lo que el planner acepta es lo que se puede
            # despachar. Catálogo de 7 filas: sin índice, declarado en DEUDA.md.
            await cur.execute("SELECT `key` FROM facet WHERE status = 'active'")
            facets = frozenset(key for (key,) in await cur.fetchall())
    return {"capabilities": capabilities, "motors": motors, "facets": facets}


# ----------------------------------------------------------------
#  Contrato de sub-pipelines (frente F, 2026-09-16)
# ----------------------------------------------------------------
# Las dos sentencias que deciden son UNA sola cada una, autocommit: nada de
# SELECT-y-después-escribir, que es exactamente la ventana de carrera que este
# contrato cierra. Si no afectan una fila, un diagnóstico APARTE nombra el
# motivo; el diagnóstico solo etiqueta el rechazo, no lo decide.

SQL_EMITIR_TOKEN = """
    INSERT INTO jacobs_subpipeline_tokens
        (token_hash, parent_pipeline_id, parent_step, depth_hijo, emitido_at, vence_at)
    SELECT %s, p.pipeline_id, s.step_id, p.depth + 1, %s, %s
      FROM jacobs_pipelines p
      JOIN jacobs_steps s ON s.step_id = %s AND s.pipeline_id = p.pipeline_id
     WHERE p.pipeline_id = %s
       AND p.status = 'running'
       AND s.facet = 'ada'
       AND p.depth + 1 <= %s
"""


async def subpipeline_token_emitir(
    token_hash: str,
    parent_pipeline_id: str,
    parent_step: str,
    emitido_at: float,
    vence_at: float,
    max_profundidad: int,
) -> int | None:
    """Inserta el hash solo si el padre está `running`, el paso existe, es de
    ese padre y es de Ada (en cualquier estado: enmienda 2026-09-16), y el
    hijo no excede la profundidad. Devuelve depth_hijo, o None si no insertó."""
    async with conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                SQL_EMITIR_TOKEN,
                (token_hash, emitido_at, vence_at, parent_step, parent_pipeline_id, max_profundidad),
            )
            if cur.rowcount != 1:
                return None
            await cur.execute(
                "SELECT depth_hijo FROM jacobs_subpipeline_tokens WHERE token_hash = %s",
                (token_hash,),
            )
            (depth_hijo,) = await cur.fetchone()
            return int(depth_hijo)


async def subpipeline_emision_diagnostico(parent_pipeline_id: str, parent_step: str) -> dict:
    async with conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT status, depth FROM jacobs_pipelines WHERE pipeline_id = %s",
                (parent_pipeline_id,),
            )
            padre = await cur.fetchone()
            await cur.execute(
                "SELECT status, facet, pipeline_id FROM jacobs_steps WHERE step_id = %s",
                (parent_step,),
            )
            paso = await cur.fetchone()
    return {
        "padre_status": padre["status"] if padre else None,
        "padre_depth": int(padre["depth"]) if padre else None,
        "paso_status": paso["status"] if paso else None,
        "paso_facet": paso["facet"] if paso else None,
        "paso_pipeline_id": paso["pipeline_id"] if paso else None,
    }


# El consumo. Un UPDATE multi-tabla autocommit: toma el candado de fila de la PK
# del token; un segundo consumo concurrente espera ese candado y, al liberarse,
# RELEE la versión confirmada (REPEATABLE-READ y READ-COMMITTED), ve usado_at
# puesto y afecta 0 filas. Probado con intercalado forzado en
# jacobs/_subpipeline_contrato_io_test.py. Plan: t por PK (const), p y s por PK.
SQL_CONSUMIR_TOKEN = """
    UPDATE jacobs_subpipeline_tokens t
      JOIN jacobs_pipelines p ON p.pipeline_id = t.parent_pipeline_id
      JOIN jacobs_steps s     ON s.step_id = t.parent_step
                             AND s.pipeline_id = t.parent_pipeline_id
       SET t.usado_at = %s, t.hijo_pipeline_id = %s
     WHERE t.token_hash = %s
       AND t.usado_at IS NULL
       AND t.vence_at > %s
       AND t.parent_pipeline_id = %s
       AND t.depth_hijo <= %s
       AND p.status = 'running'
"""

# La identidad (user_id, tenant_id) del hijo es la del PADRE (revisión final,
# I-3): sale de acá, nunca del cuerpo. t por PK, p por PK.
SQL_TOKEN_CONSUMIDO = """
    SELECT t.parent_pipeline_id, t.parent_step, t.depth_hijo, t.hijo_pipeline_id,
           p.user_id, p.tenant_id
      FROM jacobs_subpipeline_tokens t
      JOIN jacobs_pipelines p ON p.pipeline_id = t.parent_pipeline_id
     WHERE t.token_hash = %s
"""

# Identidad del padre para el que se EMITIÓ el token (no el que declara el
# cuerpo), leída antes de consumir cuando el cuerpo trae identidad. t y p por PK.
SQL_IDENTIDAD_PADRE_DEL_TOKEN = """
    SELECT t.parent_pipeline_id, p.user_id, p.tenant_id
      FROM jacobs_subpipeline_tokens t
      JOIN jacobs_pipelines p ON p.pipeline_id = t.parent_pipeline_id
     WHERE t.token_hash = %s
"""

SQL_DIAGNOSTICO_TOKEN = """
    SELECT t.usado_at, t.vence_at, t.parent_pipeline_id, t.depth_hijo,
           p.status AS padre_status, s.step_id AS paso_step_id
      FROM jacobs_subpipeline_tokens t
      LEFT JOIN jacobs_pipelines p ON p.pipeline_id = t.parent_pipeline_id
      LEFT JOIN jacobs_steps s     ON s.step_id = t.parent_step
                                  AND s.pipeline_id = t.parent_pipeline_id
     WHERE t.token_hash = %s
"""


async def subpipeline_token_consumir(
    token_hash: str,
    parent_pipeline_id: str,
    hijo_pipeline_id: str,
    ahora: float,
    max_profundidad: int,
) -> dict | None:
    """Consume el token para `hijo_pipeline_id`. Devuelve la fila consumida
    (parent_pipeline_id, parent_step, depth_hijo, hijo_pipeline_id y la
    identidad del padre: user_id, tenant_id) o None si no afectó una fila.

    La confirmación lee por PK (token_hash), no por hijo_pipeline_id: si el
    UPDATE reportó una fila afectada pero la relectura no existe o su
    hijo_pipeline_id no es el nuestro, el candado de fila y el UPDATE
    condicionado no están haciendo lo que dicen (invariante roto), y eso NO
    es un rechazo -- se revienta fuerte en vez de devolver un TOKEN_USADO
    engañoso que quemaría el token contra el hijo equivocado en silencio."""
    async with conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                SQL_CONSUMIR_TOKEN,
                (ahora, hijo_pipeline_id, token_hash, ahora, parent_pipeline_id, max_profundidad),
            )
            if cur.rowcount != 1:
                return None
            await cur.execute(SQL_TOKEN_CONSUMIDO, (token_hash,))
            fila = await cur.fetchone()
            if fila is None or fila["hijo_pipeline_id"] != hijo_pipeline_id:
                raise RuntimeError(
                    "invariante del consumo roto: el UPDATE reportó una fila afectada pero "
                    f"la relectura por PK no es del hijo esperado (hijo_pipeline_id={hijo_pipeline_id!r})"
                )
            return fila


async def subpipeline_token_identidad_padre(token_hash: str) -> dict | None:
    """(parent_pipeline_id, user_id, tenant_id) del padre del token, o None si
    el hash no existe o el padre ya no está. Solo lectura: no consume."""
    async with conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(SQL_IDENTIDAD_PADRE_DEL_TOKEN, (token_hash,))
            return await cur.fetchone()


async def subpipeline_token_diagnostico(token_hash: str) -> dict | None:
    async with conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(SQL_DIAGNOSTICO_TOKEN, (token_hash,))
            return await cur.fetchone()


# ----------------------------------------------------------------
#  Human gate de LAS MANOS (2026-09-17)
# ----------------------------------------------------------------
# Mismo patrón que el contrato de sub-pipelines: la decisión es UNA sentencia
# autocommit (nada de SELECT-y-después-escribir); el diagnóstico solo rotula.

SQL_EMITIR_TOKEN_GATE = """
    INSERT INTO las_manos_human_gate_tokens
        (token_hash, emitido_por, emitido_at, vence_at)
    VALUES (%s, %s, %s, %s)
"""

# Candado de fila por PK: un segundo consumo concurrente espera, relee la
# versión confirmada, ve usado_at puesto y afecta 0 filas. Probado con 20
# consumos a la vez en las_manos/_human_gate_io_test.py.
SQL_CONSUMIR_TOKEN_GATE = """
    UPDATE las_manos_human_gate_tokens
       SET usado_at = %s, usado_en = %s
     WHERE token_hash = %s
       AND usado_at IS NULL
       AND vence_at > %s
"""

SQL_DIAGNOSTICO_TOKEN_GATE = """
    SELECT usado_at, vence_at
      FROM las_manos_human_gate_tokens
     WHERE token_hash = %s
"""


async def human_gate_token_emitir(
    token_hash: str, emitido_por: str, emitido_at: float, vence_at: float,
) -> None:
    async with conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_EMITIR_TOKEN_GATE, (token_hash, emitido_por, emitido_at, vence_at))


async def human_gate_token_consumir(token_hash: str, usado_en: str, ahora: float) -> bool:
    """True si ESTE llamador consumió el token (una fila afectada)."""
    async with conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_CONSUMIR_TOKEN_GATE, (ahora, usado_en, token_hash, ahora))
            return cur.rowcount == 1


async def human_gate_token_diagnostico(token_hash: str) -> dict | None:
    async with conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(SQL_DIAGNOSTICO_TOKEN_GATE, (token_hash,))
            return await cur.fetchone()


# ----------------------------------------------------------------
#  Audit events
# ----------------------------------------------------------------

async def event_append(
    pipeline_id: str,
    event_type: str,
    payload: dict | None = None,
    step_id: str | None = None,
    conexion: aiomysql.Connection | None = None,
) -> None:
    async with _conexion_o_pool(conexion) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """INSERT INTO jacobs_events (pipeline_id, step_id, event_type, payload, ts)
                   VALUES (%s,%s,%s,%s,%s)""",
                (
                    pipeline_id, step_id, event_type,
                    json.dumps(payload or {}, ensure_ascii=False),
                    time.time(),
                ),
            )


async def events_by_pipeline(pipeline_id: str) -> list[dict[str, Any]]:
    # m3 de la re-revisión final (2026-09-17): por el pool, como el resto de
    # los endpoints. GET /jacobs/pipeline/{id}/events era el último que abría
    # una conexión por pedido -- la misma forma que a c=50 dio 0/13/49/61 % de
    # errores en get_motor_governance (R38).
    async with conexion_del_pool() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM jacobs_events WHERE pipeline_id=%s ORDER BY id",
                (pipeline_id,),
            )
            rows = await cur.fetchall()
    result = []
    for row in rows:
        payload_raw = row.get("payload") or "{}"
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
        result.append({
            "id":          row["id"],
            "pipeline_id": row["pipeline_id"],
            "step_id":     row["step_id"],
            "event_type":  row["event_type"],
            "payload":     payload,
            "ts":          row["ts"],
        })
    return result
