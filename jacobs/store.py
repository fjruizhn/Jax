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
from typing import Any

import aiomysql
from pymysql import err as _pymysql_err

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
# _lanzado_por lee nombres de frame de PyMySQL (raise_for_error ->
# raise_mysql_exception) y _codigo_del_servidor_sano su tabla error_map.
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


_pools: dict[asyncio.AbstractEventLoop, _PoolDelLoop] = {}


def tamanio_pool() -> int:
    """Lee y valida `JAX_JACOBS_DB_POOL_SIZE`. Ausente -> default. Presente pero
    vacia, no entera o fuera de [1, TAMANIO_POOL_MAXIMO] -> RuntimeError
    (fail-closed: un typo no se convierte en un pool de 0 o de 5000)."""
    crudo = os.environ.get(ENV_TAMANIO_POOL)
    if crudo is None:
        return TAMANIO_POOL_POR_DEFECTO
    try:
        valor = int(crudo)
    except ValueError:
        valor = None
    if valor is None or not 1 <= valor <= TAMANIO_POOL_MAXIMO:
        raise RuntimeError(
            f"{ENV_TAMANIO_POOL}={crudo!r} invalido -- tiene que ser un entero entre 1 "
            f"y {TAMANIO_POOL_MAXIMO} (max_connections de la MariaDB es 151 y la "
            "comparten todos los servicios)."
        )
    return valor


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


# Una conexion que termino el cuerpo con excepcion vuelve al pool SOLO si hay
# evidencia de tres cosas; si falta una, se cierra (fail-closed). Descartarla
# siempre abria un handshake por pedido fallido: el job de CI de 13b7759 (base
# sin la tabla `facet`, 1146) midio 20 conexiones para 20 autorizaciones, el
# mismo agotamiento de puertos que el pool vino a cerrar, disparado por una
# rafaga de errores. Las tres:
#
#   1. QUE error: uno que respondio el servidor y no termina la sesion
#      (_codigo_del_servidor_sano). LISTA BLANCA: pymysql mapea un codigo
#      desconocido a OperationalError, y ahi caen tambien los que SI cortan la
#      sesion (1927 conexion matada, 1053 apagado).
#   2. DE QUIEN: lo lanzo ESTA conexion al leer el paquete de error
#      (_lanzado_por). Un error de otra conexion propagado dentro del bloque no
#      dice nada del estado de esta.
#   3. EN QUE ESTADO: nadie mas la esta usando y no queda nada en vuelo
#      (_en_reposo). Una tarea hermana de un `gather` que sigue viva con la
#      conexion en sus frames, bytes sin leer, una lectura esperando o una
#      escritura sin vaciar: se cierra.
#
# Lo que (3) NO ve: una tarea que guarda la conexion dentro de un contenedor
# (lista, dict, atributo de otro objeto) y esta suspendida fuera de la E/S de
# la conexion. Es el mismo uso fuera del bloque que el contrato de conexion()
# ya prohibe tambien para la salida sin error.
#
# (2) y (3) leen estado interno de aiomysql 0.3.2 / pymysql 1.2 (nombres de
# frame, `_reader`, `_writer`, `_result`): las dos versiones estan fijadas y las
# vigila VersionDeAiomysqlTest. Si un nombre no esta, AttributeError -> se cierra.
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


def _lanzado_por(e: BaseException, conn: aiomysql.Connection) -> bool:
    """El final del traceback es el paquete de error leido por ESTA conexion:
    `Connection._read_packet` (self is conn) -> `raise_for_error` ->
    `raise_mysql_exception`. Nada despues: si alguien lo atrapo y relanzo
    otro, el ultimo frame ya no es ese."""
    frames = []
    tb = e.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    if len(frames) < 3:
        return False
    lectura, paquete, lanzamiento = frames[-3:]
    return (
        lanzamiento.f_code.co_name == "raise_mysql_exception"
        and paquete.f_code.co_name == "raise_for_error"
        and lectura.f_code.co_name == "_read_packet"
        and lectura.f_locals.get("self") is conn
    )


def _referencia_a(valor: Any, conn: aiomysql.Connection) -> bool:
    return valor is conn or (isinstance(valor, aiomysql.Cursor) and valor._connection is conn)


def _en_reposo(conn: aiomysql.Connection) -> bool:
    lector = conn._reader
    if (conn.closed or lector is None or lector._waiter is not None or len(lector._buffer)
            or lector.at_eof() or lector.exception() is not None):
        return False
    if conn._writer is None or conn._writer.transport.get_write_buffer_size():
        return False
    if conn._result is not None and conn._result.unbuffered_active:
        return False
    actual = asyncio.current_task()
    for tarea in asyncio.all_tasks():
        if tarea is actual:
            continue
        for frame in tarea.get_stack():
            if any(_referencia_a(v, conn) for v in frame.f_locals.values()):
                return False
    return True


def _reutilizable_tras_error(e: BaseException, conn: aiomysql.Connection) -> bool:
    try:
        return _codigo_del_servidor_sano(e) and _lanzado_por(e, conn) and _en_reposo(conn)
    except Exception:  # fail-soft: la verificacion no pudo probar el reposo; devuelve False y la conexion se CIERRA (fail-closed)
        logger.exception("jacobs.store: no se pudo verificar la conexion tras un error; se cierra")
        return False


@contextlib.asynccontextmanager
async def conexion(desechable: bool = False):
    """Una conexion del pool, devuelta al salir.

    Se DESCARTA (se cierra, el pool abre otra cuando haga falta) si el cuerpo
    termino con excepcion o cancelacion -- el socket puede haber quedado a mitad
    de una respuesta --, si la sesion quedo sucia, o si `desechable=True` (ver el
    contrato en _sesion_reutilizable). La excepcion a esa regla es un error que
    respondio el servidor a ESTA conexion y la deja en reposo
    (_reutilizable_tras_error): vuelve al pool si la sesion sigue limpia.

    Esperar un hueco tiene limite: `JAX_DB_CONNECT_TIMEOUT_SECONDS`, el mismo
    que acota abrir el socket. Con el pool lleno mas alla de eso, TimeoutError:
    fail-closed, no una espera infinita (aiomysql espera sin limite)."""
    estado = await _estado_del_loop()
    pool, permisos = estado.pool, estado.permisos
    limite = db_connect_timeout_seconds()
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
    limpia = False
    try:
        yield conn
        limpia = True
    except BaseException as e:
        limpia = _reutilizable_tras_error(e, conn)
        raise
    finally:
        try:
            if desechable or not limpia or not _sesion_reutilizable(conn):
                conn.close()
            # release() saca la conexion de `_used` antes de devolver. El aviso
            # que agenda es para el cond.wait de aiomysql, al que con el
            # semaforo nadie llega: no se espera.
            pool.release(conn)
        finally:
            permisos.release()


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

async def pipeline_create(p: Pipeline) -> None:
    async with conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO jacobs_pipelines
                    (pipeline_id, name, invoked_by, mode, status,
                     plan, current_step_index, max_steps, context_refs,
                     created_at, updated_at, user_id, tenant_id,
                     parent_pipeline_id, depth)
                VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s, %s,%s)
                """,
                (
                    p.pipeline_id, p.name, p.invoked_by, p.mode, p.status.value,
                    json.dumps([s.model_dump() for s in p.plan], ensure_ascii=False),
                    p.current_step_index, p.max_steps,
                    json.dumps(p.context, ensure_ascii=False),
                    p.created_at, p.updated_at,
                    p.user_id, p.tenant_id,
                    p.parent_pipeline_id, p.depth,
                ),
            )


async def pipeline_get(pipeline_id: str) -> Pipeline | None:
    async with conexion() as conn:
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
) -> None:
    now = time.time()
    async with conexion() as conn:
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


async def pipelines_by_status(statuses: list[PipelineStatus]) -> list[Pipeline]:
    """Usado por jacobs/reaper.py -- lista pipelines en los status dados
    para evaluar edad/estancamiento. No filtra por antigüedad acá, eso
    es criterio del reaper."""
    if not statuses:
        return []
    async with conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            placeholders = ",".join(["%s"] * len(statuses))
            await cur.execute(
                f"SELECT * FROM jacobs_pipelines WHERE status IN ({placeholders})",
                tuple(s.value for s in statuses),
            )
            rows = await cur.fetchall()
    return [_row_to_pipeline(row) for row in rows]


async def pipeline_count_active() -> int:
    async with conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM jacobs_pipelines WHERE status IN ('pending','running')"
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0


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

async def step_upsert(s: Step) -> None:
    async with conexion() as conn:
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
    async with conexion() as conn:
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
    después y NO está medido."""
    async with conexion() as conn:
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
#  Audit events
# ----------------------------------------------------------------

async def event_append(
    pipeline_id: str,
    event_type: str,
    payload: dict | None = None,
    step_id: str | None = None,
) -> None:
    async with conexion() as conn:
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
    async with conexion() as conn:
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
