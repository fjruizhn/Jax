"""
Jacobs — Almacén de pipelines en MariaDB.

Base: jax_memory. Tablas: jacobs_pipelines, jacobs_steps.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import aiomysql
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


async def get_conn(found_rows: bool = False) -> aiomysql.Connection:
    # connect_timeout explícito (no en _db_cfg()): hallazgo de revisión,
    # Tarea 2b (tanda A, ronda de arreglo 1, 2026-09-14) -- sin esto,
    # aiomysql espera sin límite si la DB se cuelga.
    #
    # found_rows (2026-09-17, época de corrida): por defecto MariaDB devuelve
    # las filas CAMBIADAS de un UPDATE, no las que cumplen el WHERE. Un UPDATE
    # condicional que escribe los mismos valores devolvería 0 y el ejecutor
    # creería haber perdido la época. Con CLIENT.FOUND_ROWS el conteo es de
    # filas encontradas.
    extra = {"client_flag": CLIENT.FOUND_ROWS} if found_rows else {}
    return await aiomysql.connect(**_db_cfg(), connect_timeout=db_connect_timeout_seconds(), **extra)


# --- Pool de lectura del pre-vuelo (Task 15b, 2026-09-17, LAS CUATRO #2) -----
# Medido (task-15b-report.md): cada pre-vuelo abría 2 conexiones nuevas
# (MotorCatalog.from_db y prevuelo_catalogo.leer_catalogo), ~0,15 ms de CPU
# del event loop cada una más el handshake en la base. Ahora el pre-vuelo
# toma UNA conexión de este pool y la pasa a los dos lectores.
#
# - Un pool por proceso y por event loop: se crea perezosamente en el primer
#   pedido y queda atado al loop que lo creó (aiomysql.Pool guarda el loop).
#   Usarlo desde otro loop se niega con RuntimeError -- fallaría más tarde con
#   un error críptico. LAS MANOS lo cierra en su shutdown y el CLI
#   (tools/jacobs_relaunch.py) antes de salir.
# - minsize=0: crear el pool no toca la base; la primera conexión se abre al
#   pedirla, así una base caída falla en el pedido (503), no en un pool roto
#   guardado.
# - SIN CLIENT.FOUND_ROWS: es de lectura. Las escrituras condicionales por
#   época (pipeline_update_status_si_epoca, step_upsert_si_epoca) siguen con
#   get_conn(found_rows=True): meterlas en este pool cambiaría en silencio el
#   conteo de filas de los UPDATE que no tienen el flag, y ningún camino
#   medido las pone en la ruta caliente.
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
_pool_de_lectura_estado: tuple[asyncio.AbstractEventLoop, aiomysql.Pool, asyncio.Semaphore] | None = None
# El candado sólo serializa la CREACIÓN (dos primeros pedidos a la vez no
# crean dos pools; hoy create_pool con minsize=0 no cede el loop antes de
# volver, pero con minsize>0 conectaría al crear y la carrera sería real --
# test_dos_primeros_pedidos_a_la_vez_crean_un_solo_pool la fuerza con una pausa). Es de un loop; si todavía no hay pool, uno de otro loop
# (el anterior ya cerrado, p. ej. tras un intento fallido) se reemplaza.
_candado_de_creacion: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None


def _pool_del_loop(
    loop: asyncio.AbstractEventLoop,
) -> tuple[aiomysql.Pool, asyncio.Semaphore] | None:
    if _pool_de_lectura_estado is None:
        return None
    duenio, pool, turno = _pool_de_lectura_estado
    if duenio is not loop:
        raise RuntimeError(
            "el pool de lectura de Jacobs se creó en otro event loop: hay que "
            "cerrarlo con store.cerrar_pool() en el loop que lo creó antes de "
            "pedir conexiones desde este"
        )
    return pool, turno


async def _pool_de_lectura() -> tuple[aiomysql.Pool, asyncio.Semaphore]:
    global _pool_de_lectura_estado, _candado_de_creacion
    loop = asyncio.get_running_loop()
    actual = _pool_del_loop(loop)
    if actual is not None:
        return actual
    if _candado_de_creacion is None or _candado_de_creacion[0] is not loop:
        _candado_de_creacion = (loop, asyncio.Lock())
    async with _candado_de_creacion[1]:
        actual = _pool_del_loop(loop)
        if actual is None:
            from jacobs.prevuelo_config import db_pool_max

            tamanio = db_pool_max()
            pool = await aiomysql.create_pool(
                minsize=0, maxsize=tamanio,
                **_db_cfg(), connect_timeout=db_connect_timeout_seconds(),
            )
            actual = (pool, asyncio.Semaphore(tamanio))
            _pool_de_lectura_estado = (loop, *actual)
    return actual


@asynccontextmanager
async def conexion_de_lectura() -> AsyncIterator[aiomysql.Connection]:
    """Una conexión del pool de lectura, devuelta al salir. Cualquier error de
    la base (al conectar, al esperar turno o a mitad de consulta) se propaga:
    quien llama responde 503, nunca un veredicto por defecto."""
    pool, turno = await _pool_de_lectura()
    async with asyncio.timeout(db_connect_timeout_seconds()):
        await turno.acquire()
        try:
            conn = await pool.acquire()
        except BaseException:
            turno.release()
            raise
    try:
        yield conn
    except BaseException:
        conn.close()  # puede tener filas sin leer o el socket roto: no vuelve al pool
        raise
    finally:
        try:
            await pool.release(conn)
        finally:
            turno.release()


async def cerrar_pool() -> None:
    """Cierra el pool de lectura (shutdown de LAS MANOS, salida del CLI).
    Sin pool, no hace nada. Después se puede volver a pedir: se crea otro."""
    global _pool_de_lectura_estado
    actual = _pool_del_loop(asyncio.get_running_loop())
    if actual is None:
        return
    pool, _ = actual
    _pool_de_lectura_estado = None
    pool.close()
    await pool.wait_closed()


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
    conn = await get_conn()
    try:
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
    finally:
        conn.close()


# ----------------------------------------------------------------
#  Pipeline CRUD
# ----------------------------------------------------------------

async def pipeline_create(p: Pipeline) -> None:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO jacobs_pipelines
                    (pipeline_id, name, invoked_by, mode, status,
                     plan, current_step_index, max_steps, context_refs,
                     created_at, updated_at, user_id, tenant_id, run_epoch)
                VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s, %s)
                """,
                (
                    p.pipeline_id, p.name, p.invoked_by, p.mode, p.status.value,
                    json.dumps([s.model_dump() for s in p.plan], ensure_ascii=False),
                    p.current_step_index, p.max_steps,
                    json.dumps(p.context, ensure_ascii=False),
                    p.created_at, p.updated_at,
                    p.user_id, p.tenant_id, p.run_epoch,
                ),
            )
    finally:
        conn.close()


async def pipeline_get(pipeline_id: str) -> Pipeline | None:
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM jacobs_pipelines WHERE pipeline_id=%s", (pipeline_id,)
            )
            row = await cur.fetchone()
    finally:
        conn.close()
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
    conn = await get_conn()
    try:
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
    finally:
        conn.close()


async def pipelines_by_status(statuses: list[PipelineStatus]) -> list[Pipeline]:
    """Usado por jacobs/reaper.py -- lista pipelines en los status dados
    para evaluar edad/estancamiento. No filtra por antigüedad acá, eso
    es criterio del reaper."""
    if not statuses:
        return []
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            placeholders = ",".join(["%s"] * len(statuses))
            await cur.execute(
                f"SELECT * FROM jacobs_pipelines WHERE status IN ({placeholders})",
                tuple(s.value for s in statuses),
            )
            rows = await cur.fetchall()
    finally:
        conn.close()
    return [_row_to_pipeline(row) for row in rows]


_SQL_CONTAR_ACTIVOS = "SELECT COUNT(*) FROM jacobs_pipelines WHERE status IN ('pending','running')"


async def pipeline_count_active(conexion: aiomysql.Connection | None = None) -> int:
    """Pipelines activos. Con `conexion` (la del candado de activos, F3) lee por
    ella y no la cierra; sin ella abre y cierra una propia."""
    conn = conexion if conexion is not None else await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_SQL_CONTAR_ACTIVOS)
            row = await cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        if conexion is None:
            conn.close()


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
        conn = await get_conn()
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
    conn = await get_conn(found_rows=True)
    try:
        async with conn.cursor() as cur:
            return await cur.execute(sql, params)
    finally:
        conn.close()


async def pipeline_epoca_y_status(pipeline_id: str) -> tuple[int, PipelineStatus] | None:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_SQL_EPOCA_Y_STATUS, (pipeline_id,))
            fila = await cur.fetchone()
    finally:
        conn.close()
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
    evento."""
    conn = await get_conn(found_rows=True)
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
            await conn.commit()
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
    conn = await get_conn()
    try:
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
    finally:
        conn.close()


async def steps_by_pipeline(pipeline_id: str) -> list[Step]:
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM jacobs_steps WHERE pipeline_id=%s ORDER BY step_index",
                (pipeline_id,),
            )
            rows = await cur.fetchall()
    finally:
        conn.close()
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
       "motors": {motor_key: has_tool_access (bool)}}

    Costo medido en vivo (2026-08-21, DB real): 3 SELECTs, 0.00024s de
    ejecución total en el servidor (motor: 4 filas, capability: ~17,
    capability_motor: ~26) -- insignificante para llamar en cada dispatch,
    no solo en plan-build."""
    conn = await get_conn()
    try:
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
    finally:
        conn.close()
    return {"capabilities": capabilities, "motors": motors}


# ----------------------------------------------------------------
#  Audit events
# ----------------------------------------------------------------

async def event_append(
    pipeline_id: str,
    event_type: str,
    payload: dict | None = None,
    step_id: str | None = None,
) -> None:
    conn = await get_conn()
    try:
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
    finally:
        conn.close()


async def events_by_pipeline(pipeline_id: str) -> list[dict[str, Any]]:
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM jacobs_events WHERE pipeline_id=%s ORDER BY id",
                (pipeline_id,),
            )
            rows = await cur.fetchall()
    finally:
        conn.close()
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
