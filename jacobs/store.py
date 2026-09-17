"""
Jacobs — Almacén de pipelines en MariaDB.

Base: jax_memory. Tablas: jacobs_pipelines, jacobs_steps.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import aiomysql

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


async def get_conn() -> aiomysql.Connection:
    # connect_timeout explícito (no en _db_cfg()): hallazgo de revisión,
    # Tarea 2b (tanda A, ronda de arreglo 1, 2026-09-14) -- sin esto,
    # aiomysql espera sin límite si la DB se cuelga.
    return await aiomysql.connect(**_db_cfg(), connect_timeout=db_connect_timeout_seconds())


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


async def pipeline_count_active() -> int:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM jacobs_pipelines WHERE status IN ('pending','running')"
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        conn.close()


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
       "motors": {motor_key: has_tool_access (bool)},
       "facets": frozenset de facet.key con status='active'}

    Costo medido en vivo (2026-08-21, DB real) con 3 SELECTs: 0.00024s de
    ejecución total en el servidor (motor: 4 filas, capability: ~17,
    capability_motor: ~26) -- insignificante para llamar en cada dispatch,
    no solo en plan-build. El 4º SELECT (facet, 7 filas, E-17) se agregó
    después y NO está medido."""
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

            # E-03 (2026-09-16): el vocabulario de facetas del planner es la
            # tabla `facet`, no una lista fija. Mismas reglas que resolve_facet
            # (status='active'): lo que el planner acepta es lo que se puede
            # despachar. Catálogo de 7 filas: sin índice, declarado en DEUDA.md.
            await cur.execute("SELECT `key` FROM facet WHERE status = 'active'")
            facets = frozenset(key for (key,) in await cur.fetchall())
    finally:
        conn.close()
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
    conn = await get_conn()
    try:
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
    finally:
        conn.close()


async def subpipeline_emision_diagnostico(parent_pipeline_id: str, parent_step: str) -> dict:
    conn = await get_conn()
    try:
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
    finally:
        conn.close()
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

SQL_TOKEN_CONSUMIDO = """
    SELECT parent_pipeline_id, parent_step, depth_hijo
      FROM jacobs_subpipeline_tokens
     WHERE token_hash = %s AND hijo_pipeline_id = %s
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
    (parent_pipeline_id, parent_step, depth_hijo) o None si no afectó una fila."""
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                SQL_CONSUMIR_TOKEN,
                (ahora, hijo_pipeline_id, token_hash, ahora, parent_pipeline_id, max_profundidad),
            )
            if cur.rowcount != 1:
                return None
            await cur.execute(SQL_TOKEN_CONSUMIDO, (token_hash, hijo_pipeline_id))
            return await cur.fetchone()
    finally:
        conn.close()


async def subpipeline_token_diagnostico(token_hash: str) -> dict | None:
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(SQL_DIAGNOSTICO_TOKEN, (token_hash,))
            return await cur.fetchone()
    finally:
        conn.close()


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
