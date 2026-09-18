# jax/las_manos/motor_registry/usage_writer.py
"""Escritura directa a axioma_usage (jax-platform) desde motor_registry.
CONEXION (2026-09-17): del pool compartido de Jacobs (`jacobs.store.conexion`),
no una conexion suelta por intento: LAS MANOS ya importa jacobs.store (worker,
tool_authority) y el pool es por proceso. Una falla del pool sube como
excepcion y toma el MISMO camino de reintento y cola durable que antes tomaba
un `aiomysql.connect` caido.

COLA DURABLE (Task 7, 2026-09-15): agotados los reintentos en línea, la fila ya
no se pierde -- se deposita en el respaldo de `cola_uso` con
`origen='motor_registry'`. **Este proceso NO DRENA**: sólo `jax-platform` lee el
respaldo e inserta, porque es la dueña de `axioma_usage` y la única con
migraciones (la columna `spool_id` y su UNIQUE, que es lo que hace idempotente
al reintento). Si este módulo también insertara, dos procesos borrarían el mismo
archivo sin coordinación y el cobro se duplicaría en la ventana entre el INSERT
y el borrado.
"""
from __future__ import annotations

import asyncio
import logging

from jacobs import store as jacobs_store

try:
    # Mismo doble import que db_connect_config, por la misma razón:
    # las_manos/cola_uso.py es un symlink a jax/core/cola_uso.py, y con
    # cwd=las_manos el paquete `jax.core` no es importable.
    from cola_uso import _ahora_iso, encolar as encolar_uso
except ImportError:
    from jax.core.cola_uso import _ahora_iso, encolar as encolar_uso

logger = logging.getLogger("motor_registry.usage_writer")

# T1.d (2026-08-22, auditoria usage_writer): 2 intentos totales (1 reintento),
# backoff corto fijo -- esto es best-effort de contabilidad, no un job real;
# no vale la pena un backoff exponencial ni más intentos. Si el 2do intento
# también falla, la DB está genuinamente inalcanzable y más intentos no
# ayudan -- escalar a logger.error y seguir es la respuesta correcta (T3,
# el chequeo de reconciliación, es la red que atrapa esto después).
_WRITE_MAX_ATTEMPTS = 2
_WRITE_RETRY_DELAY_SECONDS = 0.5


def _entero_o_none(valor) -> int | None:
    """tenant_id/user_id son INT(11) en la tabla y llegan como string. El cast
    NO puede propagar: esto corre en el camino de recuperación de una fila que
    ya se perdió una vez. Un valor no numérico entra como NULL -- exactamente
    lo mismo que hace hoy la columna cuando el INSERT lo rechaza, pero con la
    fila guardada en vez de tirada."""
    if valor is None:
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        logger.warning("tenant_id/user_id no numérico (%r) -- se encola como NULL", valor)
        return None


async def _lookup_model_price(conn, provider_id: str, model: str) -> tuple[float | None, float | None]:
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT price_input_per_1m_usd, price_output_per_1m_usd "
            "FROM model WHERE provider_id=%s AND model_id=%s",
            (provider_id, model),
        )
        row = await cur.fetchone()
    if not row:
        return None, None
    return row[0], row[1]


async def record_motor_usage(
    user_id: str | None,
    tenant_id: str | None,
    facet: str,
    provider_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    *,
    job_id: str | None = None,
    status: str = "unknown",
    pipeline_id: str | None = None,
) -> None:
    """Best-effort: nunca debe romper el flujo del worker (mismo criterio
    que record_usage en jax-platform), pero best-effort no es lo mismo que
    silencioso.

    pipeline_id (Task 7b, 2026-09-18): el Pipeline.pipeline_id de Jacobs que
    dispara este job, si vino de uno -- viaja desde
    MotorDispatchRequest.pipeline_id (routes.py) via worker.py:run(). None
    es válido y correcto cuando el caller no es un pipeline (mismo criterio
    que user_id/tenant_id en None: no se inventa, se declara ausente).
    Sin esta columna, api/pipelines.py::list_pipelines() no tiene con qué
    sumar el costo real de un pipeline -- verificado que ES el camino que
    ejecuta la mayoría de los pasos reales (kimi/jax_local, Ruling 7 de
    Task 1 en el ledger de esta ronda).

    T1.c (2026-08-22, auditoria usage_writer): antes, sin user_id/tenant_id
    esto retornaba SIN loguear nada -- un dispatch sin identidad (pruebas
    manuales, jobs internos) sigue gastando dinero real contra una API paga;
    descartar la fila en silencio lo hacía invisible. Ahora escribe con
    tenant_id/user_id NULL (la columna lo permite, sin NOT NULL -- ver
    migrations.py CREATE_AXIOMA_USAGE) y loguea WARNING -- el costo queda
    contabilizado y filtrable (`WHERE tenant_id IS NULL` = gasto sin dueño
    real), en vez de perdido.

    tenant_id/user_id (2026-08-10, verificado contra el schema real): la
    columna es INT(11) -- un valor no numerico rompe el cast. Se castea acá
    SOLO si no es None (None se inserta como NULL real, no dispara el
    DEFAULT 1 de la columna: DEFAULT solo aplica si la columna se OMITE del
    INSERT, no si se pasa NULL explícito -- confirmado contra el DDL).

    status/job_id (T1.b): status distingue el desenlace del job (completed/
    failed/killed/etc, lo que worker.py lea del estado final real vía
    store.get(), no un valor inventado acá) -- un token gastado en un fallo
    es tan real como uno en un éxito. job_id permite reconciliar esta fila
    contra motor_jobs.jsonl por igualdad exacta, no por timestamp aproximado
    (T3, el chequeo de reconciliación)."""
    if not user_id or not tenant_id:
        logger.warning(
            f"record_motor_usage job={job_id} facet={facet} sin identidad "
            f"(user_id={user_id!r} tenant_id={tenant_id!r}) -- escribe con "
            f"tenant_id/user_id NULL, no se descarta"
        )
    # La hora del TURNO, leída ANTES del primer intento: con los reintentos y
    # una caída larga de por medio, tomarla al encolar movería el costo de día.
    creado_en = _ahora_iso()
    cost = None
    last_exc: Exception | None = None
    for attempt in range(1, _WRITE_MAX_ATTEMPTS + 1):
        try:
            # Pool compartido (2026-09-17): el guard de JAX_DB_HOST/PORT, el
            # connect_timeout y el limite de espera por hueco viven en store.
            async with jacobs_store.conexion() as conn:
                price_in, price_out = await _lookup_model_price(conn, provider_id, model)
                if price_in is not None and price_out is not None:
                    cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO axioma_usage "
                        "(tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type, status, job_id, pipeline_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, 'motor', %s, %s, %s)",
                        (
                            int(tenant_id) if tenant_id is not None else None,
                            int(user_id) if user_id is not None else None,
                            facet, model, tokens_in, tokens_out, cost, status, job_id, pipeline_id,
                        ),
                    )
            return
        except Exception as e:  # fail-soft: contabilidad no debe tumbar un job ya terminado
            last_exc = e
            if attempt < _WRITE_MAX_ATTEMPTS:
                await asyncio.sleep(_WRITE_RETRY_DELAY_SECONDS)
    # T7 (2026-09-15): agotados los reintentos, la fila va al respaldo en vez
    # de perderse. Los reintentos en línea se mantienen -- son baratos y
    # resuelven el caso transitorio sin tocar el disco; la cola es para cuando
    # la base está genuinamente caída, que es el caso que T1.d no cubría.
    #
    # `status` y `job_id` VIAJAN en el archivo (Task 8, 2026-09-15: el contrato
    # compartido pasó a TRECE campos). Sin ellos, la fila recuperada entraba a
    # `axioma_usage` con esas dos columnas en NULL y la reconciliación contra
    # motor_jobs.jsonl por igualdad exacta (T3) no la podía emparejar: se
    # recuperaba el cobro y se perdía la trazabilidad.
    #
    # cost_usd va None cuando la caída fue antes del lookup de precios: sin base
    # no hay tabla `model` que consultar. La plataforma lo resuelve al insertar.
    spool_id = await encolar_uso({
        "created_at": creado_en,
        "tenant_id": _entero_o_none(tenant_id),
        "user_id": _entero_o_none(user_id),
        "facet": facet,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
        "request_type": "motor",
        "origen": "motor_registry",
        "status": status,
        "job_id": job_id,
        "pipeline_id": pipeline_id,
    })
    if spool_id:
        # INFO, no ERROR: encolada NO es pérdida. Un ERROR acá entrena a
        # ignorar el log, y entonces el ERROR de abajo -- que sí es una pérdida
        # real -- deja de distinguirse.
        logger.info(
            f"record_motor_usage AGOTÓ {_WRITE_MAX_ATTEMPTS} intentos, job={job_id} "
            f"facet={facet} status={status} tokens_in={tokens_in} "
            f"tokens_out={tokens_out} -- fila ENCOLADA en el respaldo "
            f"spool_id={spool_id} (la inserta jax-platform), "
            f"reason={type(last_exc).__name__}: {last_exc}"
        )
        return
    # T1.d: agotados los reintentos Y el respaldo -- error (no warning), máxima
    # visibilidad desde este módulo. No escribe a jacobs_events: motor_registry
    # no tiene pipeline_id en este scope (LAS MANOS no conoce el pipeline de
    # Jacobs que lo llamó, es una frontera de arquitectura real, no un
    # descuido) y jacobs_events.pipeline_id es NOT NULL -- forzar un valor
    # inventado ahí sería peor que no escribir. T3 (chequeo de reconciliación)
    # es la red que atrapa esto después, comparando motor_jobs.jsonl contra
    # esta tabla.
    logger.error(
        f"record_motor_usage AGOTÓ {_WRITE_MAX_ATTEMPTS} intentos, job={job_id} "
        f"facet={facet} tokens_in={tokens_in} tokens_out={tokens_out} status={status} "
        f"-- TAMPOCO se pudo encolar: fila PERDIDA, "
        f"reason={type(last_exc).__name__}: {last_exc}"
    )
