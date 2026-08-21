# jax/las_manos/motor_registry/usage_writer.py
"""Escritura directa a axioma_usage (jax-platform) desde motor_registry.
Mismo patron de conexion que credential_resolver.py/jacobs/store.py: cada
repo se conecta a la misma DB jax_memory con su propio conector minimo,
sin paquete compartido (repos/venvs independientes, mismo trade-off
documentado desde Fase 1).
"""
from __future__ import annotations

import asyncio
import logging
import os

import aiomysql

logger = logging.getLogger("motor_registry.usage_writer")

# T1.d (2026-08-22, auditoria usage_writer): 2 intentos totales (1 reintento),
# backoff corto fijo -- esto es best-effort de contabilidad, no un job real;
# no vale la pena un backoff exponencial ni más intentos. Si el 2do intento
# también falla, la DB está genuinamente inalcanzable y más intentos no
# ayudan -- escalar a logger.error y seguir es la respuesta correcta (T3,
# el chequeo de reconciliación, es la red que atrapa esto después).
_WRITE_MAX_ATTEMPTS = 2
_WRITE_RETRY_DELAY_SECONDS = 0.5


def _db_cfg() -> dict:
    return {
        "host": os.getenv("JAX_DB_HOST", "localhost"),
        "port": int(os.getenv("JAX_DB_PORT", "3306")),
        "user": os.getenv("JAX_DB_USER", ""),
        "password": os.getenv("JAX_DB_PASSWORD", ""),
        "db": os.getenv("JAX_DB_NAME", "jax_memory"),
        "charset": "utf8mb4",
        "autocommit": True,
    }


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
) -> None:
    """Best-effort: nunca debe romper el flujo del worker (mismo criterio
    que record_usage en jax-platform), pero best-effort no es lo mismo que
    silencioso.

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
    last_exc: Exception | None = None
    for attempt in range(1, _WRITE_MAX_ATTEMPTS + 1):
        try:
            conn = await aiomysql.connect(**_db_cfg())
            try:
                price_in, price_out = await _lookup_model_price(conn, provider_id, model)
                cost = None
                if price_in is not None and price_out is not None:
                    cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO axioma_usage "
                        "(tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type, status, job_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, 'motor', %s, %s)",
                        (
                            int(tenant_id) if tenant_id is not None else None,
                            int(user_id) if user_id is not None else None,
                            facet, model, tokens_in, tokens_out, cost, status, job_id,
                        ),
                    )
            finally:
                conn.close()
            return
        except Exception as e:  # fail-soft: contabilidad no debe tumbar un job ya terminado
            last_exc = e
            if attempt < _WRITE_MAX_ATTEMPTS:
                await asyncio.sleep(_WRITE_RETRY_DELAY_SECONDS)
    # T1.d: agotados los reintentos -- error (no warning), máxima visibilidad
    # desde este módulo. No escribe a jacobs_events: motor_registry no tiene
    # pipeline_id en este scope (LAS MANOS no conoce el pipeline de Jacobs
    # que lo llamó, es una frontera de arquitectura real, no un descuido) y
    # jacobs_events.pipeline_id es NOT NULL -- forzar un valor inventado ahí
    # sería peor que no escribir. T3 (chequeo de reconciliación) es la red
    # que atrapa esto después, comparando motor_jobs.jsonl contra esta tabla.
    logger.error(
        f"record_motor_usage AGOTÓ {_WRITE_MAX_ATTEMPTS} intentos, job={job_id} "
        f"facet={facet} tokens_in={tokens_in} tokens_out={tokens_out} status={status} "
        f"-- fila NO escrita, reason={type(last_exc).__name__}: {last_exc}"
    )
