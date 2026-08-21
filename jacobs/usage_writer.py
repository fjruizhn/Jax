# jax/jacobs/usage_writer.py
"""Escritura directa a axioma_usage (jax-platform) desde jacobs/executor.py,
para los 3 transportes HTTP directos (hipatia/gemini, jekyll+thot+ada/openai
compat, jax_local/ollama) cuando se invocan via un pipeline de Jacobs -- no
via la Mesa web (esa ruta ya escribe axioma_usage desde jax-platform/backend/
api/chat.py, Tasks 1-4).

Mismo patron de conexion que credential_resolver.py/store.py/
las_manos/motor_registry/usage_writer.py: cada repo se conecta a la misma DB
jax_memory con su propio conector minimo, sin paquete compartido. jacobs NO
importa motor_registry (no esta en su sys.path standalone, ver comentario en
jacobs/executor.py sobre el catalogo de capabilities) -- por eso este modulo
es una copia adaptada, no un import cruzado.

request_type='pipeline' (no 'chat'): distingue en /api/admin/usage estos
mismos transportes invocados DESDE un pipeline de Jacobs (posiblemente sin
supervision humana en el momento, corriendo en cadena con otros steps) de la
misma faceta invocada directo desde la Mesa web -- util para filtrar/atribuir
costo mas adelante sin tener que inferirlo de otra tabla."""
from __future__ import annotations

import logging
import os

import aiomysql

logger = logging.getLogger("jacobs.usage_writer")


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


async def record_direct_usage(
    user_id: str | None,
    tenant_id: str | None,
    facet: str,
    provider_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> None:
    """Best-effort (usage tracking no debe romper un step ya completado),
    pero no silencioso.

    T1.c (2026-08-22, auditoria usage_writer): mismo bug que
    motor_registry/usage_writer.py::record_motor_usage -- sin user_id/
    tenant_id esto retornaba SIN loguear. Ahora escribe con tenant_id/
    user_id NULL (la columna lo permite) y loguea WARNING.

    T1.b (auditoria usage_writer, alcance de esta ronda): el escritor de
    Motor Registry tenía un bug real -- la llamada vivía solo en la rama de
    éxito, así que ningún job fallido contabilizaba, confirmado 7/9 jobs
    reales sin fila. Este escritor (transportes HTTP directos) reconcilió
    4/4 en la única corrida real disponible -- sin evidencia del mismo
    problema, NO se tocó el punto de llamada en executor.py::_dispatch_step
    esta ronda (tocar _invoke_http_gemini/_invoke_http_openai_compat/
    _invoke_ollama para capturar tokens parciales antes de una excepción
    sería un cambio no verificado). Deuda declarada, no una garantía.

    tenant_id/user_id: la columna es INT(11) -- el cast a int() se hace
    SOLO si no es None (None se inserta como NULL real, mismo motivo que
    record_motor_usage documenta)."""
    if not user_id or not tenant_id:
        logger.warning(
            f"record_direct_usage facet={facet} sin identidad "
            f"(user_id={user_id!r} tenant_id={tenant_id!r}) -- escribe con "
            f"tenant_id/user_id NULL, no se descarta"
        )
    try:
        conn = await aiomysql.connect(**_db_cfg())
        try:
            price_in, price_out = await _lookup_model_price(conn, provider_id, model)
            cost = None
            if price_in is not None and price_out is not None:
                cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO axioma_usage (tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, 'pipeline')",
                    (
                        int(tenant_id) if tenant_id is not None else None,
                        int(user_id) if user_id is not None else None,
                        facet, model, tokens_in, tokens_out, cost,
                    ),
                )
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"record_direct_usage failed facet={facet} reason={type(e).__name__}: {e}")
