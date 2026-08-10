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
    """Fail-soft en dos sentidos, mismo criterio que
    motor_registry/usage_writer.py::record_motor_usage: sin user_id/tenant_id
    no escribe nada (pipeline disparado sin identidad real -- no deberia
    pasar tras Task 3/6, pero no es este modulo quien lo garantiza), y
    cualquier error de DB se loguea sin romper el step ya completado (usage
    tracking best-effort).

    tenant_id/user_id: la columna es INT(11) y la DB corre con
    STRICT_TRANS_TABLES -- el cast a int() se hace SOLO aca, en el sitio del
    INSERT (nunca en la firma publica de esta funcion ni en la de los 3
    invocadores de executor.py), mismo motivo que record_usage en
    jax-platform y record_motor_usage en las_manos ya documentan."""
    if not user_id or not tenant_id:
        return
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
                    (int(tenant_id), int(user_id), facet, model, tokens_in, tokens_out, cost),
                )
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"record_direct_usage failed facet={facet} reason={type(e).__name__}: {e}")
