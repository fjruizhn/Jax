# jax/las_manos/motor_registry/usage_writer.py
"""Escritura directa a axioma_usage (jax-platform) desde motor_registry.
Mismo patron de conexion que credential_resolver.py/jacobs/store.py: cada
repo se conecta a la misma DB jax_memory con su propio conector minimo,
sin paquete compartido (repos/venvs independientes, mismo trade-off
documentado desde Fase 1).
"""
from __future__ import annotations

import logging
import os

import aiomysql

logger = logging.getLogger("motor_registry.usage_writer")


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
) -> None:
    """Fail-soft en dos sentidos: sin user_id/tenant_id no escribe nada (job
    disparado sin identidad real todavia -- ver Task 6), y cualquier error
    de DB se loguea sin romper el flujo del worker (usage tracking best-effort,
    mismo criterio que record_usage en jax-platform).

    tenant_id/user_id (2026-08-10, verificado contra el schema real): la
    columna es INT(11) y la DB corre con STRICT_TRANS_TABLES -- un valor no
    numerico rompe el INSERT. jax-platform/backend/api/admin/usage.py::record_usage
    ya castea con int(...) por el mismo motivo; se espeja aca. En produccion
    ambos valores son siempre el id real como string (ver jacobs/models.py
    Pipeline.user_id/tenant_id), asi que el cast es un no-op salvo en datos
    corruptos -- en cuyo caso preferimos loguear y no escribir, no reventar
    el job."""
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
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, 'motor')",
                    (int(tenant_id), int(user_id), facet, model, tokens_in, tokens_out, cost),
                )
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"record_motor_usage failed facet={facet} reason={type(e).__name__}: {e}")
