"""
Resolver de facetas — Bloque C (facet/facet_binding como fuente unica
faceta->modelo). Espejo minimo en jax-platform, jax/core, las_manos, mismo
patron que credential_resolver.py. Consume resolve_credential_instrumented,
no reimplementa Fase 1. Ver jax-platform/docs/fase2-facetas-diseno.md.
"""
import logging
import os
import time
from dataclasses import dataclass

import aiomysql

from jax.core.credential_resolver import resolve_credential_instrumented, CredentialUnavailableError

logger = logging.getLogger("facet_resolver")

FACET_CACHE_TTL_SECONDS = int(os.getenv("FACET_CACHE_TTL_SECONDS", "30"))
FACET_STALE_MAX_SECONDS = int(os.getenv("FACET_STALE_MAX_SECONDS", "300"))


class FacetUnavailableError(Exception):
    """FAIL-CLOSED: sin facet activa con binding role='primary'. El llamador
    declara estado degradado — nunca cae a un default hardcodeado."""


@dataclass
class ResolvedFacet:
    key: str
    provider_id: str
    base_url: str | None
    model: str
    credential: str
    transport: str
    persona: str | None
    params: dict | None


class _CacheEntry:
    __slots__ = ("value", "fetched_at")

    def __init__(self, value: ResolvedFacet, fetched_at: float):
        self.value = value
        self.fetched_at = fetched_at


_cache: dict[str, _CacheEntry] = {}


async def _db_conn() -> aiomysql.Connection:
    return await aiomysql.connect(
        host=os.getenv("JAX_DB_HOST", "localhost"),
        port=int(os.getenv("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4",
        autocommit=True,
    )


async def _query_facet(facet_key: str) -> ResolvedFacet:
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT f.transport, f.persona, p.base_url, b.provider_id, b.model_id, b.params "
                "FROM facet f "
                "JOIN facet_binding b ON b.facet_key = f.`key` AND b.role = 'primary' "
                "JOIN provider p ON p.id = b.provider_id "
                "WHERE f.`key` = %s AND f.status = 'active'",
                (facet_key,),
            )
            row = await cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise FacetUnavailableError(f"sin binding activo para facet '{facet_key}'")
    transport, persona, base_url, provider_id, model_id, params = row

    credential = ""
    if transport not in ("ollama", "subprocess"):  # ollama/subprocess no usan credencial de proveedor gestionada aqui
        try:
            credential = await resolve_credential_instrumented(provider_id)
        except CredentialUnavailableError as e:
            raise FacetUnavailableError(f"facet '{facet_key}': {e}") from e

    return ResolvedFacet(
        key=facet_key, provider_id=provider_id, base_url=base_url, model=model_id,
        credential=credential, transport=transport, persona=persona, params=params,
    )


async def resolve_facet(facet_key: str) -> ResolvedFacet:
    now = time.monotonic()
    cached = _cache.get(facet_key)
    if cached and (now - cached.fetched_at) < FACET_CACHE_TTL_SECONDS:
        return cached.value
    try:
        value = await _query_facet(facet_key)
        _cache[facet_key] = _CacheEntry(value, now)
        return value
    except Exception as e:
        if cached and (now - cached.fetched_at) < FACET_STALE_MAX_SECONDS:
            logger.warning(f"facet_resolver key={facet_key} db_unreachable=1 serving_stale reason={type(e).__name__}")
            return cached.value
        logger.error(f"facet_resolver key={facet_key} FAIL_CLOSED reason={type(e).__name__}")
        raise FacetUnavailableError(facet_key) from e
