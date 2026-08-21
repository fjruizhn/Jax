"""
Resolver de facetas — Bloque C (facet/facet_binding como fuente unica
faceta->modelo). Archivo unico real dentro de jax (Bloque 2, 2026-08-21):
las_manos/facet_resolver.py es symlink a este -- jacobs lo importa via
PYTHONPATH apuntando a las_manos, el REPL via el paquete jax.core. jax-platform
sigue con copia propia aparte (repo distinto, con su propio
credential_resolver.py local -- un symlink cruzado de repos no sobrevive un
clone fresco); ver scripts/check_facet_resolver_sync.py para detectar drift
entre ambas. Consume resolve_credential_instrumented, no reimplementa Fase 1.
Ver jax-platform/docs/fase2-facetas-diseno.md.
"""
import logging
import os
import time
from dataclasses import dataclass

import aiomysql

try:
    # las_manos y jacobs (PYTHONPATH incluye las_manos/, sin la raiz del
    # paquete jax): credential_resolver.py vive directo en las_manos/.
    from credential_resolver import resolve_credential_instrumented, CredentialUnavailableError
except ImportError:
    # REPL (PYTHONPATH=. desde la raiz del repo, `python -m jax.core.main`):
    # las_manos/ no esta en sys.path, solo el paquete jax.core.
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
    """D1.1 paso 4 (fase2-facetas-diseno.md:233-237), completado 2026-08-19:
    lee el modelo via model_ref -> model.model_id, no b.model_id (texto
    libre, quedaba desincronizado de las aprobaciones de
    model_binding_proposal). facet_binding.model_id se conserva de
    solo-lectura un ciclo de cutover antes de dropearla, no se usa mas aca."""
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT f.transport, f.persona, p.base_url, b.provider_id, m.model_id, b.params "
                "FROM facet f "
                "JOIN facet_binding b ON b.facet_key = f.`key` AND b.role = 'primary' "
                "JOIN provider p ON p.id = b.provider_id "
                "JOIN model m ON m.id = b.model_ref "
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


async def load_facet_registry() -> dict:
    """Una sola query, para bootstrap SINCRONO-de-arranque (REPL): quien
    existe, con que modelo, label e icono. A diferencia de resolve_facet()
    (por-request, con cache/TTL), esto se llama UNA vez al arrancar un
    proceso de sesion unica — mismo patron que build_muscles() ya usaba
    para leer config.toml una vez, solo que ahora la fuente es la DB.
    Devuelve {key: {model, provider_id, display_name, icon, auto_selectable}}."""
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT f.`key`, f.display_name, f.icon, f.auto_selectable, "
                "b.provider_id, m.model_id "
                "FROM facet f JOIN facet_binding b ON b.facet_key = f.`key` AND b.role = 'primary' "
                "JOIN model m ON m.id = b.model_ref "
                "WHERE f.status = 'active'"
            )
            rows = await cur.fetchall()
    finally:
        conn.close()
    return {
        key: {
            "display_name": display_name, "icon": icon,
            "auto_selectable": bool(auto_sel),
            "provider_id": provider_id, "model": model_id,
        }
        for key, display_name, icon, auto_sel, provider_id, model_id in rows
    }


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
