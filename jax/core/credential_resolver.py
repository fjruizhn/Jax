"""
Resolver de credenciales de proveedor — Fase 1 (DB como fuente de verdad).

Espejo minimo en los 3 codebases (jax-platform, jax/core, las_manos), mismo
patron que crypto_secrets.py: repos/venvs independientes, no justifica un
paquete compartido en esta fase.

Diseño completo: jax-platform/docs/fase1-credenciales-diseno.md (B1.2/B1.4).
Resuelve R3 de la auditoria (rotar una key no la propagaba sin restart).
"""
import logging
import os
import time

import aiomysql

from jax.core.crypto_secrets import decrypt_secret

logger = logging.getLogger("credential_resolver")

# El TTL ES el SLA de revocacion: una key revocada sigue viva hasta este
# tiempo en el peor caso normal (DB sana). CREDENTIAL_STALE_MAX_SECONDS es
# el techo de tolerancia si la DB cae — pasado ese techo, fail-closed
# explicito, nunca una llamada silenciosa con credencial vieja.
CREDENTIAL_CACHE_TTL_SECONDS = int(os.getenv("CREDENTIAL_CACHE_TTL_SECONDS", "30"))
CREDENTIAL_STALE_MAX_SECONDS = int(os.getenv("CREDENTIAL_STALE_MAX_SECONDS", "300"))

# Mapeo provider_id -> env var, usado solo por resolve_credential_instrumented
# durante la ventana de doble lectura (B1.4). Se retira cuando el criterio de
# salida (7 dias sin source=env_fallback) se cumpla.
_PROVIDER_ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "moonshot": "KIMI_API_KEY",
    "zhipu": "ZAI_API_KEY",
}


class CredentialUnavailableError(Exception):
    """FAIL-CLOSED: no hay credencial valida (ni fresca ni stale dentro del
    techo). El llamador debe declarar estado degradado explicito — nunca
    caer a un default silencioso."""


class _CacheEntry:
    __slots__ = ("value", "fetched_at")

    def __init__(self, value: str, fetched_at: float):
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


async def _query_active_credential(provider_id: str) -> str:
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT encrypted_value FROM credential "
                "WHERE provider_id=%s AND state='active' "
                "ORDER BY activated_at DESC LIMIT 1",
                (provider_id,),
            )
            row = await cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise CredentialUnavailableError(f"no active credential for {provider_id}")
    return decrypt_secret(row[0])


async def resolve_credential(provider_id: str) -> str:
    """Resolucion por request, sin cache de vida de proceso (a). Cache TTL
    corto (b). Cache acotado con techo si la DB falla (c). FAIL-CLOSED sin
    credencial valida (d)/(e)."""
    now = time.monotonic()
    cached = _cache.get(provider_id)
    if cached and (now - cached.fetched_at) < CREDENTIAL_CACHE_TTL_SECONDS:
        return cached.value
    try:
        value = await _query_active_credential(provider_id)
        _cache[provider_id] = _CacheEntry(value, now)
        return value
    except Exception as e:
        if cached and (now - cached.fetched_at) < CREDENTIAL_STALE_MAX_SECONDS:
            logger.warning(
                f"credential_resolver provider={provider_id} db_unreachable=1 "
                f"serving_stale_age={now - cached.fetched_at:.0f}s reason={type(e).__name__}"
            )
            return cached.value
        logger.error(f"credential_resolver provider={provider_id} FAIL_CLOSED reason={type(e).__name__}")
        raise CredentialUnavailableError(provider_id) from e


async def resolve_credential_instrumented(provider_id: str) -> str:
    """Doble lectura DB->env con logging (B1.4). Criterio de salida: 7 dias
    consecutivos sin ninguna linea source=env_fallback, incluyendo al menos
    una rotacion real en la ventana. No es una solucion temporal — tiene
    condicion de salida medible y su proposito es medir, no tapar."""
    try:
        value = await resolve_credential(provider_id)
        logger.info(f"credential_resolution provider={provider_id} source=db")
        return value
    except CredentialUnavailableError:
        env_key = _PROVIDER_ENV_KEY_MAP.get(provider_id)
        env_value = decrypt_secret(os.environ.get(env_key, "")) if env_key else ""
        if not env_value:
            raise
        logger.warning(f"credential_resolution provider={provider_id} source=env_fallback")
        return env_value
