"""
JAX 2.0 — Worker de embeddings.

Vectoriza todos los mensajes y facts cuyo embedding sigue en vector cero, en
la columna y con el modelo configurados (jax/memory/embedding_config.py).
Corre una vez (o cuando se necesite poner al dia la base).

2026-09-12: buscaba `embedding IS NULL`, que no puede darse -- la columna es
VECTOR NOT NULL con default vector cero (el VECTOR KEY lo exige) --, asi que no
encontraba nunca nada. Ahora usa el mismo predicado de vector cero que
MemoryDB.backfill_zero_embeddings, y el UPDATE esta guardado por el.

Uso:
    set -a; source <(sudo -n cat /etc/jax/.env); set +a
    cd ~/jax && .venv/bin/python -m jax.memory.embedding_worker

En memoria de Jairo Urbina.
"""

import asyncio
import argparse
import json
import logging
import os
import uuid
from typing import Awaitable, Callable

import aiomysql

from jax.memory.db import MemoryDB, _col, _zero_embedding_sql
from jax.memory.mapping_pool import MappingPool
from jax.memory.b9 import (EmbeddingSpaceIdentity, MutationAuthorizationRequest,
                           ScopeContext, ScopeDenied, Visibility)
from jax.memory.b9_mariadb import MariaDBB9Store, PersistentMemoryAPI
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
from jax.memory.embedding_config import CONFIG
from jax.memory.recall_tripwire import recall_requiere_remedicion
from jax.core.cliente_http_compartido import cerrar_cliente_http, obtener_cliente_http
from jax.core.config_entorno import url_requerida
from jax.core.db_connect_config import db_connect_timeout_seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [embedding_worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jax.memory.embedding_worker")

BATCH_SIZE = 50

#: Intentos fallidos por (revision, espacio) antes de que el worker salte la fila.
DEFAULT_MAX_ATTEMPTS = 5
ESQUEMA_B9 = "b9-v1"

#: Codigos de salida de vector-health (aviso-fallo distingue incidentes por ellos).
EXIT_GENERAL = 1          # faltan vectores o fallo general
EXIT_RECALL_REMEDIR = 3   # el tripwire pide volver a medir el recall del HNSW (el 2 es del worker: jobs atascados)


def _max_attempts() -> int:
    value = int(os.environ.get("JAX_MEMORY_EMBED_MAX_ATTEMPTS", str(DEFAULT_MAX_ATTEMPTS)))
    if value <= 0:
        raise ValueError("JAX_MEMORY_EMBED_MAX_ATTEMPTS must be positive")
    return value


async def modelo_digest(model: str) -> str:
    """Digest real del modelo de embeddings segun Ollama (`/api/tags`).

    Auditoria 2026-10-05: el espacio de embeddings se registraba con digest None, asi que
    re-descargar `bge-m3` con otros pesos mezclaba en silencio vectores de dos modelos. Si Ollama
    no responde, no trae el modelo o no trae digest, la corrida FALLA: nunca se guarda None.
    (`/api/show` no devuelve digest; `/api/tags` si.) Sin tag, Ollama entiende `:latest`.
    """
    url = url_requerida("JAX_OLLAMA_URL") + "/api/tags"
    resp = await obtener_cliente_http().get(url, timeout=10.0)
    resp.raise_for_status()
    wanted = model if ":" in model else f"{model}:latest"
    for entry in resp.json().get("models") or []:
        if (entry.get("name") or entry.get("model")) == wanted:
            digest = entry.get("digest")
            if isinstance(digest, str) and digest.strip():
                return digest.strip()
            raise RuntimeError(f"Ollama no informa el digest del modelo {wanted}")
    raise RuntimeError(f"Ollama no tiene instalado el modelo {wanted}")


async def resolver_identidad() -> EmbeddingSpaceIdentity:
    """Identidad del espacio B9 con el digest real del modelo; la usan el worker y vector-health."""
    digest = await modelo_digest(CONFIG.model)
    return EmbeddingSpaceIdentity(ESQUEMA_B9, "ollama", CONFIG.model, digest, CONFIG.dim, "unit", "cosine")


async def avisar_si_cambio_el_digest(pool: object, identity: EmbeddingSpaceIdentity) -> None:
    """Avisa (WARNING) cuando el digest del modelo cambio respecto del ultimo espacio registrado.

    El digest es parte de embedding_space_id: un digest nuevo ES un espacio nuevo, y las
    revisiones se vectorizan de nuevo en el, sin tocar los vectores del espacio anterior (no se
    mezclan). Mientras dura esa re-vectorizacion, vector-health queda en rojo: ese es el aviso
    que llega a Telegram; este WARNING deja la causa en el journal.
    """
    async with MappingPool(pool).acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 AS ok FROM embedding_spaces WHERE embedding_space_id=%s",
                              (identity.embedding_space_id,))
            if await cur.fetchone():
                return
            await cur.execute(
                "SELECT embedding_space_id,model_version_or_digest FROM embedding_spaces "
                "WHERE model_identifier=%s AND dimension=%s AND provider_runtime_class=%s AND schema_version=%s "
                "ORDER BY created_at DESC LIMIT 1",
                (identity.model_identifier, identity.dimension, identity.provider_runtime_class,
                 identity.schema_version))
            previo = await cur.fetchone()
    if previo:
        logger.warning(
            "B9 embeddings: el digest del modelo %s cambio (%s -> %s): se abre un espacio de embeddings "
            "nuevo (%s) y las revisiones se re-vectorizan en el; no se mezclan con el espacio anterior (%s)",
            identity.model_identifier, previo["model_version_or_digest"] or "UNKNOWN",
            identity.model_version_or_digest, identity.embedding_space_id, previo["embedding_space_id"])


async def _registrar_fallo(pool: object, revision_id: str, space_id: str, motivo: str, max_attempts: int) -> None:
    """Cuenta un intento fallido de (revision, espacio); al llegar al tope la fila deja de intentarse."""
    async with MappingPool(pool).acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO embedding_generation_attempts (revision_id,embedding_space_id,attempts,last_error) "
                "VALUES (%s,%s,1,%s) ON DUPLICATE KEY UPDATE attempts=attempts+1,last_error=VALUES(last_error)",
                (revision_id, space_id, motivo[:255]))
            await cur.execute("SELECT attempts FROM embedding_generation_attempts WHERE revision_id=%s AND embedding_space_id=%s",
                              (revision_id, space_id))
            attempts = (await cur.fetchone())["attempts"]
        await conn.commit()
    if attempts >= max_attempts:
        logger.error("B9 embedding: revision %s saltada tras %s intentos fallidos (%s); no se reintenta hasta que "
                     "una persona borre su fila de embedding_generation_attempts; vector-health sigue en rojo por ella",
                     revision_id, attempts, motivo)


class PersistentEmbeddingWriter:
    """Dedicated B9 embedding sink; authority is revalidated by persistence."""
    def __init__(self, api: PersistentMemoryAPI,
                 build_request_scope: Callable[[str, str, Visibility], Awaitable[ScopeContext]]):
        self._api = api
        self._build_request_scope = build_request_scope

    async def persist(self, memory_id: str, identity: EmbeddingSpaceIdentity,
                      vector: tuple[float, ...], visibility: Visibility, *, expected_revision_id: str | None = None) -> str:
        scope = await self._build_request_scope(memory_id, "RE_EMBED", visibility)
        if scope.actor_type != "SERVICE" or scope.actor_principal != "service:embedding":
            raise ScopeDenied("embedding requires the fixed embedding service principal")
        return await self._api.reembed_memory(
            MutationAuthorizationRequest(scope, "RE_EMBED", visibility), memory_id, identity, vector,
            expected_revision_id=expected_revision_id
        )


def build_persistent_embedding_writer(pool: object) -> PersistentEmbeddingWriter:
    """Compose re-embedding from a stored revision's exact canonical scope."""
    pool = MappingPool(pool)
    api = PersistentMemoryAPI(MariaDBB9Store(pool), MariaDBScopeAuthorityResolver(pool))

    async def revision_scope(memory_id: str, _operation: str, _visibility: Visibility) -> ScopeContext:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT o.tenant_id,r.project_id,p.subject_user_id "
                    "FROM memory_objects o JOIN memory_projections pr ON pr.memory_id=o.memory_id "
                    "JOIN memory_revisions r ON r.revision_id=pr.current_revision_id "
                    "JOIN memory_provenance p ON p.revision_id=r.revision_id "
                    "WHERE o.memory_id=%s ORDER BY p.created_at DESC LIMIT 1", (memory_id,)
                )
                row = await cur.fetchone()
        if not row:
            raise ScopeDenied("embedding revision scope is unavailable")
        if isinstance(row, dict):
            tenant_id, project_id, subject_user_id = row["tenant_id"], row["project_id"], row["subject_user_id"]
        else:
            tenant_id, project_id, subject_user_id = row
        if tenant_id is None or subject_user_id is None:
            raise ScopeDenied("embedding revision lacks canonical tenant or subject")
        return ScopeContext("service:embedding", "SERVICE", str(subject_user_id), str(tenant_id),
                            str(project_id) if project_id is not None else None,
                            calling_component="embedding", request_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()))

    return PersistentEmbeddingWriter(api, revision_scope)


async def run_b9_embeddings(db: MemoryDB, *, writer: PersistentEmbeddingWriter | None = None,
                            identity: EmbeddingSpaceIdentity | None = None) -> tuple[int, int]:
    """Generate B9 embeddings without changing a revision's scope or lifecycle."""
    if not db.pool:
        raise RuntimeError("B9 embedding worker requires a connected pool")
    writer = writer or build_persistent_embedding_writer(db.pool)
    identity_resuelta_aqui = identity is None
    identity = identity or await resolver_identidad()
    await avisar_si_cambio_el_digest(db.pool, identity)
    max_attempts = _max_attempts()
    async with db.pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT o.memory_id,r.revision_id,r.visibility,p.payload FROM memory_objects o "
                "JOIN memory_projections pr ON pr.memory_id=o.memory_id "
                "JOIN memory_revisions r ON r.revision_id=pr.current_revision_id "
                "JOIN memory_revision_payloads p ON p.revision_id=r.revision_id "
                "LEFT JOIN embedding_generations g ON g.revision_id=r.revision_id AND g.embedding_space_id=%s "
                "LEFT JOIN embedding_generation_attempts a ON a.revision_id=r.revision_id AND a.embedding_space_id=%s "
                "WHERE pr.reconciliation_required=FALSE AND pr.current_lifecycle_state IN ('ACTIVE','VERIFIED') "
                "AND p.payload IS NOT NULL AND g.generation_id IS NULL "
                "AND (a.attempts IS NULL OR a.attempts<%s) "
                "ORDER BY r.created_at,r.revision_id LIMIT %s",
                (identity.embedding_space_id, identity.embedding_space_id, max_attempts, BATCH_SIZE)
            )
            rows = await cur.fetchall()
    completed = failed = 0
    for row in rows:
        payload = row["payload"]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "replace")
        try:
            vector = await asyncio.wait_for(db.get_embedding(payload or ""), float(os.getenv("JAX_MEMORY_EMBED_CALL_TIMEOUT_SECONDS", "120")))
            if vector is None:
                failed += 1
                await _registrar_fallo(db.pool, row["revision_id"], identity.embedding_space_id, "NO_VECTOR", max_attempts)
                continue
            await writer.persist(row["memory_id"], identity, tuple(vector), Visibility(row["visibility"]),
                                 expected_revision_id=row["revision_id"])
            completed += 1
        except Exception as error:  # fail-soft: one embedding failure must not stop the batch; queda contada en el intento de la fila
            logger.exception("B9 embedding failed for memory %s", row["memory_id"])
            failed += 1
            await _registrar_fallo(db.pool, row["revision_id"], identity.embedding_space_id, type(error).__name__, max_attempts)
    if identity_resuelta_aqui:
        # El digest se leyo al inicio; si Ollama cambio de pesos a mitad de la corrida, lo generado
        # despues pertenece a otro modelo pero quedo registrado con el digest viejo: la corrida falla.
        final = await modelo_digest(CONFIG.model)
        if final != identity.model_version_or_digest:
            logger.error("B9 embeddings: el digest del modelo %s cambio DURANTE la corrida (%s -> %s); los vectores "
                         "generados en ella no son confiables: se marca la corrida como fallida",
                         CONFIG.model, identity.model_version_or_digest, final)
            failed += 1
    return completed, failed


async def run_b9_vector_health() -> int:
    """Report B9 revisions that lack an embedding generation.

    This is deliberately read-only and is scheduled separately from embedding
    generation.  A non-zero result keeps missing vectors visible; it never
    manufactures a vector or changes lifecycle state.
    """
    host, port = os.environ.get("JAX_DB_HOST"), os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError("JAX_DB_HOST and JAX_DB_PORT are required for B9 vector health")
    try:
        identity = await resolver_identidad()
    finally:
        await cerrar_cliente_http()
    pool = await aiomysql.create_pool(
        host=host, port=int(port), user=os.environ.get("JAX_DB_USER", ""),
        password=os.environ.get("JAX_DB_PASSWORD", ""), db=os.environ.get("JAX_DB_NAME", "jax_memory"),
        autocommit=True, minsize=1, maxsize=1, connect_timeout=db_connect_timeout_seconds(),
    )
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) FROM memory_projections p "
                    "JOIN memory_revisions r ON r.revision_id=p.current_revision_id "
                    "JOIN memory_revision_payloads x ON x.revision_id=r.revision_id "
                    "LEFT JOIN embedding_generations g ON g.revision_id=r.revision_id AND g.embedding_space_id=%s "
                    "WHERE r.lifecycle_state IN ('ACTIVE','VERIFIED') "
                    "AND p.reconciliation_required=FALSE AND x.payload IS NOT NULL AND g.generation_id IS NULL",
                    (identity.embedding_space_id,)
                )
                (missing,) = await cur.fetchone()

                async def contar_messages():
                    await cur.execute("SELECT COUNT(*) FROM messages")
                    return (await cur.fetchone())[0]
                recall_vencido = await recall_requiere_remedicion(contar_messages)
    finally:
        pool.close()
        await pool.wait_closed()
    if recall_vencido:
        logger.error("B9 vector health: el recall del indice HNSW de messages hay que volver a medirlo (ver el ERROR anterior)")
    if missing:
        logger.error("B9 vector health: %s current revision(s) have no embedding generation", missing)
        return EXIT_GENERAL            # faltan vectores: fallo general, aunque ademas haya que remedir el recall
    if recall_vencido:
        return EXIT_RECALL_REMEDIR
    logger.info("B9 vector health: every current retrievable revision has a generation")
    return 0


async def procesar_mensajes(db: MemoryDB) -> tuple[int, int]:
    """Vectoriza todos los mensajes con embedding IS NULL.
    Devuelve (procesados, fallidos)."""
    async with db.pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT COUNT(*) FROM messages WHERE {_zero_embedding_sql(_col())}"
            )
            (total,) = await cur.fetchone()

    if total == 0:
        print("  Mensajes: ninguno pendiente.")
        return 0, 0

    print(f"  Mensajes pendientes: {total}")
    procesados = 0
    fallidos = 0
    last_id = 0  # cursor por id — estable aunque falle algún registro

    while True:
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, content FROM messages "
                    f"WHERE {_zero_embedding_sql(_col())} AND id > %s "
                    "ORDER BY id ASC LIMIT %s",
                    (last_id, BATCH_SIZE),
                )
                lote = await cur.fetchall()

        if not lote:
            break

        for msg_id, content in lote:
            print(f"  Procesando mensaje {procesados + fallidos + 1}/{total}...", end="\r", flush=True)

            embedding = await db.get_embedding(content or "")
            if embedding is None:
                logger.warning(f"Mensaje id={msg_id}: fallo embedding, se omite.")
                fallidos += 1
                last_id = msg_id  # avanzar igual para no repetir este registro
                continue

            vec_str = json.dumps(embedding)
            try:
                async with db.pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"UPDATE messages SET {_col()} = VEC_FromText(%s) "
                            f"WHERE id = %s AND {_zero_embedding_sql(_col())}",
                            (vec_str, msg_id),
                        )
                procesados += 1
            except Exception as e:  # fail-soft: el UPDATE fallido deja la fila con su embedding en ceros, asi que el propio WHERE la vuelve a seleccionar en la corrida siguiente; ademas se cuenta en `fallidos`, que se devuelve y se imprime
                logger.error(f"Mensaje id={msg_id}: error al guardar embedding: {e}")
                fallidos += 1

            last_id = msg_id  # siempre avanzar al final de cada registro

    print()  # salto de linea tras el \r
    return procesados, fallidos


async def procesar_facts(db: MemoryDB) -> tuple[int, int]:
    """Vectoriza todos los facts con embedding IS NULL.
    Devuelve (procesados, fallidos)."""
    async with db.pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT COUNT(*) FROM facts WHERE {_zero_embedding_sql(_col())}"
            )
            (total,) = await cur.fetchone()

    if total == 0:
        print("  Facts: ninguno pendiente.")
        return 0, 0

    print(f"  Facts pendientes: {total}")
    procesados = 0
    fallidos = 0
    last_id = 0

    while True:
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, fact_text FROM facts "
                    f"WHERE {_zero_embedding_sql(_col())} AND id > %s "
                    "ORDER BY id ASC LIMIT %s",
                    (last_id, BATCH_SIZE),
                )
                lote = await cur.fetchall()

        if not lote:
            break

        for fact_id, fact_text in lote:
            print(f"  Procesando fact {procesados + fallidos + 1}/{total}...", end="\r", flush=True)

            embedding = await db.get_embedding(fact_text or "")
            if embedding is None:
                logger.warning(f"Fact id={fact_id}: fallo embedding, se omite.")
                fallidos += 1
                last_id = fact_id
                continue

            vec_str = json.dumps(embedding)
            try:
                async with db.pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"UPDATE facts SET {_col()} = VEC_FromText(%s) "
                            f"WHERE id = %s AND {_zero_embedding_sql(_col())}",
                            (vec_str, fact_id),
                        )
                procesados += 1
            except Exception as e:  # fail-soft: el UPDATE fallido deja el fact con su embedding en ceros, asi que el propio WHERE lo vuelve a seleccionar en la corrida siguiente; ademas se cuenta en `fallidos`, que se devuelve y se imprime
                logger.error(f"Fact id={fact_id}: error al guardar embedding: {e}")
                fallidos += 1

            last_id = fact_id

    print()
    return procesados, fallidos


async def main() -> int:
    jax_db_host = os.environ.get("JAX_DB_HOST")
    if not jax_db_host:
        raise RuntimeError(
            "JAX_DB_HOST no está seteado -- sin default silencioso a "
            "localhost (esa instancia está muerta, ver memoria "
            "jax-dual-mariadb-instances). Sourceá /etc/jax/.env."
        )
    db = MemoryDB()
    try:
        ok = await db.connect(
            host=jax_db_host,
            user=os.getenv("JAX_DB_USER", ""),
            password=os.getenv("JAX_DB_PASSWORD", ""),
            database=os.getenv("JAX_DB_NAME", "jax_memory"), migrate_schema=False,
        )
        if not ok:
            print("[embedding_worker] No pude conectar a la base. Verificar variables de entorno.")
            return 1
        print("=" * 56)
        print("  JAX — Worker de embeddings")
        print("=" * 56)

        print("\n[1/2] Procesando mensajes...")
        m_ok, m_fail = await procesar_mensajes(db)
        print(f"  => Mensajes: {m_ok} vectorizados, {m_fail} fallidos.")

        print("\n[2/2] Procesando facts...")
        f_ok, f_fail = await procesar_facts(db)
        print(f"  => Facts: {f_ok} vectorizados, {f_fail} fallidos.")

        print("\n[3/3] Procesando revisiones B9...")
        b9_ok, b9_fail = await run_b9_embeddings(db)
        print(f"  => B9: {b9_ok} vectorizados, {b9_fail} fallidos.")

        print("\n" + "=" * 56)
        print(f"  Total procesado: {m_ok + f_ok + b9_ok} vectores guardados.")
        if m_fail + f_fail + b9_fail > 0:
            print(f"  Fallidos: {m_fail + f_fail + b9_fail} (ver logs para detalle).")
        print("=" * 56)

        return int(m_fail + f_fail + b9_fail > 0)
    finally:
        await db.close()
        await cerrar_cliente_http()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX memory embedding worker")
    parser.add_argument("--health", action="store_true", help="check B9 embedding-generation coverage only")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(asyncio.wait_for(run_b9_vector_health() if args.health else main(), float(os.getenv("JAX_MEMORY_RUN_TIMEOUT_SECONDS", "840")))) or 0)
