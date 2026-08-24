"""
JAX 2.0 — Worker de embeddings.

Vectoriza todos los mensajes y facts que tienen embedding IS NULL.
Corre una vez (o cuando se necesite poner al dia la base).

Uso:
    set -a; source /etc/jax/.env; set +a
    cd ~/jax && .venv/bin/python -m jax.memory.embedding_worker

En memoria de Jairo Urbina.
"""

import asyncio
import json
import logging
import os

from jax.memory.db import MemoryDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [embedding_worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jax.memory.embedding_worker")

BATCH_SIZE = 50


async def procesar_mensajes(db: MemoryDB) -> tuple[int, int]:
    """Vectoriza todos los mensajes con embedding IS NULL.
    Devuelve (procesados, fallidos)."""
    async with db.pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM messages WHERE embedding IS NULL"
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
                    "WHERE embedding IS NULL AND id > %s "
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
                            "UPDATE messages SET embedding = VEC_FromText(%s) WHERE id = %s",
                            (vec_str, msg_id),
                        )
                procesados += 1
            except Exception as e:
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
                "SELECT COUNT(*) FROM facts WHERE embedding IS NULL"
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
                    "WHERE embedding IS NULL AND id > %s "
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
                            "UPDATE facts SET embedding = VEC_FromText(%s) WHERE id = %s",
                            (vec_str, fact_id),
                        )
                procesados += 1
            except Exception as e:
                logger.error(f"Fact id={fact_id}: error al guardar embedding: {e}")
                fallidos += 1

            last_id = fact_id

    print()
    return procesados, fallidos


async def main() -> None:
    jax_db_host = os.environ.get("JAX_DB_HOST")
    if not jax_db_host:
        raise RuntimeError(
            "JAX_DB_HOST no está seteado -- sin default silencioso a "
            "localhost (esa instancia está muerta, ver memoria "
            "jax-dual-mariadb-instances). Sourceá /etc/jax/.env."
        )
    db = MemoryDB()
    ok = await db.connect(
        host=jax_db_host,
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        database=os.getenv("JAX_DB_NAME", "jax_memory"),
    )
    if not ok:
        print("[embedding_worker] No pude conectar a la base. Verificar variables de entorno.")
        return

    print("=" * 56)
    print("  JAX — Worker de embeddings")
    print("=" * 56)

    print("\n[1/2] Procesando mensajes...")
    m_ok, m_fail = await procesar_mensajes(db)
    print(f"  => Mensajes: {m_ok} vectorizados, {m_fail} fallidos.")

    print("\n[2/2] Procesando facts...")
    f_ok, f_fail = await procesar_facts(db)
    print(f"  => Facts: {f_ok} vectorizados, {f_fail} fallidos.")

    print("\n" + "=" * 56)
    print(f"  Total procesado: {m_ok + f_ok} vectores guardados.")
    if m_fail + f_fail > 0:
        print(f"  Fallidos: {m_fail + f_fail} (ver logs para detalle).")
    print("=" * 56)

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
