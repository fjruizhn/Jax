#!/usr/bin/env python3
"""Migración de los embeddings de memoria a otro modelo (2026-09-12: bge-m3).

POR QUE ASI. MariaDB 12.3 no admite dos índices vectoriales en una tabla
(medido: "doesn't yet support 'multiple VECTOR indexes'"). Así que no hay
forma de tener la columna vieja y la nueva indexadas a la vez, y la migración
va en tres operaciones, cada una idempotente:

  migrar    agrega la columna nueva (NOT NULL, default vector cero) en
            messages y facts, y re-embebe en lotes las filas que siguen en
            ceros. NO toca índices: el sistema sigue funcionando con la vieja.
  activar   el corte: saca el índice vectorial de la columna vieja y lo crea
            sobre la nueva. Se niega si quedan filas en ceros en la nueva --
            buscar sobre una columna a medio llenar pierde memoria sin error
            -- salvo --forzar.
  revertir  devuelve el índice a la columna vieja y borra la nueva. Se niega a
            borrar la columna que la configuración activa usa. La vieja nunca
            se toca: volver atrás no pierde datos.

La configuración que leen los servicios (JAX_MEMORY_EMBED_MODEL/_DIM/_COLUMN,
ver jax/memory/embedding_config.py) se cambia en /etc/jax/.env aparte: este
script no la toca. El orden del corte está en DEUDA.md / el plan de ejecución.

Uso (hall9000, con /etc/jax/.env cargado):
  python3 scripts/migrar_embeddings.py migrar   --modelo bge-m3 --dim 1024 --columna embedding_bge_m3
  python3 scripts/migrar_embeddings.py activar  --dim 1024 --columna embedding_bge_m3
  python3 scripts/migrar_embeddings.py revertir --columna embedding_bge_m3

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Awaitable, Callable, Optional

# Tablas con embedding -> (columna de texto, opciones del índice vectorial).
# Las opciones son las de producción hoy (SHOW CREATE TABLE, 2026-09-12):
# messages se reconstruyó con M=16 (jax#128); facts usa el M por defecto.
TABLAS = {
    "messages": ("content", "DISTANCE=cosine M=16"),
    "facts": ("fact_text", "DISTANCE=cosine"),
}
_IDENTIFICADOR = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
# Mismo tope que jax/memory/db.py::get_embedding.
MAX_CHARS = 4000

EmbedLote = Callable[[list], Awaitable[list]]


def zero_vector_text(dim: int) -> str:
    return "[" + ",".join(["0"] * dim) + "]"


def _ident(nombre: str) -> str:
    if not _IDENTIFICADOR.match(nombre or ""):
        raise ValueError(f"{nombre!r} no es un identificador simple")
    return nombre


async def _uno(pool, sql, args=()):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, args)
            return await cur.fetchone()


async def _ejecutar(pool, sql, args=()):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, args)
            return cur.rowcount


async def _columna_existe(pool, tabla, columna) -> bool:
    r = await _uno(pool, "SELECT COUNT(*) FROM information_schema.COLUMNS WHERE "
                         "TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
                   (tabla, columna))
    return r[0] == 1


async def _indice_vectorial(pool, tabla) -> Optional[tuple[str, str]]:
    """(nombre del índice, columna) del índice vectorial de la tabla, o None.
    Hay a lo sumo uno: MariaDB no admite más."""
    r = await _uno(pool, "SELECT INDEX_NAME, COLUMN_NAME FROM information_schema.STATISTICS "
                         "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_TYPE='VECTOR'",
                   (tabla,))
    return (r[0], r[1]) if r else None


async def _filas_en_ceros(pool, tabla, columna, dim) -> int:
    r = await _uno(pool, f"SELECT COUNT(*) FROM {tabla} WHERE "
                         f"VEC_DISTANCE_EUCLIDEAN({columna}, VEC_FromText(%s)) = 0",
                   (zero_vector_text(dim),))
    return r[0]


async def migrar(pool, columna: str, dim: int, embed_lote: EmbedLote, lote: int = 32,
                 tablas: tuple = tuple(TABLAS)) -> dict:
    """Agrega `columna` y re-embebe las filas que siguen en ceros. No toca
    índices. Idempotente: una fila con vector real nunca se re-embebe, y el
    UPDATE está guardado por "sigue en ceros"."""
    _ident(columna)
    cero = zero_vector_text(dim)
    resultado = {}
    for tabla in tablas:
        texto_col, _ = TABLAS[tabla]
        t0 = time.monotonic()
        if not await _columna_existe(pool, tabla, columna):
            await _ejecutar(pool, f"ALTER TABLE {tabla} ADD COLUMN {columna} VECTOR({dim}) "
                                  f"NOT NULL DEFAULT VEC_FromText('{cero}')")
        r = {"reparadas": 0, "fallidas": 0}
        ultimo = 0
        while True:
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"SELECT id, {texto_col} FROM {tabla} WHERE id > %s AND "
                        f"VEC_DISTANCE_EUCLIDEAN({columna}, VEC_FromText(%s)) = 0 "
                        f"ORDER BY id LIMIT %s", (ultimo, cero, lote))
                    filas = await cur.fetchall()
            if not filas:
                break
            ultimo = filas[-1][0]
            vectores = await embed_lote([(t or "")[:MAX_CHARS] for _, t in filas])
            for (fila_id, _), v in zip(filas, vectores):
                if not v or len(v) != dim or not any(v):
                    r["fallidas"] += 1
                    continue
                r["reparadas"] += await _ejecutar(
                    pool, f"UPDATE {tabla} SET {columna} = VEC_FromText(%s) WHERE id = %s AND "
                          f"VEC_DISTANCE_EUCLIDEAN({columna}, VEC_FromText(%s)) = 0",
                    (json.dumps(v), fila_id, cero))
        r["segundos"] = round(time.monotonic() - t0, 1)
        resultado[tabla] = r
    return resultado


async def activar(pool, columna: str, columna_vieja: str, dim: int, forzar: bool = False,
                  tablas: tuple = tuple(TABLAS)) -> dict:
    """Mueve el índice vectorial a `columna`. Idempotente."""
    _ident(columna); _ident(columna_vieja)
    for tabla in tablas:
        if not await _columna_existe(pool, tabla, columna):
            raise ValueError(f"{tabla}.{columna} no existe: correr `migrar` primero")
        pendientes = await _filas_en_ceros(pool, tabla, columna, dim)
        if pendientes and not forzar:
            raise ValueError(f"{tabla}.{columna}: {pendientes} fila(s) siguen en ceros; "
                             f"activar ahora las deja fuera de toda búsqueda. Correr "
                             f"`migrar` otra vez, o --forzar")
    hecho = {}
    for tabla in tablas:
        _, opciones = TABLAS[tabla]
        actual = await _indice_vectorial(pool, tabla)
        if actual and actual[1] == columna:
            hecho[tabla] = "ya activo"
            continue
        if actual:
            await _ejecutar(pool, f"DROP INDEX {actual[0]} ON {tabla}")
        await _ejecutar(pool, f"CREATE VECTOR INDEX idx_{columna} ON {tabla} ({columna}) {opciones}")
        hecho[tabla] = f"{actual[0] if actual else '-'} -> idx_{columna}"
    return hecho


async def revertir(pool, columna: str, activa: str, columna_vieja: str,
                   tablas: tuple = tuple(TABLAS)) -> dict:
    """Devuelve el índice a `columna_vieja` y borra `columna`. Idempotente."""
    _ident(columna); _ident(activa); _ident(columna_vieja)
    if columna == activa:
        raise ValueError(f"{columna} es la columna que usa la configuración activa "
                         f"(JAX_MEMORY_EMBED_COLUMN): cambiarla y reiniciar antes de revertir")
    hecho = {}
    for tabla in tablas:
        _, opciones = TABLAS[tabla]
        actual = await _indice_vectorial(pool, tabla)
        if actual and actual[1] == columna:
            await _ejecutar(pool, f"DROP INDEX {actual[0]} ON {tabla}")
            actual = None
        if not actual and await _columna_existe(pool, tabla, columna_vieja):
            await _ejecutar(pool, f"CREATE VECTOR INDEX idx_{columna_vieja} ON {tabla} "
                                  f"({columna_vieja}) {opciones}")
        if await _columna_existe(pool, tabla, columna):
            await _ejecutar(pool, f"ALTER TABLE {tabla} DROP COLUMN {columna}")
        hecho[tabla] = "revertida"
    return hecho


def _embebedor_ollama(modelo: str, url: str = "http://localhost:11434/api/embed") -> EmbedLote:
    import httpx

    async def embed(textos: list) -> list:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json={"model": modelo, "input": textos, "keep_alive": "60s"})
            r.raise_for_status()
            return r.json().get("embeddings") or [None] * len(textos)
    return embed


async def _pool_desde_env():
    import aiomysql
    host, port = os.environ.get("JAX_DB_HOST"), os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise SystemExit("JAX_DB_HOST/JAX_DB_PORT sin setear: sourceá /etc/jax/.env")
    return await aiomysql.create_pool(host=host, port=int(port), user=os.getenv("JAX_DB_USER", ""),
                                      password=os.getenv("JAX_DB_PASSWORD", ""),
                                      db=os.getenv("JAX_DB_NAME", "jax_memory"),
                                      autocommit=True, minsize=1, maxsize=2)


async def _main(args) -> int:
    pool = await _pool_desde_env()
    try:
        if args.op == "migrar":
            r = await migrar(pool, args.columna, args.dim, _embebedor_ollama(args.modelo), args.lote)
        elif args.op == "activar":
            r = await activar(pool, args.columna, args.columna_vieja, args.dim, args.forzar)
        else:
            activa = os.environ.get("JAX_MEMORY_EMBED_COLUMN", "embedding")
            r = await revertir(pool, args.columna, activa, args.columna_vieja)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0
    except ValueError as e:
        print(f"RECHAZADO: {e}", file=sys.stderr)
        return 1
    finally:
        pool.close()
        await pool.wait_closed()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("op", choices=["migrar", "activar", "revertir"])
    p.add_argument("--columna", required=True)
    p.add_argument("--columna-vieja", default="embedding")
    p.add_argument("--modelo", default="bge-m3")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--forzar", action="store_true")
    sys.exit(asyncio.run(_main(p.parse_args())))
