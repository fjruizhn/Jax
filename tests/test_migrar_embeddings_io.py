"""La migración de embeddings (scripts/migrar_embeddings.py), contra MariaDB.

2026-09-12: nomic-embed-text (768) -> bge-m3 (1024). MariaDB 12.3 no admite
dos índices vectoriales en una tabla (medido: "doesn't yet support 'multiple
VECTOR indexes'"), así que la migración va en dos tiempos:
- `migrar`: agrega la columna nueva y re-embebe todas las filas, SIN índice.
  La columna vieja sigue indexada: el sistema funciona igual que antes.
- `activar`: el corte. Mueve el índice vectorial a la columna nueva. Se niega
  si quedan filas en ceros (buscar sobre una columna a medio llenar pierde
  memoria en silencio), salvo `forzar`.
- `revertir`: devuelve el índice a la vieja y borra la nueva. Se niega a borrar
  la columna que la configuración usa. La vieja nunca se toca: volver atrás no
  pierde datos.

Contra una base real (jax_memory_test) y con un embebedor inyectado (sin
Ollama). Mismo patrón que test_memory_vector_zero_io.py: DDL desde
jax_memory_schema.sql y un event loop por test.
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys
import uuid
from pathlib import Path

import pytest

_nombre = os.environ.get("JAX_DB_NAME", "")
if not _nombre.endswith("_test"):
    pytest.skip(f"JAX_DB_NAME={_nombre!r}: esta prueba escribe y borra tablas, solo en *_test",
                allow_module_level=True)

aiomysql = pytest.importorskip("aiomysql")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _esquema_memoria  # noqa: E402
import migrar_embeddings as mig  # noqa: E402

_USER = 990_411  # reservado para este archivo
# Desde el corte (2026-09-12) la columna "vieja" de estas pruebas es la ACTIVA
# del esquema (embedding_bge_m3) y la "nueva" una de prueba: asi no dependen de
# `embedding` (nomic), que se retira. El mecanismo es el mismo en cualquier
# direccion; lo que se prueba es migrar/activar/revertir/retirar, no un modelo.
_COL = "embedding_prueba"
_VIEJA = "embedding_bge_m3"
_DIM = 4


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


async def _pool():
    return await aiomysql.create_pool(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.environ["JAX_DB_NAME"], autocommit=True, minsize=1, maxsize=2)


async def _sql(pool, q, args=(), fetch=False):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(q, args)
            return list(await cur.fetchall()) if fetch else cur.lastrowid


async def _columna_existe(pool, tabla, col):
    r = await _sql(pool, "SELECT COUNT(*) FROM information_schema.COLUMNS WHERE "
                         "TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
                   (tabla, col), fetch=True)
    return r[0][0] == 1


async def _columna_indexada(pool, tabla, col):
    r = await _sql(pool, "SELECT COUNT(*) FROM information_schema.STATISTICS WHERE "
                         "TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
                   (tabla, col), fetch=True)
    return r[0][0] > 0


def _embebedor(llamadas, fallar=("falla",)):
    async def embed(textos):
        llamadas.append(list(textos))
        return [None if t in fallar else [float(len(t)), 1.0, 0.5, 0.25] for t in textos]
    return embed


@pytest.fixture
def base():
    """Tablas de memoria en el estado de HOY (sin la columna nueva)."""
    async def preparar():
        pool = await _pool()
        creadas = []
        for nombre, ddl in _esquema_memoria.ddl().items():
            if not await _sql(pool, "SELECT 1 FROM information_schema.TABLES WHERE "
                                    "TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s", (nombre,), fetch=True):
                await _sql(pool, ddl)
                creadas.append(nombre)
        await mig.revertir(pool, columna=_COL, activa=_VIEJA, columna_vieja=_VIEJA)
        await _sql(pool, "DELETE FROM facts WHERE user_id=%s", (_USER,))
        await _sql(pool, "DELETE FROM conversations WHERE user_id=%s", (_USER,))
        conv = await _sql(pool, "INSERT INTO conversations (conversation_uuid, user_id) VALUES (%s, %s)",
                          (str(uuid.uuid4()), _USER))
        for i, texto in enumerate(["uno", "dos", "falla", "cuatro"]):
            await _sql(pool, "INSERT INTO messages (conversation_id, turn_number, role, content, user_id) "
                             "VALUES (%s, %s, 'user', %s, %s)", (conv, i, texto, _USER))
        await _sql(pool, "INSERT INTO facts (fact_uuid, fact_text, fact_type, user_id) "
                         "VALUES (%s, 'un hecho', 'user', %s)", (str(uuid.uuid4()), _USER))
        pool.close(); await pool.wait_closed()
        return creadas
    creadas = asyncio.run(preparar())
    yield

    async def limpiar():
        pool = await _pool()
        await mig.revertir(pool, columna=_COL, activa=_VIEJA, columna_vieja=_VIEJA)
        await _sql(pool, "DELETE FROM facts WHERE user_id=%s", (_USER,))
        await _sql(pool, "DELETE FROM conversations WHERE user_id=%s", (_USER,))
        for nombre in reversed(list(_esquema_memoria.TABLAS)):
            if nombre in creadas:
                await _sql(pool, f"DROP TABLE {nombre}")
        pool.close(); await pool.wait_closed()
    asyncio.run(limpiar())


@asincrono
async def test_migrar_llena_la_columna_nueva_sin_tocar_el_indice_viejo(base):
    pool = await _pool()
    r = await mig.migrar(pool, columna=_COL, dim=_DIM, embed_lote=_embebedor([]), lote=2)

    for tabla in ("messages", "facts"):
        assert await _columna_existe(pool, tabla, _COL)
        assert await _columna_indexada(pool, tabla, _VIEJA), "migrar no debe mover el índice"
        assert not await _columna_indexada(pool, tabla, _COL)
    assert r["messages"]["fallidas"] >= 1 and r["facts"]["fallidas"] == 0, r
    vivas = await _sql(pool, f"SELECT content FROM messages WHERE user_id=%s AND "
                             f"VEC_DISTANCE_EUCLIDEAN({_COL}, VEC_FromText(%s)) > 0",
                       (_USER, mig.zero_vector_text(_DIM)), fetch=True)
    assert sorted(x[0] for x in vivas) == ["cuatro", "dos", "uno"]
    pool.close(); await pool.wait_closed()


@asincrono
async def test_migrar_es_idempotente(base):
    pool = await _pool()
    await mig.migrar(pool, columna=_COL, dim=_DIM, embed_lote=_embebedor([]), lote=2)
    segunda = []
    await mig.migrar(pool, columna=_COL, dim=_DIM, embed_lote=_embebedor(segunda), lote=2)
    textos = [t for lote in segunda for t in lote]
    assert set(textos) <= {"falla"}, textos  # solo reintenta lo que quedó en ceros
    pool.close(); await pool.wait_closed()


@asincrono
async def test_activar_se_niega_con_filas_en_ceros(base):
    pool = await _pool()
    await mig.migrar(pool, columna=_COL, dim=_DIM, embed_lote=_embebedor([]), lote=2)
    with pytest.raises(ValueError):
        await mig.activar(pool, columna=_COL, columna_vieja=_VIEJA, dim=_DIM)
    assert await _columna_indexada(pool, "messages", _VIEJA), "un activar rechazado no mueve nada"
    pool.close(); await pool.wait_closed()


@asincrono
async def test_activar_mueve_el_indice_y_revertir_lo_devuelve(base):
    pool = await _pool()
    await mig.migrar(pool, columna=_COL, dim=_DIM, embed_lote=_embebedor([], fallar=()), lote=2)
    await mig.activar(pool, columna=_COL, columna_vieja=_VIEJA, dim=_DIM)
    for tabla in ("messages", "facts"):
        assert await _columna_indexada(pool, tabla, _COL)
        assert not await _columna_indexada(pool, tabla, _VIEJA)
    # activar dos veces no rompe
    await mig.activar(pool, columna=_COL, columna_vieja=_VIEJA, dim=_DIM)

    with pytest.raises(ValueError):
        await mig.revertir(pool, columna=_COL, activa=_COL, columna_vieja=_VIEJA)
    assert await _columna_existe(pool, "messages", _COL)

    await mig.revertir(pool, columna=_COL, activa=_VIEJA, columna_vieja=_VIEJA)
    for tabla in ("messages", "facts"):
        assert not await _columna_existe(pool, tabla, _COL)
        assert await _columna_indexada(pool, tabla, _VIEJA), "revertir debe devolver el índice viejo"
    pool.close(); await pool.wait_closed()


@asincrono
async def test_retirar_se_niega_con_la_columna_activa(base):
    # La columna que usa la configuracion no se borra, aunque no tenga indice.
    pool = await _pool()
    with pytest.raises(ValueError):
        await mig.retirar(pool, columna=_VIEJA, activa=_VIEJA)
    for tabla in ("messages", "facts"):
        assert await _columna_existe(pool, tabla, _VIEJA)
    pool.close(); await pool.wait_closed()


@asincrono
async def test_retirar_se_niega_si_la_columna_tiene_el_indice(base):
    # Una columna indexada es la que se busca: aunque la config diga otra cosa
    # (proceso con config vieja, variable mal escrita), no se borra.
    pool = await _pool()
    with pytest.raises(ValueError):
        await mig.retirar(pool, columna=_VIEJA, activa="otra_columna")
    for tabla in ("messages", "facts"):
        assert await _columna_existe(pool, tabla, _VIEJA)
        assert await _columna_indexada(pool, tabla, _VIEJA)
    pool.close(); await pool.wait_closed()


@asincrono
async def test_retirar_borra_una_columna_sin_indice_ni_uso(base):
    pool = await _pool()
    await mig.migrar(pool, columna=_COL, dim=_DIM, embed_lote=_embebedor([]), lote=2)
    r = await mig.retirar(pool, columna=_COL, activa=_VIEJA)
    for tabla in ("messages", "facts"):
        assert not await _columna_existe(pool, tabla, _COL), r
        assert await _columna_indexada(pool, tabla, _VIEJA), "retirar no toca la columna activa"
    # idempotente: una segunda vez no falla
    await mig.retirar(pool, columna=_COL, activa=_VIEJA)
    pool.close(); await pool.wait_closed()
