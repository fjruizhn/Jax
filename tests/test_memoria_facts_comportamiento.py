"""Comportamiento REAL de verify_fact, supersede_fact, expire_fact y
get_facts (jax/memory/db.py) contra una MariaDB real.

POR QUE EXISTE (auditoria adversarial 2026-09-20 sobre feat/memoria-admin).
Los 27 tests que la rama traia son de `inspect.signature`/
`inspect.getsource`: confirman que el codigo tiene ciertas palabras (un
parametro `verified_by` sin default, la palabra `expires_at` en el texto de
`get_facts`), no que el metodo HACE lo que el nombre promete. El auditor copio
la rama, aplico 5 mutaciones que revierten el proposito del PR, y los 27
tests pasaron las 5 veces.

Este archivo siembra filas con SQL directo, ejecuta el metodo bajo prueba, y
LEE LA FILA DE VUELTA -- si el metodo no escribe lo que dice, alguna
aserción sobre la fila falla. Las cinco mutaciones del auditor tienen que
poner esta suite en rojo (evidencia en el reporte de cierre / mensaje del
commit); los tests de introspeccion existentes (test_memoria_gobernanza.py,
test_repl_fact_verify_autoria.py) se quedan porque documentan el contrato de
la firma, pero no cuentan para ese criterio.

Mismo patron de aislamiento que test_memory_scope_denormalized.py: user_id
reservado propio, solo corre contra una base cuyo nombre es de tests
(`requiere_db_de_prueba`), y limpia unicamente lo que crea.
"""
from __future__ import annotations

import asyncio
import functools
import os
from datetime import datetime, timedelta

import aiomysql
import pytest

from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory import db as dbmod
import _esquema_memoria

from base_de_test import es_base_de_test  # noqa: E402

_DB = os.getenv("JAX_DB_NAME", "")
requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB),
    reason="necesita una MariaDB real y JAX_DB_NAME en una base de tests",
)

_USER = 990_020        # reservado para este archivo
_OTRO_USER = 990_021   # autor de una correccion -- distinto del dueno del hecho


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


async def _conn():
    return await aiomysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.environ["JAX_DB_NAME"], autocommit=True,
        connect_timeout=db_connect_timeout_seconds())


async def _sql(query, args=(), fetch=False):
    conn = await _conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            if fetch:
                return list(await cur.fetchall())
            return cur.lastrowid
    finally:
        conn.close()


_DDL = _esquema_memoria.ddl()


async def _preparar() -> list[str]:
    """Crea las tablas solo si faltan; devuelve las que creo para borrarlas al
    final. Igual que en test_memory_scope_denormalized.py: en hall9000 la base
    de tests la comparten varias suites."""
    creadas = []
    for nombre, ddl in _DDL.items():
        existe = await _sql(
            "SELECT 1 FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (nombre,), fetch=True)
        if not existe:
            await _sql(ddl)
            creadas.append(nombre)
    return creadas


async def _limpiar():
    await _sql("DELETE FROM facts WHERE user_id IN (%s, %s)", (_USER, _OTRO_USER))


@pytest.fixture
def limpio():
    creadas = asyncio.run(_preparar())
    asyncio.run(_limpiar())
    yield
    async def _teardown():
        await _limpiar()
        for nombre in reversed(list(_DDL)):
            if nombre in creadas:
                await _sql(f"DROP TABLE {nombre}")
    asyncio.run(_teardown())


async def _memoria() -> dbmod.MemoryDB:
    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok, "MemoryDB no conecto: el test no probaria nada"
    return m


async def _crear_fact(fact_text: str = "hecho de prueba", **overrides) -> int:
    campos = {
        "fact_type": "user",
        "confidence": 0.7,
        "is_verified": False,
        "user_id": _USER,
        "expires_at": None,
        "importance": None,
    }
    campos.update(overrides)
    return await _sql(
        "INSERT INTO facts (fact_uuid, fact_text, fact_type, confidence, "
        "is_verified, user_id, expires_at, importance) "
        "VALUES (UUID(), %s, %s, %s, %s, %s, %s, %s)",
        (fact_text, campos["fact_type"], campos["confidence"], campos["is_verified"],
         campos["user_id"], campos["expires_at"], campos["importance"]))


async def _fila(fact_id: int, *columnas: str) -> tuple:
    filas = await _sql(
        f"SELECT {', '.join(columnas)} FROM facts WHERE id = %s", (fact_id,), fetch=True)
    assert filas, f"fact {fact_id} no existe"
    return filas[0]


# ---------------------------------------------------------------------------
# verify_fact: escribe QUIEN aprobo, no solo que "affected > 0"
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_verify_fact_escribe_quien_aprobo(limpio):
    """Mutacion (a) del auditor: borrar `verified_by = %s` del UPDATE deja la
    firma intacta (el parametro se sigue recibiendo) pero la columna en NULL.
    Sin leer la fila de vuelta, ningun test de firma lo detecta."""
    fid = await _crear_fact()
    m = await _memoria()
    ok = await m.verify_fact(fid, _USER)
    assert ok is True
    is_verified, verified_by, verified_at = await _fila(
        fid, "is_verified", "verified_by", "verified_at")
    assert is_verified == 1
    assert verified_by == _USER, "verify_fact no dejo constancia de QUIEN aprobo"
    assert verified_at is not None


# ---------------------------------------------------------------------------
# supersede_fact: escribe QUIEN corrigio, no None
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_supersede_fact_registra_quien_corrigio(limpio):
    """Mutacion (b) del auditor: supersede_fact escribe None en
    superseded_by_user en vez del autor recibido."""
    viejo = await _crear_fact("hecho viejo")
    nuevo = await _crear_fact("hecho nuevo")
    m = await _memoria()
    ok = await m.supersede_fact(viejo, nuevo, _OTRO_USER)
    assert ok is True
    superseded_by, superseded_by_user, superseded_at = await _fila(
        viejo, "superseded_by", "superseded_by_user", "superseded_at")
    assert superseded_by == nuevo
    assert superseded_by_user == _OTRO_USER, (
        "supersede_fact no registro quien tomo la decision de corregir")
    assert superseded_at is not None


# ---------------------------------------------------------------------------
# expire_fact: escribe expires_at, no verified_at
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_expire_fact_escribe_expires_at_y_no_toca_verified_at(limpio):
    """Mutacion (e) del auditor: expire_fact escribe en verified_at en vez de
    expires_at."""
    fid = await _crear_fact()
    m = await _memoria()
    vence = (datetime.now() + timedelta(days=1)).replace(microsecond=0)
    ok = await m.expire_fact(fid, vence)
    assert ok is True
    expires_at, verified_at = await _fila(fid, "expires_at", "verified_at")
    assert expires_at is not None and expires_at == vence
    assert verified_at is None, "expire_fact no debe tocar verified_at"


# ---------------------------------------------------------------------------
# get_facts: excluye vencidos, pero NO todo lo que tiene expires_at
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_get_facts_excluye_vencidos_pero_incluye_los_que_vencen_a_futuro(limpio):
    """Mutacion (d) del auditor: el filtro pasa de
    `(expires_at IS NULL OR expires_at > NOW())` a `expires_at IS NULL` a
    secas -- el bug clasico de NULL. Bajo esa mutacion, un hecho que vence
    DENTRO DE UN MES (todavia vigente) desaparecería del listado igual que uno
    vencido de verdad: el test tiene que sembrar los dos casos para
    distinguirlos."""
    pasado = datetime.now() - timedelta(days=1)
    futuro = datetime.now() + timedelta(days=30)
    vencido = await _crear_fact("hecho vencido", expires_at=pasado)
    vigente_con_vencimiento = await _crear_fact(
        "hecho vigente que vence a futuro", expires_at=futuro)
    sin_vencimiento = await _crear_fact("hecho sin vencimiento")

    m = await _memoria()
    facts = await m.get_facts(only_unverified=False, user_id=_USER, limit=50)
    ids = {f["id"] for f in facts}

    assert vencido not in ids, "un hecho vencido no puede pesar en el listado normal"
    assert vigente_con_vencimiento in ids, (
        "un hecho que vence a futuro sigue activo HOY -- excluirlo es el bug "
        "clasico de NULL, no el filtro de caducidad")
    assert sin_vencimiento in ids


# ---------------------------------------------------------------------------
# idx_facts_revision: el indice que el MIGRADOR arma, no solo el declarado
# en jax_memory_schema.sql (esos dos caminos son independientes: una base
# creada por _preparar() ya trae la del .sql y no ejercita migrations.py).
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_ensure_schema_repone_idx_facts_revision_con_las_tres_columnas_en_orden(limpio):
    """Mutacion (c) del auditor: el DDL de idx_facts_revision en
    migrations.py pasa de `(is_verified, expires_at, created_at)` a
    `(created_at)` sola. `connect()` llama a `ensure_schema()`, que es el
    camino que corre en cada arranque real de JAX -- por eso el indice se
    DROPEA primero y se deja que `_memoria()` (que conecta) lo reponga,
    en vez de leerlo tal como lo dejo `_preparar()` desde el .sql."""
    await _sql("DROP INDEX idx_facts_revision ON facts")
    m = await _memoria()
    assert m.schema_ok is True
    filas = await _sql(
        "SELECT COLUMN_NAME FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='facts' "
        "AND INDEX_NAME='idx_facts_revision' ORDER BY SEQ_IN_INDEX", fetch=True)
    columnas = [c for (c,) in filas]
    assert columnas == ["is_verified", "expires_at", "created_at"], (
        f"el migrador repuso idx_facts_revision como {columnas}: la pantalla "
        f"de Memoria filtra por is_verified/expires_at y ordena por "
        f"created_at, y sin las tres columnas EN ESE ORDEN el EXPLAIN cae a "
        f"filesort")
