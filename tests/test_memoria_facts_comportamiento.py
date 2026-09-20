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


@requiere_db_de_prueba
@asincrono
async def test_verify_fact_con_verified_by_cero_escribe_cero_sin_chistar(limpio):
    """Hueco de cobertura senalado en la auditoria adversarial 2026-09-20:
    `verify_fact` NO valida `verified_by` (a diferencia de sus DOS
    llamadores, que si tratan 0 como "no se sabe quien" -- `handle_fact_
    command`/REPL en tests/test_repl_fact_verify_autoria.py, y el candado de
    `save_fact` para `supersede_fact` en tests/test_memoria_no_traga_fallos_
    de_escritura.py). Es a proposito: el candado vive en cada LLAMADOR, que
    es quien sabe de donde sale el id y si 0 es valido en ese contexto --
    `MemoryDB.verify_fact` es una escritura generica que hace lo que se le
    pide. Este test documenta esa frontera, no la mueve."""
    fid = await _crear_fact()
    m = await _memoria()
    ok = await m.verify_fact(fid, 0)
    assert ok is True
    is_verified, verified_by = await _fila(fid, "is_verified", "verified_by")
    assert is_verified == 1
    assert verified_by == 0, (
        "verify_fact tiene que escribir 0 tal cual se lo pasaron -- validar "
        "'0 es no se sabe' es responsabilidad del llamador, no de este metodo")


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


@requiere_db_de_prueba
@asincrono
async def test_supersede_fact_con_old_fact_id_inexistente_da_false(limpio):
    """M3 (auditoria adversarial 2026-09-20): antes `supersede_fact`
    devolvia `True` SIN mirar `rowcount` -- un `old_fact_id` que no existe
    (o que desaparecio entre que `save_fact` lo encontro como candidato y
    este UPDATE) da `affected=0` y el metodo devolvia `True` igual. El test
    viejo (`test_supersede_fact_registra_quien_corrigio`, arriba) solo
    ejercita el camino feliz y `assert ok is True`: pasaria identico con el
    bug, porque nunca prueba un `old_fact_id` que no exista. Este lo hace."""
    nuevo = await _crear_fact("hecho nuevo, sin nada que reemplazar")
    m = await _memoria()
    id_inexistente = 900_000_002  # no hay AUTO_INCREMENT que llegue tan alto en test
    ok = await m.supersede_fact(id_inexistente, nuevo, _OTRO_USER)
    assert ok is False, (
        "supersede_fact devolvio True para un old_fact_id que no existe -- "
        "el rowcount del UPDATE tiene que decir la verdad")


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
    try:
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
    finally:
        # m5 (auditoria adversarial 2026-09-20): si algun assert de arriba
        # revienta, el indice queda CAIDO en una base COMPARTIDA
        # (jax_memory_test, o cualquier otra base de la misma sesion que
        # otra suite use en paralelo) -- reponerlo no puede depender del
        # camino feliz de este test, o la siguiente suite que corra pierde
        # su EXPLAIN sobre idx_facts_revision. Idempotente: si ensure_schema
        # ya lo repuso, este CREATE corre sobre un indice que ya existe y no
        # se ejecuta.
        existe = await _sql(
            "SELECT 1 FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='facts' "
            "AND INDEX_NAME='idx_facts_revision'", fetch=True)
        if not existe:
            await _sql(
                "CREATE INDEX idx_facts_revision ON facts "
                "(is_verified, expires_at, created_at)")


# ---------------------------------------------------------------------------
# M2: la migracion COMPENSATORIA repara una base ya migrada torcida, no solo
# una virgen (auditoria adversarial 2026-09-20).
#
# El test estatico de test_memoria_gobernanza.py (test_el_migrador_agrega_
# las_columnas_en_la_MISMA_posicion_que_el_esquema) compara TEXTO de DDL --
# nunca mira una base. No detecta que `ensure_schema()` salteaba la
# reposicion cuando la columna YA EXISTIA: el bucle de `_COLUMNAS` solo
# miraba "existe si/no", nunca "esta en el lugar correcto". Una base
# MIGRADA antes de que el `AFTER` se agregara al DDL (medido 2026-09-20
# contra jax_memory_test: verified_by en la posicion 20, superseded_by_user
# en la 21, cuando el .sql las pone en la 10 y la 11) se quedaba torcida
# PARA SIEMPRE, aunque `connect()`/`ensure_schema()` corriera en cada
# arranque -- y como `base_de_test.py` CLONA el esquema de esa base para
# cada base de sesion nueva, la deriva se propagaba a toda base nacida
# despues.
#
# Este test usa una base TEMPORAL propia (creada y borrada aca mismo, nunca
# `jax_memory_test`): reproduce el escenario a mano (agrega las columnas SIN
# `AFTER`, tal como las dejaba el migrador viejo) y corre `ensure_schema()`
# -- el camino real de `connect()` -- contra ella.
# ---------------------------------------------------------------------------

from jax.memory import migrations as migrations_mod  # noqa: E402

_DB_TEMPORAL_M2 = f"jax_memory_test_m2gobernanza_{os.getpid()}"


async def _conn_sin_base():
    return await aiomysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        autocommit=True, connect_timeout=db_connect_timeout_seconds())


async def _posiciones_de_columnas(pool, tabla: str) -> dict:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COLUMN_NAME, ORDINAL_POSITION FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s", (tabla,))
            return {nombre: pos for nombre, pos in await cur.fetchall()}


@requiere_db_de_prueba
@asincrono
async def test_ensure_schema_repara_una_base_migrada_con_las_columnas_en_mal_lugar():
    """M2 (auditoria adversarial 2026-09-20). Ver el bloque de comentarios de
    arriba para el porque completo."""
    assert _DB_TEMPORAL_M2 != "jax_memory", "nunca DROP/CREATE sobre produccion"
    assert es_base_de_test(_DB_TEMPORAL_M2), (
        "el nombre de la base temporal no matchea el patron de bases de "
        "test -- no se crea sin esa garantia")

    conn = await _conn_sin_base()
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{_DB_TEMPORAL_M2}`")
            await cur.execute(f"CREATE DATABASE `{_DB_TEMPORAL_M2}`")
        await conn.select_db(_DB_TEMPORAL_M2)
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE facts ("
                "  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,"
                "  is_verified TINYINT(1) DEFAULT 0,"
                "  verified_at TIMESTAMP NULL DEFAULT NULL,"
                "  expires_at TIMESTAMP NULL DEFAULT NULL,"
                "  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP"
                ") ENGINE=InnoDB")
            # Lo que dejaba el migrador ANTES de que `AFTER` se agregara al
            # DDL: la columna existe, pero al FINAL de la tabla -- no entre
            # `verified_at` y `expires_at`.
            await cur.execute("ALTER TABLE facts ADD COLUMN verified_by INT NULL")
            await cur.execute(
                "ALTER TABLE facts ADD COLUMN superseded_by_user INT NULL")
    finally:
        conn.close()

    pool = await aiomysql.create_pool(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=_DB_TEMPORAL_M2, autocommit=True,
        connect_timeout=db_connect_timeout_seconds())
    try:
        antes = await _posiciones_de_columnas(pool, "facts")
        assert antes["verified_by"] != antes["verified_at"] + 1, (
            "el escenario no reproduce el bug: verified_by ya estaba en su "
            "lugar ANTES de correr ensure_schema() -- este test no probaria "
            "nada")

        await migrations_mod.ensure_schema(pool)

        despues = await _posiciones_de_columnas(pool, "facts")
        assert despues["verified_by"] == despues["verified_at"] + 1, (
            f"ensure_schema() no reposiciono verified_by: quedo en la "
            f"posicion {despues['verified_by']}, verified_at esta en la "
            f"{despues['verified_at']}")
        assert despues["superseded_by_user"] == despues["verified_by"] + 1, (
            f"ensure_schema() no reposiciono superseded_by_user: quedo en "
            f"la posicion {despues['superseded_by_user']}, verified_by "
            f"esta en la {despues['verified_by']}")
    finally:
        pool.close()
        await pool.wait_closed()
        conn = await _conn_sin_base()
        try:
            async with conn.cursor() as cur:
                await cur.execute(f"DROP DATABASE IF EXISTS `{_DB_TEMPORAL_M2}`")
        finally:
            conn.close()
