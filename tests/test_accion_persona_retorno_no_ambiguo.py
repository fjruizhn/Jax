"""`mark_action_item_done` y `touch_person_mentions` devuelven el rowcount
crudo (`cur.rowcount > 0` y `cur.rowcount` respectivamente), y aiomysql
cuenta filas CAMBIADAS, no COINCIDENTES (`MemoryDB.connect()` no pasa
`CLIENT.FOUND_ROWS` -- ver `connect()`, jax/memory/db.py). Es el MISMO
defecto que se arreglo el 2026-09-20 en `verify_fact`/`expire_fact`
(`tests/test_memoria_retorno_no_ambiguo.py`), reencarnado en estos dos
metodos vecinos -- pendiente explicito del 2026-09-27.

Verificado empiricamente contra una MariaDB real:

  - `mark_action_item_done`: `completed_at=NOW()` tiene precision de
    SEGUNDO (columna `timestamp`, sin digitos). Marcar como hecho un item
    que YA esta hecho, dos veces en el mismo segundo, no cambia ninguna
    columna -> `rowcount=0` -> `False` AUNQUE EL ITEM EXISTE.
  - `touch_person_mentions`: `last_mentioned=CURDATE()` es un no-op cuando
    ya se toco hoy -> `rowcount=0` para una persona que SI matcheo el WHERE.
  - un id/nombre que NO EXISTE -> tambien 0 / False.
  - con `db_error_handler` de por medio, la base caida -> None, tambien
    falsy.

"no existe", "ya estaba asi" y "se cayo la base" son indistinguibles.

EL ARREGLO, mismo patron que `verify_fact`/`expire_fact` (auditoria
adversarial m1, 2026-09-20): el UPDATE va PRIMERO; el SELECT/COUNT de
desempate solo corre si no hubo filas cambiadas. Con `autocommit=True` la
conexion NO es una transaccion, asi que el SELECT-antes-del-UPDATE tiene una
carrera (otra sesion podria borrar la fila entre las dos consultas) -- por
eso el orden importa y no se invierte.

Contrato nuevo:
    `mark_action_item_done`   True = el item existe (se aplico, sea o no
                               no-op) · False = no existe · None = no se
                               pudo completar.
    `touch_person_mentions`   int > 0 si matchea. Con `affected == 0` de la
                               UPDATE, se cuenta cuantas filas matchean el
                               WHERE para distinguir "nadie matchea" (0) de
                               "matcheo pero todos ya estaban al dia"
                               (> 0). None = no se pudo completar.
"""
from __future__ import annotations

import asyncio
import functools
import os
import unittest
from unittest import mock

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


# `action_items` referencia `conversations` por FK (source_conversation_id,
# nullable) -- CREATE TABLE action_items exige que `conversations` ya
# exista, aunque los tests de este archivo no le insertan filas.
_TABLAS_ACTION_ITEMS = ("conversations", "action_items")
_TABLA_PEOPLE = "people"


async def _preparar() -> list[str]:
    creadas = []
    for nombre in (*_TABLAS_ACTION_ITEMS, _TABLA_PEOPLE):
        existe = await _sql(
            "SELECT 1 FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (nombre,), fetch=True)
        if not existe:
            await _sql(_esquema_memoria.ddl_del_archivo(nombre))
            creadas.append(nombre)
    return creadas


async def _limpiar():
    await _sql("DELETE FROM action_items WHERE user_id = %s", (_USER,))
    await _sql("DELETE FROM people WHERE name LIKE %s", ("__test_retorno_no_ambiguo_%",))


@pytest.fixture
def limpio():
    creadas = asyncio.run(_preparar())
    asyncio.run(_limpiar())
    yield
    async def _teardown():
        await _limpiar()
        for nombre in reversed((*_TABLAS_ACTION_ITEMS, _TABLA_PEOPLE)):
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


_USER = 990_031  # reservado para este archivo (distinto del 990_030 de facts)


async def _crear_pendiente(status: str = "pending") -> int:
    return await _sql(
        "INSERT INTO action_items (action_uuid, description, status, user_id) "
        "VALUES (UUID(), %s, %s, %s)",
        ("pendiente de prueba", status, _USER))


async def _crear_persona(nombre: str, last_mentioned=None) -> int:
    return await _sql(
        "INSERT INTO people (person_uuid, name, last_mentioned) VALUES (UUID(), %s, %s)",
        (nombre, last_mentioned))


ID_INEXISTENTE = 900_000_001  # no hay AUTO_INCREMENT que llegue tan alto en test


# ---------------------------------------------------------------------------
# mark_action_item_done -- los cuatro casos, contra una base real
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_mark_action_item_done_marcar_dos_veces_seguidas_da_true(limpio):
    """Marcar como hecho un item ya hecho es un no-op si cae en el mismo
    segundo que la primera marca (completed_at, columna `timestamp`, sin
    microsegundos) -- el item EXISTE y la operacion es valida, no es un
    'no encontre el pendiente'."""
    item_id = await _crear_pendiente()
    m = await _memoria()
    primera = await m.mark_action_item_done(item_id)
    segunda = await m.mark_action_item_done(item_id)
    assert primera is True
    assert segunda is True, "marcar como hecho un item ya hecho no es 'no encontre el pendiente'"


@requiere_db_de_prueba
@asincrono
async def test_mark_action_item_done_item_inexistente_da_false(limpio):
    m = await _memoria()
    ok = await m.mark_action_item_done(ID_INEXISTENTE)
    assert ok is False


@requiere_db_de_prueba
@asincrono
async def test_mark_action_item_done_base_caida_da_none(limpio):
    m = dbmod.MemoryDB()
    m.pool = mock.MagicMock()
    m.pool.acquire = mock.MagicMock(side_effect=OSError("la base no responde"))
    resultado = await m.mark_action_item_done(1)
    assert resultado is None


# ---------------------------------------------------------------------------
# touch_person_mentions -- mismo mecanismo, adaptado a bulk + int
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_touch_person_mentions_tocar_dos_veces_el_mismo_dia_da_mayor_a_cero(limpio):
    """`last_mentioned=CURDATE()` puesto dos veces el mismo dia es un no-op
    de verdad (a diferencia de completed_at, CURDATE() no cambia dentro del
    mismo dia): la persona SI matcheo el WHERE las dos veces."""
    nombre = "__test_retorno_no_ambiguo_ana"
    await _crear_persona(nombre)
    m = await _memoria()
    primera = await m.touch_person_mentions([nombre])
    segunda = await m.touch_person_mentions([nombre])
    assert primera == 1
    assert segunda == 1, "tocar dos veces el mismo dia sigue siendo una mencion real, no cero"


@requiere_db_de_prueba
@asincrono
async def test_touch_person_mentions_nadie_matchea_da_cero(limpio):
    m = await _memoria()
    n = await m.touch_person_mentions(["__test_retorno_no_ambiguo_nadie_asi"])
    assert n == 0


@requiere_db_de_prueba
@asincrono
async def test_touch_person_mentions_base_caida_da_none(limpio):
    m = dbmod.MemoryDB()
    m.pool = mock.MagicMock()
    m.pool.acquire = mock.MagicMock(side_effect=OSError("la base no responde"))
    resultado = await m.touch_person_mentions(["cualquiera"])
    assert resultado is None
