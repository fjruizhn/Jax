"""`verify_fact` y `expire_fact` devuelven `affected > 0`, y aiomysql cuenta
filas CAMBIADAS, no COINCIDENTES (`MemoryDB.connect()` no pasa
`CLIENT.FOUND_ROWS` -- ver `connect()`, jax/memory/db.py). Auditoria
adversarial 2026-09-20 sobre feat/memoria-admin, verificado empiricamente
contra una MariaDB real:

  - quitar una caducidad que nunca existio (expire_fact(id, None) sobre un
    hecho que ya tenia expires_at NULL) -> `affected=0` -> devuelve False
    AUNQUE FUNCIONO.
  - poner dos veces la misma fecha de vencimiento -> tambien False.
  - un hecho que NO EXISTE -> tambien False.
  - con `db_error_handler` de por medio, la base caida -> None, tambien
    falsy.

"no existe", "ya estaba asi" y "se cayo la base" son indistinguibles --
exactamente el defecto que esta rama vino a arreglar, reencarnado en sus
propios metodos. jax-platform#hechos/caducar hace `if not ok: raise 404`
(backend/api/admin/memoria.py): un no-op idempotente hoy le devuelve al
usuario "hecho no encontrado" sobre un hecho que SI existe.

EL ARREGLO: un SELECT de existencia en la MISMA conexion, antes del UPDATE.
Se prefiere sobre habilitar CLIENT.FOUND_ROWS en el pool porque ese flag es
GLOBAL a la conexion: cambiaria el contrato de `rowcount`/`execute()` de
CUALQUIER otro escritor que comparte el pool (mark_action_item_done,
touch_person_mentions, delete_fact, ...) sin que esta ronda los haya
auditado a todos. El SELECT confina el arreglo a los dos metodos con el
bug documentado.

Contrato nuevo, verificado abajo (comportamiento REAL, no introspeccion):
    True  = el hecho existe (la operacion se aplico, sea o no un no-op)
    False = el hecho no existe
    None  = no se pudo completar (falla de conexion/consulta)
"""
from __future__ import annotations

import asyncio
import functools
import os
import unittest
from datetime import datetime, timedelta
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

_USER = 990_030   # reservado para este archivo


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
    await _sql("DELETE FROM facts WHERE user_id = %s", (_USER,))


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
    }
    campos.update(overrides)
    return await _sql(
        "INSERT INTO facts (fact_uuid, fact_text, fact_type, confidence, "
        "is_verified, user_id, expires_at) VALUES (UUID(), %s, %s, %s, %s, %s, %s)",
        (fact_text, campos["fact_type"], campos["confidence"], campos["is_verified"],
         campos["user_id"], campos["expires_at"]))


ID_INEXISTENTE = 900_000_001  # no hay AUTO_INCREMENT que llegue tan alto en test


# ---------------------------------------------------------------------------
# expire_fact -- los cuatro casos, contra una base real
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_expire_fact_quitar_una_caducidad_que_nunca_existio_da_true(limpio):
    """El hecho nunca tuvo expires_at (NULL de fabrica). Ponerlo en None de
    nuevo es un no-op: no cambia ninguna columna, pero el hecho EXISTE y la
    operacion se completo -- no es un 404."""
    fid = await _crear_fact()  # expires_at=None por defecto
    m = await _memoria()
    ok = await m.expire_fact(fid, None)
    assert ok is True, "un no-op sobre un hecho que existe no es 'no encontrado'"


@requiere_db_de_prueba
@asincrono
async def test_expire_fact_poner_dos_veces_la_misma_fecha_da_true(limpio):
    vence = (datetime.now() + timedelta(days=5)).replace(microsecond=0)
    fid = await _crear_fact(expires_at=vence)
    m = await _memoria()
    ok = await m.expire_fact(fid, vence)  # mismo valor: UPDATE no cambia nada
    assert ok is True, "repetir el mismo vencimiento sigue siendo una operacion valida"


@requiere_db_de_prueba
@asincrono
async def test_expire_fact_hecho_inexistente_da_false(limpio):
    m = await _memoria()
    ok = await m.expire_fact(ID_INEXISTENTE, None)
    assert ok is False


@requiere_db_de_prueba
@asincrono
async def test_expire_fact_base_caida_da_none(limpio):
    """Con `db_error_handler` de por medio: una excepcion en la conexion
    tiene que seguir devolviendo None, nunca False (que se leeria como 'no
    existe')."""
    m = dbmod.MemoryDB()
    m.pool = mock.MagicMock()
    m.pool.acquire = mock.MagicMock(side_effect=OSError("la base no responde"))
    resultado = await m.expire_fact(1, None)
    assert resultado is None


# ---------------------------------------------------------------------------
# verify_fact -- mismo mecanismo, mismos cuatro casos
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_verify_fact_verificar_dos_veces_con_el_mismo_aprobador_da_true(limpio):
    fid = await _crear_fact()
    m = await _memoria()
    primera = await m.verify_fact(fid, _USER)
    segunda = await m.verify_fact(fid, _USER)  # mismo verified_by: UPDATE no cambia nada
    assert primera is True
    assert segunda is True, "reafirmar la misma aprobacion no es 'hecho no encontrado'"


@requiere_db_de_prueba
@asincrono
async def test_verify_fact_hecho_inexistente_da_false(limpio):
    m = await _memoria()
    ok = await m.verify_fact(ID_INEXISTENTE, _USER)
    assert ok is False


@requiere_db_de_prueba
@asincrono
async def test_verify_fact_base_caida_da_none(limpio):
    m = dbmod.MemoryDB()
    m.pool = mock.MagicMock()
    m.pool.acquire = mock.MagicMock(side_effect=OSError("la base no responde"))
    resultado = await m.verify_fact(1, _USER)
    assert resultado is None
