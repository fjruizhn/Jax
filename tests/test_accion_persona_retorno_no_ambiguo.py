"""`mark_action_item_done` y `touch_person_mentions` devolvian el rowcount
crudo (`cur.rowcount > 0` y `cur.rowcount` respectivamente), y aiomysql
cuenta filas CAMBIADAS, no COINCIDENTES (`MemoryDB.connect()` no pasa
`CLIENT.FOUND_ROWS` -- ver `connect()`, jax/memory/db.py). Es el MISMO
defecto que se arreglo el 2026-09-20 en `verify_fact`/`expire_fact`
(`tests/test_memoria_retorno_no_ambiguo.py`), reencarnado en estos dos
metodos vecinos -- pendiente notado el 2026-09-20 (recordatorio agendado en
`claude-skills/PENDIENTES.md` para el 2026-09-27; ESE 2026-09-27 es la
fecha de AVISO del pendiente, no la del arreglo -- confundirlas fue el
hallazgo #4 de la auditoria adversarial de jax#247 sobre el commit
`c687a2e`). Arreglado de verdad el 2026-09-21.

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

EL ARREGLO -- y NO es el mismo orden en los dos metodos, a proposito
(hallazgos #2 y #3 de la auditoria adversarial de jax#247, sobre el commit
`c687a2e` que abrio este mismo archivo):

  - `mark_action_item_done`, mismo patron que `verify_fact`/`expire_fact`
    (auditoria adversarial m1, 2026-09-20): el UPDATE va PRIMERO; el
    SELECT de existencia solo corre si `affected == 0`. El WHERE filtra
    por `id` (AUTO_INCREMENT), asi que invertir el orden -- SELECT antes
    del UPDATE -- si abriria una carrera (otra sesion podria borrar la
    fila entre las dos consultas).
  - `touch_person_mentions` es DISTINTO: el WHERE filtra por NOMBRE, que
    SI puede aparecer entre dos consultas (alguien crea la persona justo
    en el medio). Por eso el COUNT va PRIMERO, SIEMPRE (no condicionado a
    nada), y el UPDATE es un efecto de lado que ni se emite si nadie
    matchea. La primera version de este arreglo (commit `c687a2e`) le
    copio el patron "UPDATE primero" al vecino sin ver que el WHERE por
    nombre invierte cual orden es seguro -- reproduciendo, en la propia
    correccion, el retorno no-monotono que decia estar cerrando (medido
    contra MariaDB real: mismo estado, misma entrada, dos numeros
    distintos -- ver `test_touch_person_mentions_caso_mixto_es_monotono`
    mas abajo). Con el pool en `None` (no confundir con "no pool y sin
    nombres", que ahora se reportan distinto), el commit original tambien
    devolvia `0` en vez de `None` -- el mismo aplastamiento "no pude" vs
    "no matcheo nadie" que todo este archivo existe para cerrar.

Contrato nuevo:
    `mark_action_item_done`   True = el item existe (se aplico, sea o no
                               no-op) · False = no existe · None = no se
                               pudo completar.
    `touch_person_mentions`   int = cuantas personas matchean
                               `names_or_nicknames`, SIEMPRE de la misma
                               consulta (no de `cur.rowcount`, y no
                               distinto segun hubo no-op o no) · 0 si la
                               lista de nombres viene vacia (no hay nada
                               que pedir, no es un fallo) · None SOLO si
                               falta el pool (no se pudo completar).
"""
from __future__ import annotations

import asyncio
import functools
import os
import time
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


def _esperar_borde_de_segundo(margen: float = 0.9) -> None:
    """Sincroniza el reloj de pared a un punto recien pasado un borde de
    segundo, dejando `margen` segundos de aire antes de que cambie de nuevo.

    Hallazgo #5 de la auditoria adversarial de jax#247 (sobre el commit
    `c687a2e`): `test_mark_action_item_done_marcar_dos_veces_seguidas_da_true`
    hacia DOS llamadas reales a `mark_action_item_done()` una tras otra y
    daba por sentado que caian en el MISMO segundo de reloj -- `completed_at`
    es `timestamp` SIN microsegundos, y el no-op que el test quiere probar
    SOLO se da si las dos llamadas comparten segundo. Si el borde de
    segundo cae justo en medio (azar del scheduler de asyncio/pytest, o un
    runner de CI mas lento), el test pasa CON EL DEFECTO PRESENTE -- rojo
    intermitente que nadie asocia con el codigo.

    LIMITE explicito (no se puede eliminar del todo sin cambiar la firma de
    `mark_action_item_done` para aceptar un `completed_at` inyectado, que
    es un cambio de diseño fuera de este alcance): si el proceso se pausa
    mas de `margen` segundos ENTRE las dos llamadas (GC largo, contencion
    de CPU del contenedor de CI), el test vuelve a ser flojo. Dos llamadas
    SQL locales tardan milisegundos, asi que en la practica el margen
    sobra; esto reduce la ventana de fallo de "cualquier borde de segundo"
    a "una pausa real de casi un segundo entre dos queries", no la anula
    matematicamente."""
    inicio = time.monotonic()
    while time.time() % 1 > (1 - margen):
        if time.monotonic() - inicio > 2:
            break  # no se pudo sincronizar en 2s reales -- seguir igual, ver LIMITE arriba
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# mark_action_item_done -- los cuatro casos, contra una base real
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_mark_action_item_done_marcar_dos_veces_seguidas_da_true(limpio):
    """Marcar como hecho un item ya hecho es un no-op si cae en el mismo
    segundo que la primera marca (completed_at, columna `timestamp`, sin
    microsegundos) -- el item EXISTE y la operacion es valida, no es un
    'no encontre el pendiente'. Sincronizado al borde de segundo (ver
    `_esperar_borde_de_segundo`): sin esto el test es flojo -- verde con el
    defecto presente si el borde cae entre las dos llamadas."""
    item_id = await _crear_pendiente()
    m = await _memoria()
    _esperar_borde_de_segundo()
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
async def test_touch_person_mentions_caso_mixto_es_monotono(limpio):
    """El caso comun de un worker que ve a las mismas personas varias veces
    al dia: una ya tocada hoy, otra no. Las DOS matchean el WHERE -- el
    retorno tiene que ser el MISMO (2) las dos veces, sin importar que una
    ya estuviera al dia. Hallazgo #2 de la auditoria adversarial de jax#247:
    los dos tests de arriba cubren solo los extremos (nadie tocado / todos
    ya tocados); ninguno cubria la mezcla, que es donde el retorno viejo
    dejaba de ser monotono (medido contra MariaDB real, commit `c687a2e`:
    1 la primera vez, 2 la segunda, mismo estado y misma entrada)."""
    ana = "__test_retorno_no_ambiguo_ana_mixto"
    beto = "__test_retorno_no_ambiguo_beto_mixto"
    await _crear_persona(ana)
    await _crear_persona(beto)
    m = await _memoria()
    await m.touch_person_mentions([ana])  # "ana" queda tocada HOY; "beto" no
    primera = await m.touch_person_mentions([ana, beto])
    segunda = await m.touch_person_mentions([ana, beto])
    assert primera == 2, "las DOS matchean el WHERE, aunque ana ya estuviera al dia"
    assert segunda == primera, "mismo estado, misma entrada: el retorno no puede cambiar"


@requiere_db_de_prueba
@asincrono
async def test_touch_person_mentions_cuenta_antes_de_actualizar(limpio):
    """Hallazgo #3 de la auditoria adversarial de jax#247: con el WHERE por
    NOMBRE (a diferencia del vecino, que filtra por `id`), una fila
    insertada por OTRA sesion entre el UPDATE y un SELECT/COUNT tardio
    matchearia sin que el UPDATE la haya tocado -- una mencion afirmada que
    nadie escribio. No se puede fabricar esa carrera de verdad con una sola
    sesion; lo que SI es verificable es el orden: el COUNT tiene que ser la
    PRIMERA consulta, siempre, nunca una segunda consulta que mira el
    estado DESPUES de escribir -- y con `matches == 0` el UPDATE no se
    emite en absoluto (nada que tocar)."""
    nombre = "__test_retorno_no_ambiguo_orden"
    await _crear_persona(nombre)
    m = await _memoria()

    consultas = []
    original = aiomysql.Cursor.execute

    async def _rastrear(self, query, *args, **kwargs):
        consultas.append(query.strip().split(None, 1)[0].upper())
        return await original(self, query, *args, **kwargs)

    with mock.patch.object(aiomysql.Cursor, "execute", _rastrear):
        n = await m.touch_person_mentions([nombre])
    assert n == 1
    assert consultas == ["SELECT", "UPDATE"], (
        f"el orden tiene que ser SELECT (conteo) primero, UPDATE despues: {consultas}")

    consultas.clear()
    with mock.patch.object(aiomysql.Cursor, "execute", _rastrear):
        n = await m.touch_person_mentions(["__test_retorno_no_ambiguo_orden_nadie"])
    assert n == 0
    assert consultas == ["SELECT"], (
        f"con matches == 0 no hay nada que tocar, el UPDATE no se emite: {consultas}")


@requiere_db_de_prueba
@asincrono
async def test_touch_person_mentions_sin_nombres_da_cero(limpio):
    """Sin nombres que buscar, no hay nada que pedir -- distinto de "no
    pude completar la operacion" (hallazgo #1 de la auditoria adversarial
    de jax#247: el commit `c687a2e` mezclaba las dos condiciones,
    `not self.pool or not names_or_nicknames`, en un solo `return 0`)."""
    m = await _memoria()
    n = await m.touch_person_mentions([])
    assert n == 0


@requiere_db_de_prueba
@asincrono
async def test_touch_person_mentions_sin_pool_da_none(limpio):
    """EL CAMINO REAL de "base caida": `self.pool` nunca se establecio
    (falta `connect()`, o `connect()` fallo). Hallazgo #1 de la auditoria
    adversarial de jax#247, BLOQUEANTE: el commit `c687a2e` probaba "base
    caida" solo simulando un `pool.acquire()` que explota (ver
    `test_touch_person_mentions_acquire_explota_da_none`, mas abajo) --
    eso ejercita el `except` de `db_error_handler`, pero el `if not
    self.pool: return 0` de la primera linea del metodo NUNCA se llegaba a
    correr, porque el mock siempre dejaba `self.pool` en un MagicMock
    truthy. Un test de "base caida" que evita el UNICO camino de base
    caida que estaba mal."""
    m = dbmod.MemoryDB()  # pool nunca se conecto -- self.pool es None de fabrica
    assert m.pool is None
    resultado = await m.touch_person_mentions(["cualquiera"])
    assert resultado is None


@requiere_db_de_prueba
@asincrono
async def test_touch_person_mentions_acquire_explota_da_none(limpio):
    """El OTRO camino de "base caida": el pool existe pero se cayo la
    conexion al intentar usarla (`acquire()` explota). `db_error_handler`
    lo atrapa y devuelve `None` igual que el caso de arriba."""
    m = dbmod.MemoryDB()
    m.pool = mock.MagicMock()
    m.pool.acquire = mock.MagicMock(side_effect=OSError("la base no responde"))
    resultado = await m.touch_person_mentions(["cualquiera"])
    assert resultado is None
