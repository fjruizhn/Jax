# tests/test_indice_vectorial.py
"""El centinela del indice vectorial: detectar que miente, y repararlo.

**Por que existe (2026-09-20).** Medido en hall9000 contra MariaDB 12.3.3:
despues de un DELETE masivo, un indice HNSW deja de devolver filas que SI
estan en la tabla, y no se recupera solo. `OPTIMIZE TABLE` no lo repara.

Lo grave no es perder resultados de busqueda. El que se queda ciego es
`_find_nearest_fact`, el dedup de la memoria -- y para un dedup, "no encontre
nada parecido" significa "es nuevo". O sea que no deja de encontrar: DUPLICA,
en silencio y para siempre.

**Y la ceguera es PARCIAL.** En una corrida medida devolvio 1 de 10, no 0. Por
eso el centinela NO puede preguntar "devolvio vacio?": tiene que comparar el
conteo por el indice contra el mismo conteo con `IGNORE INDEX`. Un centinela
que solo mire el vacio da verde con el indice roto.

Los tests usan su propia tabla, no `messages` ni `facts`: el que vigila no
ensucia lo vigilado.
"""
from __future__ import annotations

import asyncio
import functools
import os
import random

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory import indice_vectorial as iv

_DB = os.getenv("JAX_DB_NAME", "")
_PADRE = "prueba_iv_conversaciones"
_TABLA = "prueba_iv_mensajes"
_COL = "embedding_bge_m3"
_DIM = 1024

requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not es_base_de_test(_DB),
    reason="necesita una MariaDB real y JAX_DB_NAME en una base de tests")


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


def _vec(semilla: int) -> str:
    r = random.Random(semilla)
    v = [r.random() for _ in range(_DIM)]
    n = sum(x * x for x in v) ** 0.5
    return "[" + ",".join(str(x / n) for x in v) + "]"


async def _crear_tabla(cur):
    """Dos tablas con la MISMA forma que `conversations` y `messages`.

    Lo que importa y no es decorado:
      * la FK con `ON DELETE CASCADE` -- es el disparador del defecto;
      * el indice vectorial con `DISTANCE='cosine'` y `M='16'`, igual que
        produccion. Con los valores por omision el defecto NO se reproduce.

    Son tablas propias y no `messages`: envenenar el indice de la base de test
    compartida dejaria mintiendo a todo lo que corra despues. El que vigila no
    ensucia lo vigilado.
    """
    await cur.execute(f"DROP TABLE IF EXISTS {_TABLA}")
    await cur.execute(f"DROP TABLE IF EXISTS {_PADRE}")
    await cur.execute(
        f"CREATE TABLE {_PADRE} ("
        f"  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
        f"  user_id INT NOT NULL"
        f") ENGINE=InnoDB")
    await cur.execute(
        f"CREATE TABLE {_TABLA} ("
        f"  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
        f"  conversation_id BIGINT NOT NULL,"
        f"  user_id INT NOT NULL,"
        f"  {_COL} VECTOR({_DIM}) NOT NULL,"
        f"  VECTOR INDEX idx_{_COL} ({_COL}) `DISTANCE`='cosine' `M`='16',"
        f"  CONSTRAINT fk_prueba_iv FOREIGN KEY (conversation_id)"
        f"    REFERENCES {_PADRE} (id) ON DELETE CASCADE"
        f") ENGINE=InnoDB")


async def _borrar_tablas(cur):
    await cur.execute(f"DROP TABLE IF EXISTS {_TABLA}")
    await cur.execute(f"DROP TABLE IF EXISTS {_PADRE}")


async def _poner(cur, cuantas: int, desde: int = 0, uid: int = 77):
    await cur.execute(f"INSERT INTO {_PADRE} (user_id) VALUES (%s)", (uid,))
    conv = cur.lastrowid
    for i in range(cuantas):
        await cur.execute(
            f"INSERT INTO {_TABLA} (conversation_id, user_id, {_COL}) "
            f"VALUES (%s, %s, VEC_FromText(%s))", (conv, uid, _vec(desde + i)))


async def _envenenar(cur):
    """Reproduce el defecto EXACTO, medido el 2026-09-20.

    El disparador no es "un DELETE masivo": es el borrado EN CASCADA. Medido
    con las mismas filas y la misma consulta:

        DELETE FROM <padre>   (cascada)  -> indice 1 / scan 10  ENVENENADO
        DELETE FROM <hijo>    (directo)  -> indice 10 / scan 10  sano

    Y basta UNA conversacion: no hace falta vaciar la tabla.
    """
    await _poner(cur, 300, desde=1000, uid=1)
    await cur.execute(
        # marcador-propio: `_PADRE` es una tabla EXCLUSIVA de este archivo -- la
        # crea `_crear_tabla()` y la borra el fixture al terminar. No es una tabla
        # compartida de `jax_memory_test`: no puede haber filas de otra sesion.
        f"DELETE FROM {_PADRE} WHERE user_id = 1")
    await _poner(cur, 25)


@pytest.fixture
def tabla():
    """Crea y borra la tabla de prueba.

    Cada `asyncio.run()` levanta su propio bucle y aiomysql ata la conexion al
    bucle donde nacio, asi que NO se comparte una conexion entre el fixture y
    el test: cada uno abre la suya. Mismo patron que `_sql()` en los otros
    tests de memoria.
    """
    asyncio.run(_con_cursor(_crear_tabla))
    yield
    asyncio.run(_con_cursor(_borrar_tablas))


async def _con_cursor(fn):
    """Abre conexion, corre `fn(cur)` y cierra. Devuelve lo que devuelva `fn`."""
    conn = await _conn()
    try:
        async with conn.cursor() as cur:
            return await fn(cur)
    finally:
        conn.close()


@requiere_db_de_prueba
@asincrono
async def test_una_tabla_sana_no_se_reporta(tabla):
    """El centinela no puede gritar cuando no pasa nada: una alarma que suena
    siempre termina apagada, y entonces no hay alarma."""
    async def _caso(cur):
        await _poner(cur, 25)
        return await iv.revisar_uno(cur, _TABLA, f"idx_{_COL}", _COL)

    informe = await _con_cursor(_caso)
    assert informe.sano, informe
    assert informe.por_el_indice == informe.por_scan == informe.muestra


@requiere_db_de_prueba
@asincrono
async def test_detecta_el_indice_envenenado(tabla):
    async def _caso(cur):
        await _envenenar(cur)
        return await iv.revisar_uno(cur, _TABLA, f"idx_{_COL}", _COL)

    informe = await _con_cursor(_caso)
    assert not informe.sano, informe
    assert informe.por_el_indice < informe.por_scan, informe


@requiere_db_de_prueba
@asincrono
async def test_un_centinela_que_solo_mire_el_vacio_no_sirve(tabla):
    """Centinela de la CEGUERA PARCIAL. Si el indice envenenado devuelve algo
    -- no cero -- el detector tiene que seguir marcandolo roto. Es el caso
    medido (1 de 10) y el que haria fallar a la version ingenua del detector.
    """
    async def _caso(cur):
        await _envenenar(cur)
        return await iv.revisar_uno(cur, _TABLA, f"idx_{_COL}", _COL)

    informe = await _con_cursor(_caso)
    if informe.por_el_indice == 0:
        pytest.skip("esta corrida dio ceguera TOTAL; este test cubre la parcial")
    assert not informe.sano, (
        f"el indice devolvio {informe.por_el_indice} de {informe.por_scan} y se dio por sano: "
        "un detector que solo mire el vacio deja pasar la ceguera parcial")


@requiere_db_de_prueba
@asincrono
async def test_reparar_devuelve_el_indice_sin_perder_filas(tabla):
    async def _caso(cur):
        await _envenenar(cur)
        await cur.execute(f"SELECT COUNT(*) FROM {_TABLA}")
        antes = (await cur.fetchone())[0]
        roto = await iv.revisar_uno(cur, _TABLA, f"idx_{_COL}", _COL)

        await iv.reparar_uno(cur, _TABLA, f"idx_{_COL}", _COL)

        sano = await iv.revisar_uno(cur, _TABLA, f"idx_{_COL}", _COL)
        await cur.execute(f"SELECT COUNT(*) FROM {_TABLA}")
        return antes, roto, sano, (await cur.fetchone())[0]

    antes, roto, sano, despues = await _con_cursor(_caso)
    assert not roto.sano, f"no se logro envenenar: el test no probaria nada ({roto})"
    assert sano.sano, sano
    assert despues == antes, f"la reparacion perdio filas: {antes} -> {despues}"


@requiere_db_de_prueba
@asincrono
async def test_encuentra_los_indices_vectoriales_solo(tabla):
    """Descubre por `INDEX_TYPE='VECTOR'`, no por el nombre de la columna: el
    dia que la columna cambie -- ya paso al pasar a bge-m3 -- esto sigue
    encontrandolos sin que nadie lo actualice."""
    async def _caso(cur):
        await _poner(cur, 1)
        return await iv.indices_vectoriales(cur)

    hallados = await _con_cursor(_caso)
    assert (_TABLA, f"idx_{_COL}", _COL) in hallados, hallados


@requiere_db_de_prueba
@asincrono
async def test_una_tabla_vacia_no_puede_mentir(tabla):
    """Sin filas no hay nada que devolver: no se reporta envenenada."""
    informe = await _con_cursor(
        lambda cur: iv.revisar_uno(cur, _TABLA, f"idx_{_COL}", _COL))
    assert informe.sano and informe.filas == 0, informe


@requiere_db_de_prueba
@asincrono
async def test_la_reparacion_no_degrada_la_definicion_del_indice(tabla):
    """El indice se recrea IGUAL, con su distancia y su M.

    Un `ADD VECTOR INDEX (col)` pelado lo recrearia con los valores por
    omision: otra funcion de distancia y otro M. La busqueda seguiria
    "funcionando" y daria resultados distintos, sin un solo error -- una
    reparacion que degrada en silencio lo que vino a reparar.
    """
    async def _caso(cur):
        await _poner(cur, 25)
        antes = await iv.definicion_del_indice(cur, _TABLA, f"idx_{_COL}")
        await iv.reparar_uno(cur, _TABLA, f"idx_{_COL}", _COL)
        return antes, await iv.definicion_del_indice(cur, _TABLA, f"idx_{_COL}")

    antes, despues = await _con_cursor(_caso)
    assert "cosine" in antes.lower() and "16" in antes, (
        f"la tabla de prueba no reproduce la definicion de produccion: {antes}")
    assert despues == antes, f"la reparacion cambio el indice:\n  antes:  {antes}\n  despues: {despues}"
