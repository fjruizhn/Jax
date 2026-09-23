"""Caducar un hecho no puede volverlo un agujero permanente en la memoria.

POR QUE EXISTE (auditoria adversarial 2026-09-20 sobre feat/memoria-admin).
`_find_nearest_fact` (jax/memory/db.py) filtra `superseded_by IS NULL` pero
NO `expires_at`. Escenario real: se caduca un hecho; semanas despues el
extractor de memoria vuelve a producir el MISMO hecho (la conversacion lo
menciona de nuevo); `save_fact` busca el candidato mas cercano para
deduplicar, encuentra el hecho VENCIDO, lo clasifica como duplicado exacto,
y NO inserta el nuevo. El hecho vencido sigue invisible para la busqueda
(get_facts ya excluye vencidos) y el hecho nuevo nunca llega a existir: se
pierde en silencio y para siempre.

`get_scopes_with_verified_facts` tiene el mismo problema por el otro lado:
cuenta hechos vencidos para decidir si un scope tiene suficientes hechos
verificados como para que valga la pena sintetizar sobre el -- y despues el
sintetizador (item #8) recibe MENOS hechos verificados de los que el conteo
prometio.

EL CRITERIO (los caminos de lectura de `facts` en los DOS repos, y por que
cada uno filtra o no filtra `expires_at`).

m4 (auditoria adversarial 2026-09-20, SEGUNDA ronda): esta lista decia "los
CINCO caminos" y nacio desactualizada -- ya habia mas incluso el dia que se
escribio. Un criterio que se declara completo y no lo es es peor que uno que
no promete nada: alguien lo lee, confia en el numero, y no busca el resto.
Que el documento describa el CODIGO, no lo que el codigo tenia cuando se
escribio el comentario:

  1. `_find_nearest_fact` (dedup para save_fact) -- FILTRA. Un hecho vencido
     ya no es autoridad: tratarlo como "todavia ahi" bloquea la reextraccion
     del mismo hecho para siempre (este archivo).
  2. `get_facts` (listado de la pantalla de Memoria) -- FILTRA por defecto,
     con `incluir_vencidos=True` como escape explicito (spec §2.4; ya
     arreglado en una ronda anterior de esta rama).
  3. `get_scopes_with_verified_facts` (decide si un scope vale la pena para
     el sintetizador de segundo orden) -- FILTRA. Contar vencidos infla el
     conteo y el sintetizador recibe menos de lo que el conteo prometio
     (este archivo).
  4. `get_fact_text` (muestra el texto antes de confirmar un borrado, `/fact
     delete`) -- NO FILTRA a proposito. Un hecho vencido sigue siendo
     borrable -- de hecho es el caso mas probable de querer borrarlo -- y
     ocultar su texto rompería esa pantalla.
  5. `embedding_worker.procesar_facts` (backfill de embeddings en cero) --
     NO FILTRA a proposito. Vectoriza TODAS las filas con embedding en cero,
     vencidas o no: si mas adelante alguien quita el vencimiento
     (`expire_fact(id, None)`), el hecho tiene que volver a ser encontrable
     por busqueda semantica de inmediato, sin depender de que el backfill
     vuelva a pasar por el.
  6. `verify_fact`/`expire_fact` -- su `SELECT 1 FROM facts WHERE id = %s`
     (jax/memory/db.py, agregado en la primera ronda de esta auditoria para
     distinguir "no existe" de "no-op idempotente") NO FILTRA a proposito.
     Es una pregunta de EXISTENCIA, no de vigencia: un hecho vencido sigue
     siendo un hecho real que se puede aprobar/re-caducar (tests/
     test_memoria_retorno_no_ambiguo.py cubre este camino especifico).
  7. `SQL_LISTAR`/`SQL_CONTAR` (jax-platform, backend/api/admin/memoria.py
     -- la cola de revision de la pantalla de Memoria) -- FILTRA por
     defecto, con `incluir_vencidos` como escape explicito, MISMO criterio
     que `get_facts` (2), pero es SQL propio, no pasa por `MemoryDB`: es un
     camino de lectura aparte, no una llamada a (2) con otro nombre. Ver el
     comentario de ese archivo sobre por que no reusa `get_facts()`.

Mismo patron de aislamiento que los demas archivos de memoria: user_id
reservado, solo corre contra una base de tests, limpia lo que crea.
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

_USER = 990_040   # reservado para este archivo
_TENANT = 990_040

# This I/O module bootstraps the memory tables itself because the CI MariaDB
# service is empty.  B9's scope query now intentionally resolves the tenant
# from the DB-backed user row, so the fixture must provide that authority too.
_IDENTITY_DDL = """
CREATE TABLE IF NOT EXISTS jax_users (
    user_id BIGINT NOT NULL PRIMARY KEY,
    tenant_id BIGINT NOT NULL,
    status VARCHAR(32) NOT NULL,
    role VARCHAR(32) NOT NULL
)
"""


def _vec(pos: int, valor: float = 1.0) -> list[float]:
    v = [0.0] * dbmod.EMBEDDING_DIM
    v[pos] = valor
    return v


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
    existe_usuario = await _sql(
        "SELECT 1 FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'jax_users'", fetch=True)
    if not existe_usuario:
        await _sql(_IDENTITY_DDL)
        creadas.append("jax_users")
    await _sql(
        "INSERT INTO jax_users (user_id, tenant_id, status, role) VALUES (%s, %s, 'active', 'viewer') "
        "ON DUPLICATE KEY UPDATE tenant_id=VALUES(tenant_id), status='active', role='viewer'",
        (_USER, _TENANT))
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
    await _sql("DELETE FROM jax_users WHERE user_id = %s", (_USER,))


@pytest.fixture
def limpio():
    creadas = asyncio.run(_preparar())
    asyncio.run(_limpiar())
    yield
    async def _teardown():
        await _limpiar()
        for nombre in reversed([*list(_DDL), "jax_users"]):
            if nombre in creadas:
                await _sql(f"DROP TABLE {nombre}")
    asyncio.run(_teardown())


async def _memoria(monkeypatch, vec: list[float]) -> dbmod.MemoryDB:
    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok, "MemoryDB no conecto: el test no probaria nada"

    async def _emb(_texto):
        return vec

    monkeypatch.setattr(m, "get_embedding", _emb)
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


# ---------------------------------------------------------------------------
# 1. _find_nearest_fact: un hecho vencido no puede bloquear la reextraccion
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_un_hecho_vencido_no_bloquea_la_reextraccion_del_mismo_hecho(limpio, monkeypatch):
    vec = _vec(0)
    m = await _memoria(monkeypatch, vec)
    texto = "Fernando vive en Tegucigalpa"

    primero = await m.save_fact(texto, "user", user_id=_USER)
    assert primero is True

    filas = await _sql(
        "SELECT id FROM facts WHERE user_id=%s ORDER BY id DESC LIMIT 1",
        (_USER,), fetch=True)
    fid_viejo = filas[0][0]

    # Se caduca -- semanas despues, por ejemplo, porque Fernando se mudo y
    # alguien lo marco vencido desde la pantalla de Memoria.
    await m.expire_fact(fid_viejo, datetime.now() - timedelta(days=1))

    # El extractor vuelve a producir el MISMO hecho en una conversacion
    # posterior (Fernando lo vuelve a mencionar).
    segundo = await m.save_fact(texto, "user", user_id=_USER)
    assert segundo is True, (
        "el hecho vencido actuo como duplicado activo y bloqueo la "
        "reinsercion -- el hecho nuevo se perdio en silencio y para siempre")

    filas = await _sql(
        "SELECT id FROM facts WHERE user_id=%s AND fact_text=%s",
        (_USER, texto), fetch=True)
    assert len(filas) == 2, (
        f"esperaba el hecho vencido MAS el reinsertado, hay {len(filas)}")


@requiere_db_de_prueba
@asincrono
async def test_un_hecho_vigente_SI_sigue_deduplicando(limpio, monkeypatch):
    """Control: sin caducar, el dedup sigue funcionando como siempre -- el
    arreglo no puede volverse 'nunca deduplicar'."""
    vec = _vec(0)
    m = await _memoria(monkeypatch, vec)
    texto = "Fernando vive en San Pedro Sula"

    primero = await m.save_fact(texto, "user", user_id=_USER)
    assert primero is True
    segundo = await m.save_fact(texto, "user", user_id=_USER)
    assert segundo is False, "sin caducar, el mismo texto sigue siendo duplicado"

    filas = await _sql(
        "SELECT id FROM facts WHERE user_id=%s AND fact_text=%s",
        (_USER, texto), fetch=True)
    assert len(filas) == 1


@requiere_db_de_prueba
@asincrono
async def test_expire_fact_con_none_devuelve_el_hecho_al_juego_del_dedup(limpio, monkeypatch):
    """Hueco de cobertura senalado en la auditoria adversarial 2026-09-20:
    es la justificacion del camino 5 de la lista de arriba ("si mas adelante
    alguien quita el vencimiento, el hecho tiene que volver a ser encontrable
    por busqueda semantica de inmediato") pero nadie lo probaba. Si
    `expire_fact(id, None)` no devolviera al hecho al filtro `expires_at IS
    NULL OR expires_at > NOW()` de `_find_nearest_fact`, "reactivar" un hecho
    desde la pantalla de Memoria seria un boton que no hace lo que promete."""
    vec = _vec(0)
    m = await _memoria(monkeypatch, vec)
    texto = "Fernando vive en Comayaguela"

    primero = await m.save_fact(texto, "user", user_id=_USER)
    assert primero is True
    filas = await _sql(
        "SELECT id FROM facts WHERE user_id=%s ORDER BY id DESC LIMIT 1",
        (_USER,), fetch=True)
    fid = filas[0][0]

    # Vencido: reextraer el mismo texto ya NO lo ve como duplicado (test de
    # arriba, con un hecho distinto).
    ok = await m.expire_fact(fid, datetime.now() - timedelta(days=1))
    assert ok is True

    # Se le QUITA el vencimiento (equivalente a lo que hace la pantalla de
    # Memoria al "reactivar" un hecho).
    quitado = await m.expire_fact(fid, None)
    assert quitado is True

    # Y tiene que volver a actuar como duplicado activo DE INMEDIATO, sin
    # depender de que nada mas corra (el backfill de embeddings, un
    # reindexado, etc.).
    tercero = await m.save_fact(texto, "user", user_id=_USER)
    assert tercero is False, (
        "expire_fact(id, None) no devolvio el hecho al juego del dedup -- "
        "'reactivar' un hecho desde la pantalla de Memoria no lo reactivo "
        "de verdad para la busqueda semantica")

    filas = await _sql(
        "SELECT id FROM facts WHERE user_id=%s AND fact_text=%s",
        (_USER, texto), fetch=True)
    assert len(filas) == 1, "el reactivado deberia seguir siendo el UNICO hecho con ese texto"


@requiere_db_de_prueba
@asincrono
async def test_correccion_sobre_un_candidato_vencido_no_lo_supersede_inserta_como_nuevo(
        limpio, monkeypatch):
    """Hueco de cobertura + DECISION de diseno (auditoria adversarial
    2026-09-20). Con el filtro nuevo, `_find_nearest_fact` ya no ve un
    candidato vencido -- ni para el camino normal (arriba) ni para
    `is_correction=True`. Consecuencia real: una "correccion" contra un
    hecho que ya vencio no encuentra nada que corregir, y `save_fact` la
    trata como un hecho NUEVO SIN RELACION (`band == "unrelated"`): no llama
    a `supersede_fact`, no revierte nada, no deja registro de que hubo un
    intento de correccion. La cadena de versiones (`superseded_by`) se
    "rompe" en el sentido de que el hecho nuevo no queda enlazado al viejo.

    EL CRITERIO (decision de esta ronda): es ACEPTABLE, a proposito. Un
    hecho vencido ya no es autoridad (docstring de `_find_nearest_fact`, mas
    arriba en este archivo) -- encadenarlo via `superseded_by` a una
    correccion nueva mezclaria dos conceptos distintos: "este hecho quedo
    OBSOLETO por el paso del tiempo" (expires_at) y "este hecho quedo
    REEMPLAZADO por otro mas preciso" (superseded_by). El hecho vencido
    sigue existiendo, sigue siendo visible con `incluir_vencidos=True` y
    `/fact delete`, y el hecho "corregido" queda como un registro nuevo,
    correcto e independiente: no hay perdida de informacion, solo una
    cadena de versiones que no se conecta con un eslabon que ya estaba fuera
    de vigencia. Es el MISMO criterio que
    `test_un_hecho_vencido_no_bloquea_la_reextraccion_del_mismo_hecho`
    (arriba): un hecho vencido no bloquea, pero tampoco encadena.

    Si este criterio cambia (por ejemplo, "una correccion SI tiene que
    poder atar un hecho nuevo a uno vencido"), lo que cambia es
    `_find_nearest_fact` -- una busqueda de candidato SEPARADA para
    `is_correction=True` que si mire vencidos, no volver a dejar de filtrar
    `expires_at` en general (eso reabre el agujero que el resto de este
    archivo cierra). No es parte de esta ronda: no hay pedido de producto
    para "corregir un hecho vencido" hoy."""
    vec = _vec(0)
    m = await _memoria(monkeypatch, vec)
    texto = "Fernando vive en Choluteca"

    original = await m.save_fact(texto, "user", user_id=_USER)
    assert original is True
    filas = await _sql(
        "SELECT id FROM facts WHERE user_id=%s ORDER BY id DESC LIMIT 1",
        (_USER,), fetch=True)
    fid_viejo = filas[0][0]
    ok = await m.expire_fact(fid_viejo, datetime.now() - timedelta(days=1))
    assert ok is True

    correccion = await m.save_fact(
        "Fernando se mudo de Choluteca", "user", user_id=_USER, is_correction=True)
    assert correccion is True, "la correccion tiene que insertarse igual, como hecho nuevo"

    filas = await _sql(
        "SELECT superseded_by FROM facts WHERE id=%s", (fid_viejo,), fetch=True)
    assert filas[0][0] is None, (
        "el hecho vencido quedo superseded por la 'correccion' -- eso "
        "contradice el criterio de esta ronda: un vencido no es candidato "
        "de correccion, no se encadena")

    filas = await _sql(
        "SELECT id FROM facts WHERE user_id=%s", (_USER,), fetch=True)
    assert len(filas) == 2, "el hecho vencido original MAS la correccion como hecho nuevo"


# ---------------------------------------------------------------------------
# 2. get_scopes_with_verified_facts: no cuenta vencidos
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_get_scopes_with_verified_facts_no_cuenta_vencidos(limpio):
    ids = []
    for i in range(5):
        ids.append(await _crear_fact(f"hecho verificado {i}", is_verified=True))
    # 3 de los 5 estan vencidos: solo quedan 2 verificados VIGENTES.
    for fid in ids[:3]:
        await _sql("UPDATE facts SET expires_at = %s WHERE id = %s",
                    (datetime.now() - timedelta(days=1), fid))

    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok

    scopes = await m.get_scopes_with_verified_facts(min_facts=5)
    assert not any(s["user_id"] == _USER for s in scopes), (
        "el scope aparecio con min_facts=5 contando 3 hechos VENCIDOS: solo "
        "hay 2 verificados vigentes, el sintetizador recibiria menos de lo "
        "que el conteo prometio")

    scopes = await m.get_scopes_with_verified_facts(min_facts=2)
    assert any(s["user_id"] == _USER and s["n_facts"] == 2 for s in scopes), (
        "con min_facts=2 el scope tiene que aparecer con EXACTAMENTE 2 "
        "(los vigentes), no 5")
