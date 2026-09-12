"""El scope de la busqueda semantica vive en `messages`, no detras de un JOIN.

POR QUE EXISTE
--------------
`messages` tiene `idx_embedding` VECTOR (HNSW). `search_similar_messages` NO lo
usaba: filtraba el scope (`user_id`/`project_id`) uniendo con `conversations`, y
ese JOIN saca al optimizador del indice vectorial. Medido el 2026-09-11 contra
`jax_memory` (1.149 filas):

    JOIN + scope + filtro anti-NaN ..... 58,5 ms   (Using temporary; filesort)
    lo mismo SIN el JOIN ...............  0,4 ms   (usa idx_embedding)

145x, y crece lineal porque un scan es O(n). El arreglo desnormaliza
`user_id`/`project_id` a `messages` para que el WHERE y el ORDER BY vivan en la
misma tabla que el indice.

POR QUE SE PUEDE DESNORMALIZAR SIN RIESGO DE DESINCRONIZACION
-------------------------------------------------------------
El scope de una conversacion es INMUTABLE: se fija en `start_conversation()` y
ningun `UPDATE conversations` del arbol lo toca (los cuatro que existen tocan
`ended_at`, `total_turns` y `memory_processed`). Verificado tambien en
jax-platform, que no escribe estas tablas: usa esta misma clase. Si algun dia
alguien hace mutable el scope, `test_el_scope_de_una_conversacion_es_inmutable`
se pone rojo y avisa que esta copia necesita mantenimiento.

SEGURIDAD
---------
Solo corre contra una base cuyo nombre termina en `_test`, y borra unicamente
lo que crea (cuelga de un user_id reservado propio).
"""
from __future__ import annotations

import asyncio
import functools
import os
import re
import uuid
from pathlib import Path

import aiomysql
import pytest

from jax.memory import db as dbmod
import _esquema_memoria

_DB = os.getenv("JAX_DB_NAME", "")
requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not _DB.endswith("_test"),
    reason="necesita una MariaDB real con JAX_DB_NAME terminado en '_test'",
)

_DIM = dbmod.EMBEDDING_DIM
_USER = 990_002        # reservado para este archivo
_OTRO_USER = 990_003   # el vecino: sus mensajes NUNCA pueden aparecer
_PROYECTO = 990_004


def _vec(pos: int, valor: float = 1.0) -> list[float]:
    v = [0.0] * _DIM
    v[pos] = valor
    return v


def _txt(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def asincrono(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


async def _conn():
    return await aiomysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.environ["JAX_DB_NAME"], autocommit=True)


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
    final. Igual que en test_memory_vector_zero_io.py: en hall9000 la base
    `jax_memory_test` la comparten varias suites."""
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
    for u in (_USER, _OTRO_USER):
        await _sql("DELETE FROM conversations WHERE user_id = %s", (u,))  # messages: ON DELETE CASCADE
    await _sql("DELETE FROM conversations WHERE project_id = %s", (_PROYECTO,))


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


async def _memoria(monkeypatch, vec_consulta: list[float]) -> dbmod.MemoryDB:
    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok, "MemoryDB no conecto: el test no probaria nada"

    async def _emb(_texto):
        return vec_consulta

    monkeypatch.setattr(m, "get_embedding", _emb)
    return m


# ---------------------------------------------------------------------------
# 1. La copia existe y la escribe el camino real
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_save_message_copia_el_scope_de_la_conversacion(limpio, monkeypatch):
    """El mensaje se guarda con el user_id/project_id de SU conversacion.

    Es la unica forma de que el WHERE del scope viva en la misma tabla que el
    indice vectorial. Si esto no pasa, la busqueda deja de encontrar los
    mensajes del propio usuario -- una falla silenciosa y total del recuerdo.
    """
    m = await _memoria(monkeypatch, _vec(0))
    conv = await m.start_conversation(source="test", user_id=_USER, project_id=_PROYECTO)
    assert conv
    await m.save_message(conv, "user", "hola desde el scope")

    filas = await _sql(
        "SELECT mm.user_id, mm.project_id FROM messages mm "
        "JOIN conversations c ON mm.conversation_id = c.id WHERE c.user_id = %s",
        (_USER,), fetch=True)
    assert filas, "no se guardo el mensaje"
    assert filas[0] == (_USER, _PROYECTO), (
        f"el mensaje no heredo el scope de su conversacion: {filas[0]}")


@requiere_db_de_prueba
@asincrono
async def test_el_scope_de_una_conversacion_es_inmutable(limpio):
    """Centinela de la premisa que hace segura la desnormalizacion.

    La copia en `messages` no se mantiene sincronizada con nada porque hoy NADA
    cambia el scope de una conversacion despues de crearla. Si alguien agrega
    ese UPDATE, este test se pone rojo -- y ese rojo es la instruccion de
    propagar el cambio a `messages`, no de borrar el test.
    """
    raiz = Path(__file__).resolve().parents[1]
    sospechosos = []
    for py in list(raiz.glob("jax/memory/*.py")) + list(raiz.glob("jax/**/*.py")):
        texto = py.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"UPDATE\s+conversations\s+SET\s+(.{0,120})", texto, re.I | re.S):
            fragmento = m.group(1)
            if re.search(r"\b(user_id|project_id)\s*=", fragmento, re.I):
                sospechosos.append(f"{py.relative_to(raiz)}: {fragmento[:80]!r}")
    assert sospechosos == [], (
        "alguien hizo mutable el scope de una conversacion:\n  " + "\n  ".join(sospechosos) +
        "\nLa copia desnormalizada en messages.user_id/project_id hay que "
        "actualizarla en el mismo UPDATE, o la busqueda devolvera el scope viejo.")


# ---------------------------------------------------------------------------
# 2. El plan: sin JOIN, con el indice vectorial
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_la_busqueda_usa_el_indice_vectorial_y_no_une_con_conversations(limpio, monkeypatch):
    """El EXPLAIN de la consulta REAL tiene que nombrar el indice vectorial.

    Se afirma sobre el PLAN y no sobre el reloj: a la escala de una base de
    test cualquier consulta es rapida, y un test de milisegundos seria un flake
    que ademas no dice por que. El plan es la afirmacion honesta.
    """
    m = await _memoria(monkeypatch, _vec(0))
    conv = await m.start_conversation(source="test", user_id=_USER)
    await m.save_message(conv, "user", "una fila para que la tabla no este vacia")

    consultas = []
    original = dbmod.aiomysql.cursors.DictCursor.execute

    async def espia(self, query, args=None):
        if "VEC_DISTANCE_COSINE" in (query or ""):
            consultas.append((query, args))
        return await original(self, query, args)

    monkeypatch.setattr(dbmod.aiomysql.cursors.DictCursor, "execute", espia)
    await m.search_similar_messages("hola", limit=3, user_id=_USER)
    assert consultas, "no se capturo la consulta de busqueda"

    query, args = consultas[0]
    assert "JOIN conversations" not in query, (
        "la busqueda sigue uniendo con conversations: ese JOIN es lo que saca "
        "al optimizador del indice vectorial (58,5 ms contra 0,4 ms medidos)")

    plan = await _sql("EXPLAIN " + query, args or (), fetch=True)
    texto = " | ".join(str(c) for fila in plan for c in fila)
    assert "idx_embedding" in texto, f"el plan no usa el indice vectorial: {texto}"
    assert "conversations" not in texto, f"el plan todavia toca conversations: {texto}"


# ---------------------------------------------------------------------------
# 3. Lo que NO puede cambiar: el aislamiento
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_no_devuelve_mensajes_de_otro_usuario(limpio, monkeypatch):
    """Optimizar no puede ampliar lo que se ve. Es la misma clase de fuga que
    ya costo un incidente (memoria jax-platform-session-leak)."""
    m = await _memoria(monkeypatch, _vec(0))

    ajena = await m.start_conversation(source="test", user_id=_OTRO_USER)
    await m.save_message(ajena, "user", "SECRETO DEL VECINO")
    propia = await m.start_conversation(source="test", user_id=_USER)
    await m.save_message(propia, "user", "cosa mia")

    filas = await m.search_similar_messages("hola", limit=10, user_id=_USER)
    contenidos = [f["content"] for f in filas]
    assert "SECRETO DEL VECINO" not in contenidos, (
        f"fuga de scope: la busqueda devolvio mensajes de otro usuario: {contenidos}")


@requiere_db_de_prueba
@asincrono
async def test_encuentra_el_mensaje_del_proyecto_compartido(limpio, monkeypatch):
    """El otro lado del scope: la memoria de proyecto tiene que seguir
    visible. Un aislamiento que tambien esconde lo propio no es aislamiento,
    es una busqueda rota."""
    m = await _memoria(monkeypatch, _vec(0))
    conv = await m.start_conversation(source="test", user_id=_OTRO_USER, project_id=_PROYECTO)
    await m.save_message(conv, "user", "dato del proyecto")

    filas = await m.search_similar_messages("hola", limit=10, project_id=_PROYECTO)
    assert any(f["content"] == "dato del proyecto" for f in filas), (
        f"no encontro la memoria del proyecto compartido: {[f['content'] for f in filas]}")


# ---------------------------------------------------------------------------
# 4. La calidad de la aproximacion no puede degradarse en silencio
# ---------------------------------------------------------------------------

@requiere_db_de_prueba
@asincrono
async def test_la_conexion_fija_ef_search_por_encima_del_default_de_mariadb(limpio, monkeypatch):
    """Desde que la busqueda usa el indice HNSW, el resultado es APROXIMADO, y
    cuanto se aproxima lo decide `mhnsw_ef_search`.

    El default de MariaDB es 20, y con 20 el recall@5 medido contra la busqueda
    exacta fue **50,7 %**: la mitad de los recuerdos mas parecidos no aparecen.
    No da error, no aparece en ningun log: JAX simplemente recuerda peor. Con
    400 sube a 93,3 % y cuesta 0,92 ms contra los 49 ms de la exacta.

    Este test es el unico aviso posible ante esa degradacion silenciosa: si
    alguien saca el `init_command`, el `mhnsw_ef_search` de la sesion vuelve a
    20 y esto se pone rojo.
    """
    m = await _memoria(monkeypatch, _vec(0))
    async with m.pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT @@SESSION.mhnsw_ef_search")
            (ef,) = await cur.fetchone()
    esperado = int(os.getenv("JAX_MEMORY_HNSW_EF_SEARCH", "400"))
    assert ef == esperado, (
        f"la conexion quedo con mhnsw_ef_search={ef}, se esperaba {esperado}. "
        "Con el default de MariaDB (20) el recall@5 medido fue 50,7 %: la "
        "busqueda semantica pierde la mitad de los vecinos y no avisa."
    )
    assert ef > 20, "20 es el default de MariaDB, medido en 50,7 % de recall"
