"""Busqueda vectorial de la memoria contra filas con embedding "vector cero",
contra una MariaDB REAL.

POR QUE EXISTE
--------------
`messages.embedding` y `facts.embedding` son `VECTOR(768) NOT NULL DEFAULT`
vector cero en produccion: el `VECTOR KEY` exige NOT NULL, asi que una fila
sin embedding real no queda en NULL sino en ceros. Dos caminos las producen:

  1. **Permanentes**: `save_message` inserta y DESPUES pide el embedding a
     Ollama; si Ollama falla, la fila se queda en ceros para siempre.
     Medido 2026-09-11: 24 filas asi en `jax_memory`, todas del 2026-06-09.
  2. **Transitorias**: el turno en curso. `/api/chat` guarda el mensaje del
     usuario fire-and-forget y en paralelo corre la busqueda semantica: la
     fila recien insertada todavia tiene el default.

`VEC_DISTANCE_COSINE` contra un vector de norma cero da NaN (medido: la
distancia no es NULL para el servidor y no es igual a si misma), aiomysql
la entrega como `None`, y como las comparaciones con NaN son siempre falsas
el `ORDER BY ... ASC` las ubica en cualquier lado -- en produccion, primero.
Medido 2026-09-11 por el camino real: `search_similar_messages("hola",
user_id=1)` devolvia 5 filas, las 5 con `distancia=None`, y
`api/chat.py::_semantic_context` hacia `None < 0.8` -> TypeError -> el turno
entero de chat en 500, en cualquier faceta, para todo el scope individual.

POR QUE EL ROJO ES DETERMINISTICO
---------------------------------
Con NaN el orden es indefinido, asi que un test con muchas filas podia pasar
por suerte. Cada caso inserta pocas filas y pide un `limit` que las cubre a
todas (o, en facts, SOLO filas cero): el resultado no depende de donde caen
los NaN.

SEGURIDAD
---------
Solo corre contra una base cuyo nombre termina en `_test`. Crea las tablas
solo si no existen y al final borra unicamente lo que creo: sus filas (por un
user_id reservado) y las tablas que no estaban. En hall9000 `jax_memory_test`
la comparte la suite de jax-platform.

El DDL es el de PRODUCCION (`SHOW CREATE TABLE`, 2026-09-11), no el de
`jax_memory_schema.sql`: ese archivo esta desactualizado (sin
`conversations.user_id/project_id`, que el scope de la busqueda necesita, y
con `embedding NULL` sin default) -- anotado en DEUDA.md.
"""
import asyncio
import functools
import math
import os
import uuid

import aiomysql
import pytest

from jax.memory import db as dbmod

_DB = os.getenv("JAX_DB_NAME", "")
requiere_db_de_prueba = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST") or not _DB.endswith("_test"),
    reason="necesita una MariaDB real con JAX_DB_NAME terminado en '_test'",
)

_DIM = dbmod.EMBEDDING_DIM
_CERO = "[" + ",".join(["0.0"] * _DIM) + "]"
# Usuario reservado para este archivo: todo lo que inserta cuelga de el y es
# lo unico que borra al terminar.
_USER = 990_001


def _vec(*pares: tuple[int, float]) -> list[float]:
    """Vector de _DIM dimensiones con los valores dados en esas posiciones."""
    v = [0.0] * _DIM
    for i, x in pares:
        v[i] = x
    return v


def _txt(v: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


# Orden de creacion (facts depende de messages, messages de conversations).
_DDL = {
    "conversations": """
        CREATE TABLE conversations (
          id int(11) NOT NULL AUTO_INCREMENT,
          conversation_uuid char(36) NOT NULL,
          started_at timestamp NULL DEFAULT current_timestamp(),
          ended_at timestamp NULL DEFAULT NULL,
          total_turns int(11) DEFAULT 0,
          source varchar(20) DEFAULT 'terminal',
          tenant_id int(11) DEFAULT NULL,
          user_id int(11) DEFAULT NULL,
          project_id int(11) DEFAULT NULL,
          memory_processed tinyint(1) DEFAULT 0,
          memory_processed_at timestamp NULL DEFAULT NULL,
          PRIMARY KEY (id),
          UNIQUE KEY conversation_uuid (conversation_uuid),
          KEY idx_conv_user (user_id),
          KEY idx_conv_project (project_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    "messages": f"""
        CREATE TABLE messages (
          id int(11) NOT NULL AUTO_INCREMENT,
          conversation_id int(11) NOT NULL,
          turn_number int(11) NOT NULL,
          role enum('user','jax_local','jekyll','hyde','hipatia','thot','kimi','ada') NOT NULL,
          content text NOT NULL,
          facet_used varchar(20) DEFAULT NULL,
          model varchar(50) DEFAULT NULL,
          latency_ms int(11) DEFAULT NULL,
          embedding vector({_DIM}) NOT NULL DEFAULT VEC_FromText('{_CERO}'),
          created_at timestamp NULL DEFAULT current_timestamp(),
          PRIMARY KEY (id),
          KEY idx_conversation (conversation_id),
          VECTOR KEY idx_embedding (embedding) DISTANCE=cosine,
          CONSTRAINT messages_ibfk_1 FOREIGN KEY (conversation_id)
            REFERENCES conversations (id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    "facts": f"""
        CREATE TABLE facts (
          id int(11) NOT NULL AUTO_INCREMENT,
          fact_uuid char(36) NOT NULL,
          fact_text text NOT NULL,
          fact_type enum('user','technical','social','preference','project','financial') NOT NULL,
          confidence float DEFAULT 1,
          source_message_id int(11) DEFAULT NULL,
          source_facet varchar(20) DEFAULT NULL,
          is_verified tinyint(1) DEFAULT 0,
          verified_at timestamp NULL DEFAULT NULL,
          expires_at timestamp NULL DEFAULT NULL,
          embedding vector({_DIM}) NOT NULL DEFAULT VEC_FromText('{_CERO}'),
          created_at timestamp NULL DEFAULT current_timestamp(),
          updated_at timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
          user_id int(11) DEFAULT NULL,
          project_id int(11) DEFAULT NULL,
          superseded_by int(11) DEFAULT NULL,
          superseded_at timestamp NULL DEFAULT NULL,
          source_fact_ids longtext DEFAULT NULL,
          importance tinyint(4) DEFAULT NULL,
          PRIMARY KEY (id),
          UNIQUE KEY fact_uuid (fact_uuid),
          KEY idx_facts_user (user_id),
          VECTOR KEY idx_embedding (embedding) DISTANCE=cosine,
          CONSTRAINT facts_ibfk_1 FOREIGN KEY (source_message_id)
            REFERENCES messages (id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
}


def asincrono(fn):
    """Corre el test en un event loop propio (sin pytest-asyncio, mismo
    criterio que jacobs/_facet_health_io_test.py)."""
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


async def _preparar() -> list[str]:
    creadas = []
    for nombre, ddl in _DDL.items():
        existe = await _sql(
            "SELECT 1 FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (nombre,), fetch=True)
        if not existe:
            await _sql(ddl)
            creadas.append(nombre)
    await _limpiar_filas()
    return creadas


async def _limpiar_filas():
    await _sql("DELETE FROM facts WHERE user_id = %s", (_USER,))
    # messages cae por ON DELETE CASCADE.
    await _sql("DELETE FROM conversations WHERE user_id = %s", (_USER,))


@pytest.fixture
def tablas():
    creadas = asyncio.run(_preparar())
    yield
    async def _teardown():
        await _limpiar_filas()
        for nombre in reversed(list(_DDL)):
            if nombre in creadas:
                await _sql(f"DROP TABLE {nombre}")
    asyncio.run(_teardown())


async def _conversacion() -> int:
    return await _sql(
        "INSERT INTO conversations (conversation_uuid, user_id, project_id) "
        "VALUES (%s, %s, NULL)", (str(uuid.uuid4()), _USER))


async def _mensaje(conv: int, turno: int, contenido: str, vec: list[float] | None):
    """vec=None inserta SIN la columna embedding: el default, exactamente lo
    que deja save_message() hasta que llega (si llega) el embedding."""
    if vec is None:
        await _sql(
            "INSERT INTO messages (conversation_id, turn_number, role, content) "
            "VALUES (%s, %s, 'user', %s)", (conv, turno, contenido))
    else:
        await _sql(
            "INSERT INTO messages (conversation_id, turn_number, role, content, embedding) "
            "VALUES (%s, %s, 'user', %s, VEC_FromText(%s))", (conv, turno, contenido, _txt(vec)))


async def _fact(texto: str, vec: list[float]):
    await _sql(
        "INSERT INTO facts (fact_uuid, fact_text, fact_type, user_id, embedding) "
        "VALUES (%s, %s, 'user', %s, VEC_FromText(%s))",
        (str(uuid.uuid4()), texto, _USER, _txt(vec)))


async def _memoria(monkeypatch, vec_consulta: list[float]) -> dbmod.MemoryDB:
    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok, "MemoryDB no conecto: el test no probaria nada"

    async def _emb(_texto):
        return vec_consulta

    # Sin Ollama: el embedding de la consulta es un dato del test.
    monkeypatch.setattr(m, "get_embedding", _emb)
    return m


async def _cerrar(m: dbmod.MemoryDB):
    m.pool.close()
    await m.pool.wait_closed()


def _finita(d) -> bool:
    return isinstance(d, float) and math.isfinite(d)


@requiere_db_de_prueba
@asincrono
async def test_la_busqueda_excluye_mensajes_con_vector_cero(tablas, monkeypatch):
    q = _vec((0, 1.0))
    conv = await _conversacion()
    await _mensaje(conv, 1, "cero permanente A", _vec())
    await _mensaje(conv, 2, "cero permanente B", _vec())
    await _mensaje(conv, 3, "recien guardado, embedding todavia en default", None)
    await _mensaje(conv, 4, "mensaje real", q)

    m = await _memoria(monkeypatch, q)
    try:
        filas = await m.search_similar_messages("hola", limit=5, user_id=_USER)
    finally:
        await _cerrar(m)

    assert all(_finita(f["distancia"]) for f in filas), filas
    # Contrapositivo: la fila real SIGUE volviendo -- un arreglo que
    # devolviera siempre [] no pasa.
    assert [f["content"] for f in filas] == ["mensaje real"], filas
    assert filas[0]["distancia"] == pytest.approx(0.0, abs=1e-6)


@requiere_db_de_prueba
@asincrono
async def test_la_busqueda_con_consulta_degenerada_no_devuelve_distancias_no_finitas(
        tablas, monkeypatch):
    """Un embedding de consulta de norma cero (Ollama devolviendo ceros) da
    NaN contra TODAS las filas, incluidas las sanas: ninguna distancia se
    puede usar, y devolverlas es entregarle `None` al consumidor."""
    conv = await _conversacion()
    await _mensaje(conv, 1, "mensaje real", _vec((0, 1.0)))

    m = await _memoria(monkeypatch, _vec())
    try:
        filas = await m.search_similar_messages("hola", limit=5, user_id=_USER)
    finally:
        await _cerrar(m)

    assert filas == [], filas


@requiere_db_de_prueba
@asincrono
async def test_nearest_fact_sin_candidatos_reales_devuelve_none(tablas, monkeypatch):
    """Solo hay un fact con vector cero: no es un candidato. Devolverlo
    (con distancia None) le haria creer a add_fact que existe un duplicado
    o una correccion posible."""
    await _fact("fact con vector cero", _vec())

    m = await _memoria(monkeypatch, _vec((0, 1.0)))
    try:
        cand = await m._find_nearest_fact(_vec((0, 1.0)), _USER, None)
    finally:
        await _cerrar(m)

    assert cand is None, cand


@requiere_db_de_prueba
@asincrono
async def test_nearest_fact_elige_el_real_aunque_haya_ceros(tablas, monkeypatch):
    await _fact("fact con vector cero", _vec())
    await _fact("fact real", _vec((0, 1.0), (1, 1.0)))

    m = await _memoria(monkeypatch, _vec((0, 1.0)))
    try:
        cand = await m._find_nearest_fact(_vec((0, 1.0)), _USER, None)
    finally:
        await _cerrar(m)

    assert cand is not None and cand["fact_text"] == "fact real", cand
    assert _finita(cand["distancia"]), cand


# ---------------------------------------------------------------------------
# backfill_zero_embeddings: el reintento que save_message()/add_fact() no
# hacen. Sin el, una fila que nace en ceros (Ollama caido al guardar) queda
# excluida de toda busqueda para siempre, en silencio.
#
# Los contadores asumen que en estas tablas solo hay filas de este archivo:
# en CI la base nace vacia, y en hall9000 las tablas las crea y las borra el
# fixture (jax_memory_test no las tiene).
# ---------------------------------------------------------------------------

async def _es_cero(tabla: str, texto_col: str, texto: str) -> bool:
    filas = await _sql(
        f"SELECT VEC_DISTANCE_EUCLIDEAN(embedding, VEC_FromText(%s)) = 0 "
        f"FROM {tabla} WHERE {texto_col} = %s", (_CERO, texto), fetch=True)
    assert len(filas) == 1, (tabla, texto, filas)
    return bool(filas[0][0])


async def _embedding_texto(tabla: str, texto_col: str, texto: str) -> str:
    filas = await _sql(
        f"SELECT VEC_ToText(embedding) FROM {tabla} WHERE {texto_col} = %s",
        (texto,), fetch=True)
    return filas[0][0]


@requiere_db_de_prueba
@asincrono
async def test_backfill_repara_mensajes_y_facts_en_cero_sin_tocar_los_reales(
        tablas, monkeypatch):
    conv = await _conversacion()
    await _mensaje(conv, 1, "mensaje en cero", None)
    await _mensaje(conv, 2, "mensaje real", _vec((0, 1.0)))
    await _fact("fact en cero", _vec())
    await _fact("fact real", _vec((1, 1.0)))
    antes_msg = await _embedding_texto("messages", "content", "mensaje real")
    antes_fact = await _embedding_texto("facts", "fact_text", "fact real")

    m = await _memoria(monkeypatch, _vec((7, 1.0)))
    try:
        r_msg = await m.backfill_zero_embeddings("messages", limit=50)
        r_fact = await m.backfill_zero_embeddings("facts", limit=50)
    finally:
        await _cerrar(m)

    assert r_msg == {"pendientes": 1, "reparadas": 1, "fallidas": 0}, r_msg
    assert r_fact == {"pendientes": 1, "reparadas": 1, "fallidas": 0}, r_fact
    assert not await _es_cero("messages", "content", "mensaje en cero")
    assert not await _es_cero("facts", "fact_text", "fact en cero")
    # Las filas sanas no se tocan: el UPDATE esta guardado por "sigue en ceros".
    assert await _embedding_texto("messages", "content", "mensaje real") == antes_msg
    assert await _embedding_texto("facts", "fact_text", "fact real") == antes_fact


@requiere_db_de_prueba
@asincrono
async def test_backfill_con_ollama_caido_deja_la_fila_y_la_repara_en_la_proxima(
        tablas, monkeypatch):
    """Ollama caido: la fila se queda en ceros (no se inventa un vector) y se
    cuenta como fallida; la corrida siguiente, con Ollama de vuelta, la
    repara. Es exactamente el ciclo que el worker cada 20 min va a repetir."""
    conv = await _conversacion()
    await _mensaje(conv, 1, "mensaje en cero", None)

    m = await _memoria(monkeypatch, _vec((7, 1.0)))
    try:
        async def _caido(_texto):
            return None

        monkeypatch.setattr(m, "get_embedding", _caido)
        r1 = await m.backfill_zero_embeddings("messages", limit=50)
        assert r1 == {"pendientes": 1, "reparadas": 0, "fallidas": 1}, r1
        assert await _es_cero("messages", "content", "mensaje en cero")

        async def _vuelve(_texto):
            return _vec((7, 1.0))

        monkeypatch.setattr(m, "get_embedding", _vuelve)
        r2 = await m.backfill_zero_embeddings("messages", limit=50)
    finally:
        await _cerrar(m)

    assert r2 == {"pendientes": 1, "reparadas": 1, "fallidas": 0}, r2
    assert not await _es_cero("messages", "content", "mensaje en cero")


@requiere_db_de_prueba
@asincrono
async def test_backfill_respeta_el_limite_por_corrida(tablas, monkeypatch):
    conv = await _conversacion()
    for i in range(3):
        await _mensaje(conv, i + 1, f"mensaje en cero {i}", None)

    m = await _memoria(monkeypatch, _vec((7, 1.0)))
    try:
        r1 = await m.backfill_zero_embeddings("messages", limit=2)
        r2 = await m.backfill_zero_embeddings("messages", limit=2)
    finally:
        await _cerrar(m)

    assert r1 == {"pendientes": 2, "reparadas": 2, "fallidas": 0}, r1
    assert r2 == {"pendientes": 1, "reparadas": 1, "fallidas": 0}, r2


def test_backfill_rechaza_una_tabla_no_permitida():
    """La tabla se interpola en el SQL: solo las dos con embedding real, por
    lista cerrada. Sin DB -- se rechaza antes de tocar el pool."""
    m = dbmod.MemoryDB()
    with pytest.raises(ValueError):
        asyncio.run(m.backfill_zero_embeddings("messages; DROP TABLE facts", limit=1))
