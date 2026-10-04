# tests/test_save_message_deadlock.py
"""Deadlocks 1213 en save_message: ningun mensaje se pierde por un deadlock.

DIAGNOSTICO 2026-10-04 (rama fix/deadlocks-save-message), reproducido contra
una base aislada clonada de jax_memory_test (MariaDB 12.3.3, REPEATABLE-READ,
pool autocommit): 12 workers x 200 mensajes sobre 4 conversaciones
compartidas dieron 1563 deadlocks de 2400 intentos, turn_numbers duplicados
y mensajes PERDIDOS EN SILENCIO (el @db_error_handler traga el 1213 y el
guardado es fire-and-forget).

Grafo capturado (LATEST DETECTED DEADLOCK, SHOW ENGINE INNODB STATUS):

  T1 (paso 5, `UPDATE messages SET embedding... WHERE conversation_id AND
      turn_number`): posee una capa del INDICE VECTORIAL HNSW (tabla
      interna `messages#i#NN`) y espera el lock X del PRIMARY de una fila
      ajena. Medido con 1272 row locks y 35 undo entries: el WHERE matcheaba
      ~35 filas con el MISMO turn_number, duplicado por la carrera MAX+1.
  T2 (paso 3, `INSERT INTO messages`): posee su fila nueva (vector default
      en CEROS, que cae siempre en la misma zona del grafo HNSW) y espera
      justo la capa del indice que tiene T1.

  Ciclo: T1 espera la fila de T2  <->  T2 espera la capa vectorial de T1.

Causa raiz (dos condiciones que se potencian):
  1. Carrera MAX+1 sin UNIQUE(conversation_id, turn_number): dos writers
     concurrentes calculan el mismo turn_number; el UPDATE de embedding de
     uno bloquea las filas del otro, cerrando el ciclo con el indice HNSW.
     Colateral medido: turnos duplicados y el embedding de un mensaje
     pisando a los demas del mismo turn.
  2. INSERT (con embedding default) y UPDATE de embedding (reescritura del
     grafo) en transacciones distintas: el indice HNSW es un recurso
     compartido entre ambas sentencias, y con vectores en cero todas las
     inserciones nuevas compiten por la misma zona del grafo.

Ronda 2 (misma sesion, tras atomizar el guardado): el candado por
conversacion tampoco basta. Con el turn bajo FOR UPDATE del rango, el
next-key lock ataba la FRONTERA entre conversaciones adyacentes en el
indice (writer conv 6 vs writer conv 5, record frontera). Se movio el
candado a la FILA de la conversacion.

Ronda 3 (misma sesion): el indice HNSW ES el recurso compartido final.
Todo INSERT escribe el vector default en ceros y todas esas entradas
cayeron en el supremo de la capa (tabla interna messages#i#06): INSERT
(conv 1) vs INSERT (conv 2) interbloqueados aun sin compartir conversacion
NI candado, con 1020 ("record changed since last read; try restarting
transaction") escapando al caller. MariaDB exige la columna del VECTOR KEY
NOT NULL: no existe default NULL que diferia el indice. Arreglo: mutex
GET_LOCK por base de datos que serializa TODOS los escritores del indice
(INSERT, vectorizacion por id y backfill); 1020 pasa al set reintentable.

Arreglo (ver jax/memory/db.py::_insertar_turno_atomico): transaccion corta
BEGIN...COMMIT con el turn asignado bajo la fila de la conversacion
(FOR UPDATE; serializa por conversacion), UNIQUE KEY como respaldo de la
carrera, mutex GET_LOCK del indice HNSW (un solo escritor a la vez, el
ciclo es imposible por construccion), reintento acotado e idempotente ante
1213/1205/1062/1020 (cada intento recalcula desde cero; el rollback de la
transaccion muerta garantiza que el mismo mensaje no se guarda dos veces)
y vectorizacion por id de fila (una sola fila, sin rango).
"""
from __future__ import annotations

import asyncio
import functools
import os

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

_USER = 990_060  # reservado para este archivo


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


async def _sql(conn, query, args=(), fetch=False):
    async with conn.cursor() as cur:
        await cur.execute(query, args)
        if fetch:
            return list(await cur.fetchall())
        return cur.lastrowid


_DDL = _esquema_memoria.ddl()


@pytest.fixture
def limpio():
    creadas = asyncio.run(_preparar())
    asyncio.run(_vaciar())
    yield
    asyncio.run(_teardown(creadas))


async def _preparar() -> list[str]:
    creadas = []
    conn = await _conn()
    try:
        for nombre, ddl in _DDL.items():
            existe = await _sql(
                conn,
                "SELECT 1 FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
                (nombre,), fetch=True)
            if not existe:
                await _sql(conn, ddl)
                creadas.append(nombre)
    finally:
        conn.close()
    return creadas


async def _vaciar():
    conn = await _conn()
    try:
        base = os.environ["JAX_DB_NAME"]
        assert es_base_de_test(base), f"me negue a vaciar tablas en {base!r}"
        await _esquema_memoria.vaciar(conn)
    finally:
        conn.close()


async def _teardown(creadas: list[str]):
    await _vaciar()
    conn = await _conn()
    try:
        for nombre in reversed(list(_DDL)):
            if nombre in creadas:
                await _sql(conn, f"DROP TABLE {nombre}")
    finally:
        conn.close()


async def _memoria(monkeypatch) -> dbmod.MemoryDB:
    m = dbmod.MemoryDB()
    ok = await m.connect(
        os.environ["JAX_DB_HOST"], os.getenv("JAX_DB_USER", ""),
        os.getenv("JAX_DB_PASSWORD", ""), os.environ["JAX_DB_NAME"],
        port=int(os.environ["JAX_DB_PORT"]))
    assert ok, "MemoryDB no conecto: el test no probaria nada"

    async def _emb(texto):
        # deterministico y distinto por mensaje: no todos los vectores nuevos
        # caen en el mismo punto del grafo HNSW
        v = [0.0] * dbmod.EMBEDDING_DIM
        h = abs(hash(texto)) % (10 ** 6)
        v[0] = (h % 997) / 997.0
        v[1] = ((h // 997) % 991) / 991.0
        v[2] = 1.0
        return v

    monkeypatch.setattr(m, "get_embedding", _emb)
    return m


@asincrono
@requiere_db_de_prueba
async def test_la_carga_concurrente_no_pierde_ni_duplica(limpio, monkeypatch):
    """La prueba de carga del bug: muchos writers sobre las mismas
    conversaciones. ANTES del arreglo reproduce el deadlock 1213 y pierde
    mensajes en silencio (medido: 1563/2400 intentos en deadlock); DESPUES
    todos los mensajes quedan guardados exactamente una vez, sin turnos
    duplicados y con el contador consistente."""

    db = await _memoria(monkeypatch)
    conn = await _conn()
    try:
        convs = [await db.start_conversation(source="test", user_id=_USER)
                 for _ in range(4)]
        workers, por_worker = 10, 120

        async def worker(w):
            for i in range(por_worker):
                conv = convs[(w * 7 + i) % len(convs)]
                task = db.save_message(conv, "user", f"w{w}-{i}", "kimi",
                                       "kimi-k3", 5)
                r = await task
                assert r is not None, (
                    f"mensaje w{w}-{i} perdido: save_message devolvio None "
                    f"(deadlock sin reintento efectivo)")

        await asyncio.gather(*[worker(w) for w in range(workers)])
        await db.close()

        esperados = workers * por_worker
        total = (await _sql(conn, "SELECT COUNT(*) FROM messages", fetch=True))[0][0]
        distintos = (await _sql(conn, "SELECT COUNT(DISTINCT content) FROM messages", fetch=True))[0][0]
        duplicados = await _sql(
            conn,
            "SELECT conversation_id, turn_number, COUNT(*) FROM messages "
            "GROUP BY 1, 2 HAVING COUNT(*) > 1", fetch=True)
        marcador = ",".join(["%s"] * len(convs))
        contador = (await _sql(
            conn,
            f"SELECT COALESCE(SUM(total_turns), 0) FROM conversations "
            f"WHERE conversation_uuid IN ({marcador})", tuple(convs),
            fetch=True))[0][0]

        assert total == esperados, (
            f"se perdieron mensajes: {total} filas de {esperados} enviados "
            f"(deadlocks 1213 tragados por db_error_handler)")
        assert distintos == esperados, (
            f"hay {total - distintos} mensajes duplicados en contenido")
        assert duplicados == [], (
            f"turn_numbers duplicados (carrera MAX+1): {duplicados[:5]}")
        assert contador == esperados, (
            f"total_turns={contador} != {esperados}: contador desincronizado")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reintento idempotente ante 1213: con mocks, sin base de datos
# ---------------------------------------------------------------------------

class _CursorTramposo:
    """Cursor que levanta un error (default 1213) en la ejecucion numero
    `fallo_en` (1-based) de SU cursor. Registra todo lo ejecutado."""

    def __init__(self, conn, fallo_en=None, codigo=1213):
        self.conn = conn
        self.fallo_en = fallo_en
        self.codigo = codigo
        self.n = 0
        self.sql = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, sql, args=()):
        self.n += 1
        self.conn.ejecutados.append((sql, args))
        if self.fallo_en is not None and self.n == self.fallo_en:
            self.fallo_en = None
            self.conn._fallo_en = None  # el fallo de una sola vez NO se re-arma en el cursor del reintento
            raise aiomysql.OperationalError(self.codigo, "error inyectado")

    async def fetchone(self):
        for sql, _ in reversed(self.conn.ejecutados):
            if "GET_LOCK" in sql:
                return (1,)  # el mutex siempre se adquiere en los mocks
            if "DATABASE()" in sql:
                return ("jax_memory_test_mock",)
            if "MAX(turn_number)" in sql:
                return (3,)
            if "FROM conversations" in sql:
                return (5, _USER, None)
        return None

    @property
    def lastrowid(self):
        return 42


class _ConnTramposo:
    """Secuencia de executes por cursor dentro de _insertar_turno_atomico:
    1 = SELECT DATABASE(), 2 = SELECT GET_LOCK (mutex HNSW), 3 = candado de
    la fila de la conversacion, 4 = SELECT MAX, 5 = INSERT, 6 = UPDATE del
    contador."""

    def __init__(self, fallo_en=None, codigo=1213):
        self.ejecutados = []
        self.begins = 0
        self.commits = 0
        self.rollbacks = 0
        self._fallo_en = fallo_en
        self._codigo = codigo

    async def begin(self):
        self.begins += 1

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1

    def cursor(self):
        return _CursorTramposo(self, self._fallo_en, self._codigo)


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_):
        return False


class _PoolTramposo:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


def _db_con(con):
    db = dbmod.MemoryDB()
    db.pool = _PoolTramposo(con)
    return db


async def _sin_embedding(_texto):
    return None


@asincrono
async def test_reintento_1213_guarda_exactamente_una_vez(monkeypatch):
    """El INSERT muere una vez con 1213: la transaccion muerta se revierte,
    se reintenta y el mensaje queda guardado UNA vez (idempotencia del
    reintento: sin el rollback, el segundo intento duplicaria la fila)."""
    con = _ConnTramposo(fallo_en=5)  # dentro del 1er intento: muere el INSERT
    db = _db_con(con)
    monkeypatch.setattr(db, "get_embedding", _sin_embedding)

    r = await db._save_message_impl("uuid-x", "user", "hola", None, None, None)

    assert r == {"conversation_id": 5, "turn_number": 3}
    inserts = [s for s, _ in con.ejecutados if s.startswith("INSERT INTO messages")]
    assert len(inserts) == 2, "se esperaba el INSERT fallido + el del reintento"
    assert con.begins == 2 and con.commits == 1 and con.rollbacks == 1, (
        f"begins={con.begins} commits={con.commits} rollbacks={con.rollbacks}: "
        "la transaccion del deadlock no se revierte completa")
    # GET_LOCK no es transaccional: sobrevive al rollback de la victima.
    # Sin RELEASE_LOCK tras revertir, el mutex quedaria tomado para todo el
    # proceso (hasta cerrar la conexion del pool).
    liberaciones = [s for s, _ in con.ejecutados if "RELEASE_LOCK" in s]
    assert len(liberaciones) == 2, (
        f"RELEASE_LOCK ejecutado {len(liberaciones)} veces (esperadas 2, "
        "una por intento): el mutex del intento fallido quedo tomado")


@asincrono
async def test_si_1213_persiste_devuelve_None_sin_duplicar(monkeypatch):
    """Tres intentos, tres 1213: falla de verdad (None), pero cada intento
    revierte su INSERT parcial -- ninguna fila queda a medias."""
    class _ConnPersistente(_ConnTramposo):
        async def begin(self):
            self.begins += 1
            self._fallo_en = 5  # el INSERT muere SIEMPRE

    con = _ConnPersistente(codigo=1213)
    db = _db_con(con)
    monkeypatch.setattr(db, "get_embedding", _sin_embedding)

    r = await db._save_message_impl("uuid-x", "user", "hola", None, None, None)

    assert r is None
    assert con.rollbacks == 3 and con.commits == 0, (
        f"rollbacks={con.rollbacks} commits={con.commits}: un intento sin "
        "revertir dejaria la fila INSERTada a pesar del fallo")
    inserts = [s for s, _ in con.ejecutados if s.startswith("INSERT INTO messages")]
    assert len(inserts) == 3, "los 3 intentos llegaron al INSERT (y revierten)"


@asincrono
async def test_el_insert_y_el_contador_son_una_sola_transaccion(monkeypatch):
    """Si el UPDATE del contador muere, el INSERT del mensaje se revierte
    tambien: los dos siempre avanzan juntos o ninguno. Y la vectorizacion no
    corre si el mensaje no quedo guardado."""
    class _CursorContadorTramposo(_CursorTramposo):
        async def execute(self, sql, args=()):
            self.n += 1
            self.conn.ejecutados.append((sql, args))
            if "UPDATE conversations" in sql:
                raise aiomysql.OperationalError(1205, "Lock wait timeout")

    class _Conn2(_ConnTramposo):
        def cursor(self):
            return _CursorContadorTramposo(self)

    con = _Conn2()
    db = _db_con(con)
    monkeypatch.setattr(db, "get_embedding", _sin_embedding)

    r = await db._save_message_impl("uuid-x", "user", "hola", None, None, None)

    assert r is None
    assert con.rollbacks == 3 and con.commits == 0
    assert not any("UPDATE messages SET" in s for s, _ in con.ejecutados), (
        "la vectorizacion no debe correr si el mensaje no quedo guardado")


@asincrono
async def test_timeout_del_mutex_hnsw_se_reintenta(monkeypatch):
    """GET_LOCK sin conceder (timeout) lo traduce el helper a
    OperationalError(1205), que esta en el set reintentable: el guardado
    reintenta y completa. El timeout del mutex NO puede tragarse el
    mensaje: si el helper dejara pasar el (0,) como fila comun, el INSERT
    correria SIN el mutex y el bug volveria."""
    class _CursorMutex(_CursorTramposo):
        async def fetchone(self):
            for sql, _ in reversed(self.conn.ejecutados):
                if "GET_LOCK" in sql:
                    if self.conn._get_lock_fallara:
                        self.conn._get_lock_fallara = False
                        return (0,)  # timeout: el mutex no se concede
                    return (1,)
                if "DATABASE()" in sql:
                    return ("jax_memory_test_mock",)
                if "MAX(turn_number)" in sql:
                    return (3,)
                if "FROM conversations" in sql:
                    return (5, _USER, None)
            return None

    class _ConnMutex(_ConnTramposo):
        def __init__(self):
            super().__init__()
            self._get_lock_fallara = True  # el PRIMER GET_LOCK da timeout

        def cursor(self):
            return _CursorMutex(self)

    con = _ConnMutex()
    db = _db_con(con)
    monkeypatch.setattr(db, "get_embedding", _sin_embedding)

    r = await db._save_message_impl("uuid-x", "user", "hola", None, None, None)

    assert r == {"conversation_id": 5, "turn_number": 3}
    inserts = [s for s, _ in con.ejecutados if s.startswith("INSERT INTO messages")]
    assert len(inserts) == 1, "solo el 2do intento llego al INSERT"
    assert con.rollbacks == 1 and con.commits == 1


def test_el_unique_de_turns_esta_en_el_esquema_y_en_el_migrador():
    """jax_memory_schema.sql es la fuente de verdad (checker de deriva) y
    migrations.py lleva las bases existentes hacia adelante: el UNIQUE
    (conversation_id, turn_number) tiene que estar en los DOS con el mismo
    nombre, o una base nueva y una migrada quedan distintas."""
    import pathlib
    import re

    from jax.memory import migrations

    esquema = (pathlib.Path(__file__).resolve().parent.parent
               / "jax_memory_schema.sql").read_text(encoding="utf-8")
    m = re.search(r"CREATE TABLE `messages` \((.*?)\n\)[^;\n]*", esquema, re.S)
    assert m, "messages no esta en jax_memory_schema.sql"
    assert "UNIQUE KEY `uq_messages_conversation_turn`" in m.group(1), (
        "el UNIQUE (conversation_id, turn_number) no esta en el esquema")
    assert migrations.UNIQUE_MESSAGES_TURN in migrations._DDL_UNIQUE_MESSAGES_TURN
    assert "UNIQUE" in migrations._DDL_UNIQUE_MESSAGES_TURN


@asincrono
@requiere_db_de_prueba
async def test_la_migracion_sanea_duplicados_sin_perder_mensajes(limpio):
    """Sobre una base VIEJA (sin el UNIQUE): crea duplicados a mano, corre
    el paso del migrador y verifica que reenumera SIN PERDER filas, crea el
    indice, es idempotente y que despues la carrera revienta con 1062."""
    import aiomysql as _am

    from jax.memory import migrations

    conn = await _conn()
    try:
        await _sql(conn,
                   "INSERT INTO conversations (conversation_uuid, source) "
                   "VALUES ('uuid-mig', 'test')")
        conv_id = (await _sql(conn, "SELECT id FROM conversations "
                              "WHERE conversation_uuid='uuid-mig'", fetch=True))[0][0]

        # simula la base vieja: el fixture crea messages desde el .sql (que
        # ya trae el UNIQUE) y este test ejercita el camino de migracion de
        # una base que todavia no lo tiene
        await _sql(conn, "ALTER TABLE messages DROP INDEX uq_messages_conversation_turn")

        for turn in (1, 1, 2, 2):  # dos grupos duplicados, ids crecientes
            await _sql(conn,
                       "INSERT INTO messages (conversation_id, turn_number, role, "
                       "content, user_id) VALUES (%s, %s, 'user', %s, %s)",
                       (conv_id, turn, f"msg-t{turn}-i", _USER))

        async with conn.cursor() as cur:
            await migrations._asegurar_unique_messages_turn(cur)

        turnos = sorted(fila[0] for fila in await _sql(
            conn, "SELECT turn_number FROM messages WHERE conversation_id=%s",
            (conv_id,), fetch=True))
        assert turnos == [1, 2, 3, 4], (
            f"turnos tras saneamiento: {turnos} -- se esperaban 1..4 "
            "(reenumeracion al final, orden por id preservado)")
        hay_indice = await _sql(
            conn,
            "SELECT COUNT(*) FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='messages' AND "
            "INDEX_NAME='uq_messages_conversation_turn'", fetch=True)
        assert hay_indice[0][0] >= 1, "el migrador no creo el indice unico"

        # idempotente: segunda corrida no revienta ni reenumera de nuevo
        async with conn.cursor() as cur:
            await migrations._asegurar_unique_messages_turn(cur)
        turnos2 = sorted(fila[0] for fila in await _sql(
            conn, "SELECT turn_number FROM messages WHERE conversation_id=%s",
            (conv_id,), fetch=True))
        assert turnos2 == [1, 2, 3, 4]

        # y ahora la carrera revienta con 1062 en vez de duplicar en silencio
        with pytest.raises(_am.IntegrityError) as excinfo:
            await _sql(conn,
                       "INSERT INTO messages (conversation_id, turn_number, role, "
                       "content, user_id) VALUES (%s, 1, 'user', 'carrera', %s)",
                       (conv_id, _USER))
        assert excinfo.value.args[0] == 1062
    finally:
        conn.close()
