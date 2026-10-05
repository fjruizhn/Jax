"""La lectura B9 del Faro corre solo contra MariaDB aislada de prueba.

El esquema se crea como TEMPORARY TABLE en una única conexión del pool; el
test no crea ni modifica objetos persistentes y cierra el pool al terminar.
"""
import asyncio
import os

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.faro.config import ConfigFaroInvalida, ConfigMemoria
from jax.faro.herramientas.memoria import AdaptadorMemoria, crear_pool_memoria_prueba
from jax.faro.identidad import Ejecucion, Identidad
from jax.memory.b9 import ScopeContext
from jax.memory.b9_mariadb import MariaDBB9Reader
from jax.memory.b9_mariadb import _rollback_bounded


class CursorObservado:
    def __init__(self, context, sentencias):
        self._context = context
        self._cursor = None
        self._sentencias = sentencias

    async def __aenter__(self):
        self._cursor = await self._context.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._context.__aexit__(*args)

    async def execute(self, sql, args=()):
        self._sentencias.append((sql.strip().upper(), tuple(args)))
        return await self._cursor.execute(sql, args)

    async def fetchall(self):
        return await self._cursor.fetchall()

    async def fetchone(self):
        return await self._cursor.fetchone()


class ConnObservada:
    def __init__(self, conn, sentencias):
        self._conn, self._sentencias = conn, sentencias

    async def begin(self):
        return await self._conn.begin()

    async def rollback(self):
        return await self._conn.rollback()

    def close(self):
        return self._conn.close()

    def cursor(self):
        return CursorObservado(self._conn.cursor(), self._sentencias)


def test_cursor_observado_delega_fetchone():
    class Cursor:
        async def fetchone(self):
            return {"revision_id": "source-rev"}

    async def caso():
        observado = CursorObservado(None, [])
        observado._cursor = Cursor()
        assert await observado.fetchone() == {"revision_id": "source-rev"}

    asyncio.run(caso())


def test_rollback_colgado_cierra_la_conexion_y_respeta_el_limite():
    class Conexion:
        cerrada = False

        async def rollback(self):
            await asyncio.Event().wait()

        def close(self):
            self.cerrada = True

    async def caso():
        conn = Conexion()
        with pytest.raises(TimeoutError):
            await _rollback_bounded(conn, timeout_s=0.02)
        assert conn.cerrada

    asyncio.run(caso())


def test_validar_sintesis_consulta_existencia_de_payload_sin_leer_blob():
    class Cursor:
        def __init__(self, has_payload):
            self.has_payload = has_payload
        sql = ""
        args = ()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, sql, args=()):
            self.sql, self.args = sql.upper(), tuple(args)

        async def fetchall(self):
            if self.sql.startswith("SELECT STRAIGHT_JOIN") and "USER_PRIVATE" in self.args:
                return [{
                    "memory_id": "synth-id", "object_kind": "SYNTHESIS", "tenant_id": "tenant-db",
                    "object_created_at": 1.0, "revision_id": "synth-rev", "content_digest": "digest",
                    "visibility": "USER_PRIVATE", "user_id": "user-db", "project_id": None,
                    "lifecycle_state": "ACTIVE", "revision_created_at": 2.0, "payload": b"summary",
                    "payload_truncated": False, "provenance_status": "VERIFIED", "prior_revision_id": None,
                }]
            if self.sql.startswith("SELECT PROVENANCE_ID"):
                return [{
                    "provenance_id": "prov-id", "source_revisions": ["source-rev"],
                    "transformation_id": "synthesis", "transformation_version": "v1",
                    "actor_principal": "service:test", "actor_type": "SERVICE", "subject_user_id": None,
                    "provider": None, "model": None, "created_at": 3.0, "limitations": None,
                }]
            return []

        async def fetchone(self):
            assert "(P.PAYLOAD IS NOT NULL) AS HAS_PAYLOAD" in self.sql
            return {"revision_id": "source-rev", "lifecycle_state": "ACTIVE",
                    "has_payload": self.has_payload, "current_revision_id": "source-rev"}

    class Connection:
        def __init__(self, has_payload):
            self.has_payload = has_payload

        async def begin(self):
            return None

        async def rollback(self):
            return None

        def cursor(self):
            return Cursor(self.has_payload)

    class Pool:
        def __init__(self, has_payload):
            self.has_payload = has_payload

        class Lease:
            def __init__(self, has_payload):
                self.has_payload = has_payload

            async def __aenter__(self):
                return Connection(self.has_payload)

            async def __aexit__(self, *args):
                return None

        def acquire(self):
            return self.Lease(self.has_payload)

    async def caso():
        scope = ScopeContext("user:user-db", "USER", "user-db", "tenant-db")
        present = await MariaDBB9Reader(Pool(True), max_payload_bytes=8192).retrieve(scope)
        absent = await MariaDBB9Reader(Pool(False), max_payload_bytes=8192).retrieve(scope)
        assert [item.revision.revision_id for item in present] == ["synth-rev"]
        assert absent == ()

    asyncio.run(caso())


class PoolObservado:
    def __init__(self, pool, sentencias):
        self._pool, self._sentencias = pool, sentencias

    def acquire(self):
        pool, sentencias = self._pool, self._sentencias

        class Acquire:
            async def __aenter__(self):
                self.context = pool.acquire()
                self.conn = await self.context.__aenter__()
                return ConnObservada(self.conn, sentencias)

            async def __aexit__(self, *args):
                return await self.context.__aexit__(*args)

        return Acquire()


def test_reader_mariadb_solo_lee_scope_del_socket_en_base_de_prueba():
    async def caso():
        host = os.environ.get("JAX_DB_HOST", "")
        port = int(os.environ.get("JAX_DB_PORT", "0"))
        user = os.environ.get("JAX_DB_USER", "")
        password = os.environ.get("JAX_DB_PASSWORD", "")
        database = os.environ.get("JAX_DB_NAME", "")
        puertos_permitidos = {3306, 3308} if os.getenv("CI") else {3308}
        if (host != "127.0.0.1" or user != "jax_test" or not password or not es_base_de_test(database)
                or port not in puertos_permitidos):
            pytest.fail("memoria DB test requires 127.0.0.1, jax_test, and jax_memory_test[_suffix] only")

        pool = await aiomysql.create_pool(host=host, port=port, user=user, password=password, db=database,
                                          minsize=1, maxsize=1, cursorclass=aiomysql.DictCursor,
                                          autocommit=True, connect_timeout=5)
        statements = []
        try:
            ddls = (
                "CREATE TEMPORARY TABLE memory_objects (memory_id CHAR(36), object_kind VARCHAR(32), tenant_id VARCHAR(128), created_at DATETIME(6))",
                "CREATE TEMPORARY TABLE memory_revisions (revision_id CHAR(36), memory_id CHAR(36), content_digest CHAR(71), visibility VARCHAR(32), user_id VARCHAR(128), project_id VARCHAR(128), lifecycle_state VARCHAR(32), created_at DATETIME(6), payload LONGBLOB, provenance_status VARCHAR(64), prior_revision_id CHAR(36), tenant_id VARCHAR(128))",
                "CREATE TEMPORARY TABLE memory_projections (memory_id CHAR(36), current_revision_id CHAR(36), reconciliation_required BOOLEAN)",
                "CREATE TEMPORARY TABLE memory_revision_payloads (revision_id CHAR(36), payload LONGBLOB)",
                "CREATE TEMPORARY TABLE memory_provenance (provenance_id CHAR(36), revision_id CHAR(36), source_revisions JSON, transformation_id VARCHAR(128), transformation_version VARCHAR(64), actor_principal VARCHAR(255), actor_type VARCHAR(64), subject_user_id VARCHAR(128), provider VARCHAR(128), model VARCHAR(255), created_at DATETIME(6), limitations TEXT)",
            )
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    for ddl in ddls:
                        await cur.execute(ddl)
                    await cur.execute("INSERT INTO memory_objects VALUES ('synth-id','SYNTHESIS','tenant-db',NOW(6))")
                    await cur.execute("INSERT INTO memory_revisions VALUES ('synth-rev','synth-id','digest','USER_PRIVATE','user-db',NULL,'ACTIVE',NOW(6),NULL,'VERIFIED',NULL,'tenant-db')")
                    await cur.execute("INSERT INTO memory_projections VALUES ('synth-id','synth-rev',FALSE)")
                    await cur.execute("INSERT INTO memory_revision_payloads VALUES ('synth-rev',%s)", (b'needle summary',))
                    await cur.execute("INSERT INTO memory_revisions VALUES ('source-rev','source-id','digest','USER_PRIVATE','user-db',NULL,'ACTIVE',NOW(6),NULL,'VERIFIED',NULL,'tenant-db')")
                    await cur.execute("INSERT INTO memory_projections VALUES ('source-id','source-rev',FALSE)")
                    await cur.execute("INSERT INTO memory_revision_payloads VALUES ('source-rev',%s)", (b'x' * 100000,))
                    await cur.execute("INSERT INTO memory_provenance VALUES ('prov-id','synth-rev',JSON_ARRAY('source-rev'),'synthesis','v1','service:test','SERVICE',NULL,NULL,NULL,NOW(6),NULL)")
            reader = MariaDBB9Reader(PoolObservado(pool, statements), max_payload_bytes=8192)
            execution = Ejecucion("run-db", "user-db", "tenant-db", "hyde", "codex", "p", "repl", "corr-db", 1)
            result = await AdaptadorMemoria(reader).buscar(Identidad(execution, "conn", 1, 1, 1), "needle", 5)
            assert [item["revision_id"] for item in result] == ["synth-rev"]
            assert statements
            assert all(sql.startswith(("SELECT", "SET TRANSACTION")) for sql, _ in statements)
            queries = [(sql, args) for sql, args in statements if sql.startswith("SELECT")]
            candidate_queries = [(sql, args) for sql, args in queries if "SELECT STRAIGHT_JOIN" in sql]
            assert all("WHERE R.TENANT_ID=%S" in sql and "tenant-db" in args for sql, args in candidate_queries)
            assert any("user-db" in args for _, args in candidate_queries)
            payload_queries = [(sql, args) for sql, args in candidate_queries if "LEFT(P.PAYLOAD,%S) AS PAYLOAD" in sql]
            assert payload_queries
            assert all(args[:2] == (8192, 8192) for _, args in payload_queries)
            synthesis_source_queries = [sql for sql, _ in queries if "CURRENT_REVISION_ID" in sql]
            assert synthesis_source_queries
            assert all("(P.PAYLOAD IS NOT NULL) AS HAS_PAYLOAD" in sql for sql in synthesis_source_queries)
        finally:
            pool.close()
            await pool.wait_closed()

    asyncio.run(caso())


def test_pool_memoria_revalida_perfil_incluso_si_se_elude_el_dataclass(monkeypatch):
    async def caso():
        calls = []

        async def create_pool(**kwargs):
            calls.append(kwargs)
            raise AssertionError("no debe abrirse una conexión fuera de la allowlist")

        monkeypatch.setattr(aiomysql, "create_pool", create_pool)
        cfg = object.__new__(ConfigMemoria)
        for key, value in {
            "habilitada": True, "host": "10.0.0.9", "port": 3306,
            "usuario": "root", "clave": "secret", "base": "jax_memory", "timeout_s": 5.0,
        }.items():
            object.__setattr__(cfg, key, value)
        with pytest.raises(ConfigFaroInvalida, match="solo permite el perfil local de prueba"):
            await crear_pool_memoria_prueba(cfg)
        assert calls == []

    asyncio.run(caso())


def test_pool_memoria_abre_solo_perfil_local_aprobado_y_limita_payload(monkeypatch):
    async def caso():
        pool_falso = object()
        calls = []

        async def create_pool(**kwargs):
            calls.append(kwargs)
            return pool_falso

        monkeypatch.setattr(aiomysql, "create_pool", create_pool)
        cfg = ConfigMemoria(True, "127.0.0.1", 3308, "jax_test", "test-password", "jax_memory_test")
        pool, reader = await crear_pool_memoria_prueba(cfg)
        assert pool is pool_falso
        assert reader._max_payload_bytes == 8192
        assert len(calls) == 1
        assert calls[0]["maxsize"] == 4
        assert {key: calls[0][key] for key in ("host", "port", "user", "db")} == {
            "host": "127.0.0.1", "port": 3308, "user": "jax_test", "db": "jax_memory_test"}

    asyncio.run(caso())
