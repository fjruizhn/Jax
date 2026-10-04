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
from jax.memory.b9_mariadb import MariaDBB9Reader


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


class ConnObservada:
    def __init__(self, conn, sentencias):
        self._conn, self._sentencias = conn, sentencias

    async def begin(self):
        return await self._conn.begin()

    async def rollback(self):
        return await self._conn.rollback()

    def cursor(self):
        return CursorObservado(self._conn.cursor(), self._sentencias)


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
            )
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    for ddl in ddls:
                        await cur.execute(ddl)
            reader = MariaDBB9Reader(PoolObservado(pool, statements), max_payload_bytes=8192)
            execution = Ejecucion("run-db", "user-db", "tenant-db", "hyde", "codex", "p", "repl", "corr-db", 1)
            result = await AdaptadorMemoria(reader).buscar(Identidad(execution, "conn", 1, 1, 1), "no existe", 5)
            assert result == []
            assert statements
            assert all(sql.startswith(("SELECT", "SET TRANSACTION")) for sql, _ in statements)
            queries = [(sql, args) for sql, args in statements if sql.startswith("SELECT")]
            assert all("WHERE R.TENANT_ID=%S" in sql and "tenant-db" in args for sql, args in queries)
            assert any("user-db" in args for _, args in queries)
            payload_queries = [(sql, args) for sql, args in queries if "LEFT(P.PAYLOAD,%S) AS PAYLOAD" in sql]
            assert payload_queries
            assert all(args[:2] == (8192, 8192) for _, args in payload_queries)
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
