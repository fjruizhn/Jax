"""La lectura B9 del Faro corre solo contra MariaDB aislada de prueba.

El esquema se crea como TEMPORARY TABLE en una única conexión del pool; el
test no crea ni modifica objetos persistentes y cierra el pool al terminar.
"""
import asyncio
import os

import aiomysql
import pytest

from base_de_test import es_base_de_test
from jax.faro.herramientas.memoria import AdaptadorMemoria
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
        self._sentencias.append(sql.strip().upper())
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
        if (host != "127.0.0.1" or user != "jax_test" or not password or not es_base_de_test(database)
                or port not in {3308, 3306}):
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
            reader = MariaDBB9Reader(PoolObservado(pool, statements))
            execution = Ejecucion("run-db", "user-db", "tenant-db", "hyde", "codex", "p", "repl", "corr-db", 1)
            result = await AdaptadorMemoria(reader).buscar(Identidad(execution, "conn", 1, 1, 1), "no existe", 5)
            assert result == []
            assert statements
            assert all(sql.startswith(("SELECT", "SET TRANSACTION")) for sql in statements)
            assert any("WHERE R.TENANT_ID=%S" in sql for sql in statements)
        finally:
            pool.close()
            await pool.wait_closed()

    asyncio.run(caso())
