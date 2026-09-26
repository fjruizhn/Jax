from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
import pytest
from jax.memory import db as module

class Pool:
    def __init__(self): self.closed=False
    def close(self): self.closed=True
    async def wait_closed(self): pass

@pytest.mark.asyncio
async def test_worker_connect_never_invokes_legacy_migration(monkeypatch):
    pool=Pool()
    monkeypatch.setattr(module.aiomysql,'create_pool',AsyncMock(return_value=pool))
    migrate=AsyncMock(side_effect=AssertionError('startup migration forbidden'))
    monkeypatch.setattr(module,'ensure_schema',migrate)
    readiness=AsyncMock(return_value=True)
    monkeypatch.setattr(module.MemoryDB,'check_schema_readiness',readiness,raising=False)
    memory=module.MemoryDB()
    assert await memory.connect('test','test','unused','jax_memory_test_isolated',port=3308,migrate_schema=False)
    migrate.assert_not_awaited(); readiness.assert_awaited_once()
    await memory.close()

@pytest.mark.asyncio
async def test_readonly_startup_missing_schema_fails_closed_and_closes_pool(monkeypatch):
    pool=Pool()
    monkeypatch.setattr(module.aiomysql,'create_pool',AsyncMock(return_value=pool))
    monkeypatch.setattr(module,'ensure_schema',AsyncMock(side_effect=AssertionError('startup migration forbidden')))
    monkeypatch.setattr(module.MemoryDB,'check_schema_readiness',AsyncMock(return_value=False),raising=False)
    memory=module.MemoryDB()
    assert not await memory.connect('test','test','unused','jax_memory_test_isolated',port=3308,migrate_schema=False)
    assert pool.closed and memory.pool is None

@pytest.mark.asyncio
async def test_schema_readiness_only_reads_metadata_and_never_repairs():
    from contextlib import asynccontextmanager
    class Cursor:
        def __init__(self): self.sql=[]
        async def execute(self,query): self.sql.append(query)
        async def fetchall(self): return [('messages','user_id')]
    cursor=Cursor()
    class Connection:
        @asynccontextmanager
        async def cursor(self): yield cursor
    class ReadPool:
        @asynccontextmanager
        async def acquire(self): yield Connection()
    memory=module.MemoryDB(); memory.pool=ReadPool()
    assert not await memory.check_schema_readiness()
    assert cursor.sql and all(query.startswith('SELECT ') and 'information_schema' in query for query in cursor.sql)
