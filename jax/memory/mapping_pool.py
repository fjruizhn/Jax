"""Mapping cursor view over a shared pool; leaves legacy tuple consumers intact."""
from contextlib import asynccontextmanager

class MappingPool:
    def __init__(self, pool): self.pool = pool.pool if isinstance(pool, MappingPool) else pool
    @asynccontextmanager
    async def acquire(self):
        async with self.pool.acquire() as connection:
            yield MappingConnection(connection)

class MappingConnection:
    def __init__(self, connection): self.connection = connection
    def __getattr__(self, name): return getattr(self.connection, name)
    def cursor(self):
        import aiomysql
        return self.connection.cursor(aiomysql.DictCursor)
