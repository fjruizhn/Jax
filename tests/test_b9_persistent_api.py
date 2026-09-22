import pytest

from jax.memory.b9 import (
    AuthorizationDenied, MutationAuthorizationContext, ObjectKind, ScopeContext,
    Visibility,
)
from jax.memory.b9_mariadb import MariaDBB9Store, PersistentMemoryAPI


class Cursor:
    def __init__(self, fail_at=None): self.fail_at, self.calls = fail_at, []
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass
    async def execute(self, sql, args=()):
        self.calls.append((sql, args))
        if self.fail_at == len(self.calls): raise RuntimeError("write failure")

class Conn:
    def __init__(self, fail_at=None): self.cursor_obj=Cursor(fail_at); self.committed=False; self.rolled=False
    async def begin(self): pass
    async def commit(self): self.committed=True
    async def rollback(self): self.rolled=True
    def cursor(self): return self.cursor_obj

class Acquire:
    def __init__(self, conn): self.conn=conn
    async def __aenter__(self): return self.conn
    async def __aexit__(self, *_): pass

class Pool:
    def __init__(self, conn): self.conn=conn
    def acquire(self): return Acquire(self.conn)


def auth(operation="CREATE", visibility=Visibility.USER_PRIVATE):
    scope=ScopeContext("user-1", "USER", "user-1", "tenant-1")
    return MutationAuthorizationContext(scope, operation, visibility, frozenset(), frozenset(), "test-authority")


@pytest.mark.asyncio
async def test_persistent_create_builds_complete_atomic_bundle():
    conn=Conn(); api=PersistentMemoryAPI(MariaDBB9Store(Pool(conn)))
    memory_id=await api.create_memory(auth(), ObjectKind.FACT, "payload", Visibility.USER_PRIVATE, user_id="user-1")
    assert memory_id
    assert conn.committed and not conn.rolled
    text="\n".join(sql for sql, _ in conn.cursor_obj.calls)
    for table in ("memory_objects", "memory_revisions", "memory_revision_payloads", "memory_provenance", "memory_events", "memory_projections"):
        assert table in text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [2, 3, 4, 5, 6])
async def test_each_canonical_write_failure_rolls_back(failure):
    conn=Conn(failure); api=PersistentMemoryAPI(MariaDBB9Store(Pool(conn)))
    with pytest.raises(RuntimeError):
        await api.create_memory(auth(), ObjectKind.FACT, "payload", Visibility.USER_PRIVATE, user_id="user-1")
    assert conn.rolled and not conn.committed


@pytest.mark.asyncio
async def test_persistent_api_rejects_raw_authority_input():
    api=PersistentMemoryAPI(MariaDBB9Store(Pool(Conn())))
    with pytest.raises(AuthorizationDenied):
        await api.create_memory({"admin": True}, ObjectKind.FACT, "x", Visibility.USER_PRIVATE, user_id="user-1")

