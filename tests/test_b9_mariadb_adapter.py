import pytest

from jax.memory.b9 import (
    EventKind, Lifecycle, MemoryEvent, MemoryObject, MemoryProjection,
    MemoryProvenance, MemoryRevision, ObjectKind, Visibility,
)
from jax.memory.b9_mariadb import MariaDBB9Store


class Cursor:
    def __init__(self, fail_at=None): self.sql=[]; self.fail_at=fail_at
    async def __aenter__(self): return self
    async def __aexit__(self,*_): pass
    async def execute(self, sql, args=()):
        self.sql.append((sql,args))
        if self.fail_at and len(self.sql)==self.fail_at: raise RuntimeError("injected write failure")
    async def fetchone(self): return None

class Conn:
    def __init__(self, fail_at=None): self.cur=Cursor(fail_at); self.begun=self.committed=self.rolled=False
    async def begin(self): self.begun=True
    async def commit(self): self.committed=True
    async def rollback(self): self.rolled=True
    def cursor(self): return self.cur

class Acquire:
    def __init__(self,c): self.c=c
    async def __aenter__(self): return self.c
    async def __aexit__(self,*_): pass
class Pool:
    def __init__(self,c): self.c=c
    def acquire(self): return Acquire(self.c)

def bundle():
    o=MemoryObject("m",ObjectKind.FACT,"t",1)
    r=MemoryRevision("r","m","sha256:x",Visibility.USER_PRIVATE,"u",None,Lifecycle.ACTIVE,1,"x")
    p=MemoryProvenance("p","r",(),"test","1","a","USER","u",None,None,1)
    e=MemoryEvent("e","m","r",EventKind.CREATE,"a","u","auth",1)
    x=MemoryProjection("m","r",Lifecycle.ACTIVE,False,"sha256:h")
    return o,r,p,e,x

@pytest.mark.asyncio
async def test_bundle_is_one_transaction():
    c=Conn(); await MariaDBB9Store(Pool(c)).persist_bundle(*bundle())
    assert c.begun and c.committed and not c.rolled and len(c.cur.sql)==6
    assert any("memory_revision_payloads" in sql for sql, _ in c.cur.sql)

@pytest.mark.asyncio
async def test_bundle_failure_rolls_back_every_step():
    c=Conn(fail_at=4)
    with pytest.raises(RuntimeError): await MariaDBB9Store(Pool(c)).persist_bundle(*bundle())
    assert c.begun and c.rolled and not c.committed
