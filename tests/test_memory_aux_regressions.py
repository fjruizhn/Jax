"""Isolated worker regressions; no production config or provider calls."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from jax.memory import synthesis_worker as synthesis
from jax.memory.embedding_worker import PersistentEmbeddingWriter
from jax.memory.b9 import ScopeContext, Visibility, EmbeddingSpaceIdentity


def test_embedding_carries_revision_fence_to_api():
    async def run():
        api = SimpleNamespace(reembed_memory=AsyncMock(return_value="g1"))
        async def scope(*args):
            return ScopeContext("service:embedding","SERVICE","1","1",None,calling_component="embedding")
        writer = PersistentEmbeddingWriter(api,scope)
        identity = EmbeddingSpaceIdentity("1","runtime","m",None,2,"unit","cosine")
        await writer.persist("m1",identity,(.1,.2),Visibility.USER_PRIVATE,expected_revision_id="r1")
        assert api.reembed_memory.call_args.kwargs["expected_revision_id"] == "r1"
    asyncio.run(run())


def test_duplicate_or_invented_sources_cannot_create_synthesis():
    facts = [{"b9_revision_id":"r1"},{"b9_revision_id":"r2"}]
    assert synthesis.validated_insights({"insights":[{"text":"bad","source_ids":[1,1]}]},facts) == []
    assert synthesis.validated_insights({"insights":[{"text":"bad","source_ids":[1,99]}]},facts) == []
    assert synthesis.validated_insights({"insights":[{"text":"good","source_ids":[2,1,2]}]},facts)[0]["source_revision_ids"] == ["r1","r2"]


def test_insufficient_b9_sources_never_call_model(monkeypatch):
    async def run():
        read = AsyncMock(return_value=[{"b9_revision_id":"r1"}])
        monkeypatch.setattr(synthesis,"read_b9_facts",read)
        llm = SimpleNamespace(invoke=AsyncMock())
        jobs = SimpleNamespace(claim=AsyncMock())
        assert await synthesis.process_scope(SimpleNamespace(pool=object()),llm,1,None,tenant_id=1,b9_writer=object(),jobs=jobs) == 0
        llm.invoke.assert_not_called()
        jobs.claim.assert_not_called()
    asyncio.run(run())


def test_frozen_output_retry_never_pays_again_and_fails_visible(monkeypatch):
    async def run():
        monkeypatch.setattr(synthesis,"read_b9_facts",AsyncMock(return_value=[{"b9_revision_id":f"r{i}","fact_text":"synthetic"} for i in range(5)]))
        item = {"item_key":"i","text":"derived","source_revision_ids":["r1","r2"]}
        jobs = SimpleNamespace(claim=AsyncMock(return_value=("job","token",[item])),freeze=AsyncMock())
        llm = SimpleNamespace(invoke=AsyncMock())
        writer = SimpleNamespace(preflight=AsyncMock(),persist=AsyncMock(side_effect=RuntimeError("source unverified")))
        with pytest.raises(RuntimeError,match="source unverified"):
            await synthesis.process_scope(SimpleNamespace(pool=object()),llm,1,None,tenant_id=1,b9_writer=writer,jobs=jobs)
        llm.invoke.assert_not_called()
        assert writer.persist.call_args.kwargs == {"job_key":"job","job_token":"token","item_key":"i"}
    asyncio.run(run())


def test_lifecycle_rereads_history_under_object_and_projection_locks():
    from jax.memory.lifecycle_worker import scan_and_mark
    from contextlib import asynccontextmanager
    class Cursor:
        def __init__(self): self.calls=[]; self.sql=""; self.batches=0
        async def execute(self,sql,args):
            self.sql=sql; self.calls.append(sql)
        async def fetchall(self):
            if self.sql.startswith("SELECT memory_id FROM memory_objects"):
                self.batches+=1
                return [{"memory_id":"m1"}] if self.batches==1 else []
            return []
    class Conn:
        def __init__(self): self.cur=Cursor(); self.commits=0; self.rollbacks=0
        async def begin(self): pass
        async def commit(self): self.commits+=1
        async def rollback(self): self.rollbacks+=1
        @asynccontextmanager
        async def cursor(self,*args): yield self.cur
    class Pool:
        def __init__(self): self.conn=Conn()
        @asynccontextmanager
        async def acquire(self): yield self.conn
    async def run():
        pool=Pool()
        assert await scan_and_mark(pool)==("m1",)
        calls=pool.conn.cur.calls
        assert "memory_objects" in calls[0] and calls[0].endswith("FOR UPDATE")
        assert "memory_projections" in calls[1] and calls[1].endswith("FOR UPDATE")
        assert "memory_revisions" in calls[2] and calls[2].endswith("FOR UPDATE")
        assert "memory_events" in calls[3] and calls[3].endswith("FOR UPDATE")
        assert calls[4].startswith("UPDATE memory_projections SET reconciliation_required=TRUE")
        assert pool.conn.commits==2 and pool.conn.rollbacks==0
    asyncio.run(run())


def test_embedding_partial_failure_exits_nonzero_and_closes(monkeypatch):
    from jax.memory import embedding_worker as embedding
    async def run():
        db = SimpleNamespace(connect=AsyncMock(return_value=True),close=AsyncMock())
        monkeypatch.setenv("JAX_DB_HOST","test-only")
        monkeypatch.setattr(embedding,"MemoryDB",lambda: db)
        monkeypatch.setattr(embedding,"procesar_mensajes",AsyncMock(return_value=(0,0)))
        monkeypatch.setattr(embedding,"procesar_facts",AsyncMock(return_value=(0,0)))
        monkeypatch.setattr(embedding,"run_b9_embeddings",AsyncMock(return_value=(0,1)))
        close_http=AsyncMock()
        monkeypatch.setattr(embedding,"cerrar_cliente_http",close_http)
        assert await embedding.main()==1
        db.close.assert_awaited_once()
        close_http.assert_awaited_once()
    asyncio.run(run())


def test_synthesis_denied_preflight_never_calls_provider_or_claims(monkeypatch):
    async def run():
        monkeypatch.setattr(synthesis,"read_b9_facts",AsyncMock(return_value=[{"b9_revision_id":f"r{i}","fact_text":"test"} for i in range(5)]))
        writer=SimpleNamespace(preflight=AsyncMock(side_effect=RuntimeError("inactive subject")))
        provider=SimpleNamespace(invoke=AsyncMock())
        jobs=SimpleNamespace(claim=AsyncMock())
        with pytest.raises(RuntimeError,match="inactive subject"):
            await synthesis.process_scope(SimpleNamespace(pool=object()),provider,1,None,tenant_id=1,b9_writer=writer,jobs=jobs)
        provider.invoke.assert_not_called()
        jobs.claim.assert_not_called()
    asyncio.run(run())


def test_synthesis_response_and_item_budgets(monkeypatch):
    monkeypatch.setenv("JAX_MEMORY_SYNTHESIS_MAX_ITEMS","1")
    facts=[{"b9_revision_id":"r1"},{"b9_revision_id":"r2"}]
    with pytest.raises(ValueError,match="item limit"):
        synthesis.validated_insights({"insights":[{"text":"one","source_ids":[1,2]},{"text":"two","source_ids":[1,2]}]},facts)
    monkeypatch.setenv("JAX_MEMORY_SYNTHESIS_MAX_ITEM_CHARS","2")
    with pytest.raises(ValueError,match="text limit"):
        synthesis.validated_insights({"insights":[{"text":"long","source_ids":[1,2]}]},facts)
