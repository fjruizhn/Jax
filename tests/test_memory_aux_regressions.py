"""Isolated worker regressions; no production config or provider calls."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from jax.memory import synthesis_worker as synthesis
from jax.memory.embedding_worker import PersistentEmbeddingWriter
from jax.memory.b9 import ScopeContext, Visibility, EmbeddingSpaceIdentity


def test_legacy_delete_rejects_active_b9_binding_and_never_issues_delete():
    """M1 locks fact/owner before testing its tenant-qualified B9 binding."""
    from contextlib import asynccontextmanager
    from jax.memory.db import MemoryDB
    class Cursor:
        def __init__(self): self.calls=[]; self.rows=[(7, 3), ('ACTIVE',)]; self.rowcount=0
        async def execute(self, sql, args=()):
            self.calls.append((sql,args)); self.rowcount=0
            if sql.startswith('DELETE'): self.rowcount=1
        async def fetchone(self): return self.rows.pop(0)
        async def __aenter__(self): return self
        async def __aexit__(self,*_): return False
    class Conn:
        def __init__(self): self.cursor_obj=Cursor(); self.commits=0; self.rollbacks=0
        async def begin(self): pass
        async def commit(self): self.commits+=1
        async def rollback(self): self.rollbacks+=1
        def cursor(self): return self.cursor_obj
    class Pool:
        def __init__(self): self.conn=Conn()
        @asynccontextmanager
        async def acquire(self): yield self.conn
    async def run():
        db=MemoryDB(); db.pool=Pool()
        assert await db.delete_fact(7) is None  # decorator reports safe failure
        calls=[sql for sql,_ in db.pool.conn.cursor_obj.calls]
        assert calls[0].endswith('FOR UPDATE') and 'jax_users' in calls[0]
        assert calls[1].endswith('FOR UPDATE') and 'memory_legacy_bindings' in calls[1]
        assert not any(sql.startswith('DELETE FROM facts') for sql in calls)
        assert db.pool.conn.rollbacks==1 and db.pool.conn.commits==0
    asyncio.run(run())


def test_legacy_delete_without_active_binding_commits_after_locked_checks():
    from contextlib import asynccontextmanager
    from jax.memory.db import MemoryDB
    class Cursor:
        def __init__(self): self.calls=[]; self.rows=[(7, 3), None]; self.rowcount=0
        async def execute(self, sql, args=()):
            self.calls.append((sql,args)); self.rowcount=1 if sql.startswith('DELETE') else 0
        async def fetchone(self): return self.rows.pop(0)
        async def __aenter__(self): return self
        async def __aexit__(self,*_): return False
    class Conn:
        def __init__(self): self.cursor_obj=Cursor(); self.commits=0; self.rollbacks=0
        async def begin(self): pass
        async def commit(self): self.commits+=1
        async def rollback(self): self.rollbacks+=1
        def cursor(self): return self.cursor_obj
    class Pool:
        def __init__(self): self.conn=Conn()
        @asynccontextmanager
        async def acquire(self): yield self.conn
    async def run():
        db=MemoryDB(); db.pool=Pool()
        assert await db.delete_fact(7) is True
        calls=[sql for sql,_ in db.pool.conn.cursor_obj.calls]
        assert calls[-1].startswith('DELETE FROM facts')
        assert db.pool.conn.commits==1 and db.pool.conn.rollbacks==0
    asyncio.run(run())


def test_chunked_extraction_keeps_each_exact_turn_allowlist():
    from jax.memory.worker import _chunk_locked_messages
    messages=[{'message_id':11,'turn_number':1,'role':'user','content':'a'*80},
              {'message_id':12,'turn_number':2,'role':'jax_local','content':'b'*80}]
    chunks=_chunk_locked_messages(messages,50)
    assert len(chunks)>2
    first=('11',1,'user'); second=('12',2,'jax_local')
    assert all(turns <= {first,second} for _,turns in chunks)
    assert any(turns=={first} for _,turns in chunks)
    assert any(turns=={second} for _,turns in chunks)


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


# --- Auditoria 2026-10-05, menores del worker de embeddings ---------------------------------

def _cliente_ollama(monkeypatch, *, tags=None, falla=None):
    """Sustituye el cliente HTTP compartido por uno que responde `tags` en /api/tags."""
    from jax.memory import embedding_worker as embedding
    monkeypatch.setenv("JAX_OLLAMA_URL", "http://ollama.invalid:11434")
    llamadas = []
    class Resp:
        def raise_for_status(self):
            if falla: raise falla
        def json(self): return {"models": tags or []}
    class Cliente:
        async def get(self, url, timeout=None):
            llamadas.append(url)
            if falla and not isinstance(falla, Exception): raise falla
            return Resp()
    monkeypatch.setattr(embedding, "obtener_cliente_http", lambda: Cliente())
    return llamadas


def test_el_digest_del_modelo_sale_de_api_tags_de_ollama(monkeypatch):
    from jax.memory import embedding_worker as embedding
    llamadas = _cliente_ollama(monkeypatch, tags=[
        {"name": "nomic-embed-text:latest", "digest": "otro"}, {"name": "bge-m3:latest", "digest": "7907aaaa"}])
    assert asyncio.run(embedding.modelo_digest("bge-m3")) == "7907aaaa"      # sin tag = :latest, como Ollama
    assert asyncio.run(embedding.modelo_digest("bge-m3:latest")) == "7907aaaa"
    assert llamadas[0] == "http://ollama.invalid:11434/api/tags"


@pytest.mark.parametrize("tags,falla", [
    ([{"name": "otro:latest", "digest": "x"}], None),                       # el modelo no esta instalado
    ([{"name": "bge-m3:latest"}], None),                                    # sin digest
    ([{"name": "bge-m3:latest", "digest": ""}], None),                      # digest vacio
    ([], RuntimeError("HTTP 500")),                                         # Ollama responde error
    ([], ConnectionError("Ollama caido")),                                  # Ollama no responde
])
def test_si_ollama_no_da_el_digest_la_corrida_falla_y_no_guarda_none(monkeypatch, tags, falla):
    from jax.memory import embedding_worker as embedding
    _cliente_ollama(monkeypatch, tags=tags, falla=falla)
    with pytest.raises(Exception):
        asyncio.run(embedding.resolver_identidad())


def test_la_identidad_lleva_el_digest_real_y_otro_digest_es_otro_espacio(monkeypatch):
    from jax.memory import embedding_worker as embedding
    ids = {}
    for digest in ("d1", "d2"):
        monkeypatch.setattr(embedding, "modelo_digest", AsyncMock(return_value=digest))
        ident = asyncio.run(embedding.resolver_identidad())
        assert ident.model_version_or_digest == digest
        ids[digest] = ident.embedding_space_id
    assert ids["d1"] != ids["d2"]                    # el cambio de modelo no se mezcla en el espacio viejo


def test_vector_health_tambien_falla_si_no_puede_consultar_ollama(monkeypatch):
    from jax.memory import embedding_worker as embedding
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1"); monkeypatch.setenv("JAX_DB_PORT", "1")
    monkeypatch.setattr(embedding, "modelo_digest", AsyncMock(side_effect=RuntimeError("Ollama caido")))
    monkeypatch.setattr(embedding, "cerrar_cliente_http", AsyncMock())
    with pytest.raises(RuntimeError, match="Ollama caido"):
        asyncio.run(embedding.run_b9_vector_health())


async def _pool_de_embeddings():
    """Tablas minimas de embeddings sobre el pool temporal de CI (una conexion)."""
    from pathlib import Path
    from test_b9_persistent_api import _b9_ci_test_pool
    pool = await _b9_ci_test_pool()
    mig = Path(__file__).resolve().parents[1] / "jax/memory/b9_migrations"
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TEMPORARY TABLE embedding_generations")
            await cur.execute("DROP TEMPORARY TABLE embedding_generation_attempts")
            await cur.execute("CREATE TEMPORARY TABLE embedding_generations (generation_id CHAR(36) PRIMARY KEY, revision_id CHAR(36), embedding_space_id CHAR(71), generated_at DATETIME(6), embedding_payload LONGBLOB, KEY idx_embedding_generation_revision (revision_id, embedding_space_id))")
            await cur.execute("CREATE TEMPORARY TABLE embedding_spaces (embedding_space_id CHAR(71) PRIMARY KEY, schema_version VARCHAR(64), provider_runtime_class VARCHAR(64), model_identifier VARCHAR(255), model_version_or_digest VARCHAR(255) NULL, dimension INT, normalization VARCHAR(64), distance_semantics VARCHAR(64), created_at DATETIME(6))")
            for nombre in ("008_embedding_generation_unique.sql", "009_embedding_generation_attempts.sql"):
                sql = "\n".join(l for l in (mig / nombre).read_text().splitlines() if not l.lstrip().startswith("--"))
                for st in sql.split(";"):
                    if st.strip():
                        await cur.execute(st.replace("CREATE TABLE IF NOT EXISTS", "CREATE TEMPORARY TABLE"))
        await conn.commit()
    return pool


async def _sembrar_revisiones(pool, n):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for i in range(1, n + 1):
                await cur.execute("INSERT INTO memory_objects (memory_id,tenant_id) VALUES (%s,'t')", (f"m{i}",))
                await cur.execute("INSERT INTO memory_revisions (revision_id,memory_id,visibility,lifecycle_state,created_at) VALUES (%s,%s,'USER_PRIVATE','ACTIVE',%s)", (f"r{i}", f"m{i}", f"2026-01-01 00:00:0{i}"))
                await cur.execute("INSERT INTO memory_revision_payloads (revision_id,payload) VALUES (%s,%s)", (f"r{i}", f"texto {i}".encode()))
                await cur.execute("INSERT INTO memory_projections (memory_id,current_revision_id,current_lifecycle_state,reconciliation_required) VALUES (%s,%s,'ACTIVE',FALSE)", (f"m{i}", f"r{i}"))
        await conn.commit()


@pytest.mark.asyncio
async def test_real_una_fila_que_siempre_falla_no_bloquea_la_cola_y_se_registra(monkeypatch, caplog):
    """BATCH_SIZE=1: sin contador de intentos, r1 ocupa la cabeza de la cola para siempre."""
    import uuid
    from jax.memory import embedding_worker as embedding
    pool = await _pool_de_embeddings()
    try:
        await _sembrar_revisiones(pool, 3)
        monkeypatch.setattr(embedding, "BATCH_SIZE", 1)
        monkeypatch.setenv("JAX_MEMORY_EMBED_MAX_ATTEMPTS", "2")
        identity = EmbeddingSpaceIdentity("b9-v1", "ollama", "bge-m3", "d1", 2, "unit", "cosine")

        async def get_embedding(texto):
            return None if texto == "texto 1" else [0.1, 0.2]
        class Writer:
            async def persist(self, memory_id, ident, vector, visibility, *, expected_revision_id=None):
                async with pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("INSERT INTO embedding_generations (generation_id,revision_id,embedding_space_id,generated_at) VALUES (%s,%s,%s,NOW(6))",
                                          (str(uuid.uuid4()), expected_revision_id, ident.embedding_space_id))
                    await conn.commit()
                return "g"
        db = SimpleNamespace(pool=pool, get_embedding=get_embedding)
        resultados = []
        with caplog.at_level("ERROR", logger="jax.memory.embedding_worker"):
            for _ in range(5):
                resultados.append(await embedding.run_b9_embeddings(db, writer=Writer(), identity=identity))
        assert resultados == [(0, 1), (0, 1), (1, 0), (1, 0), (0, 0)], resultados
        assert any("r1" in r.getMessage() and "saltad" in r.getMessage() for r in caplog.records)   # queda registrado
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT attempts FROM embedding_generation_attempts WHERE revision_id='r1'")
                assert (await cur.fetchone())["attempts"] == 2
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_real_un_digest_nuevo_del_modelo_avisa_y_no_se_mezcla(caplog):
    from jax.memory import embedding_worker as embedding
    pool = await _pool_de_embeddings()
    try:
        viejo = EmbeddingSpaceIdentity("b9-v1", "ollama", "bge-m3", "d1", 1024, "unit", "cosine")
        nuevo = EmbeddingSpaceIdentity("b9-v1", "ollama", "bge-m3", "d2", 1024, "unit", "cosine")
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO embedding_spaces VALUES (%s,'b9-v1','ollama','bge-m3','d1',1024,'unit','cosine',NOW(6))", (viejo.embedding_space_id,))
            await conn.commit()
        with caplog.at_level("WARNING", logger="jax.memory.embedding_worker"):
            await embedding.avisar_si_cambio_el_digest(pool, viejo)          # mismo digest: nada que avisar
            assert not caplog.records
            await embedding.avisar_si_cambio_el_digest(pool, nuevo)
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "d1" in msg and "d2" in msg and "espacio" in msg
        assert viejo.embedding_space_id != nuevo.embedding_space_id
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_real_migraciones_008_y_009_son_idempotentes_y_la_unica_rechaza_duplicados():
    import aiomysql
    pool = await _pool_de_embeddings()
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                from pathlib import Path
                sql = "\n".join(l for l in (Path(__file__).resolve().parents[1] / "jax/memory/b9_migrations/008_embedding_generation_unique.sql").read_text().splitlines() if not l.lstrip().startswith("--"))
                await cur.execute(sql.strip().rstrip(";"))                  # segunda vez: no falla
                await cur.execute("INSERT INTO embedding_generations (generation_id,revision_id,embedding_space_id,generated_at) VALUES ('g1','r1','s1',NOW(6))")
                with pytest.raises(aiomysql.IntegrityError):
                    await cur.execute("INSERT INTO embedding_generations (generation_id,revision_id,embedding_space_id,generated_at) VALUES ('g2','r1','s1',NOW(6))")
                await cur.execute("INSERT INTO embedding_generations (generation_id,revision_id,embedding_space_id,generated_at) VALUES ('g3','r1','s2',NOW(6))")   # otro espacio: permitido
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_real_migracion_010_quita_el_indice_redundante_y_las_consultas_siguen_usando_la_unica():
    """La UNIQUE 008 cubre (revision_id, embedding_space_id): idx_embedding_generation_revision sobra."""
    from pathlib import Path
    pool = await _pool_de_embeddings()
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW INDEX FROM embedding_generations")
                assert "idx_embedding_generation_revision" in {r["Key_name"] for r in await cur.fetchall()}
                sql = "\n".join(l for l in (Path(__file__).resolve().parents[1] / "jax/memory/b9_migrations/010_embedding_generation_drop_redundant_index.sql").read_text().splitlines() if not l.lstrip().startswith("--"))
                for _ in range(2):                                    # idempotente
                    await cur.execute(sql.strip().rstrip(";"))
                await cur.execute("SHOW INDEX FROM embedding_generations")
                assert "idx_embedding_generation_revision" not in {r["Key_name"] for r in await cur.fetchall()}
                await cur.execute("INSERT INTO embedding_generations (generation_id,revision_id,embedding_space_id,generated_at) VALUES ('g1','r1','s1',NOW(6)),('g2','r2','s1',NOW(6))")
                await cur.execute("EXPLAIN SELECT generation_id FROM embedding_generations WHERE revision_id=%s AND embedding_space_id=%s", ("r1", "s1"))
                plan = await cur.fetchone()
                assert plan["key"] == "uq_embedding_generation_revision_space", plan
    finally:
        pool.close(); await pool.wait_closed()
