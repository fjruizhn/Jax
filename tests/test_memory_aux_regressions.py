"""Isolated worker regressions; no production config or provider calls."""
import asyncio
import os
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


# --- Auditoria de Jax#354, 8 MINOR (2026-10-05) ---------------------------------------------

@pytest.mark.asyncio
async def test_real_si_el_digest_cambia_a_mitad_de_la_corrida_la_corrida_falla(monkeypatch, caplog):
    """MINOR 7: el digest se lee al inicio y otra vez al final; si difiere, hay vectores de dos modelos."""
    from jax.memory import embedding_worker as embedding
    pool = await _pool_de_embeddings()
    try:
        db = SimpleNamespace(pool=pool, get_embedding=AsyncMock(return_value=[0.1, 0.2]))
        for lecturas, esperado in ((["d1", "d1"], (0, 0)), (["d1", "d2"], (0, 1))):
            monkeypatch.setattr(embedding, "modelo_digest", AsyncMock(side_effect=lecturas))
            with caplog.at_level("ERROR", logger="jax.memory.embedding_worker"):
                assert await embedding.run_b9_embeddings(db, writer=object()) == esperado
        assert any("digest" in r.getMessage() and "d2" in r.getMessage() for r in caplog.records)
    finally:
        pool.close(); await pool.wait_closed()


async def _pool_de_messages():
    from test_b9_persistent_api import _b9_ci_test_pool
    pool = await _b9_ci_test_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("CREATE TEMPORARY TABLE messages (id INT AUTO_INCREMENT PRIMARY KEY, conversation_id INT NOT NULL, turn_number INT NOT NULL, role VARCHAR(16), content TEXT, KEY idx_conversation (conversation_id))")
            await cur.execute("INSERT INTO messages (conversation_id,turn_number,role,content) SELECT seq%20, seq, 'user', 'x' FROM seq_1_to_400")
        await conn.commit()
    return pool


@pytest.mark.asyncio
async def test_real_migracion_011_quita_el_filesort_de_las_dos_lecturas_de_mensajes():
    from pathlib import Path
    pool = await _pool_de_messages()
    asc = "EXPLAIN SELECT id, turn_number, role, content FROM messages WHERE conversation_id = 5 ORDER BY turn_number ASC, id ASC"
    desc = "EXPLAIN SELECT role, content FROM messages WHERE conversation_id = 5 ORDER BY turn_number DESC, id DESC LIMIT 20"
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for q in (asc, desc):
                    await cur.execute(q); assert "filesort" in (await cur.fetchone())["Extra"]     # el problema, medido
                sql = "\n".join(l for l in (Path(__file__).resolve().parents[1] / "jax/memory/b9_migrations/011_messages_conversation_turn_index.sql").read_text().splitlines() if not l.lstrip().startswith("--"))
                for _ in range(2):                                                                     # idempotente
                    for st in (x.strip() for x in sql.split(";") if x.strip()):
                        await cur.execute(st)
                for q in (asc, desc):
                    await cur.execute(q); plan = await cur.fetchone()
                    assert "filesort" not in plan["Extra"] and plan["key"] == "idx_messages_conversation_turn", plan
    finally:
        pool.close(); await pool.wait_closed()


_RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_readme_de_migraciones_trae_la_marcha_atras_exacta_de_007_a_013():
    texto = (_RAIZ / "jax/memory/b9_migrations/README.md").read_text(encoding="utf-8")
    assert "Marcha atrás" in texto and "ALGORITHM=COPY" in texto and "lock_wait_timeout" in texto
    pasos = [
        "ALTER TABLE messages ADD INDEX IF NOT EXISTS idx_conversation (conversation_id)",              # 013 (antes que 011: la FK lo necesita)
        "ALTER TABLE conversations DROP INDEX IF EXISTS idx_conversations_open",                        # 012
        "ALTER TABLE messages DROP INDEX IF EXISTS idx_messages_conversation_turn",                     # 011
        "ADD INDEX IF NOT EXISTS idx_embedding_generation_revision (revision_id, embedding_space_id)",  # 010 (antes que 008)
        "DROP TABLE IF EXISTS embedding_generation_attempts",                                           # 009
        "DROP INDEX IF EXISTS uq_embedding_generation_revision_space",                                  # 008
        "DROP TABLE IF EXISTS memory_extraction_job_events",                                            # 007
    ]
    posiciones = [texto.index(p) for p in pasos]
    assert posiciones == sorted(posiciones), "marcha atras en orden inverso: 013, 012, 011, 010, 009, 008, 007"
    assert texto.index("SET SESSION lock_wait_timeout=10;") < texto.index(pasos[0])      # la marcha atras tambien limita la espera


def test_install_memory_scope_no_habilita_el_worker_sin_bandera_explicita():
    import subprocess
    script = _RAIZ / "config/systemd/install-memory-scope.sh"
    texto = script.read_text(encoding="utf-8")
    assert "enable --now" not in texto and "--now" not in texto
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=20, cwd="/")
    assert r.returncode != 0 and "obsoleto" in (r.stdout + r.stderr).lower()
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


_QUE_CALLAN_CODIGOS = ("SuccessExitStatus", "RestartPreventExitStatus")


def _opciones_que_callan_codigos(texto: str) -> list[str]:
    """Claves de la seccion [Service] que harian que systemd NO trate 2 o 3 como fallo (callarian el aviso)."""
    return [k for k in _QUE_CALLAN_CODIGOS if _opciones_servicio(texto, k)]


def _opciones_servicio(texto: str, clave: str) -> list[str]:
    actual, valores = None, []
    for linea in texto.splitlines():
        linea = linea.strip()
        if linea.startswith("[") and linea.endswith("]"):
            actual = linea[1:-1]
        elif actual == "Service" and linea.partition("=")[0].strip() == clave and "=" in linea:
            valores.append(linea.partition("=")[2].strip())
    return valores


def test_ninguna_unidad_jax_memory_calla_los_codigos_de_salida_2_o_3():
    """Re-auditoria Jax#354: SuccessExitStatus/RestartPreventExitStatus en [Service] de CUALQUIER unidad
    (o de sus drop-ins versionados) convertirian el 2 o el 3 en 'exito' y aviso-fallo no se enteraria."""
    unidades = sorted((_RAIZ / "config/systemd").glob("jax-memory-*.service"))
    dropins = sorted((_RAIZ / "config/systemd").glob("jax-memory-*.service.d/*.conf"))
    assert len(unidades) == 5
    for archivo in [*unidades, *dropins]:
        assert _opciones_que_callan_codigos(archivo.read_text(encoding="utf-8")) == [], archivo.name


def test_el_detector_de_codigos_silenciados_detecta_de_verdad():
    # un control que no falla cuando debe no valida nada
    assert _opciones_que_callan_codigos("[Service]\nSuccessExitStatus=2 3\n") == ["SuccessExitStatus"]
    assert _opciones_que_callan_codigos("[Service]\nRestartPreventExitStatus=3\n") == ["RestartPreventExitStatus"]
    assert _opciones_que_callan_codigos("[Service]\nSuccessExitStatus = 2 3\n") == ["SuccessExitStatus"]   # systemd admite espacios
    assert _opciones_que_callan_codigos("[Service]\n  RestartPreventExitStatus\t=\t3\n") == ["RestartPreventExitStatus"]
    assert _opciones_que_callan_codigos("[Unit]\nSuccessExitStatus=2\n[Service]\nType=oneshot\n") == []


def test_unidades_y_runbook_documentan_los_codigos_de_salida():
    for unidad in ("jax-memory-worker.service", "jax-memory-vector-health.service"):
        t = (_RAIZ / "config/systemd" / unidad).read_text(encoding="utf-8")
        assert "Códigos de salida" in t or "Codigos de salida" in t, unidad
    assert "2 =" in (_RAIZ / "config/systemd/jax-memory-worker.service").read_text(encoding="utf-8")
    assert "3 =" in (_RAIZ / "config/systemd/jax-memory-vector-health.service").read_text(encoding="utf-8")
    rb = (_RAIZ / "docs/runbooks/memoria-cola-atascada.md").read_text(encoding="utf-8")
    for frase in ("Códigos de salida", "INCONSISTENT", "UNKNOWN", "digest", "memory_processed"):
        assert frase in rb, frase


# --- Re-auditoria Jax#354: 011/012/013 con la DDL REAL de produccion (VECTOR KEY + FK ON DELETE CASCADE) ---

import re as _re

_MIG = _RAIZ / "jax/memory/b9_migrations"


def _ddl_produccion(tabla: str) -> str:
    """DDL de `tabla` tal como esta HOY en produccion (antes de 011/012/013): el esquema versionado, con
    los indices de esas migraciones quitados y `idx_conversation` repuesto."""
    texto = (_RAIZ / "jax_memory_schema.sql").read_text(encoding="utf-8")
    ddl = _re.search(r"CREATE TABLE `%s` \(.*?\) ENGINE=InnoDB[^;]*;" % tabla, texto, _re.S).group(0)
    lineas = [l for l in ddl.splitlines()
              if "idx_messages_conversation_turn" not in l and "idx_conversations_open" not in l and "`idx_conversation`" not in l]
    if tabla == "messages":
        i = next(n for n, l in enumerate(lineas) if "PRIMARY KEY" in l)
        lineas.insert(i + 1, "  KEY `idx_conversation` (`conversation_id`),")
    return "\n".join(lineas).rstrip(";")


def _sentencias(nombre: str, *, timeout: int | None = None) -> list[str]:
    sql = "\n".join(l for l in (_MIG / nombre).read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("--"))
    if timeout is not None:
        sql = sql.replace("lock_wait_timeout=10", f"lock_wait_timeout={timeout}")
    return [x.strip() for x in sql.split(";") if x.strip()]


async def _conexion_real():
    import aiomysql
    from base_de_test import exigir_base_de_test
    if not os.getenv("JAX_DB_HOST"):
        pytest.skip("requires an isolated CI test database")
    return await aiomysql.connect(host=os.environ["JAX_DB_HOST"], port=int(os.getenv("JAX_DB_PORT", "3306")),
                                  user=os.getenv("JAX_DB_USER", "root"), password=os.getenv("JAX_DB_PASSWORD", ""),
                                  db=exigir_base_de_test(), autocommit=True, connect_timeout=5,
                                  cursorclass=aiomysql.DictCursor)


async def _con_tablas_reales(escenario):
    """Crea conversations/messages persistentes con la DDL real en la base DESECHABLE y las borra al final."""
    c1 = await _conexion_real()
    try:
        async with c1.cursor() as cur:
            await cur.execute("SET SESSION lock_wait_timeout=5")        # red de seguridad de la prueba: nunca colgarse
            await cur.execute("DROP TABLE IF EXISTS messages"); await cur.execute("DROP TABLE IF EXISTS conversations")
            await cur.execute(_ddl_produccion("conversations")); await cur.execute(_ddl_produccion("messages"))
        await escenario(c1)
    finally:
        async with c1.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS messages"); await cur.execute("DROP TABLE IF EXISTS conversations")
        c1.close()


def test_011_012_013_fijan_lock_wait_timeout_dentro_del_propio_sql_y_documentan_la_copia():
    for nombre in ("011_messages_conversation_turn_index.sql", "012_conversations_open_index.sql",
                   "013_messages_drop_redundant_conversation_index.sql"):
        crudo = (_MIG / nombre).read_text(encoding="utf-8")
        sentencias = _sentencias(nombre)
        assert sentencias[0] == "SET SESSION lock_wait_timeout=10", nombre            # ANTES del ALTER
        assert sentencias[1].startswith("ALTER TABLE"), nombre
        for frase in ("ALGORITHM=COPY", "LOCK=SHARED", "bloquea escrituras", "sin subir el limite"):
            assert frase in crudo, (nombre, frase)
        assert "ALGORITHM=COPY, LOCK=SHARED" in sentencias[1], nombre                    # explicito, no a criterio del motor


@pytest.mark.asyncio
async def test_real_011_sobre_la_ddl_de_produccion_con_vector_key_y_fk_cascade():
    async def escenario(c1):
        async with c1.cursor() as cur:
            await cur.execute("SHOW CREATE TABLE messages"); ddl = (await cur.fetchone())["Create Table"]
            assert "VECTOR KEY" in ddl.upper() and "ON DELETE CASCADE" in ddl.upper()     # la tabla ES la real
            await cur.execute("INSERT INTO conversations (conversation_uuid) VALUES ('u1')")
            await cur.execute("INSERT INTO messages (conversation_id,turn_number,role,content) VALUES (1,1,'user','x'),(1,1,'user','y')")
            for st in _sentencias("011_messages_conversation_turn_index.sql", timeout=10):
                await cur.execute(st)
            for st in _sentencias("011_messages_conversation_turn_index.sql", timeout=10):      # idempotente
                await cur.execute(st)
            await cur.execute("SHOW INDEX FROM messages")
            assert "idx_messages_conversation_turn" in {r["Key_name"] for r in await cur.fetchall()}
            await cur.execute("DELETE FROM conversations WHERE id=1")                              # la FK CASCADE sigue viva
            await cur.execute("SELECT COUNT(*) AS n FROM messages"); assert (await cur.fetchone())["n"] == 0
    await _con_tablas_reales(escenario)


@pytest.mark.asyncio
async def test_real_011_con_una_escritura_abierta_se_agota_el_tiempo_sin_cambiar_nada_y_se_reintenta():
    import pymysql
    async def escenario(c1):
        c2 = await _conexion_real()
        try:
            async with c2.cursor() as cur2:
                await cur2.execute("SET SESSION lock_wait_timeout=5")
            async with c1.cursor() as cur:
                await cur.execute("INSERT INTO conversations (conversation_uuid) VALUES ('u1')")
            async with c2.cursor() as cur2:
                await cur2.execute("START TRANSACTION")
                await cur2.execute("INSERT INTO messages (conversation_id,turn_number,role,content) VALUES (1,1,'user','abierta')")
            async with c1.cursor() as cur:
                with pytest.raises(pymysql.err.OperationalError) as e:
                    for st in _sentencias("011_messages_conversation_turn_index.sql", timeout=1):
                        await cur.execute(st)
                assert e.value.args[0] == 1205                                                        # lock wait timeout
                await cur.execute("SHOW INDEX FROM messages")
                assert "idx_messages_conversation_turn" not in {r["Key_name"] for r in await cur.fetchall()}   # nada cambio
            async with c2.cursor() as cur2:
                await cur2.execute("ROLLBACK")
            async with c1.cursor() as cur:                                                            # reintento, mismo limite
                for st in _sentencias("011_messages_conversation_turn_index.sql", timeout=1):
                    await cur.execute(st)
                await cur.execute("SHOW INDEX FROM messages")
                assert "idx_messages_conversation_turn" in {r["Key_name"] for r in await cur.fetchall()}
        finally:
            c2.close()
    await _con_tablas_reales(escenario)


@pytest.mark.asyncio
async def test_real_012_el_indice_de_abiertas_cambia_el_plan_de_stale_open_conversations():
    consulta = ("EXPLAIN SELECT COUNT(*) AS n FROM conversations c WHERE c.ended_at IS NULL "
                "AND c.started_at<NOW(6)-INTERVAL 7 DAY")
    async def escenario(c1):
        async with c1.cursor() as cur:
            await cur.execute("INSERT INTO conversations (conversation_uuid,started_at,ended_at) "
                              "SELECT CONCAT('c',seq), NOW()-INTERVAL (seq%400) DAY, IF(seq%50=0,NULL,NOW()) FROM seq_1_to_2000")
            await cur.execute("ANALYZE TABLE conversations")
            await cur.fetchall()
            await cur.execute(consulta); antes = await cur.fetchone()
            for st in _sentencias("012_conversations_open_index.sql", timeout=10):
                await cur.execute(st)
            for st in _sentencias("012_conversations_open_index.sql", timeout=10):                  # idempotente
                await cur.execute(st)
            await cur.execute("ANALYZE TABLE conversations"); await cur.fetchall()
            await cur.execute(consulta); despues = await cur.fetchone()
        assert despues["key"] == "idx_conversations_open", (antes["key"], antes["rows"], despues["key"], despues["rows"])
        assert int(despues["rows"]) < int(antes["rows"]), (antes["type"], antes["rows"], despues["type"], despues["rows"])                                    # EXPLAIN antes/despues
    await _con_tablas_reales(escenario)


@pytest.mark.asyncio
async def test_real_013_quita_idx_conversation_sin_que_ninguna_consulta_pierda_plan():
    consultas = {
        "bounds (COUNT/SUM por conversacion)": "EXPLAIN SELECT COUNT(*), COALESCE(SUM(CHAR_LENGTH(content)+CHAR_LENGTH(role)+3),0) FROM messages WHERE conversation_id=5",
        "mensajes ASC": "EXPLAIN SELECT id, turn_number, role, content FROM messages WHERE conversation_id = 5 ORDER BY turn_number ASC, id ASC",
        "mensajes DESC": "EXPLAIN SELECT role, content FROM messages WHERE conversation_id = 5 ORDER BY turn_number DESC, id DESC LIMIT 20",
        "borrado en cascada": "EXPLAIN DELETE FROM messages WHERE conversation_id = 5",
    }
    async def escenario(c1):
        async with c1.cursor() as cur:
            await cur.execute("INSERT INTO conversations (conversation_uuid) SELECT CONCAT('c',seq) FROM seq_1_to_30")
            await cur.execute("INSERT INTO messages (conversation_id,turn_number,role,content) SELECT 1+(seq%29), seq, 'user','x' FROM seq_1_to_900")
            await cur.execute("ANALYZE TABLE messages"); await cur.fetchall()
            for n in ("011_messages_conversation_turn_index.sql",):
                for st in _sentencias(n, timeout=10): await cur.execute(st)
            antes = {}
            for nombre, q in consultas.items():
                await cur.execute(q); antes[nombre] = await cur.fetchone()
            for st in _sentencias("013_messages_drop_redundant_conversation_index.sql", timeout=10): await cur.execute(st)
            for st in _sentencias("013_messages_drop_redundant_conversation_index.sql", timeout=10): await cur.execute(st)   # idempotente
            await cur.execute("SHOW INDEX FROM messages")
            nombres = {r["Key_name"] for r in await cur.fetchall()}
            assert "idx_conversation" not in nombres and "idx_messages_conversation_turn" in nombres
            for nombre, q in consultas.items():
                await cur.execute(q); plan = await cur.fetchone()
                assert plan["key"] == "idx_messages_conversation_turn", (nombre, plan)
                assert "filesort" not in (plan["Extra"] or ""), (nombre, plan)
            await cur.execute("DELETE FROM conversations WHERE id=5")                                # la FK sigue indexada y viva
            await cur.execute("SELECT COUNT(*) AS n FROM messages WHERE conversation_id=5"); assert (await cur.fetchone())["n"] == 0
    await _con_tablas_reales(escenario)


@pytest.mark.asyncio
async def test_real_la_marcha_atras_del_readme_restaura_el_estado_de_produccion_dos_veces():
    """Ejecuta los pasos 013/012/011 del README sobre la DDL real despues de aplicar las tres migraciones."""
    bloque = _re.search(r"```sql\n(.*?)```", (_MIG / "README.md").read_text(encoding="utf-8"), _re.S).group(1)
    pasos = [x.strip() for x in "\n".join(l for l in bloque.splitlines() if not l.lstrip().startswith("--")).split(";") if x.strip()]
    sobre_tablas = [p for p in pasos if "messages" in p or "conversations" in p or "lock_wait" in p]
    assert len(sobre_tablas) == 4
    async def escenario(c1):
        async with c1.cursor() as cur:
            for n in ("011_messages_conversation_turn_index.sql", "012_conversations_open_index.sql", "013_messages_drop_redundant_conversation_index.sql"):
                for st in _sentencias(n, timeout=10): await cur.execute(st)
            for _ in range(2):
                for st in sobre_tablas: await cur.execute(st)
            await cur.execute("SHOW INDEX FROM messages"); m = {r["Key_name"] for r in await cur.fetchall()}
            await cur.execute("SHOW INDEX FROM conversations"); c = {r["Key_name"] for r in await cur.fetchall()}
        assert "idx_conversation" in m and "idx_messages_conversation_turn" not in m and "idx_conversations_open" not in c
    await _con_tablas_reales(escenario)


def test_el_esquema_versionado_ordena_los_indices_como_los_deja_show_create_tras_011_a_013():
    """check_memory_schema_drift.py compara texto: tras 011-013 los indices agregados van al final de los KEY
    (despues de idx_conv_tenant / idx_msg_scope), no donde se declararon a mano."""
    texto = (_RAIZ / "jax_memory_schema.sql").read_text(encoding="utf-8")
    conv = _re.search(r"CREATE TABLE `conversations` \(.*?\) ENGINE", texto, _re.S).group(0)
    msgs = _re.search(r"CREATE TABLE `messages` \(.*?\) ENGINE", texto, _re.S).group(0)
    assert conv.index("`idx_conv_tenant`") < conv.index("`idx_conversations_open`")
    assert msgs.index("`idx_msg_scope`") < msgs.index("`idx_messages_conversation_turn`") < msgs.index("VECTOR KEY")
