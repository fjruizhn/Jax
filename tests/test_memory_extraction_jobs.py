import pytest
from jax.memory.b9 import ScopeDenied

def test_normalization_rejects_wrong_shape_and_never_silently_drops_items():
    from jax.memory.extraction_jobs import normalize_extraction
    for data in ({'facts': {}}, {'facts': [{'text': ''}]}, {'decisions': [{'title': 'a'}]}, {'action_items': [None]}):
        with pytest.raises(ValueError): normalize_extraction(data, max_items=10, max_text_chars=100)

def test_normalization_is_bounded_and_stable():
    from jax.memory.extraction_jobs import normalize_extraction
    turn={'message_id':'1','turn_number':1,'role':'user'}
    data={'facts':[{'text':'known','source_turns':[turn]}], 'decisions':[], 'action_items':[]}
    assert normalize_extraction(data, max_items=1, max_text_chars=100)==[{'kind':'FACT','content':'known','source_turns':[turn]}]
    with pytest.raises(ValueError): normalize_extraction(data, max_items=0, max_text_chars=100)

def test_extraction_rejects_invented_or_unlocked_source_turns():
    from jax.memory.extraction_jobs import normalize_extraction
    allowed={('11', 2, 'user')}
    good={'facts':[{'text':'known','source_turns':[{'message_id':'11','turn_number':2,'role':'user'}]}], 'decisions':[], 'action_items':[]}
    assert normalize_extraction(good,max_items=2,max_text_chars=100,allowed_turns=allowed)[0]['source_turns'][0]['message_id']=='11'
    bad={'facts':[{'text':'known','source_turns':[{'message_id':'12','turn_number':2,'role':'user'}]}], 'decisions':[], 'action_items':[]}
    with pytest.raises(ValueError,match='outside locked input'):
        normalize_extraction(bad,max_items=2,max_text_chars=100,allowed_turns=allowed)


@pytest.mark.parametrize("message_id", ["0123", " 123 ", "١٢٣", "-123", "999999999999999999999"])
def test_extraction_source_ids_have_one_canonical_locked_representation(message_id):
    """Only the decimal database identity may cite this locked turn."""
    from jax.memory.extraction_jobs import normalize_extraction
    allowed={("123", 7, "user")}
    data={"facts":[{"text":"known", "source_turns":[{"message_id":message_id,"turn_number":7,"role":"user"}]}]}
    with pytest.raises(ValueError, match="outside locked input"):
        normalize_extraction(data, max_items=1, max_text_chars=100, allowed_turns=allowed)


def test_extraction_accepts_integer_and_canonical_string_database_id_only():
    from jax.memory.extraction_jobs import normalize_extraction
    allowed={("123", 7, "user")}
    for raw_id in (123, "123"):
        data={"facts":[{"text":"known", "source_turns":[{"message_id":raw_id,"turn_number":7,"role":"user"}]}]}
        item=normalize_extraction(data, max_items=1, max_text_chars=100, allowed_turns=allowed)[0]
        assert item["source_turns"] == [{"message_id":"123", "turn_number":7, "role":"user"}]


def test_extraction_rejects_role_spoof_and_valid_turn_outside_chunk_allowlist():
    from jax.memory.extraction_jobs import normalize_extraction
    first=("123", 7, "user")
    other_chunk=("124", 8, "assistant")
    allowed={first}
    for turn in (
        {"message_id":"123", "turn_number":7, "role":"assistant"},
        {"message_id":"124", "turn_number":8, "role":"assistant"},
    ):
        data={"facts":[{"text":"known", "source_turns":[turn]}]}
        with pytest.raises(ValueError, match="outside locked input"):
            normalize_extraction(data, max_items=1, max_text_chars=100, allowed_turns=allowed)

def test_extraction_requires_structured_turn_provenance():
    from jax.memory.extraction_jobs import normalize_extraction
    with pytest.raises(ValueError,match='source_turns'):
        normalize_extraction({'facts':[{'text':'known'}],'decisions':[],'action_items':[]},max_items=1,max_text_chars=100)

@pytest.mark.asyncio
async def test_reembed_cannot_attach_old_payload_vector_to_new_revision():
    from test_b9_persistent_api import lifecycle_api, auth
    from jax.memory.b9 import EmbeddingSpaceIdentity
    conn, api=lifecycle_api()
    identity=EmbeddingSpaceIdentity('1','runtime','model','digest',2,'unit','cosine')
    with pytest.raises(ScopeDenied):
        await api.reembed_memory(auth('RE_EMBED'),'m1',identity,(.1,.2),expected_revision_id='superseded')
    assert conn.rolled and not conn.committed

@pytest.mark.asyncio
async def test_completed_extraction_never_bypasses_live_authority_or_source_scope():
    from test_b9_persistent_api import ScriptConn, make_api, TxResolver
    from jax.memory.b9 import MutationAuthorizationRequest, ScopeContext, Visibility, AuthorizationDenied, _digest
    from jax.memory.extraction_jobs import canonical
    job={'state':'COMPLETED','frozen_output':canonical([]),'output_digest':_digest(canonical([])),
         'request_id':'request','trace_id':'trace','run_id':'run'}
    conv={'id':1,'uuid':'uuid','tenant_id':'t1','user_id':'u1','project_id':None,'memory_processed':1}
    scope=ScopeContext('service:memory-extraction','SERVICE','u1','t1',calling_component='memory-extraction')
    conn=ScriptConn([job,conv], [[]]); api=make_api(conn,TxResolver(denied=True))
    with pytest.raises(AuthorizationDenied):
        await api.persist_conversation_extraction(MutationAuthorizationRequest(scope,'CREATE',Visibility.USER_PRIVATE),1,'old')
    assert conn.rolled and not conn.committed
    cross=ScopeContext('service:memory-extraction','SERVICE','u1','t2',calling_component='memory-extraction')
    conn=ScriptConn([job,conv],[[]]); api=make_api(conn)
    with pytest.raises(ScopeDenied):
        await api.persist_conversation_extraction(MutationAuthorizationRequest(cross,'CREATE',Visibility.USER_PRIVATE),1,'old')
    assert conn.rolled

@pytest.mark.asyncio
async def test_new_claim_quarantine_is_visible_failure_and_committed():
    from test_b9_persistent_api import ScriptConn, Pool
    from jax.memory.extraction_jobs import ExtractionJobs
    row={'state':'UNKNOWN','retry_wait':False,'attempts':1,'frozen_output':'[]'}
    conn=ScriptConn([row,{'id':1,'memory_processed':True}],[[]])
    result=await ExtractionJobs(Pool(conn)).claim(1,run_id='run')
    assert result=={'quarantined':True} and conn.committed
    assert any('INCONSISTENT_COMMIT_MARKERS' in sql for sql,_ in conn.cursor_obj.calls)

@pytest.mark.asyncio
async def test_internal_legacy_import_also_requires_admin_and_current_owner():
    from test_b9_persistent_api import ScriptConn, make_api, auth, TxResolver
    from jax.memory.b9 import AuthorizationDenied, ObjectKind, Visibility
    source={'id':1,'fact_text':'known','user_id':'u1','project_id':None}
    conn=ScriptConn([source]); api=make_api(conn)
    with pytest.raises(AuthorizationDenied):
        await api.import_legacy_memory(auth('IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL),'facts','legacy','1',ObjectKind.FACT,'known')
    assert conn.rolled and not conn.committed
    conn=ScriptConn([source,{'user_id':'u1','tenant_id':'other','status':'ACTIVE'}])
    api=make_api(conn,TxResolver(roles={'memory_admin'},caps={'memory:admin'}))
    with pytest.raises(ScopeDenied):
        await api.import_legacy_memory(auth('IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL),'facts','legacy','1',ObjectKind.FACT,'known')
    assert conn.rolled

@pytest.mark.asyncio
async def test_adoption_canonical_order_survives_reverse_uuid_order(monkeypatch):
    import jax.memory.b9_mariadb as module
    from test_b9_persistent_api import ScriptConn, make_api, auth, TxResolver
    from jax.memory.b9 import ObjectKind, Visibility, MemoryRevision, MemoryEvent, MemoryProjection, _derive_projection
    source={'id':1,'fact_text':'known','user_id':'user-1','project_id':None}
    owner={'user_id':'user-1','tenant_id':'tenant-1','status':'ACTIVE'}
    conn=ScriptConn([source,owner,None]); api=make_api(conn,TxResolver(roles={'memory_admin'},caps={'memory:admin'}))
    monkeypatch.setattr(module.time,'time',lambda:1000.123456)
    ids=iter(['mid','z-internal','prov-a','z-event','a-published','prov-b','a-event'])
    monkeypatch.setattr(module,'_uuid7',lambda:next(ids))
    bundles=[]
    original=api._write
    async def record(cur,obj,revision,provenance,event,projection,**kwargs):
        bundles.append((revision,event,projection))
        return await original(cur,obj,revision,provenance,event,projection,**kwargs)
    monkeypatch.setattr(api,'_write',record)
    await api.import_legacy_memory(auth('IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL),'facts','legacy','1',ObjectKind.FACT,'known',visibility=Visibility.USER_PRIVATE,user_id='user-1')
    revisions=sorted([item[0] for item in bundles],key=lambda r:(round(r.created_at,6),r.revision_id))
    events=sorted([item[1] for item in bundles],key=lambda e:(round(e.occurred_at,6),e.event_id))
    assert revisions[-1].visibility is Visibility.USER_PRIVATE
    assert _derive_projection('mid',revisions,events)==bundles[-1][2]


async def real_extraction_fixture():
    """One isolated test connection; no DDL or credentials from production."""
    from test_b9_persistent_api import _b9_ci_test_pool, TxResolver
    from jax.memory.b9 import ScopeContext, MutationAuthorizationRequest, Visibility
    from jax.memory.b9_mariadb import PersistentMemoryAPI, MariaDBB9Store
    from jax.memory.extraction_jobs import ExtractionJobs, source_digest
    pool=await _b9_ci_test_pool()
    from pathlib import Path
    migration=Path(__file__).parents[1]/'jax/memory/b9_migrations/006_memory_jobs.sql'
    sql='\n'.join(line for line in migration.read_text().splitlines() if not line.lstrip().startswith('--'))
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute('CREATE TEMPORARY TABLE conversations (id BIGINT PRIMARY KEY,conversation_uuid CHAR(36),tenant_id BIGINT,user_id BIGINT,project_id BIGINT,ended_at DATETIME(6),memory_processed BOOLEAN DEFAULT FALSE,memory_processed_at DATETIME(6))')
            await cur.execute('CREATE TEMPORARY TABLE messages (id BIGINT PRIMARY KEY,conversation_id BIGINT,turn_number INT,role VARCHAR(32),content TEXT,KEY ix_messages_conversation(conversation_id,turn_number))')
            for statement in sql.split(';'):
                if statement.strip().startswith('CREATE TABLE IF NOT EXISTS'):
                    await cur.execute(statement.replace('CREATE TABLE IF NOT EXISTS','CREATE TEMPORARY TABLE'))
            await cur.execute("INSERT INTO conversations VALUES (1,'source-uuid',1,1,NULL,NOW(6),FALSE,NULL)")
            await cur.execute("INSERT INTO messages VALUES (1,1,1,'user','known')")
        await conn.commit()
    api=PersistentMemoryAPI(MariaDBB9Store(pool),TxResolver())
    scope=ScopeContext('service:memory-extraction','SERVICE','1','1',calling_component='memory-extraction')
    request=MutationAuthorizationRequest(scope,'CREATE',Visibility.USER_PRIVATE)
    jobs=ExtractionJobs(pool)
    conv={'id':1,'uuid':'source-uuid','tenant_id':1,'user_id':1,'project_id':None}
    job=await jobs.claim(1,run_id='test-run')
    turn={'message_id':'1','turn_number':1,'role':'user'}
    items=[{'kind':'FACT','content':'one','source_turns':[turn]},{'kind':'FACT','content':'two','source_turns':[turn]}]
    # The locked DB snapshot retains its numeric id; provenance is canonicalized
    # to a string before the model output is frozen.
    db_turn={'message_id':1,'turn_number':1,'role':'user','content':'known'}
    await jobs.freeze(1,job['claim_token'],source_digest(conv,[db_turn]),items)
    return pool,api,jobs,job,request

async def real_counts(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute('SELECT (SELECT COUNT(*) FROM memory_events) events,(SELECT COUNT(*) FROM memory_extraction_results) results,(SELECT memory_processed FROM conversations WHERE id=1) processed,(SELECT state FROM memory_extraction_jobs WHERE conversation_id=1) state')
            return await cur.fetchone()

@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['second_item','processed_marker'])
async def test_real_extraction_complete_bundle_rolls_back_every_partial_write(monkeypatch,fault):
    pool,api,jobs,job,request=await real_extraction_fixture()
    try:
        if fault=='second_item':
            original=api._write; calls=0
            async def fail(cur,*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2: raise RuntimeError('second item failure')
                return await original(cur,*args,**kwargs)
            monkeypatch.setattr(api,'_write',fail)
        else:
            # Fail the real SQL statement after all B9 rows and bindings were inserted.
            import aiomysql
            original=aiomysql.cursors.DictCursor.execute
            async def fail(cur,query,args=None):
                if query.startswith('UPDATE conversations SET memory_processed=TRUE'): raise RuntimeError('mark failure')
                return await original(cur,query,args)
            monkeypatch.setattr(aiomysql.cursors.DictCursor,'execute',fail)
        with pytest.raises(RuntimeError): await api.persist_conversation_extraction(request,1,job['claim_token'])
        counts=await real_counts(pool)
        assert counts=={'events':0,'results':0,'processed':0,'state':'RUNNING'}
        monkeypatch.undo()
        await api.persist_conversation_extraction(request,1,job['claim_token'])
        assert await real_counts(pool)=={'events':2,'results':2,'processed':1,'state':'COMPLETED'}
        await api.persist_conversation_extraction(request,1,job['claim_token'])
        assert (await real_counts(pool))['events']==2
    finally:
        pool.close(); await pool.wait_closed()

@pytest.mark.asyncio
async def test_real_confirmed_commit_lost_response_resolves_without_new_events(monkeypatch):
    pool,api,jobs,job,request=await real_extraction_fixture()
    try:
        original=api._store.mutation
        async def lost_response(operation):
            await original(operation)
            raise ConnectionError('commit response lost')
        monkeypatch.setattr(api._store,'mutation',lost_response)
        with pytest.raises(ConnectionError): await api.persist_conversation_extraction(request,1,job['claim_token'])
        await jobs.fail(1,job['claim_token'],'ConnectionError',unknown=True)
        assert (await real_counts(pool))['state']=='COMPLETED'
        assert await jobs.claim(1,run_id='retry') is None
        assert await real_counts(pool)=={'events':2,'results':2,'processed':1,'state':'COMPLETED'}
    finally:
        pool.close(); await pool.wait_closed()

@pytest.mark.asyncio
async def test_worker_message_read_failure_is_not_empty_or_processed():
    import time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from jax.memory import worker
    from jax.memory.b9 import ScopeContext
    scope=ScopeContext('service:memory-extraction','SERVICE','1','1',calling_component='memory-extraction')
    async def op(operation): return await operation(None)
    api=SimpleNamespace(_store=SimpleNamespace(mutation=op),_auth=AsyncMock(return_value=SimpleNamespace(scope=scope)),_project_permissions=Mock())
    writer=SimpleNamespace(_api=api,_build_request_scope=AsyncMock(return_value=scope),persist_frozen=AsyncMock())
    db=SimpleNamespace(get_conversation_messages=AsyncMock(return_value=None),mark_processed=AsyncMock())
    jobs=SimpleNamespace(freeze=AsyncMock(),fail=AsyncMock(),validate_message_bounds=AsyncMock())
    llm=SimpleNamespace(invoke=AsyncMock())
    job={'claim_token':'token','request_id':'r','trace_id':'t','frozen_output':None}
    assert not await worker.process_claimed(db,llm,{'id':1,'user_id':1,'tenant_id':1},writer,jobs,job,{'calls':0,'max_calls':1},time.monotonic()+10)
    db.mark_processed.assert_not_called(); llm.invoke.assert_not_called(); writer.persist_frozen.assert_not_called()
    jobs.fail.assert_awaited_once_with(1,'token','RuntimeError',unknown=False)

@pytest.mark.asyncio
async def test_worker_empty_conversation_still_uses_atomic_completion():
    import time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from jax.memory import worker
    from jax.memory.b9 import ScopeContext
    scope=ScopeContext('service:memory-extraction','SERVICE','1','1',calling_component='memory-extraction')
    async def op(operation): return await operation(None)
    api=SimpleNamespace(_store=SimpleNamespace(mutation=op),_auth=AsyncMock(return_value=SimpleNamespace(scope=scope)),_project_permissions=Mock())
    writer=SimpleNamespace(_api=api,_build_request_scope=AsyncMock(return_value=scope),persist_frozen=AsyncMock())
    db=SimpleNamespace(get_conversation_messages=AsyncMock(return_value=[]),mark_processed=AsyncMock())
    jobs=SimpleNamespace(freeze=AsyncMock(),fail=AsyncMock(),validate_message_bounds=AsyncMock())
    llm=SimpleNamespace(invoke=AsyncMock())
    job={'claim_token':'token','request_id':'r','trace_id':'t','frozen_output':None}
    assert await worker.process_claimed(db,llm,{'id':1,'user_id':1,'tenant_id':1},writer,jobs,job,{'calls':0,'max_calls':1},time.monotonic()+10)
    db.mark_processed.assert_not_called(); llm.invoke.assert_not_called()
    assert jobs.freeze.call_args.args[-1]==[]
    writer.persist_frozen.assert_awaited_once()

@pytest.mark.asyncio
async def test_worker_poison_does_not_starve_next_and_exit_is_failure(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from jax.memory import worker
    db=SimpleNamespace(pool=object(),connect=AsyncMock(return_value=True),close=AsyncMock())
    jobs=SimpleNamespace(pending=AsyncMock(return_value=[{'id':1},{'id':2}]),claim=AsyncMock(return_value={'claim_token':'token'}))
    monkeypatch.setenv('JAX_DB_HOST','isolated-test')
    monkeypatch.setattr(worker,'MemoryDB',lambda:db)
    monkeypatch.setattr(worker,'ExtractionJobs',lambda *a,**k:jobs)
    monkeypatch.setattr(worker,'build_extractor',AsyncMock(return_value=object()))
    process=AsyncMock(side_effect=[False,True]); monkeypatch.setattr(worker,'process_claimed',process)
    monkeypatch.setattr(worker,'cerrar_cliente_http',AsyncMock())
    with pytest.raises(RuntimeError,match='failures: 1'): await worker.run_once(b9_writer=object())
    assert process.await_count==2
    db.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_worker_claim_quarantine_exits_failed(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from jax.memory import worker
    db=SimpleNamespace(pool=object(),connect=AsyncMock(return_value=True),close=AsyncMock())
    jobs=SimpleNamespace(pending=AsyncMock(return_value=[{'id':1}]),claim=AsyncMock(return_value={'quarantined':True}))
    monkeypatch.setenv('JAX_DB_HOST','isolated-test')
    monkeypatch.setattr(worker,'MemoryDB',lambda:db)
    monkeypatch.setattr(worker,'ExtractionJobs',lambda *a,**k:jobs)
    monkeypatch.setattr(worker,'build_extractor',AsyncMock(return_value=object()))
    monkeypatch.setattr(worker,'cerrar_cliente_http',AsyncMock())
    with pytest.raises(RuntimeError,match='failures: 1'): await worker.run_once(b9_writer=object())
    db.close.assert_awaited_once()


def test_oversized_single_message_is_always_bounded():
    from jax.memory.worker import _chunk_conversation
    chunks=_chunk_conversation('a'*12001,12000)
    assert [len(chunk) for chunk in chunks]==[12000,1]
    assert ''.join(chunks)=='a'*12001

@pytest.mark.asyncio
async def test_legacy_project_owner_revoked_after_plan_cannot_publish():
    from test_b9_persistent_api import ScriptConn, make_api, auth, TxResolver
    from jax.memory.b9 import ObjectKind, Visibility
    source={'id':1,'fact_text':'known','user_id':'user-1','project_id':'project-a'}
    owner={'user_id':'user-1','tenant_id':'tenant-1','status':'ACTIVE'}
    membership={'project_role':'OWNER','status':'REVOKED','tenant_id':'tenant-1'}
    conn=ScriptConn([source,owner,membership])
    api=make_api(conn,TxResolver(roles={'memory_admin'},caps={'memory:admin'},project_role='OWNER'))
    with pytest.raises(ScopeDenied,match='owner membership'):
        await api.import_legacy_memory(auth('IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL,project_id='project-a'),
            'facts','legacy','1',ObjectKind.FACT,'known',visibility=Visibility.PROJECT_SHARED,project_id='project-a')
    assert conn.rolled and not conn.committed
    assert not any(sql.startswith('INSERT INTO memory_') for sql,_ in conn.cursor_obj.calls)

@pytest.mark.asyncio
async def test_reader_merges_all_scope_branches_in_one_snapshot_with_stable_top_limit():
    from test_b9_persistent_api import ScriptConn, Pool, TxResolver, auth
    from jax.memory.b9_mariadb import MariaDBB9Reader
    from jax.memory.b9 import Visibility
    def row(mid,rev,visibility,project,created):
        return {'memory_id':mid,'object_kind':'FACT','tenant_id':'tenant-1','object_created_at':1,
            'revision_id':rev,'content_digest':'sha256:x','visibility':visibility,'user_id':'user-1' if visibility=='USER_PRIVATE' else None,
            'project_id':project,'lifecycle_state':'ACTIVE','revision_created_at':created,'payload':mid,
            'provenance_status':'COMPLETE','prior_revision_id':None}
    # Per visibility: global, project. The project-shared global case is kept
    # because the preexisting predicate permits it; no invariant is assumed.
    batches=[[row('private','r1','USER_PRIVATE',None,1)],[],[],[],[row('shared-global','r9','PROJECT_SHARED',None,5)],[]]
    conn=ScriptConn(many=batches+[[]])
    reader=MariaDBB9Reader(Pool(conn),TxResolver(project_role='VIEWER'))
    result=await reader.retrieve_authorized(auth('RETRIEVE',Visibility.PROJECT_SHARED,project_id='project-a'),limit=1)
    assert [item.identity.memory_id for item in result]==['shared-global']
    statements=[(sql,args) for sql,args in conn.cursor_obj.calls if 'AS revision_created_at' in sql]
    assert len(statements)==6 and all('SELECT STRAIGHT_JOIN ' in sql and 'r.tenant_id=%s' in sql and 'ORDER BY r.created_at DESC,r.revision_id DESC LIMIT %s' in sql for sql,_ in statements)
    assert all(args[-1]==1 for _,args in statements)
    assert conn.rolled

@pytest.mark.asyncio
async def test_new_revision_tenant_comes_from_resolved_object_and_sql_binds_every_value():
    from test_b9_persistent_api import Conn, make_api, auth
    from jax.memory.b9 import ObjectKind, Visibility
    conn=Conn(); api=make_api(conn)
    await api.create_memory(auth(),ObjectKind.FACT,'tenant-looking text: attacker',Visibility.USER_PRIVATE,user_id='user-1')
    statement,args=next((sql,values) for sql,values in conn.cursor_obj.calls if sql.startswith('INSERT INTO memory_revisions'))
    assert 'prior_revision_id,tenant_id)' in statement
    assert args[-1]=='tenant-1'
    assert statement.count('%s')==len(args)


# --- Auditoria 2026-10-05, MAJOR-3: el digest de la fuente no puede depender del azar ---
# `messages` tiene grupos (conversacion, turno) duplicados (152 medidos). Con
# `ORDER BY turn_number` a secas, dos lecturas del mismo origen podian devolver
# los empatados en orden distinto -> digest distinto -> cuarentena espuria.

class _CursorQueGrabaSQL:
    def __init__(self, rows): self.sql=[]; self.rows=rows
    async def __aenter__(self): return self
    async def __aexit__(self,*a): return False
    async def execute(self, sql, args=None): self.sql.append(sql)
    async def fetchall(self): return self.rows


def test_las_dos_lecturas_de_mensajes_del_digest_desempatan_por_id():
    import asyncio
    from types import SimpleNamespace
    from jax.memory.db import MemoryDB
    cur=_CursorQueGrabaSQL([])
    conn=SimpleNamespace(cursor=lambda *a,**k: cur)
    class _Acq:
        async def __aenter__(self): return conn
        async def __aexit__(self,*a): return False
    db=MemoryDB(); db.pool=SimpleNamespace(acquire=lambda: _Acq())
    asyncio.run(db.get_conversation_messages(1))
    assert any('ORDER BY turn_number ASC, id ASC' in s for s in cur.sql), cur.sql
    import inspect
    from jax.memory import b9_mariadb
    src=inspect.getsource(b9_mariadb)
    assert 'ORDER BY turn_number ASC LIMIT' not in src
    assert 'ORDER BY turn_number ASC, id ASC LIMIT' in src


@pytest.mark.asyncio
async def test_real_mensajes_empatados_en_turno_salen_siempre_por_id():
    """Con una tabla SIN indice por turno (orden fisico = orden de insercion), dos mensajes
    del mismo turn_number insertados en orden inverso deben salir por id en ambas lecturas."""
    from test_b9_persistent_api import _b9_ci_test_pool
    from jax.memory.db import MemoryDB
    pool=await _b9_ci_test_pool()
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute('CREATE TEMPORARY TABLE messages (id BIGINT,conversation_id BIGINT,turn_number INT,role VARCHAR(32),content TEXT) ENGINE=Aria')
                await cur.execute("INSERT INTO messages VALUES (2,1,1,'assistant','segundo'),(1,1,1,'user','primero')")
            await conn.commit()
        import aiomysql
        class _Conn:  # MemoryDB lee tuplas; el pool de test es DictCursor
            def __init__(self,c): self._c=c
            def cursor(self): return self._c.cursor(aiomysql.Cursor)
        class _Acq:
            async def __aenter__(self_): self_.c=await pool.acquire(); return _Conn(self_.c)
            async def __aexit__(self_,*a): pool.release(self_.c); return False
        db=MemoryDB(); db.pool=type('P',(),{'acquire':staticmethod(lambda:_Acq())})()
        got=await db.get_conversation_messages(1)
        assert [m['message_id'] for m in got]==[1,2]
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                import inspect
                from jax.memory import b9_mariadb
                src=inspect.getsource(b9_mariadb)
                start=src.index('SELECT id AS message_id,turn_number,role,content FROM messages')
                sql=src[start:src.index('"',start)].replace('conversation_id=%s','conversation_id=1').replace('LIMIT %s','LIMIT 10')
                await cur.execute(sql)
                rows=await cur.fetchall()
        assert [r['message_id'] for r in rows]==[1,2]
    finally:
        pool.close(); await pool.wait_closed()


# --- Auditoria 2026-10-05, MAJOR-1: salida mal formada del extractor != fuente invalida ---

def _process_claimed_con(raw, *, bounds_error=None):
    """Corre process_claimed con un extractor doble que responde `raw`."""
    import asyncio, time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from jax.memory import worker
    from jax.memory.b9 import ScopeContext
    scope=ScopeContext('service:memory-extraction','SERVICE','1','1',calling_component='memory-extraction')
    async def op(operation): return await operation(None)
    api=SimpleNamespace(_store=SimpleNamespace(mutation=op),_auth=AsyncMock(return_value=SimpleNamespace(scope=scope)),_project_permissions=Mock())
    writer=SimpleNamespace(_api=api,_build_request_scope=AsyncMock(return_value=scope),persist_frozen=AsyncMock())
    messages=[{'message_id':1,'turn_number':1,'role':'user','content':'hola'}]
    db=SimpleNamespace(get_conversation_messages=AsyncMock(return_value=messages))
    jobs=SimpleNamespace(freeze=AsyncMock(),fail=AsyncMock(),
        validate_message_bounds=AsyncMock(side_effect=bounds_error))
    llm=SimpleNamespace(invoke=AsyncMock(return_value=raw))
    job={'claim_token':'token','request_id':'r','trace_id':'t','frozen_output':None}
    ok=asyncio.run(worker.process_claimed(db,llm,{'id':1,'user_id':1,'tenant_id':1},writer,jobs,job,
                                          {'calls':0,'max_calls':5},time.monotonic()+10))
    return ok,jobs,writer


@pytest.mark.parametrize('raw',[
    '{"facts":[{"text":"sin procedencia"}]}',                      # fact sin source_turns
    'esto no es json',                                            # no parseable
    '{"facts":[{"text":"x","source_turns":[{"message_id":"9","turn_number":9,"role":"user"}]}]}',  # turno inventado
    '{"facts":{"no":"es lista"}}',
])
def test_salida_mal_formada_del_extractor_reintenta_y_no_pone_en_cuarentena(raw):
    ok,jobs,writer=_process_claimed_con(raw)
    assert ok is False
    jobs.fail.assert_awaited_once()
    args,kwargs=jobs.fail.await_args
    assert kwargs.get('quarantine') is False, (args,kwargs)   # RETRY: fail() cuarentena solo al agotar intentos
    assert kwargs.get('unknown') in (None,False)
    assert args[2]=='EXTRACTOR_OUTPUT_INVALID'
    jobs.freeze.assert_not_called(); writer.persist_frozen.assert_not_called()


def test_limites_de_la_fuente_siguen_en_cuarentena_inmediata():
    ok,jobs,_=_process_claimed_con('{}',bounds_error=ValueError('conversation input limit exceeded'))
    assert ok is False
    assert jobs.fail.await_args.kwargs.get('quarantine') is True


@pytest.mark.asyncio
async def test_real_salida_invalida_agota_intentos_y_recien_ahi_cuarentena():
    """Con MariaDB: RETRY mientras queden intentos; QUARANTINED al agotarlos (JAX_MEMORY_MAX_ATTEMPTS)."""
    pool,api,jobs,job,request=await real_extraction_fixture()
    try:
        async def estado():
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT state,error_code,attempts FROM memory_extraction_jobs WHERE conversation_id=1')
                    return await cur.fetchone()
        jobs.max_attempts=2
        await jobs.fail(1,job['claim_token'],'EXTRACTOR_OUTPUT_INVALID',quarantine=False)   # intento 1 de 2
        assert (await estado())['state']=='RETRY'
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE memory_extraction_jobs SET next_attempt_at=NULL WHERE conversation_id=1")
            await conn.commit()
        job2=await jobs.claim(1,run_id='r2')                                                  # intento 2 de 2
        assert job2 and job2['attempts']==2
        await jobs.fail(1,job2['claim_token'],'EXTRACTOR_OUTPUT_INVALID',quarantine=False)
        assert (await estado())['state']=='QUARANTINED'
    finally:
        pool.close(); await pool.wait_closed()


# --- MAJOR-1 (cont.): la unidad queda roja mientras haya jobs atascados ---

def _run_once_con(monkeypatch, *, pending, stuck, claim=None):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from jax.memory import worker
    db=SimpleNamespace(pool=object(),connect=AsyncMock(return_value=True),close=AsyncMock())
    jobs=SimpleNamespace(pending=AsyncMock(return_value=pending),claim=AsyncMock(return_value=claim),
                         stuck_count=AsyncMock(return_value=stuck))
    monkeypatch.setenv('JAX_DB_HOST','isolated-test')
    monkeypatch.setattr(worker,'MemoryDB',lambda:db)
    monkeypatch.setattr(worker,'ExtractionJobs',lambda *a,**k:jobs)
    monkeypatch.setattr(worker,'build_extractor',AsyncMock(return_value=object()))
    monkeypatch.setattr(worker,'cerrar_cliente_http',AsyncMock())
    return worker,db,jobs


@pytest.mark.asyncio
async def test_cola_vacia_pero_con_jobs_en_cuarentena_sale_con_error(monkeypatch):
    """pending() excluye QUARANTINED: sin esto la cola 'vacia' salia verde para siempre."""
    worker,db,jobs=_run_once_con(monkeypatch,pending=[],stuck=2)
    with pytest.raises(RuntimeError,match='stuck.*2'): await worker.run_once(b9_writer=object())
    db.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_corrida_sana_sin_atascados_sale_verde(monkeypatch):
    worker,db,jobs=_run_once_con(monkeypatch,pending=[],stuck=0)
    await worker.run_once(b9_writer=object())


@pytest.mark.asyncio
async def test_atascados_se_cuentan_aunque_la_corrida_procese_bien(monkeypatch):
    worker,db,jobs=_run_once_con(monkeypatch,pending=[{'id':1}],stuck=1,claim={'claim_token':'t'})
    monkeypatch.setattr(worker,'process_claimed',__import__('unittest.mock',fromlist=['AsyncMock']).AsyncMock(return_value=True))
    with pytest.raises(RuntimeError,match='stuck.*1'): await worker.run_once(b9_writer=object())


@pytest.mark.asyncio
async def test_real_stuck_count_cuenta_quarantined_y_unknown():
    pool,api,jobs,job,request=await real_extraction_fixture()
    try:
        assert await jobs.stuck_count()==0
        await jobs.fail(1,job['claim_token'],'X',quarantine=True)
        assert await jobs.stuck_count()==1
    finally:
        pool.close(); await pool.wait_closed()


# --- MAJOR-1 (cont.): re-encolar un job en cuarentena tras revisarlo, con auditoria ---

async def _fixture_con_eventos():
    pool,api,jobs,job,request=await real_extraction_fixture()
    from pathlib import Path
    sql=(Path(__file__).parents[1]/'jax/memory/b9_migrations/007_extraction_job_events.sql').read_text()
    sql='\n'.join(l for l in sql.splitlines() if not l.lstrip().startswith('--'))
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for st in sql.split(';'):
                if st.strip().startswith('CREATE TABLE IF NOT EXISTS'):
                    await cur.execute(st.replace('CREATE TABLE IF NOT EXISTS','CREATE TEMPORARY TABLE'))
        await conn.commit()
    return pool,api,jobs,job,request


@pytest.mark.asyncio
async def test_real_reencolar_deja_el_job_listo_y_registra_el_evento():
    pool,api,jobs,job,request=await _fixture_con_eventos()
    try:
        await jobs.fail(1,job['claim_token'],'ValueError',quarantine=True)
        res=await jobs.requeue(1,actor='fernando',reason='revisado: salida mal formada, reintentar')
        assert res['previous_state']=='QUARANTINED'
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute('SELECT state,attempts,error_code,next_attempt_at,claim_token,lease_until FROM memory_extraction_jobs WHERE conversation_id=1')
                row=await cur.fetchone()
                assert row=={'state':'READY','attempts':0,'error_code':None,'next_attempt_at':None,'claim_token':None,'lease_until':None}
                await cur.execute('SELECT event_kind,actor,reason,details,conversation_id FROM memory_extraction_job_events')
                ev=await cur.fetchall()
        assert len(ev)==1 and ev[0]['event_kind']=='REQUEUE' and ev[0]['actor']=='fernando'
        assert ev[0]['conversation_id']==1 and 'ValueError' in ev[0]['details'] and 'QUARANTINED' in ev[0]['details']
        assert [c['id'] for c in await jobs.pending(5)]==[1]          # vuelve a la cola
        assert await jobs.stuck_count()==0
    finally:
        pool.close(); await pool.wait_closed()


@pytest.mark.asyncio
async def test_real_reencolar_rechaza_lo_que_no_esta_en_cuarentena_y_exige_actor_y_motivo():
    pool,api,jobs,job,request=await _fixture_con_eventos()
    try:
        with pytest.raises(ValueError,match='QUARANTINED'):            # esta RUNNING
            await jobs.requeue(1,actor='fernando',reason='x')
        await jobs.fail(1,job['claim_token'],'ValueError',quarantine=True)
        for actor,reason in (('',"r"),('f','')):
            with pytest.raises(ValueError): await jobs.requeue(1,actor=actor,reason=reason)
        with pytest.raises(ValueError,match='no existe'): await jobs.requeue(999,actor='f',reason='r')
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute('SELECT COUNT(*) AS n FROM memory_extraction_job_events')
                assert (await cur.fetchone())['n']==0                  # nada rechazado deja evento
    finally:
        pool.close(); await pool.wait_closed()


def test_cli_reencolar_y_atascados_estan_documentados_en_el_modulo():
    from jax.memory import worker
    assert '--reencolar' in worker.__doc__ and '--atascados' in worker.__doc__
    args=worker._parse_args(['--reencolar','7','--motivo','revisado','--actor','fernando'])
    assert (args.reencolar,args.motivo,args.actor)==(7,'revisado','fernando')
    with pytest.raises(SystemExit): worker._parse_args(['--reencolar','7'])          # sin motivo no hay re-encolado
