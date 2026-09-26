"""Explicit integration driver for a newly provisioned isolated MariaDB database.

Run as a Python file, never through pytest/conftest. No schema provisioning,
production credentials, providers, database drop, or global cleanup.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math
import time
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import aiomysql
from jax.memory import worker as W
from jax.memory.b9 import ScopeContext, MutationAuthorizationRequest, Visibility, EmbeddingSpaceIdentity, ScopeDenied, AuthorizationDenied, ObjectKind
from jax.memory.b9_mariadb import MariaDBB9Store, PersistentMemoryAPI, MariaDBB9Reader
from jax.memory.mapping_pool import MappingPool
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
from jax.memory.extraction_jobs import ExtractionJobs, normalize_extraction, source_digest
from jax.memory.lifecycle_worker import scan_and_mark
from jax.memory.legacy_adoption import source_content, row_digest
from jax.memory.synthesis_jobs import SynthesisJobs, digest as synthesis_digest
from jax.memory.synthesis_worker import build_persistent_synthesis_writer_for_scope

PAYLOAD={"facts":[{"text":"integration fact one"},{"text":"integration fact two"}],"decisions":[{"title":"integration decision","chosen":"chosen","reasoning":"reason"}],"action_items":[{"description":"integration action"}]}


OLD_RETRIEVAL_SELECT = "SELECT o.memory_id,o.object_kind,o.tenant_id,UNIX_TIMESTAMP(o.created_at) AS object_created_at,r.revision_id,r.content_digest,r.visibility,r.user_id,r.project_id,r.lifecycle_state,UNIX_TIMESTAMP(r.created_at) AS revision_created_at,p.payload,r.provenance_status,r.prior_revision_id FROM memory_objects o JOIN memory_projections pr ON pr.memory_id=o.memory_id JOIN memory_revisions r ON r.revision_id=pr.current_revision_id LEFT JOIN memory_revision_payloads p ON p.revision_id=r.revision_id WHERE o.tenant_id=%s AND pr.reconciliation_required=FALSE AND r.lifecycle_state NOT IN ('TOMBSTONED','PURGED','EXPIRED') AND (r.visibility <> 'USER_PRIVATE' OR r.user_id=%s) AND (r.project_id IS NULL OR r.project_id=%s) ORDER BY r.created_at DESC LIMIT %s"

class Extractor:
    def __init__(self): self.calls=0; self.invalid_once=False; self.empty=False
    async def invoke(self,*args,**kwargs):
        self.calls+=1
        await asyncio.sleep(0)
        if self.invalid_once:
            self.invalid_once=False
            return "not valid json"
        return json.dumps({"facts":[],"decisions":[],"action_items":[]} if self.empty else PAYLOAD)


class FaultPool:
    def __init__(self,pool,mode): self.pool=pool;self.mode=mode;self.fired=False;self.recorded={}
    @asynccontextmanager
    async def acquire(self):
        async with self.pool.acquire() as conn: yield FaultConnection(conn,self)


class FaultConnection:
    def __init__(self,conn,fault): self.conn=conn;self.fault=fault;self.marker=False
    def __getattr__(self,name): return getattr(self.conn,name)
    @asynccontextmanager
    async def cursor(self,*args):
        async with self.conn.cursor(*args) as cur: yield FaultCursor(cur,self)
    async def commit(self):
        await self.conn.commit()
        if self.marker and self.fault.mode=="lost_ack" and not self.fault.fired:
            self.fault.fired=True
            raise RuntimeError("injected lost commit acknowledgement")


class FaultCursor:
    def __init__(self,cur,connection): self.cur=cur;self.connection=connection
    def __getattr__(self,name): return getattr(self.cur,name)
    async def execute(self,sql,args=None):
        if self.connection.fault.mode=="capture" and ((sql.startswith("SELECT") and "memory_objects" in sql and "LIMIT" in sql) or "FROM memory_provenance WHERE" in sql):
            self.connection.fault.recorded.setdefault(sql,args)
        if sql.startswith("UPDATE conversations SET memory_processed=TRUE"):
            self.connection.marker=True
            fault=self.connection.fault
            if fault.mode=="processed_update" and not fault.fired:
                fault.fired=True
                raise RuntimeError("injected processed update failure")
        return await self.cur.execute(sql,args)


class Driver:
    def __init__(self,pool):
        self.pool=pool;self.mapping=MappingPool(pool);self.fake=Extractor();self.results={};self.none_messages=set()
        self.api=PersistentMemoryAPI(MariaDBB9Store(self.mapping),MariaDBScopeAuthorityResolver(self.mapping))
        self.reader=MariaDBB9Reader(self.mapping,MariaDBScopeAuthorityResolver(self.mapping))
        driver=self
        class ReadDB:
            def __init__(self): self.pool=driver.pool
            async def connect(self,**kwargs): return True
            async def close(self): pass
            async def get_conversation_messages(self,cid):
                if cid in driver.none_messages: return None
                return await driver.query("SELECT role,content FROM messages WHERE conversation_id=%s ORDER BY turn_number",(cid,))
        W.MemoryDB=ReadDB
        async def factory(): return driver.fake
        W.build_extractor=factory

    async def query(self,sql,args=()):
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql,args)
                rows=await cur.fetchall() if cur.description else None
                inserted=cur.lastrowid
            await conn.commit()
        return rows if rows is not None else inserted

    async def scalar(self,sql,args=()): return next(iter((await self.query(sql,args))[0].values()))

    async def seed(self,uid=1,project=None,empty=False):
        cid=await self.query("INSERT INTO conversations (conversation_uuid,tenant_id,user_id,project_id,ended_at,source) VALUES (%s,1,%s,%s,NOW(6),'b9-integration')",(str(uuid.uuid4()),uid,project))
        if not empty:
            await self.query("INSERT INTO messages (conversation_id,turn_number,role,content,user_id,project_id) VALUES (%s,1,'user','integration source',%s,%s)",(cid,uid,project))
        return cid

    async def state(self,cid):
        rows=await self.query("SELECT c.memory_processed,j.state,j.frozen_output IS NOT NULL frozen,(SELECT COUNT(*) FROM memory_extraction_results r WHERE r.conversation_id=c.id) results FROM conversations c LEFT JOIN memory_extraction_jobs j ON j.conversation_id=c.id WHERE c.id=%s",(cid,))
        return rows[0]

    async def total(self): return await self.scalar("SELECT COUNT(*) FROM memory_events")

    async def run(self,writer=None,expect_failure=False):
        try:
            await W.run_once(limit=100,b9_writer=writer)
        except Exception as exc:
            if not expect_failure: raise
            return type(exc).__name__
        if expect_failure: raise AssertionError("expected_visible_failure")
        return None

    async def retry_ready(self,cid):
        await self.query("UPDATE memory_extraction_jobs SET next_attempt_at=DATE_SUB(NOW(6),INTERVAL 1 SECOND),lease_until=NULL WHERE conversation_id=%s",(cid,))

    async def case(self,name,fn):
        self.current=name
        before=await self.total();calls=self.fake.calls
        await fn()
        self.results[name]={"events_delta":await self.total()-before,"llm_calls":self.fake.calls-calls,"status":"PASS"}

    async def cases(self):
        async def a1():
            cid=await self.seed();await self.run()
            state=await self.state(cid)
            assert state["memory_processed"]==1 and state["state"]=="COMPLETED" and state["results"]==4,"A1_atomic_markers"
            count=await self.scalar("SELECT COUNT(*) FROM memory_extraction_results x JOIN memory_events e ON e.revision_id=x.revision_id JOIN memory_revisions r ON r.revision_id=x.revision_id JOIN memory_provenance p ON p.revision_id=x.revision_id WHERE x.conversation_id=%s AND e.actor_principal='service:memory-extraction' AND e.actor_type='SERVICE' AND e.request_id IS NOT NULL AND e.trace_id IS NOT NULL AND r.visibility='USER_PRIVATE' AND r.user_id='1' AND r.project_id IS NULL AND p.transformation_id='conversation-extraction' AND JSON_EXTRACT(p.limitations,'$.conversation_id')=%s",(cid,cid))
            assert count==4,"A1_origin_scope"
            rows=await self.reader.retrieve(ScopeContext('user:4','USER','4','1',None),limit=100)
            assert not rows,"A1_cross_user_leak"
            self.first_cid=cid
        await self.case("A1_happy",a1)
        async def idle():
            total=await self.total();calls=self.fake.calls;await self.run()
            assert await self.total()==total and self.fake.calls==calls,"A2_idle_paid"
        await self.case("A2_idle",idle)
        async def poison(uid=None,project=None):
            bad=await self.seed(uid=uid,project=project);good=await self.seed()
            await self.run(expect_failure=True)
            assert (await self.state(bad))["state"]=="QUARANTINED","poison_not_quarantined"
            assert (await self.state(good))["results"]==4,"poison_blocks_healthy"
        await self.case("B_unbound_project",lambda:poison(1,900001))
        await self.case("C_inactive",lambda:poison(9))
        await self.case("C_missing_user",lambda:poison(None))
        async def fail_second():
            cid=await self.seed();writer=W.build_persistent_extraction_writer(self.pool)
            original=writer._api._write;n=0
            async def injected(*args,**kwargs):
                nonlocal n
                n+=1
                if n==2: raise RuntimeError("injected second write failure")
                return await original(*args,**kwargs)
            writer._api._write=injected
            before=await self.total();await self.run(writer,expect_failure=True)
            state=await self.state(cid)
            assert await self.total()==before and not state["memory_processed"] and state["results"]==0 and state["state"]=="UNKNOWN" and state["frozen"],"D_partial_write"
            calls=self.fake.calls;await self.retry_ready(cid);await self.run()
            assert self.fake.calls==calls and (await self.state(cid))["results"]==4,"D_retry_paid_or_duplicated"
        await self.case("D_second_write",fail_second)
        async def fail_marker():
            cid=await self.seed();fault=FaultPool(self.pool,"processed_update");writer=W.build_persistent_extraction_writer(fault)
            before=await self.total();await self.run(writer,expect_failure=True)
            assert fault.fired and await self.total()==before and not (await self.state(cid))["memory_processed"],"E_update_partial"
            calls=self.fake.calls;await self.retry_ready(cid);await self.run()
            assert self.fake.calls==calls and (await self.state(cid))["results"]==4,"E_retry_not_frozen"
        await self.case("E_processed_update",fail_marker)
        async def read_none():
            cid=await self.seed();self.none_messages.add(cid);calls=self.fake.calls
            await self.run(expect_failure=True);state=await self.state(cid)
            assert not state["memory_processed"] and state["state"]=="RETRY" and self.fake.calls==calls,"F_read_failed_open"
            self.none_messages.remove(cid);await self.retry_ready(cid);await self.run()
        await self.case("F_read_none",read_none)
        async def concurrent():
            cid=await self.seed();before=await self.total()
            await asyncio.gather(self.run(),self.run())
            assert await self.total()-before==4 and (await self.state(cid))["results"]==4,"H_duplicate_extraction"
        await self.case("H_concurrent",concurrent)
        async def lost_ack():
            cid=await self.seed();fault=FaultPool(self.pool,"lost_ack");writer=W.build_persistent_extraction_writer(fault)
            before=await self.total();await self.run(writer,expect_failure=True)
            state=await self.state(cid)
            assert fault.fired and state["state"]=="COMPLETED" and state["memory_processed"] and state["results"]==4,"ACK_commit_markers"
            calls=self.fake.calls;await self.run();assert self.fake.calls==calls and await self.total()-before==4,"ACK_retry_duplicate"
            job=(await self.query("SELECT * FROM memory_extraction_jobs WHERE conversation_id=%s",(cid,)))[0]
            auth=MutationAuthorizationRequest(ScopeContext('service:memory-extraction','SERVICE','1','1',None,calling_component='memory-extraction'),'CREATE',Visibility.USER_PRIVATE)
            assert len(await self.api.persist_conversation_extraction(auth,cid,job['claim_token']))==4,"ACK_authorized_idempotency"
            forged=MutationAuthorizationRequest(ScopeContext('service:memory-extraction','SERVICE','4','1',None,calling_component='memory-extraction'),'CREATE',Visibility.USER_PRIVATE)
            try:await self.api.persist_conversation_extraction(forged,cid,job['claim_token'])
            except ScopeDenied:  # fail-soft: expected denial of a forged subject retry; the else branch fails if accepted
                pass
            else:raise AssertionError("ACK_unauthorized_retry_accepted")
        await self.case("ACK_lost_commit_response",lost_ack)
        async def invalid():
            bad=await self.seed();good=await self.seed();self.fake.invalid_once=True
            await self.run(expect_failure=True)
            assert (await self.state(bad))["state"]=="QUARANTINED" and (await self.state(good))["results"]==4,"JSON_blocks_queue"
        await self.case("JSON_invalid",invalid)
        async def empty():
            cid=await self.seed(empty=True);calls=self.fake.calls;before=await self.total();await self.run()
            state=await self.state(cid)
            assert state["state"]=="COMPLETED" and state["memory_processed"] and state["results"]==0 and self.fake.calls==calls and await self.total()==before,"EMPTY_not_completed"
        await self.case("EMPTY_source",empty)
        async def expired_token():
            cid=await self.seed();jobs=ExtractionJobs(self.mapping)
            first=await jobs.claim(cid,run_id=str(uuid.uuid4()))
            conv=(await self.query("SELECT id,conversation_uuid AS uuid,tenant_id,user_id,project_id FROM conversations WHERE id=%s",(cid,)))[0]
            messages=await self.query("SELECT role,content FROM messages WHERE conversation_id=%s ORDER BY turn_number",(cid,))
            await jobs.freeze(cid,first['claim_token'],source_digest(conv,messages),normalize_extraction(PAYLOAD,max_items=100,max_text_chars=12000))
            await self.query("UPDATE memory_extraction_jobs SET lease_until=DATE_SUB(NOW(6),INTERVAL 1 SECOND) WHERE conversation_id=%s",(cid,))
            second=await jobs.claim(cid,run_id=str(uuid.uuid4()));assert first['claim_token']!=second['claim_token'],"LEASE_token_not_replaced"
            writer=W.build_persistent_extraction_writer(self.pool)
            try: await writer.persist_frozen(conv,first)
            except ScopeDenied:  # fail-soft: expected denial of a replaced lease token; the else branch fails if accepted
                pass
            else: raise AssertionError("LEASE_old_writer_accepted")
            await self.query("UPDATE messages SET content='changed integration source' WHERE conversation_id=%s",(cid,))
            budget={'calls':0,'max_calls':40}
            import time
            assert not await W.process_claimed(W.MemoryDB(),self.fake,conv,writer,jobs,second,budget,time.monotonic()+840),"SOURCE_changed_accepted"
            state=await self.state(cid)
            assert state["state"]=="QUARANTINED" and state["results"]==0,"SOURCE_not_quarantined"
        await self.case("LEASE_and_source_changed",expired_token)
        async def partial_marker():
            cid=await self.seed();jobs=ExtractionJobs(self.mapping);await jobs.claim(cid,run_id=str(uuid.uuid4()))
            await self.query("UPDATE memory_extraction_jobs SET state='UNKNOWN',lease_until=NULL WHERE conversation_id=%s",(cid,))
            # A persisted origin row without its processed/completed markers is ambiguous.
            await self.query("INSERT INTO memory_extraction_results (conversation_id,item_index,content_digest,memory_id,revision_id) VALUES (%s,0,%s,%s,%s)",(cid,'sha256:'+('0'*64),str(uuid.uuid4()),str(uuid.uuid4())))
            await self.run(expect_failure=True)
            assert (await self.state(cid))["state"]=="QUARANTINED","UNKNOWN_partial_not_quarantined"

        await self.case("UNKNOWN_partial_marker",partial_marker)
        async def adoption():
            fid=await self.query("INSERT INTO facts (fact_uuid,fact_text,fact_type,user_id,project_id) VALUES (%s,'synthetic legacy integration fact','technical',1,NULL)",(str(uuid.uuid4()),))
            source=(await self.query("SELECT * FROM facts WHERE id=%s",(fid,)))[0]
            source_hash=row_digest(source)
            content=source_content('facts',source)
            auth=MutationAuthorizationRequest(ScopeContext('user:1','USER','1','1',None,calling_component='legacy-adoption'),'IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL)
            before=await self.total()
            mid=await self.api.import_legacy_memory(auth,'facts','legacy',str(fid),ObjectKind.FACT,content,visibility=Visibility.USER_PRIVATE,user_id='1',expected_source_digest=source_hash)
            assert await self.total()-before==2,"ADOPT_import_scope_not_atomic"
            second=await self.api.import_legacy_memory(auth,'facts','legacy',str(fid),ObjectKind.FACT,content,visibility=Visibility.USER_PRIVATE,user_id='1',expected_source_digest=source_hash)
            assert second==mid and await self.total()-before==2,"ADOPT_repeated_events"
            operator=MutationAuthorizationRequest(ScopeContext('user:4','USER','4','1',None,calling_component='legacy-adoption'),'IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL)
            try:await self.api.import_legacy_memory(operator,'facts','legacy',str(fid),ObjectKind.FACT,content,visibility=Visibility.USER_PRIVATE,user_id='1',expected_source_digest=source_hash)
            except AuthorizationDenied:  # fail-soft: expected denial of operator legacy import; the else branch fails if accepted
                pass
            else:raise AssertionError("ADOPT_operator_import_allowed")
            assert not await self.reader.retrieve(ScopeContext('user:4','USER','4','1',None),limit=100),"ADOPT_private_leak"
            await self.query("UPDATE facts SET fact_text='changed synthetic legacy source' WHERE id=%s",(fid,))
            try:await self.api.import_legacy_memory(auth,'facts','legacy',str(fid),ObjectKind.FACT,content,visibility=Visibility.USER_PRIVATE,user_id='1',expected_source_digest=source_hash)
            except ScopeDenied:  # fail-soft: expected denial of changed legacy content; the else branch fails if accepted
                pass
            else:raise AssertionError("ADOPT_changed_source_allowed")
            assert await self.total()-before==2 and await scan_and_mark(self.pool)==(),"ADOPT_noncanonical_history"
        await self.case("ADOPT_real_source_and_dedup",adoption)
        async def synthesis():
            sources=await self.query("SELECT x.memory_id FROM memory_extraction_results x JOIN memory_objects o ON o.memory_id=x.memory_id WHERE x.conversation_id=%s AND o.object_kind='FACT' ORDER BY x.item_index",(self.first_cid,))
            authuser=ScopeContext('user:1','USER','1','1',None)
            for source in sources:
                await self.api.verify_memory(MutationAuthorizationRequest(authuser,'VERIFY',Visibility.USER_PRIVATE),source['memory_id'],method='integration controlled verification')
            revisions=[]
            for source in sources:
                revisions.append(await self.scalar("SELECT current_revision_id FROM memory_projections WHERE memory_id=%s",(source['memory_id'],)))
            revisions=sorted(revisions)
            writer=build_persistent_synthesis_writer_for_scope(self.pool,1)
            await writer.preflight(1,None,revisions,'USER_PRIVATE')
            try:await writer.preflight(9,None,revisions,'USER_PRIVATE')
            except ScopeDenied:  # fail-soft: expected denial of inactive synthesis subject; the else branch fails if accepted
                pass
            else:raise AssertionError('SYNTH_inactive_preflight_accepted')
            jobs=SynthesisJobs(self.mapping)
            claim=await jobs.claim(1,1,None,revisions,'b9-worker-v1');assert claim is not None,"SYNTH_claim_missing"
            key,token,_=claim
            item={'text':'synthetic integration insight','source_revision_ids':revisions}
            item['item_key']=synthesis_digest(item)
            await jobs.freeze(key,token,[item])
            auth=MutationAuthorizationRequest(ScopeContext('service:memory-synthesis','SERVICE','1','1',None,calling_component='memory-synthesis'),'SYNTHESIZE',Visibility.SYSTEM_INTERNAL)
            params={'provider':'test','model':'test','transformation_version':'b9-worker-v1','synthesis_job_key':key,'synthesis_job_token':token,'synthesis_item_key':item['item_key']}
            before=await self.total()
            mid=await self.api.synthesize_memory(auth,item['text'],tuple(revisions),**params)
            repeat=await self.api.synthesize_memory(auth,item['text'],tuple(revisions),**params)
            assert mid==repeat and await self.total()-before==1,"SYNTH_duplicate_publication"
            await self.api.correct_memory(MutationAuthorizationRequest(authuser,'CORRECT',Visibility.USER_PRIVATE),sources[0]['memory_id'],'corrected synthetic synthesis source')
            try:await self.api.synthesize_memory(auth,'insight using revoked revision',tuple(revisions),provider='test',model='test',transformation_version='integration-revoked')
            except ScopeDenied:  # fail-soft: expected denial of revoked synthesis revision; the else branch fails if accepted
                pass
            else:raise AssertionError("SYNTH_revoked_source_accepted")
            assert await scan_and_mark(self.pool)==(),"SYNTH_noncanonical_history"
        await self.case("SYNTH_verified_dedup_and_revocation",synthesis)
        async def load():
            # Dataset is canonical API output under explicit test fixture identities.
            for tenant,user in (('1','1'),('2','2')):
                active=await self.scalar("SELECT COUNT(*) FROM jax_users WHERE user_id=%s AND tenant_id=%s AND LOWER(status)='active'",(user,tenant))
                assert active==1,'LOAD_test_fixture_missing'
                request=MutationAuthorizationRequest(ScopeContext('user:'+user,'USER',user,tenant,None),'CREATE',Visibility.USER_PRIVATE)
                for index in range(500):
                    await self.api.create_memory(request,ObjectKind.FACT,'synthetic retrieval load item',Visibility.USER_PRIVATE,user_id=user)
            assert await self.scalar('SELECT COUNT(*) FROM memory_objects')>=1000,'LOAD_dataset_too_small'
            auth=MutationAuthorizationRequest(ScopeContext('user:1','USER','1','1',None),'RETRIEVE',Visibility.USER_PRIVATE)
            recorder=FaultPool(self.pool,'capture')
            captured=MappingPool(recorder)
            reader=MariaDBB9Reader(captured,MariaDBScopeAuthorityResolver(captured))
            await reader.retrieve_authorized(auth,limit=20)
            main_queries=[(sql,args) for sql,args in recorder.recorded.items() if 'memory_objects' in sql]
            assert main_queries,'LOAD_real_query_not_captured'
            new_sql,new_args=main_queries[0]
            old_args=('1','1',None,20)
            old_rows=await self.query(OLD_RETRIEVAL_SELECT,old_args)
            new_rows=await self.query(new_sql,new_args)
            assert [r['memory_id'] for r in old_rows]==[r['memory_id'] for r in new_rows],'LOAD_query_results_differ'
            assert all(str(r['tenant_id'])=='1' and str(r['user_id'])=='1' for r in new_rows),'LOAD_cross_tenant_leak'
            query_measurements=[]
            for label,sql,args in (('original_select',OLD_RETRIEVAL_SELECT,old_args),('current_select',new_sql,new_args)):
                for concurrency in (1,10,25):
                    timings=[]
                    async def sql_worker():
                        for _ in range(10):
                            started=time.perf_counter();await self.query(sql,args);timings.append(time.perf_counter()-started)
                    started=time.perf_counter()
                    await asyncio.gather(*(sql_worker() for _ in range(concurrency)))
                    elapsed=time.perf_counter()-started;ordered=sorted(timings)
                    query_measurements.append({'query':label,'concurrency':concurrency,'requests':len(timings),'rps':round(len(timings)/elapsed,2),'p95_ms':round(ordered[math.ceil(.95*len(ordered))-1]*1000,2)})
            end_to_end=[]
            async def reader_worker(timings):
                for _ in range(10):
                    started=time.perf_counter();await self.reader.retrieve_authorized(auth,limit=20);timings.append(time.perf_counter()-started)
            for concurrency in (1,10,25):
                timings=[];started=time.perf_counter()
                await asyncio.gather(*(reader_worker(timings) for _ in range(concurrency)))
                elapsed=time.perf_counter()-started;ordered=sorted(timings)
                end_to_end.append({'concurrency':concurrency,'requests':len(timings),'rps':round(len(timings)/elapsed,2),'p95_ms':round(ordered[math.ceil(.95*len(ordered))-1]*1000,2)})
            plans=[]
            for label,sql,args in (('original_select',OLD_RETRIEVAL_SELECT,old_args),('current_select',new_sql,new_args)):
                explained=await self.query('EXPLAIN '+sql,args)
                plans.append({'query':label,'plan':[{field:row.get(field) for field in ('table','type','key','rows','Extra')} for row in explained]})
            self.load={'workload':'canonical synthetic B9, >=1000 objects, two tenants, retrieval limit20','pool_maxsize':8,'semantic_ids_equal':True,'original_timestamp_ties':'original query has no tie-breaker; dataset timestamps are sequential API writes','query_measurements':query_measurements,'current_reader_measurements':end_to_end,'explain':plans}
        await self.case("LOAD_retrieval_and_explain",load)
        async def embedding():
            result=(await self.query("SELECT x.memory_id,p.current_revision_id AS revision_id FROM memory_extraction_results x JOIN memory_projections p ON p.memory_id=x.memory_id WHERE x.conversation_id=%s ORDER BY x.item_index LIMIT 1",(self.first_cid,)))[0]
            mid,rid=result['memory_id'],result['revision_id']
            service=ScopeContext('service:embedding','SERVICE','1','1',None,calling_component='embedding')
            auth=MutationAuthorizationRequest(service,'RE_EMBED',Visibility.USER_PRIVATE)
            identity=EmbeddingSpaceIdentity('integration','test','test-model',None,2,'unit','cosine')
            first=await self.api.reembed_memory(auth,mid,identity,(.1,.2),expected_revision_id=rid)
            second=await self.api.reembed_memory(auth,mid,identity,(.1,.2),expected_revision_id=rid)
            assert first==second,"EMBED_duplicate_generation"
            userauth=MutationAuthorizationRequest(ScopeContext('user:1','USER','1','1',None),'CORRECT',Visibility.USER_PRIVATE)
            await self.api.correct_memory(userauth,mid,'corrected integration source')
            try:await self.api.reembed_memory(auth,mid,identity,(.1,.2),expected_revision_id=rid)
            except ScopeDenied:  # fail-soft: expected denial of obsolete embedding revision; the else branch fails if accepted
                pass
            else:raise AssertionError("EMBED_stale_revision_accepted")
        await self.case("EMBED_stale_and_dedup",embedding)
        async def lifecycle():
            assert await scan_and_mark(self.pool)==(),"LIFECYCLE_false_positive"
            mid=await self.scalar("SELECT memory_id FROM memory_extraction_results WHERE conversation_id=%s ORDER BY item_index LIMIT 1",(self.first_cid,))
            await self.query("UPDATE memory_projections SET current_lifecycle_state='EXPIRED' WHERE memory_id=%s",(mid,))
            mismatches=await scan_and_mark(self.pool)
            assert mid in mismatches,"LIFECYCLE_missed_divergence"
            assert await self.scalar("SELECT reconciliation_required FROM memory_projections WHERE memory_id=%s",(mid,))==1,"LIFECYCLE_flag_missing"
        await self.case("LIFECYCLE_reconciliation",lifecycle)


async def main():
    env=os.environ
    if env.get('JAX_DB_USER')!='jax_test' or not env.get('JAX_DB_NAME','').startswith('jax_memory_test_memb9_'):
        raise RuntimeError('test_database_guard')
    for name in ('JAX_DB_HOST','JAX_DB_PORT','JAX_DB_PASSWORD'):
        if not env.get(name):raise RuntimeError('test_configuration_missing')
    pool=await aiomysql.create_pool(host=env['JAX_DB_HOST'],port=int(env['JAX_DB_PORT']),user=env['JAX_DB_USER'],password=env['JAX_DB_PASSWORD'],db=env['JAX_DB_NAME'],minsize=1,maxsize=8,autocommit=False,charset='utf8mb4')
    driver=Driver(pool)
    try:
        assert await driver.scalar('SELECT COUNT(*) FROM conversations')==0,'fresh_database_required'
        assert await driver.scalar('SELECT COUNT(*) FROM memory_objects')==0,'fresh_database_required'
        await driver.cases()
    except Exception as error:  # fail-closed: emit sanitized failing-case metadata and return rc=1 to the process entry point
        print(json.dumps({'status':'FAIL','error_type':type(error).__name__,'failed_case':getattr(driver,'current','setup'),'completed':driver.results},sort_keys=True))
        return 1
    finally:
        pool.close();await pool.wait_closed()
    print(json.dumps({'status':'PASS','cases':driver.results,'load':getattr(driver,'load',{})},sort_keys=True))
    return 0


if __name__=='__main__':
    logging.getLogger().setLevel(logging.CRITICAL)
    try:raise SystemExit(asyncio.run(main()))
    except Exception as error:
        print(json.dumps({'status':'FAIL','error_type':type(error).__name__}))
        raise SystemExit(1)
