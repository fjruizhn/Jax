"""Durable extraction claims, frozen normalized output and conservative recovery."""
from __future__ import annotations
import json
import uuid
from .b9 import _digest
from .b9_mariadb import MariaDBB9Store


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def source_digest(conversation, messages):
    scope={key: str(conversation.get(key)) if conversation.get(key) is not None else None
           for key in ('id','tenant_id','user_id','project_id')}
    scope['uuid']=conversation.get('uuid') or conversation.get('conversation_uuid')
    return _digest(canonical({'conversation':scope,'messages':messages}))


def normalize_extraction(data, *, max_items, max_text_chars):
    if not isinstance(data, dict) or set(data)-{'facts','decisions','action_items'}:
        raise ValueError('invalid extraction object')
    result=[]
    def text(item, field, required=True):
        value=item.get(field, '' if not required else None)
        if not isinstance(value,str) or (required and not value.strip()) or len(value)>max_text_chars:
            raise ValueError('invalid extraction text')
        return value
    for category, kind in (('facts','FACT'),('decisions','DECISION_MEMORY'),('action_items','ACTION_ITEM')):
        items=data.get(category, [])
        if not isinstance(items,list): raise ValueError('extraction category must be list')
        for item in items:
            if not isinstance(item,dict): raise ValueError('extraction item must be object')
            if category=='decisions':
                content=f"{text(item,'title')}\nChosen: {text(item,'chosen')}\nReasoning: {text(item,'reasoning',False)}"
            else: content=text(item, 'text' if category=='facts' else 'description')
            if len(content)>max_text_chars: raise ValueError('extraction content exceeds limit')
            result.append({'kind':kind,'content':content})
            if len(result)>max_items: raise ValueError('extraction item limit exceeded')
    return result


class ExtractionJobs:
    def __init__(self, pool, *, lease_seconds=900, max_attempts=3):
        self.store=MariaDBB9Store(pool)
        self.lease_seconds=lease_seconds
        self.max_attempts=max_attempts

    async def pending(self, limit):
        async def op(cur):
            await cur.execute("SELECT c.id,c.conversation_uuid AS uuid,c.tenant_id,c.user_id,c.project_id "
                "FROM conversations c LEFT JOIN memory_extraction_jobs j ON j.conversation_id=c.id "
                "WHERE c.ended_at IS NOT NULL AND c.memory_processed=FALSE "
                "AND (j.conversation_id IS NULL OR j.state='READY' OR j.state='UNKNOWN' "
                "OR (j.state='RETRY' AND (j.next_attempt_at IS NULL OR j.next_attempt_at<=NOW(6))) "
                "OR (j.state='RUNNING' AND j.lease_until<=NOW(6))) ORDER BY c.ended_at,c.id LIMIT %s",(limit,))
            return list(await cur.fetchall())
        return await self.store.mutation(op)

    async def validate_message_bounds(self, conversation_id, max_messages, max_chars):
        async def op(cur):
            await cur.execute("SELECT COUNT(*) AS message_count,COALESCE(SUM(CHAR_LENGTH(content)+CHAR_LENGTH(role)+3),0) AS character_count FROM messages WHERE conversation_id=%s",(conversation_id,))
            row=await cur.fetchone()
            if not row: raise RuntimeError('conversation message statistics unavailable')
            if row['message_count']>max_messages or row['character_count']>max_chars:
                raise ValueError('conversation input limit exceeded')
        await self.store.mutation(op)

    async def claim(self, conversation_id, *, run_id):
        async def op(cur):
            await cur.execute("INSERT INTO memory_extraction_jobs (conversation_id,state) VALUES (%s,'READY') ON DUPLICATE KEY UPDATE conversation_id=conversation_id",(conversation_id,))
            await cur.execute("SELECT *,lease_until>NOW(6) AS lease_active,next_attempt_at>NOW(6) AS retry_wait FROM memory_extraction_jobs WHERE conversation_id=%s FOR UPDATE",(conversation_id,))
            job=await cur.fetchone()
            if job['state']=='QUARANTINED' or job['retry_wait']:
                return None
            if job['state']=='RUNNING' and job['lease_active']: return None
            # Recovery observes committed markers, never deduces outcome from lease expiry.
            await cur.execute("SELECT id,memory_processed FROM conversations WHERE id=%s FOR UPDATE",(conversation_id,))
            source=await cur.fetchone()
            await cur.execute("SELECT item_index,content_digest,memory_id,revision_id FROM memory_extraction_results WHERE conversation_id=%s ORDER BY item_index FOR UPDATE",(conversation_id,))
            results=list(await cur.fetchall())
            if job['state']=='COMPLETED':
                output=job['frozen_output']
                if isinstance(output,bytes): output=output.decode('utf-8')
                try:
                    items=json.loads(output)
                    consistent=(source is not None and bool(source['memory_processed'])
                        and _digest(output)==job['output_digest'] and len(results)==len(items)
                        and all(row['item_index']==index and row['content_digest']==_digest(canonical(items[index])) for index,row in enumerate(results)))
                except (TypeError,ValueError,KeyError): consistent=False
                if consistent: return None
                await cur.execute("UPDATE memory_extraction_jobs SET state='QUARANTINED',error_code='INCONSISTENT_COMPLETION' WHERE conversation_id=%s",(conversation_id,))
                return {'quarantined':True}
            if source is None or source['memory_processed'] or results:
                await cur.execute("UPDATE memory_extraction_jobs SET state='QUARANTINED',error_code='INCONSISTENT_COMMIT_MARKERS' WHERE conversation_id=%s",(conversation_id,))
                return {'quarantined':True}
            if job['attempts']>=self.max_attempts:
                await cur.execute("UPDATE memory_extraction_jobs SET state='QUARANTINED',error_code='ATTEMPTS_EXHAUSTED' WHERE conversation_id=%s",(conversation_id,))
                return {'quarantined':True}
            token=str(uuid.uuid4()); request=str(uuid.uuid4()); trace=str(uuid.uuid4())
            await cur.execute("UPDATE memory_extraction_jobs SET state='RUNNING',claim_token=%s,lease_until=DATE_ADD(NOW(6), INTERVAL %s SECOND),attempts=attempts+1,request_id=%s,trace_id=%s,run_id=%s,error_code=NULL WHERE conversation_id=%s",(token,self.lease_seconds,request,trace,run_id,conversation_id))
            return dict(job, claim_token=token,request_id=request,trace_id=trace,run_id=run_id,state='RUNNING',attempts=job['attempts']+1)
        return await self.store.mutation(op)

    async def freeze(self, conversation_id, token, digest, items):
        output=canonical(items)
        async def op(cur):
            await cur.execute("SELECT state,claim_token,input_digest,frozen_output,output_digest,lease_until>NOW(6) AS lease_active FROM memory_extraction_jobs WHERE conversation_id=%s FOR UPDATE",(conversation_id,))
            job=await cur.fetchone()
            if not job or job['state']!='RUNNING' or job['claim_token']!=token or not job['lease_active']:
                raise RuntimeError('extraction claim lost')
            if job['frozen_output'] is not None:
                if job['input_digest']!=digest or job['frozen_output']!=output or job['output_digest']!=_digest(output):
                    raise ValueError('frozen extraction differs')
                return
            await cur.execute("UPDATE memory_extraction_jobs SET input_digest=%s,frozen_output=%s,output_digest=%s WHERE conversation_id=%s",(digest,output,_digest(output),conversation_id))
        await self.store.mutation(op)

    async def fail(self, conversation_id, token, error_code, *, quarantine=False, unknown=False):
        async def op(cur):
            await cur.execute("SELECT state,claim_token,attempts FROM memory_extraction_jobs WHERE conversation_id=%s FOR UPDATE",(conversation_id,))
            job=await cur.fetchone()
            if not job or job['claim_token']!=token or job['state']=='COMPLETED': return
            state='UNKNOWN' if unknown else ('QUARANTINED' if quarantine or job['attempts']>=self.max_attempts else 'RETRY')
            backoff=min(3600,30*2**min(job['attempts'],6))
            await cur.execute("UPDATE memory_extraction_jobs SET state=%s,error_code=%s,next_attempt_at=DATE_ADD(NOW(6), INTERVAL %s SECOND),lease_until=NULL WHERE conversation_id=%s",(state,error_code[:128],backoff,conversation_id))
        await self.store.mutation(op)
