"""Durable synthesis claim/freeze boundary; schema is installed by migration only.

An abandoned paid call is UNKNOWN, never automatically paid for again. Frozen
output can be resumed without the model. Publication dedup lives in the B9 TX.
"""
import hashlib
import json
import os
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class SynthesisJobs:
    def __init__(self, pool):
        self.pool = pool

    async def claim(self, tenant, user, project, sources, version):
        sources = sorted(set(sources))
        key = digest([str(tenant), str(user), None if project is None else str(project), sources, version])
        token = str(uuid.uuid4())
        lease = int(os.getenv("JAX_MEMORY_SYNTHESIS_LEASE_SECONDS", "900"))
        if lease <= float(os.getenv("JAX_MEMORY_RUN_TIMEOUT_SECONDS", "840")):
            raise ValueError("synthesis lease must exceed run deadline")
        async with self.pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute("INSERT IGNORE INTO memory_synthesis_jobs (job_key,tenant_id,user_id,project_id,source_revision_ids,transformation_version,state,claim_token,lease_until) VALUES (%s,%s,%s,%s,%s,%s,'CLAIMED',%s,TIMESTAMPADD(SECOND,%s,NOW()))", (key,tenant,user,project,canonical(sources),version,token,lease))
                    new = cur.rowcount == 1
                    await cur.execute("SELECT state,claim_token,frozen_output,lease_until<=NOW() expired FROM memory_synthesis_jobs WHERE job_key=%s FOR UPDATE", (key,))
                    row = await cur.fetchone()
                    if new:
                        result = (key, token, None)
                    elif row["state"] == "FROZEN" and row["expired"]:
                        await cur.execute("UPDATE memory_synthesis_jobs SET claim_token=%s,lease_until=TIMESTAMPADD(SECOND,%s,NOW()) WHERE job_key=%s", (token,lease,key))
                        result = (key,token,json.loads(row["frozen_output"]))
                    elif row["state"] == "CLAIMED" and row["expired"]:
                        await cur.execute("UPDATE memory_synthesis_jobs SET state='UNKNOWN' WHERE job_key=%s", (key,))
                        raise RuntimeError("synthesis paid-call outcome unknown; controlled reconciliation required")
                    elif row["state"] == "UNKNOWN":
                        raise RuntimeError("synthesis job requires controlled reconciliation")
                    else:
                        result = None
                await conn.commit()
                return result
            except BaseException:
                # UNKNOWN must persist even though the caller observes failure.
                if 'row' in locals() and row["state"] == "CLAIMED" and row["expired"]:
                    await conn.commit()
                else:
                    await conn.rollback()
                raise

    async def freeze(self, key, token, items):
        async with self.pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute("UPDATE memory_synthesis_jobs SET frozen_output=%s,state=%s WHERE job_key=%s AND claim_token=%s AND state='CLAIMED' AND lease_until>NOW()", (canonical(items),"FROZEN" if items else "COMPLETED",key,token))
                    if cur.rowcount != 1:
                        raise RuntimeError("synthesis claim no longer owned")
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
