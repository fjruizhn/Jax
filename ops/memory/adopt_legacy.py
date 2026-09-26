"""Explicit administrator operation from a reviewed merged checkout.

No credentials are loaded at import. Output contains counts and error types
only. Reader checks are administrator diagnostics, not simulated chat logins.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import stat
import sys


def positive_user(value):
    parsed=int(value)
    if parsed<=0:raise argparse.ArgumentTypeError('user IDs must be positive')
    return parsed


def bounded_timeout(value):
    parsed=int(value)
    if not 1<=parsed<=900:raise argparse.ArgumentTypeError('timeout must be between 1 and 900 seconds')
    return parsed


def parser():
    result=argparse.ArgumentParser(description='Apply a private reviewed legacy adoption plan, repeat it to check idempotency, and report administrator reader diagnostics.')
    result.add_argument('--env-file',type=Path,required=True,help='explicit private credential file; never loaded at import')
    result.add_argument('--plan',type=Path,required=True,help='reviewed 0600 dry-run plan')
    result.add_argument('--backup-manifest',type=Path,required=True,help='0600 verified restoration manifest')
    result.add_argument('--actor-user-id',type=positive_user,required=True,help='actual administrator performing this operation')
    result.add_argument('--expected-database',required=True,help='exact required database name')
    result.add_argument('--verify-owner-id',type=positive_user,required=True,help='owner identity checked by administrator diagnostic')
    result.add_argument('--verify-other-user-id',type=positive_user,required=True,help='different identity in the same tenant')
    result.add_argument('--timeout-seconds',type=bounded_timeout,default=300,help='total operation deadline, 1..900 seconds (default: 300)')
    return result


def read_credentials(path):
    from dotenv import dotenv_values
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'r',encoding='utf-8') as source:
        info=os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode)&0o007:
            raise ValueError('credential_file_not_private')
        values=dotenv_values(stream=source,interpolate=False)
    required=('JAX_DB_HOST','JAX_DB_PORT','JAX_DB_USER','JAX_DB_PASSWORD','JAX_DB_NAME')
    if any(not values.get(key) for key in required):raise ValueError('credential_configuration_incomplete')
    return values


async def operate(args,credentials):
    import aiomysql
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from jax.memory.b9 import ScopeContext,MutationAuthorizationRequest,Visibility,AuthorizationDenied
    from jax.memory.b9_mariadb import MariaDBB9Store,PersistentMemoryAPI,MariaDBB9Reader
    from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
    from jax.memory.legacy_adoption import apply_plan,private_read,validate_plan
    if credentials['JAX_DB_NAME']!=args.expected_database:raise RuntimeError('operation_database_guard')
    if args.verify_owner_id==args.verify_other_user_id:raise ValueError('reader_diagnostic_requires_distinct_identities')
    # Validate the reviewed plan before opening a connection or attempting writes.
    plan=private_read(args.plan);validate_plan(plan)
    pool=await aiomysql.create_pool(host=credentials['JAX_DB_HOST'],port=int(credentials['JAX_DB_PORT']),user=credentials['JAX_DB_USER'],password=credentials['JAX_DB_PASSWORD'],db=credentials['JAX_DB_NAME'],cursorclass=aiomysql.DictCursor,minsize=1,maxsize=3,autocommit=False,connect_timeout=10)
    async def count(table):
        if table not in {'memory_events','memory_objects','memory_legacy_bindings'}:raise ValueError('count_allowlist')
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute('SELECT COUNT(*) AS n FROM '+table)
                row=await cur.fetchone()
            await conn.rollback()
        return row['n']
    async def identity(uid):
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute('SELECT tenant_id,status FROM jax_users WHERE user_id=%s',(uid,))
                row=await cur.fetchone()
            await conn.rollback()
        if not row or str(row['status']).lower()!='active' or row['tenant_id'] is None:
            raise AuthorizationDenied('diagnostic_identity_unavailable')
        return str(row['tenant_id'])
    try:
        resolver=MariaDBScopeAuthorityResolver(pool)
        api=PersistentMemoryAPI(MariaDBB9Store(pool),resolver)
        actor_tenant=await identity(args.actor_user_id)
        if any(item['destination']['tenant_id']!=actor_tenant for item in plan['items'] if not item['quarantine_reason']):
            raise AuthorizationDenied('plan_destination_outside_actor_tenant')
        owner_tenant=await identity(args.verify_owner_id)
        other_tenant=await identity(args.verify_other_user_id)
        if owner_tenant!=actor_tenant or other_tenant!=actor_tenant:
            raise AuthorizationDenied('reader_diagnostic_outside_actor_tenant')
        # Diagnostic access is conditioned on the actual administrator, not
        # conferred by constructing requested owner/other ScopeContexts.
        actor_scope=ScopeContext(f'user:{args.actor_user_id}','USER',str(args.actor_user_id),actor_tenant,None,calling_component='legacy-adoption')
        async def admin_guard(cur):
            resolved=await api._auth(cur,MutationAuthorizationRequest(actor_scope,'IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL),'IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL)
            if 'memory:admin' not in resolved.resolved_capabilities:
                raise AuthorizationDenied('operation_requires_administrator')
        await api._store.mutation(admin_guard)
        before=await count('memory_events')
        applied=await apply_plan(api,args.plan,args.backup_manifest,actor_user_id=args.actor_user_id)
        after=await count('memory_events')
        repeated=await apply_plan(api,args.plan,args.backup_manifest,actor_user_id=args.actor_user_id)
        final=await count('memory_events')
        if final!=after:raise RuntimeError('adoption_repeat_created_events')
        reader=MariaDBB9Reader(pool,resolver)
        visible={}
        for label,uid in (('owner',args.verify_owner_id),('other',args.verify_other_user_id)):
            await api._store.mutation(admin_guard)
            tenant=await identity(uid)
            if tenant!=actor_tenant:raise AuthorizationDenied('diagnostic_identity_tenant_changed')
            request=MutationAuthorizationRequest(ScopeContext(f'user:{uid}','USER',str(uid),tenant,None),'RETRIEVE',Visibility.USER_PRIVATE)
            visible[label]=len(await reader.retrieve_authorized(request,limit=100))
        return {'status':'PASS','applied':applied,'repeat_returned':repeated,'events_added':after-before,'repeat_events_added':final-after,'objects':await count('memory_objects'),'bindings':await count('memory_legacy_bindings'),'administrator_reader_counts':visible,'reader_limit':100}
    finally:
        pool.close()
        await asyncio.wait_for(pool.wait_closed(),15)


def main(argv=None):
    args=parser().parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        credentials=read_credentials(args.env_file)
        result=asyncio.run(asyncio.wait_for(operate(args,credentials),args.timeout_seconds))
        print(json.dumps(result,sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({'status':'FAIL','error_type':type(error).__name__}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
