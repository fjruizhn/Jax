"""MariaDB persistence adapter for B9 atomic memory bundles.

It is deliberately not coupled to legacy ``MemoryDB`` writes: callers migrate
through the B9 API/adoption path.  The adapter never applies migrations and
never falls back to partial autocommit writes.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Mapping

from .b9 import (
    AuthorizationDenied, B9Error, EmbeddingGeneration, EmbeddingSpaceIdentity,
    EventKind, Lifecycle, MemoryEnvelope, MemoryEvent, MemoryObject, MemoryProjection,
    MemoryProvenance, MemoryRevision, MutationAuthorizationContext, MutationAuthorizationRequest, ObjectKind,
    PromptMemoryContext, ReconciliationRequired, ScopeContext, ScopeDenied, Visibility,
    SYNTHESIS_SOURCE_KINDS, _derive_projection, _digest, _uuid7,
)


class MariaDBB9Store:
    def __init__(self, pool: Any):
        self._pool = pool

    async def persist_bundle(self, obj: MemoryObject, revision: MemoryRevision,
                             provenance: MemoryProvenance, event: MemoryEvent,
                             projection: MemoryProjection) -> None:
        """Persist one logical mutation atomically.

        There is intentionally no public method that inserts a state row or
        event row independently.  Any failure rolls back the complete bundle.
        """
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO memory_objects "
                        "(memory_id,object_kind,tenant_id,created_at,legacy_source_type,legacy_source_namespace,legacy_source_key) "
                        "VALUES (%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s) "
                        "ON DUPLICATE KEY UPDATE memory_id=memory_id",
                        (obj.memory_id,obj.kind.value,obj.tenant_id,obj.created_at,
                         *(obj.legacy_binding or (None,None,None))),
                    )
                    await cur.execute(
                        "INSERT INTO memory_revisions "
                        "(revision_id,memory_id,content_digest,visibility,user_id,project_id,lifecycle_state,created_at,payload,provenance_status,prior_revision_id,tenant_id) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s)",
                        (revision.revision_id,revision.memory_id,revision.content_digest,revision.visibility.value,
                         revision.user_id,revision.project_id,revision.lifecycle.value,revision.created_at,
                        revision.payload,revision.provenance_status,revision.prior_revision_id,obj.tenant_id),
                    )
                    # Payload is separately addressable for privacy purge.  The
                    # legacy revision column remains populated only during the
                    # additive transition and is not the B9 retrieval source.
                    await cur.execute(
                        "INSERT INTO memory_revision_payloads (revision_id,payload) VALUES (%s,%s)",
                        (revision.revision_id, revision.payload),
                    )
                    await cur.execute(
                        "INSERT INTO memory_provenance "
                        "(provenance_id,revision_id,source_revisions,transformation_id,transformation_version,actor_principal,actor_type,subject_user_id,provider,model,created_at,limitations) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s)",
                        (provenance.provenance_id,provenance.revision_id,json.dumps(provenance.source_revisions),
                         provenance.transformation_id,provenance.transformation_version,provenance.actor_principal,
                         provenance.actor_type,provenance.subject_user_id,provenance.provider,provenance.model,
                         provenance.created_at,provenance.limitations),
                    )
                    await cur.execute(
                        "INSERT INTO memory_events "
                        "(event_id,memory_id,revision_id,event_kind,actor_principal,subject_user_id,authority_source,occurred_at,details,compensates_event_id,actor_type,delegation,calling_component,request_id,trace_id) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s,%s,%s,%s)",
                        (event.event_id,event.memory_id,event.revision_id,event.kind.value,event.actor_principal,
                         event.subject_user_id,event.authority_source,event.occurred_at,json.dumps(dict(event.details)),
                         event.compensates_event_id,event.actor_type,event.delegation,event.calling_component,
                         event.request_id,event.trace_id),
                    )
                    await cur.execute(
                        "INSERT INTO memory_projections "
                        "(memory_id,current_revision_id,current_lifecycle_state,current_verification_state,canonical_history_digest,reconciliation_required) "
                        "VALUES (%s,%s,%s,%s,%s,%s) "
                        "ON DUPLICATE KEY UPDATE current_revision_id=VALUES(current_revision_id),"
                        "current_lifecycle_state=VALUES(current_lifecycle_state),"
                        "current_verification_state=VALUES(current_verification_state),"
                        "canonical_history_digest=VALUES(canonical_history_digest),"
                        "reconciliation_required=VALUES(reconciliation_required)",
                        (projection.memory_id,projection.current_revision_id,
                         projection.current_lifecycle.value if projection.current_lifecycle else None,
                         projection.current_verification,projection.canonical_history_digest,
                         projection.reconciliation_required),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def mutation(self, operation: Any) -> Any:
        """Run a complete B9 mutation on one MariaDB transaction.

        ``operation`` receives the open mapping cursor.  It is intentionally
        the only primitive used by ``PersistentMemoryAPI``; callers cannot
        expose a partially persisted revision/event bundle.
        """
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    value = await operation(cur)
                await conn.commit()
                return value
            except BaseException:
                await conn.rollback()
                raise


class PersistentMemoryAPI:
    """MariaDB-backed B9 mutation boundary.

    ``authorization_resolver`` is the authority boundary.  The context passed
    to a public method is only a request description (actor, subject, target
    and trace metadata); it is deliberately re-resolved after the mutation
    transaction has begun.  A dataclass supplied by a caller is therefore not
    a capability and cannot survive a membership revocation.
    """
    def __init__(self, store: MariaDBB9Store, authorization_resolver: Any | None=None):
        self._store = store
        self._authorization_resolver = authorization_resolver

    @staticmethod
    def _request_scope(request: MutationAuthorizationRequest) -> ScopeContext:
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("mutation authorization request required")
        return request.scope

    async def _auth(self, cur: Any, request: MutationAuthorizationRequest,
                    operation: str, visibility: Visibility) -> MutationAuthorizationContext:
        scope = self._request_scope(request)
        resolver = self._authorization_resolver
        if resolver is None:
            raise AuthorizationDenied("persistent mutation requires transaction authority resolver")
        resolve = getattr(resolver, "resolve_mutation_in_transaction", None)
        if resolve is None:
            raise AuthorizationDenied("authority resolver does not support transaction revalidation")
        resolved = await resolve(cur, scope, operation, visibility)
        if not isinstance(resolved, MutationAuthorizationContext):
            raise AuthorizationDenied("authority resolver returned invalid decision")
        resolved_scope = resolved.scope
        resolved_scope.validate()
        if (resolved.operation != operation or resolved.target_visibility is not visibility
                or not resolved.authority_source):
            raise AuthorizationDenied("authority resolver returned mismatched decision")
        # The resolver, not the request, chooses the authoritative identity
        # and namespace.  It may normalize IDs but cannot change the actor or
        # subject into an implicit delegation.
        if (resolved_scope.actor_principal != scope.actor_principal
                or resolved_scope.actor_type != scope.actor_type
                or resolved_scope.subject_user_id != scope.subject_user_id):
            raise AuthorizationDenied("authority decision actor/subject mismatch")
        return resolved

    @staticmethod
    def _project_permissions(auth: MutationAuthorizationContext, operation: str,
                             visibility: Visibility) -> None:
        """Defence in depth for the project role matrix.

        The DB resolver remains authoritative.  These checks ensure a
        resolver regression cannot turn a VIEWER/CONTRIBUTOR decision into a
        persistent verify/admin operation.
        """
        if not auth.scope.project_id or visibility is not Visibility.PROJECT_SHARED:
            return
        caps = auth.resolved_capabilities
        if operation == "VERIFY" and "memory:project:verify" not in caps:
            raise AuthorizationDenied("project verification requires REVIEWER")
        if operation in {"CREATE", "CORRECT", "SUPERSEDE", "EXPIRE", "TOMBSTONE", "CONTENT_PURGE", "RE_SCOPE", "SYNTHESIZE"} and "memory:project:write" not in caps:
            if (operation == "SYNTHESIZE" and auth.scope.actor_type == "SERVICE"
                    and auth.scope.actor_principal == "service:memory-synthesis"
                    and "memory:service:synthesize" in caps):
                return
            raise AuthorizationDenied("project mutation requires CONTRIBUTOR")

    @staticmethod
    def _assert_project_binding(scope: ScopeContext, revision: MemoryRevision,
                                *, requested_project_id: str | None=None) -> None:
        """Keep every revision in the exact project namespace being authorized."""
        target = revision.project_id if requested_project_id is None else requested_project_id
        if scope.project_id:
            if revision.project_id != scope.project_id or target != scope.project_id:
                raise ScopeDenied("authorized project does not match memory revision")
        elif revision.project_id is not None or target is not None:
            raise ScopeDenied("project memory requires project authorization")

    @staticmethod
    def _event(scope: ScopeContext, auth: MutationAuthorizationContext, memory_id: str,
               revision_id: str | None, kind: EventKind, now: float, details: Mapping[str, Any] | None=None,
               compensates: str | None=None) -> MemoryEvent:
        return MemoryEvent(_uuid7(), memory_id, revision_id, kind, scope.actor_principal,
                           scope.subject_user_id, auth.authority_source, now, details or {}, compensates,
                           scope.actor_type, scope.delegation, scope.calling_component,
                           scope.request_id, scope.trace_id)

    async def _write(self, cur: Any, obj: MemoryObject, revision: MemoryRevision,
                     provenance: MemoryProvenance, event: MemoryEvent,
                     projection: MemoryProjection, *, binding: tuple[str,str,str] | None=None) -> None:
        # Kept here rather than calling persist_bundle: this cursor is owned by
        # the enclosing transaction and can additionally lock/check history.
        await cur.execute("INSERT INTO memory_objects (memory_id,object_kind,tenant_id,created_at,legacy_source_type,legacy_source_namespace,legacy_source_key) VALUES (%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s) ON DUPLICATE KEY UPDATE memory_id=memory_id",
                          (obj.memory_id,obj.kind.value,obj.tenant_id,obj.created_at,*(obj.legacy_binding or (None,None,None))))
        await cur.execute("INSERT INTO memory_revisions (revision_id,memory_id,content_digest,visibility,user_id,project_id,lifecycle_state,created_at,payload,provenance_status,prior_revision_id,tenant_id) VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s)",
                          (revision.revision_id,revision.memory_id,revision.content_digest,revision.visibility.value,revision.user_id,revision.project_id,revision.lifecycle.value,revision.created_at,revision.payload,revision.provenance_status,revision.prior_revision_id,obj.tenant_id))
        await cur.execute("INSERT INTO memory_revision_payloads (revision_id,payload) VALUES (%s,%s)", (revision.revision_id,revision.payload))
        await cur.execute("INSERT INTO memory_provenance (provenance_id,revision_id,source_revisions,transformation_id,transformation_version,actor_principal,actor_type,subject_user_id,provider,model,created_at,limitations) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s)",
                          (provenance.provenance_id,provenance.revision_id,json.dumps(provenance.source_revisions),provenance.transformation_id,provenance.transformation_version,provenance.actor_principal,provenance.actor_type,provenance.subject_user_id,provenance.provider,provenance.model,provenance.created_at,provenance.limitations))
        await cur.execute("INSERT INTO memory_events (event_id,memory_id,revision_id,event_kind,actor_principal,subject_user_id,authority_source,occurred_at,details,compensates_event_id,actor_type,delegation,calling_component,request_id,trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s,%s,%s,%s)",
                          (event.event_id,event.memory_id,event.revision_id,event.kind.value,event.actor_principal,event.subject_user_id,event.authority_source,event.occurred_at,json.dumps(dict(event.details)),event.compensates_event_id,event.actor_type,event.delegation,event.calling_component,event.request_id,event.trace_id))
        await cur.execute("INSERT INTO memory_projections (memory_id,current_revision_id,current_lifecycle_state,current_verification_state,canonical_history_digest,reconciliation_required) VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE current_revision_id=VALUES(current_revision_id),current_lifecycle_state=VALUES(current_lifecycle_state),current_verification_state=VALUES(current_verification_state),canonical_history_digest=VALUES(canonical_history_digest),reconciliation_required=VALUES(reconciliation_required)",
                          (projection.memory_id,projection.current_revision_id,projection.current_lifecycle.value if projection.current_lifecycle else None,projection.current_verification,projection.canonical_history_digest,projection.reconciliation_required))
        if binding:
            await cur.execute("INSERT INTO memory_legacy_bindings (tenant_id,legacy_source_type,legacy_source_namespace,legacy_source_key,memory_id,binding_state,created_at) VALUES (%s,%s,%s,%s,%s,'ACTIVE',FROM_UNIXTIME(%s))", (obj.tenant_id,*binding,obj.memory_id,obj.created_at))

    async def create_memory(self, auth: MutationAuthorizationRequest, kind: ObjectKind, content: str,
                            visibility: Visibility, *, user_id: str | None=None, project_id: str | None=None,
                            transformation_id: str="application", provider: str | None=None, model: str | None=None,
                            provenance_status: str="COMPLETE") -> str:
        async def op(cur: Any) -> str:
            resolved=await self._auth(cur,auth,"CREATE",visibility); scope=resolved.scope
            self._project_permissions(resolved, "CREATE", visibility)
            if visibility is Visibility.USER_PRIVATE and user_id != scope.subject_user_id: raise ScopeDenied("private memory subject mismatch")
            # A project ID in a request is a lookup target, never an ambient
            # namespace.  Project-shared writes require exact equality.
            candidate=MemoryRevision("", "", "", visibility, user_id, project_id, Lifecycle.ACTIVE, 0, None)
            self._assert_project_binding(scope, candidate)
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,kind,scope.tenant_id,now)
            rev=MemoryRevision(rid,mid,_digest(content),visibility,user_id,project_id,Lifecycle.ACTIVE,now,content,provenance_status)
            prov=MemoryProvenance(_uuid7(),rid,(),transformation_id,"1",scope.actor_principal,scope.actor_type,scope.subject_user_id,provider,model,now)
            event=self._event(scope,resolved,mid,rid,EventKind.CREATE,now)
            projection=_derive_projection(mid,[rev],[event]); await self._write(cur,obj,rev,prov,event,projection); return mid
        return await self._store.mutation(op)

    async def persist_conversation_extraction(self, auth: MutationAuthorizationRequest,
                                              conversation_id: int, claim_token: str) -> tuple[str, ...]:
        """Commit frozen extraction, origin bindings and processed marker as one unit."""
        from .extraction_jobs import canonical, source_digest
        async def op(cur: Any) -> tuple[str, ...]:
            await cur.execute("SELECT *,lease_until>NOW(6) AS lease_active FROM memory_extraction_jobs WHERE conversation_id=%s FOR UPDATE", (conversation_id,))
            job=await cur.fetchone()
            if not isinstance(job,Mapping): raise B9Error("extraction job missing")
            await cur.execute("SELECT id,conversation_uuid AS uuid,tenant_id,user_id,project_id,ended_at,memory_processed FROM conversations WHERE id=%s FOR UPDATE", (conversation_id,))
            conv=await cur.fetchone()
            await cur.execute("SELECT item_index,content_digest,memory_id,revision_id FROM memory_extraction_results WHERE conversation_id=%s ORDER BY item_index FOR UPDATE", (conversation_id,))
            existing=list(await cur.fetchall())
            output=job["frozen_output"]
            if isinstance(output,bytes): output=output.decode("utf-8")
            if output is None or _digest(output)!=job["output_digest"]: raise B9Error("invalid frozen extraction")
            items=json.loads(output)
            if not isinstance(items,list) or len(items)>int(__import__('os').getenv('JAX_MEMORY_MAX_ITEMS','100')):
                raise B9Error("invalid frozen extraction items")
            for item in items:
                if (not isinstance(item,dict) or set(item)!={'kind','content'}
                        or item['kind'] not in {'FACT','DECISION_MEMORY','ACTION_ITEM'}
                        or not isinstance(item['content'],str) or not item['content'].strip()
                        or len(item['content'])>int(__import__('os').getenv('JAX_MEMORY_MAX_ITEM_CHARS','12000'))):
                    raise B9Error("invalid frozen extraction item")
            if not conv: raise ScopeDenied('extraction source missing')
            scope=auth.scope
            if (scope.actor_principal!='service:memory-extraction' or scope.actor_type!='SERVICE'
                    or scope.calling_component!='memory-extraction'
                    or str(conv['tenant_id'])!=str(scope.tenant_id)
                    or str(conv['user_id'])!=str(scope.subject_user_id)
                    or (str(conv['project_id']) if conv['project_id'] is not None else None)!=scope.project_id):
                raise ScopeDenied('extraction canonical source namespace mismatch')
            visibility=Visibility.PROJECT_SHARED if conv['project_id'] is not None else Visibility.USER_PRIVATE
            resolved=await self._auth(cur,auth,'CREATE',visibility)
            self._project_permissions(resolved,'CREATE',visibility)
            scope=replace(resolved.scope,request_id=job['request_id'],trace_id=job['trace_id'])
            if job['state']=='COMPLETED':
                if not conv or not conv['memory_processed'] or len(existing)!=len(items):
                    raise ReconciliationRequired('extraction completion markers differ')
                if any(row['item_index']!=index or row['content_digest']!=_digest(canonical(items[index])) for index,row in enumerate(existing)):
                    raise ReconciliationRequired('extraction origin results differ')
                return tuple(row['memory_id'] for row in existing)
            if (job['state']!='RUNNING' or job['claim_token']!=claim_token or not job['lease_active']):
                raise ScopeDenied('extraction claim expired or replaced')
            if not conv or conv['ended_at'] is None or conv['memory_processed'] or existing:
                raise ScopeDenied('extraction source state changed')
            await cur.execute("SELECT COUNT(*) AS message_count,COALESCE(SUM(CHAR_LENGTH(content)+CHAR_LENGTH(role)+3),0) AS character_count FROM messages WHERE conversation_id=%s",(conversation_id,))
            counts=await cur.fetchone()
            limits={'max_messages':int(__import__('os').getenv('JAX_MEMORY_MAX_MESSAGES','1000')),
                    'max_chars':int(__import__('os').getenv('JAX_MEMORY_MAX_CHARS','96000')),
                    'max_items':int(__import__('os').getenv('JAX_MEMORY_MAX_ITEMS','100'))}
            if not counts or counts['message_count']>limits['max_messages'] or counts['character_count']>limits['max_chars']:
                raise ScopeDenied('extraction source exceeds input limits')
            await cur.execute("SELECT role,content FROM messages WHERE conversation_id=%s ORDER BY turn_number ASC LIMIT %s FOR UPDATE", (conversation_id,limits['max_messages']+1))
            messages=list(await cur.fetchall())
            if source_digest(conv,messages)!=job['input_digest']: raise ScopeDenied('extraction source changed')
            now=time.time(); memory_ids=[]
            for index,item in enumerate(items):
                mid,rid=_uuid7(),_uuid7(); content=item['content']
                obj=MemoryObject(mid,ObjectKind(item['kind']),scope.tenant_id,now)
                rev=MemoryRevision(rid,mid,_digest(content),visibility,
                    scope.subject_user_id if visibility is Visibility.USER_PRIVATE else None,
                    scope.project_id if visibility is Visibility.PROJECT_SHARED else None,Lifecycle.ACTIVE,now,content,'COMPLETE')
                self._assert_project_binding(scope,rev)
                provenance_details=canonical({'conversation_id':conversation_id,'conversation_uuid':conv['uuid'],
                    'item_index':index,'run_id':job['run_id'],'input_digest':job['input_digest'],
                    'output_digest':job['output_digest'],'limits':limits,'verification':'unverified extraction'})
                prov=MemoryProvenance(_uuid7(),rid,(),'conversation-extraction','b9-worker-v2',
                    scope.actor_principal,scope.actor_type,scope.subject_user_id,'deepseek','deepseek-v4-flash',now,provenance_details)
                event=self._event(scope,resolved,mid,rid,EventKind.CREATE,now,json.loads(provenance_details))
                await self._write(cur,obj,rev,prov,event,_derive_projection(mid,[rev],[event]))
                await cur.execute("INSERT INTO memory_extraction_results (conversation_id,item_index,content_digest,memory_id,revision_id) VALUES (%s,%s,%s,%s,%s)",(conversation_id,index,_digest(canonical(item)),mid,rid))
                memory_ids.append(mid)
            await cur.execute("UPDATE conversations SET memory_processed=TRUE,memory_processed_at=NOW(6) WHERE id=%s AND memory_processed=FALSE",(conversation_id,))
            if cur.rowcount!=1: raise B9Error('extraction processed marker not updated')
            await cur.execute("UPDATE memory_extraction_jobs SET state='COMPLETED',lease_until=NULL,error_code=NULL WHERE conversation_id=%s AND claim_token=%s",(conversation_id,claim_token))
            if cur.rowcount!=1: raise B9Error('extraction completion marker not updated')
            return tuple(memory_ids)
        return await self._store.mutation(op)

    async def append_conversation_message(self, auth: MutationAuthorizationRequest, content: str, *, project_id: str | None=None) -> str:
        return await self.create_memory(auth,ObjectKind.MESSAGE,content,Visibility.USER_PRIVATE,user_id=auth.scope.subject_user_id,project_id=project_id,transformation_id="conversation-message")

    async def import_legacy_memory(self, auth: MutationAuthorizationRequest, legacy_type: str, legacy_namespace: str,
                                   legacy_key: str, kind: ObjectKind, content: str | None, *,
                                   visibility: Visibility=Visibility.SYSTEM_INTERNAL,
                                   user_id: str | None=None, project_id: str | None=None,
                                   expected_source_digest: str | None=None) -> str:
        """Adopt a real legacy row under locked source ownership; never trust caller labels."""
        from .legacy_adoption import source_content, source_eligible, row_digest
        source_kinds={'facts':ObjectKind.FACT,'decisions':ObjectKind.DECISION_MEMORY,'action_items':ObjectKind.ACTION_ITEM}
        if legacy_type not in source_kinds or kind is not source_kinds[legacy_type] or legacy_namespace!='legacy' or not str(legacy_key).isdigit():
            raise ScopeDenied('unsupported legacy source identity')
        if visibility not in {Visibility.SYSTEM_INTERNAL,Visibility.USER_PRIVATE,Visibility.PROJECT_SHARED}:
            raise ScopeDenied('unsupported legacy adoption destination')
        binding=(legacy_type,legacy_namespace,str(legacy_key))
        async def op(cur: Any) -> str:
            # Resolve actor and lock real source before considering the supplied content/scope.
            resolved=await self._auth(cur,auth,'IMPORT_LEGACY',Visibility.SYSTEM_INTERNAL); scope=resolved.scope
            await cur.execute(f"SELECT *"+(", (expires_at IS NULL OR expires_at>NOW(6)) AS _not_expired" if legacy_type=="facts" else "")+f" FROM {legacy_type} WHERE id=%s FOR UPDATE",(int(legacy_key),))
            source=await cur.fetchone()
            if 'memory:admin' not in resolved.resolved_capabilities:
                raise AuthorizationDenied('legacy import requires resolved administrator')
            if not isinstance(source,Mapping): raise ScopeDenied('legacy source missing')
            source=dict(source)
            if legacy_type=='facts' and not source.pop('_not_expired',True): raise ScopeDenied('legacy source expired')
            eligibility_source=dict(source)
            if legacy_type=='facts': eligibility_source['expires_at']=None
            if not source_eligible(legacy_type,eligibility_source): raise ScopeDenied('legacy source ineligible')
            if source.get('user_id') is None: raise ScopeDenied('legacy source owner missing')
            await cur.execute("SELECT user_id,tenant_id,status FROM jax_users WHERE user_id=%s FOR UPDATE",(source['user_id'],))
            owner=await cur.fetchone()
            if (not isinstance(owner,Mapping) or str(owner['tenant_id'])!=str(scope.tenant_id)
                    or str(owner['status']).lower()!='active'):
                raise ScopeDenied('legacy source owner or tenant mismatch')
            if expected_source_digest is not None and row_digest(source)!=expected_source_digest:
                raise ScopeDenied('legacy source metadata changed')
            actual_content=source_content(legacy_type,source)
            if actual_content!=content: raise ScopeDenied('legacy source content changed')
            actual_project=str(source['project_id']) if source.get('project_id') is not None else None
            refs=[]
            if legacy_type=='facts' and source.get('source_message_id') is not None:
                await cur.execute("SELECT m.user_id,m.project_id,c.tenant_id,c.user_id AS conversation_user_id,c.project_id AS conversation_project_id FROM messages m JOIN conversations c ON c.id=m.conversation_id WHERE m.id=%s FOR UPDATE",(source['source_message_id'],))
                reference=await cur.fetchone()
                if not reference or str(reference['conversation_user_id'])!=str(owner['user_id']) or (str(reference['conversation_project_id']) if reference['conversation_project_id'] is not None else None)!=actual_project:
                    raise ScopeDenied('legacy message conversation scope differs')
                refs.append(reference)
            if legacy_type=='action_items' and source.get('source_conversation_id') is not None:
                await cur.execute("SELECT user_id,tenant_id,project_id FROM conversations WHERE id=%s FOR UPDATE",(source['source_conversation_id'],))
                refs.append(await cur.fetchone())
            for reference in refs:
                if (not reference or str(reference['user_id'])!=str(owner['user_id'])
                        or str(reference['tenant_id'])!=str(owner['tenant_id'])
                        or (str(reference['project_id']) if reference['project_id'] is not None else None)!=actual_project):
                    raise ScopeDenied('legacy source reference namespace differs')
            if actual_project!=scope.project_id: raise ScopeDenied('legacy source project mismatch')
            if actual_project is not None:
                await cur.execute("SELECT project_role,status,tenant_id FROM jax_project_membership WHERE project_id=%s AND user_id=%s AND tenant_id=%s FOR UPDATE",(actual_project,owner['user_id'],owner['tenant_id']))
                membership=await cur.fetchone()
                if (not membership or membership['status']!='ACTIVE' or str(membership['tenant_id'])!=str(owner['tenant_id'])
                        or membership['project_role'] not in {'CONTRIBUTOR','REVIEWER','OWNER'}):
                    raise ScopeDenied('legacy source project owner membership unavailable')
            if visibility is Visibility.USER_PRIVATE:
                if actual_project is not None or str(user_id)!=str(owner['user_id']) or project_id is not None:
                    raise ScopeDenied('legacy private destination mismatch')
            if visibility is Visibility.PROJECT_SHARED:
                if actual_project is None or project_id!=actual_project or user_id is not None:
                    raise ScopeDenied('legacy project destination mismatch')
            destination_auth=None
            if visibility is not Visibility.SYSTEM_INTERNAL:
                destination_auth=await self._auth(cur,auth,'RE_SCOPE',visibility)
                self._project_permissions(destination_auth,'RE_SCOPE',visibility)
                if 'memory_admin' not in destination_auth.resolved_roles:
                    raise AuthorizationDenied('legacy adoption scope publication requires administrator')
            # Serialize competing imports by the source lock as well as the tenant-qualified binding.
            await cur.execute("SELECT memory_id,binding_state FROM memory_legacy_bindings WHERE tenant_id=%s AND legacy_source_type=%s AND legacy_source_namespace=%s AND legacy_source_key=%s FOR UPDATE",(scope.tenant_id,*binding))
            existing=await cur.fetchone()
            if existing:
                if existing['binding_state']!='ACTIVE': raise ReconciliationRequired('legacy binding inactive')
                obj,revision=await self._current(cur,existing['memory_id'])
                await self._assert_reconciled(cur,obj.memory_id)
                if (obj.kind is not kind or obj.tenant_id!=scope.tenant_id or revision.payload!=actual_content
                        or revision.visibility is not visibility or revision.user_id!=user_id
                        or revision.project_id!=project_id or revision.lifecycle not in {Lifecycle.ACTIVE,Lifecycle.VERIFIED}):
                    raise ScopeDenied('legacy binding content or destination differs')
                return obj.memory_id
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,kind,scope.tenant_id,now,binding)
            rev=MemoryRevision(rid,mid,_digest(content or ''),Visibility.SYSTEM_INTERNAL,None,None,Lifecycle.ACTIVE,now,content,'LEGACY_PROVENANCE_INCOMPLETE')
            limitations=json.dumps({'legacy_source_type':legacy_type,'legacy_source_key':str(legacy_key),'legacy_verification_not_adopted':True})
            prov=MemoryProvenance(_uuid7(),rid,(),'legacy-import','2',scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,limitations)
            event=self._event(scope,resolved,mid,rid,EventKind.IMPORT_LEGACY,now,{'legacy_source_type':legacy_type,'legacy_source_key':str(legacy_key)})
            await self._write(cur,obj,rev,prov,event,_derive_projection(mid,[rev],[event]),binding=binding)
            if destination_auth is not None:
                # DATETIME(6) cannot order random fallback UUID7 IDs. Make the
                # second canonical step strictly later at persisted precision.
                published_at=now+0.001
                published=MemoryRevision(_uuid7(),mid,rev.content_digest,visibility,user_id,project_id,Lifecycle.ACTIVE,published_at,content,rev.provenance_status,rid)
                published_prov=MemoryProvenance(_uuid7(),published.revision_id,(rid,),'legacy-adoption-scope','1',scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,published_at,limitations)
                scope_event=self._event(destination_auth.scope,destination_auth,mid,published.revision_id,EventKind.RE_SCOPE,published_at,{'reason':'authoritative legacy owner adoption'})
                self._assert_project_binding(destination_auth.scope,published)
                await self._write(cur,obj,published,published_prov,scope_event,_derive_projection(mid,[rev,published],[event,scope_event]))
            return mid
        return await self._store.mutation(op)

    async def _current(self, cur: Any, memory_id: str) -> tuple[MemoryObject, MemoryRevision]:
        await cur.execute("SELECT o.memory_id,o.object_kind,o.tenant_id,UNIX_TIMESTAMP(o.created_at) object_created_at,r.revision_id,r.content_digest,r.visibility,r.user_id,r.project_id,r.lifecycle_state,UNIX_TIMESTAMP(r.created_at) revision_created_at,p.payload,r.provenance_status,r.prior_revision_id FROM memory_objects o JOIN memory_projections pr ON pr.memory_id=o.memory_id JOIN memory_revisions r ON r.revision_id=pr.current_revision_id LEFT JOIN memory_revision_payloads p ON p.revision_id=r.revision_id WHERE o.memory_id=%s FOR UPDATE", (memory_id,))
        row=await cur.fetchone()
        if not row: raise KeyError(memory_id)
        if not isinstance(row,Mapping): raise TypeError("persistent B9 API requires mapping rows")
        obj=MemoryObject(row["memory_id"],ObjectKind(row["object_kind"]),row["tenant_id"],float(row["object_created_at"]))
        payload=row["payload"]
        if isinstance(payload,bytes): payload=payload.decode("utf-8","replace")
        rev=MemoryRevision(row["revision_id"],memory_id,row["content_digest"],Visibility(row["visibility"]),row["user_id"],row["project_id"],Lifecycle(row["lifecycle_state"]),float(row["revision_created_at"]),payload,row["provenance_status"],row["prior_revision_id"])
        return obj,rev

    async def _assert_reconciled(self, cur: Any, memory_id: str) -> tuple[list[MemoryRevision], list[MemoryEvent]]:
        """Compare the stored projection with the immutable canonical rows.

        This lock/check occurs before every sensitive persistent mutation;
        a mismatch is marked repair-needed and fails closed.
        """
        await cur.execute("SELECT revision_id,memory_id,content_digest,visibility,user_id,project_id,lifecycle_state,UNIX_TIMESTAMP(created_at) created_at,payload,provenance_status,prior_revision_id FROM memory_revisions WHERE memory_id=%s ORDER BY created_at,revision_id FOR UPDATE", (memory_id,))
        revisions=[]
        for row in await cur.fetchall():
            if not isinstance(row,Mapping): raise TypeError("persistent B9 API requires mapping rows")
            revisions.append(MemoryRevision(row["revision_id"],row["memory_id"],row["content_digest"],Visibility(row["visibility"]),row["user_id"],row["project_id"],Lifecycle(row["lifecycle_state"]),float(row["created_at"]),row["payload"],row["provenance_status"],row["prior_revision_id"]))
        await cur.execute("SELECT event_id,memory_id,revision_id,event_kind,actor_principal,subject_user_id,authority_source,UNIX_TIMESTAMP(occurred_at) occurred_at,details,compensates_event_id,actor_type,delegation,calling_component,request_id,trace_id FROM memory_events WHERE memory_id=%s ORDER BY occurred_at,event_id FOR UPDATE", (memory_id,))
        events=[]
        for row in await cur.fetchall():
            details=row["details"]
            if isinstance(details,bytes): details=details.decode("utf-8")
            if isinstance(details,str): details=json.loads(details)
            events.append(MemoryEvent(row["event_id"],row["memory_id"],row["revision_id"],EventKind(row["event_kind"]),row["actor_principal"],row["subject_user_id"],row["authority_source"],float(row["occurred_at"]),details or {},row["compensates_event_id"],row["actor_type"],row["delegation"],row["calling_component"],row["request_id"],row["trace_id"]))
        expected=_derive_projection(memory_id,revisions,events)
        await cur.execute("SELECT current_revision_id,current_lifecycle_state,current_verification_state,canonical_history_digest,reconciliation_required FROM memory_projections WHERE memory_id=%s FOR UPDATE", (memory_id,))
        actual=await cur.fetchone()
        mismatch=(not actual or actual["current_revision_id"] != expected.current_revision_id or actual["current_lifecycle_state"] != (expected.current_lifecycle.value if expected.current_lifecycle else None) or bool(actual["current_verification_state"]) != expected.current_verification or actual["canonical_history_digest"] != expected.canonical_history_digest or bool(actual["reconciliation_required"]))
        if mismatch:
            await cur.execute("UPDATE memory_projections SET reconciliation_required=TRUE WHERE memory_id=%s", (memory_id,))
            raise ReconciliationRequired(memory_id)
        return revisions,events

    async def _revision_mutation(self, auth: MutationAuthorizationRequest, memory_id: str, event_kind: EventKind,
                                 *, content: str | None=None, lifecycle: Lifecycle | None=None,
                                 visibility: Visibility | None=None, user_id: str | None=None, project_id: str | None=None,
                                 reason: str | None=None, transformation: str | None=None,
                                 limitations: str | None=None, purge_payload: bool=False) -> str:
        # The expected visibility is obtained under the same lock as the
        # write.  The caller never supplies a replacement authority context.
        async def op(cur: Any) -> str:
            obj,old=await self._current(cur,memory_id)
            # Re-scope authorization is explicitly for the destination
            # visibility; all other mutations authorize the existing scope.
            resolved=await self._auth(cur,auth,event_kind.value,visibility if event_kind is EventKind.RE_SCOPE and visibility else old.visibility)
            scope=resolved.scope
            revisions,events=await self._assert_reconciled(cur,memory_id)
            if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
            self._assert_project_binding(scope, old, requested_project_id=project_id if project_id is not None else old.project_id)
            self._project_permissions(resolved, event_kind.value, old.visibility)
            if event_kind is EventKind.VERIFY and not (
                resolved.resolved_roles.intersection({"memory_reviewer","memory_admin"})
                or "memory:project:verify" in resolved.resolved_capabilities
            ):
                raise AuthorizationDenied("verification requires resolved reviewer authority")
            if event_kind is EventKind.VERIFY and old.lifecycle not in {Lifecycle.ACTIVE,Lifecycle.VERIFIED}:
                raise ScopeDenied("memory lifecycle is not eligible for verification")
            if event_kind in {EventKind.CORRECT,EventKind.SUPERSEDE,EventKind.RE_SCOPE} and old.payload is None and content is None:
                raise ScopeDenied("purged payload cannot be revised")
            now=time.time(); rid=_uuid7(); payload=None if purge_payload else (content if content is not None else old.payload)
            rev=MemoryRevision(rid,memory_id,_digest(payload or ""),visibility or old.visibility,user_id if user_id is not None else old.user_id,project_id if project_id is not None else old.project_id,lifecycle or Lifecycle.ACTIVE,now,payload,old.provenance_status,old.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(old.revision_id,),transformation or event_kind.value.lower(),"1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,limitations or reason)
            self._assert_project_binding(scope, rev)
            event=self._event(scope,resolved,memory_id,rid,event_kind,now,{"reason":reason} if reason else {})
            # The event/revision pair is the new canonical tail.  The digest
            # includes both and is recomputed from locked canonical history by
            # the reconciliation worker; this write is never ahead of it.
            projection=_derive_projection(memory_id,[*revisions,rev],[*events,event])
            if purge_payload:
                # Delete all dereferenceable payload/vector material while
                # retaining the safe immutable revision/event tombstone.
                await cur.execute("DELETE p FROM memory_revision_payloads p JOIN memory_revisions r ON r.revision_id=p.revision_id WHERE r.memory_id=%s", (memory_id,))
                await cur.execute("DELETE e FROM embedding_generations e JOIN memory_revisions r ON r.revision_id=e.revision_id WHERE r.memory_id=%s", (memory_id,))
                await cur.execute("UPDATE memory_revisions SET payload=NULL WHERE memory_id=%s", (memory_id,))
            await self._write(cur,obj,rev,prov,event,projection); return rid
        return await self._store.mutation(op)

    async def verify_memory(self, auth: MutationAuthorizationRequest, memory_id: str, *, method: str, limitations: str | None=None) -> str:
        return await self._revision_mutation(auth,memory_id,EventKind.VERIFY,lifecycle=Lifecycle.VERIFIED,transformation="human-verification",reason=method,limitations=limitations)
    async def correct_memory(self, auth: MutationAuthorizationRequest, memory_id: str, content: str) -> str:
        return await self._revision_mutation(auth,memory_id,EventKind.CORRECT,content=content)
    async def supersede_memory(self, auth: MutationAuthorizationRequest, memory_id: str, content: str) -> str:
        return await self._revision_mutation(auth,memory_id,EventKind.SUPERSEDE,content=content)
    async def expire_memory(self, auth: MutationAuthorizationRequest, memory_id: str, *, reason: str) -> str:
        return await self._revision_mutation(auth,memory_id,EventKind.EXPIRE,lifecycle=Lifecycle.EXPIRED,reason=reason)
    async def tombstone_memory(self, auth: MutationAuthorizationRequest, memory_id: str, *, reason: str) -> str:
        return await self._revision_mutation(auth,memory_id,EventKind.TOMBSTONE,lifecycle=Lifecycle.TOMBSTONED,reason=reason,purge_payload=True)
    async def content_purge(self, auth: MutationAuthorizationRequest, memory_id: str, *, reason: str="retention") -> str:
        # A purged revision has no separately retrievable payload; the
        # lifecycle event remains an intentionally safe tombstone.
        return await self._revision_mutation(auth,memory_id,EventKind.CONTENT_PURGE,lifecycle=Lifecycle.PURGED,reason=reason,purge_payload=True)
    async def re_scope_memory(self, auth: MutationAuthorizationRequest, memory_id: str, *, visibility: Visibility, user_id: str | None=None, project_id: str | None=None) -> str:
        # Cross-tenant transfer is a distinct create operation and requires a
        # destination resolved context; this API never mutates tenant identity.
        return await self._revision_mutation(auth,memory_id,EventKind.RE_SCOPE,visibility=visibility,user_id=user_id,project_id=project_id)
    async def synthesize_memory(self, auth: MutationAuthorizationRequest, content: str, source_revision_ids: tuple[str,...], *, provider: str | None, model: str | None, transformation_version: str,
                                synthesis_job_key: str | None=None, synthesis_job_token: str | None=None,
                                synthesis_item_key: str | None=None) -> str:
        if len(set(source_revision_ids))<2: raise B9Error("synthesis needs two distinct source revisions")
        async def op(cur: Any) -> str:
            job=None; frozen=[]
            if synthesis_job_key is not None:
                if not synthesis_job_token or not synthesis_item_key: raise B9Error('synthesis job identity missing')
                await cur.execute("SELECT *,lease_until>NOW() AS lease_active FROM memory_synthesis_jobs WHERE job_key=%s FOR UPDATE",(synthesis_job_key,))
                job=await cur.fetchone()
                if not job or job['state'] not in {'FROZEN','COMPLETED'}: raise B9Error('synthesis job not frozen')
                frozen=json.loads(job['frozen_output'])
                matches=[item for item in frozen if item.get('item_key')==synthesis_item_key]
                if len(matches)!=1 or matches[0]['text']!=content or tuple(sorted(matches[0]['source_revision_ids']))!=tuple(sorted(source_revision_ids)):
                    raise ScopeDenied('synthesis frozen result differs')
                scope=auth.scope
                if (str(job['tenant_id'])!=scope.tenant_id or str(job['user_id'])!=scope.subject_user_id
                        or (str(job['project_id']) if job['project_id'] is not None else None)!=scope.project_id
                        or job['transformation_version']!=transformation_version):
                    raise ScopeDenied('synthesis job scope differs')
                await cur.execute("SELECT memory_id FROM memory_synthesis_job_items WHERE job_key=%s AND item_key=%s FOR UPDATE",(synthesis_job_key,synthesis_item_key))
                published=await cur.fetchone()
                if published:
                    resolved=await self._auth(cur,auth,'SYNTHESIZE',Visibility.SYSTEM_INTERNAL)
                    obj,revision=await self._current(cur,published['memory_id'])
                    await self._assert_reconciled(cur,obj.memory_id)
                    if (obj.tenant_id!=resolved.scope.tenant_id or obj.kind is not ObjectKind.SYNTHESIS
                            or (revision.visibility is Visibility.USER_PRIVATE and revision.user_id!=resolved.scope.subject_user_id)):
                        raise ScopeDenied('published synthesis namespace differs')
                    self._assert_project_binding(resolved.scope,revision)
                    self._project_permissions(resolved,'SYNTHESIZE',revision.visibility)
                    return obj.memory_id
                if job['state']=='COMPLETED' or job['claim_token']!=synthesis_job_token or not job['lease_active']:
                    raise ScopeDenied('synthesis claim no longer owned')
            resolved=await self._auth(cur,auth,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL); scope=resolved.scope
            sources=[]
            for source_id in source_revision_ids:
                await cur.execute(
                    "SELECT r.revision_id,r.memory_id,o.object_kind,o.tenant_id,r.visibility,r.user_id,r.project_id,"
                    "r.lifecycle_state,p.payload,pr.current_revision_id,pr.current_verification_state FROM memory_revisions r "
                    "JOIN memory_objects o ON o.memory_id=r.memory_id "
                    "JOIN memory_projections pr ON pr.memory_id=o.memory_id "
                    "LEFT JOIN memory_revision_payloads p ON p.revision_id=r.revision_id "
                    "WHERE r.revision_id=%s AND pr.reconciliation_required=FALSE FOR UPDATE", (source_id,))
                row=await cur.fetchone()
                if (not isinstance(row,Mapping) or row["tenant_id"] != scope.tenant_id
                        or row["revision_id"] != row["current_revision_id"]
                        or row["lifecycle_state"] not in {"ACTIVE","VERIFIED"}
                        or not row.get("current_verification_state")
                        or row["payload"] is None or row["object_kind"] not in {kind.value for kind in SYNTHESIS_SOURCE_KINDS}):
                    raise ScopeDenied("source revision is ineligible for synthesis")
                if row["visibility"] == Visibility.USER_PRIVATE.value and row["user_id"] != scope.subject_user_id:
                    raise ScopeDenied("source subject mismatch")
                if row["project_id"] is not None and row["project_id"] != scope.project_id:
                    raise ScopeDenied("source project mismatch")
                sources.append(row)
            effective={(r["visibility"],r["user_id"],r["project_id"]) for r in sources}
            if len(effective) != 1: raise ScopeDenied("synthesis sources have different scopes")
            visibility_value,user_id,project_id=effective.pop()
            visibility=Visibility(visibility_value)
            self._project_permissions(resolved,"SYNTHESIZE",visibility)
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,ObjectKind.SYNTHESIS,scope.tenant_id,now)
            rev=MemoryRevision(rid,mid,_digest(content),visibility,user_id,project_id,Lifecycle.ACTIVE,now,content,"COMPLETE")
            self._assert_project_binding(scope, rev)
            prov=MemoryProvenance(_uuid7(),rid,source_revision_ids,"synthesis",transformation_version,scope.actor_principal,scope.actor_type,scope.subject_user_id,provider,model,now)
            event=self._event(scope,resolved,mid,rid,EventKind.SYNTHESIZE,now,{"derivation_depth":1})
            await self._write(cur,obj,rev,prov,event,_derive_projection(mid,[rev],[event]))
            if job is not None:
                await cur.execute("INSERT INTO memory_synthesis_job_items (job_key,item_key,memory_id) VALUES (%s,%s,%s)",(synthesis_job_key,synthesis_item_key,mid))
                await cur.execute("SELECT item_key FROM memory_synthesis_job_items WHERE job_key=%s FOR UPDATE",(synthesis_job_key,))
                published_keys={row['item_key'] for row in await cur.fetchall()}
                expected_keys={item['item_key'] for item in frozen}
                if published_keys==expected_keys:
                    await cur.execute("UPDATE memory_synthesis_jobs SET state='COMPLETED' WHERE job_key=%s",(synthesis_job_key,))
            return mid
        return await self._store.mutation(op)

    async def reembed_memory(self, auth: MutationAuthorizationRequest, memory_id: str,
                             identity: EmbeddingSpaceIdentity, vector: tuple[float,...] | None=None, *,
                             expected_revision_id: str | None=None) -> str:
        async def op(cur: Any) -> str:
            obj,revision=await self._current(cur,memory_id)
            if expected_revision_id is not None and revision.revision_id != expected_revision_id:
                raise ScopeDenied("embedding source revision changed")
            if revision.payload is None or revision.lifecycle not in {Lifecycle.ACTIVE,Lifecycle.VERIFIED}:
                raise ScopeDenied("embedding source is ineligible")
            resolved=await self._auth(cur,auth,"RE_EMBED",revision.visibility); scope=resolved.scope
            revisions,events=await self._assert_reconciled(cur,memory_id)
            if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
            self._assert_project_binding(scope, revision)
            if vector is not None and len(vector) != identity.dimension: raise B9Error("embedding dimension mismatch")
            space_id=identity.embedding_space_id
            await cur.execute("SELECT generation_id FROM embedding_generations WHERE revision_id=%s AND embedding_space_id=%s ORDER BY generated_at,generation_id LIMIT 1 FOR UPDATE", (revision.revision_id,space_id))
            existing=await cur.fetchone()
            if existing: return existing['generation_id']
            generation_id=_uuid7(); now=time.time()
            await cur.execute("INSERT INTO embedding_spaces (embedding_space_id,schema_version,provider_runtime_class,model_identifier,model_version_or_digest,dimension,normalization,distance_semantics,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s)) ON DUPLICATE KEY UPDATE embedding_space_id=embedding_space_id", (space_id,identity.schema_version,identity.provider_runtime_class,identity.model_identifier,identity.model_version_or_digest,identity.dimension,identity.normalization,identity.distance_semantics,now))
            await cur.execute("INSERT INTO embedding_generations (generation_id,revision_id,embedding_space_id,generated_at,embedding_payload) VALUES (%s,%s,%s,FROM_UNIXTIME(%s),%s)", (generation_id,revision.revision_id,space_id,now,json.dumps(vector) if vector is not None else None))
            event=self._event(scope,resolved,memory_id,revision.revision_id,EventKind.RE_EMBED,now,{"embedding_space_id":space_id,"generation_id":generation_id})
            # Re-embedding is append-only operational history; it cannot alter
            # revision identity or lifecycle.
            projection=_derive_projection(memory_id,revisions,[*events,event])
            await cur.execute("INSERT INTO memory_events (event_id,memory_id,revision_id,event_kind,actor_principal,subject_user_id,authority_source,occurred_at,details,compensates_event_id,actor_type,delegation,calling_component,request_id,trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s,%s,%s,%s)", (event.event_id,event.memory_id,event.revision_id,event.kind.value,event.actor_principal,event.subject_user_id,event.authority_source,event.occurred_at,json.dumps(dict(event.details)),None,event.actor_type,event.delegation,event.calling_component,event.request_id,event.trace_id))
            await cur.execute("UPDATE memory_projections SET canonical_history_digest=%s WHERE memory_id=%s", (projection.canonical_history_digest,memory_id))
            return generation_id
        return await self._store.mutation(op)

    async def compensating_event(self, auth: MutationAuthorizationRequest, memory_id: str, compensates_event_id: str, *, reason: str) -> str:
        async def op(cur: Any) -> str:
            obj,revision=await self._current(cur,memory_id); resolved=await self._auth(cur,auth,"COMPENSATE",revision.visibility); scope=resolved.scope
            revisions,events=await self._assert_reconciled(cur,memory_id)
            if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
            self._assert_project_binding(scope, revision)
            await cur.execute("SELECT event_id FROM memory_events WHERE memory_id=%s AND event_id=%s FOR UPDATE", (memory_id,compensates_event_id))
            if not await cur.fetchone(): raise B9Error("compensation target is not an event of this memory")
            event=self._event(scope,resolved,memory_id,revision.revision_id,EventKind.COMPENSATE,time.time(),{"reason":reason},compensates_event_id)
            await cur.execute("INSERT INTO memory_events (event_id,memory_id,revision_id,event_kind,actor_principal,subject_user_id,authority_source,occurred_at,details,compensates_event_id,actor_type,delegation,calling_component,request_id,trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s,%s,%s,%s)", (event.event_id,event.memory_id,event.revision_id,event.kind.value,event.actor_principal,event.subject_user_id,event.authority_source,event.occurred_at,json.dumps(dict(event.details)),event.compensates_event_id,event.actor_type,event.delegation,event.calling_component,event.request_id,event.trace_id))
            projection=_derive_projection(memory_id,revisions,[*events,event])
            await cur.execute("UPDATE memory_projections SET canonical_history_digest=%s WHERE memory_id=%s", (projection.canonical_history_digest,memory_id))
            return event.event_id
        return await self._store.mutation(op)

    async def assert_event_immutable(self, event_id: str) -> None:
        """Events are readable only; UPDATE/DELETE are intentionally absent."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT event_id FROM memory_events WHERE event_id=%s", (event_id,))
                await cur.fetchone()


class MariaDBB9Reader:
    """Production read boundary for B9-managed memory.

    It returns envelopes only.  It deliberately has no API returning raw B9
    content rows to model-facing callers.  The caller must already possess a
    ScopeContext constructed by its authenticated JAX boundary.
    """
    def __init__(self, pool: Any, authorization_resolver: Any | None=None):
        self._pool = pool
        self._authorization_resolver = authorization_resolver

    async def retrieve_authorized(self, request: MutationAuthorizationRequest, *, limit: int=20) -> tuple[MemoryEnvelope, ...]:
        """Resolve project access from current DB state before retrieval.

        ``ScopeContext`` is never accepted as project-read authority here.
        The resolver locks/revalidates the identity, project and membership
        rows in a short transaction; the subsequent read is a fresh snapshot.
        """
        if not isinstance(request, MutationAuthorizationRequest):
            raise AuthorizationDenied("read requires mutation authorization request")
        resolver=self._authorization_resolver
        if resolver is None or not hasattr(resolver, "resolve_mutation_in_transaction"):
            raise AuthorizationDenied("read requires DB-backed authority resolver")
        async with self._pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    target = Visibility.PROJECT_SHARED if request.scope.project_id else Visibility.TENANT_SHARED
                    decision=await resolver.resolve_mutation_in_transaction(cur, request.scope, "RETRIEVE", target)
                    if not isinstance(decision, MutationAuthorizationContext):
                        raise AuthorizationDenied("invalid read authority decision")
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return await self._retrieve_scoped(decision.scope, limit=limit)

    async def retrieve(self, scope: ScopeContext, *, limit: int = 20) -> tuple[MemoryEnvelope, ...]:
        if scope.project_id:
            raise AuthorizationDenied("project retrieval requires retrieve_authorized")
        return await self._retrieve_scoped(scope, limit=limit)

    async def _retrieve_scoped(self, scope: ScopeContext, *, limit: int = 20) -> tuple[MemoryEnvelope, ...]:
        scope.validate()
        if limit < 1 or limit > 100:
            raise ScopeDenied("retrieval limit outside bounded range")
        # A repeatable read transaction makes object/revision/payload/
        # provenance one snapshot.  No result is constructed after it closes.
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                # ``SET TRANSACTION`` affects the next transaction only, so a
                # pooled connection cannot leak read-only mode to its next
                # borrower.
                await cur.execute("SET TRANSACTION READ ONLY")
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    # Equality branches preserve the previous visibility/project
                    # predicate while allowing each scope to use an ordered
                    # tenant index. Merge at most six bounded streams inside
                    # this same repeatable-read snapshot.
                    candidates=[]
                    projects=[None] if scope.project_id is None else [None,scope.project_id]
                    for visibility in (Visibility.USER_PRIVATE,Visibility.TENANT_SHARED,Visibility.PROJECT_SHARED):
                        for project in projects:
                            sql=(
                                # Measured optimizer chose object-first plus temp/filesort.
                                # Fix only join order so each equality branch drives the
                                # ordered revision index; retain optimizer's index choice.
                                "SELECT STRAIGHT_JOIN o.memory_id,o.object_kind,o.tenant_id,UNIX_TIMESTAMP(o.created_at) AS object_created_at,"
                                "r.revision_id,r.content_digest,r.visibility,r.user_id,r.project_id,r.lifecycle_state,"
                                "UNIX_TIMESTAMP(r.created_at) AS revision_created_at,p.payload,r.provenance_status,r.prior_revision_id "
                                "FROM memory_revisions r "
                                "JOIN memory_objects o ON o.memory_id=r.memory_id AND o.tenant_id=r.tenant_id "
                                "JOIN memory_projections pr ON pr.memory_id=r.memory_id AND pr.current_revision_id=r.revision_id "
                                "LEFT JOIN memory_revision_payloads p ON p.revision_id=r.revision_id "
                                "WHERE r.tenant_id=%s AND r.visibility=%s AND pr.reconciliation_required=FALSE "
                                "AND r.lifecycle_state NOT IN ('TOMBSTONED','PURGED','EXPIRED') ")
                            args=[scope.tenant_id,visibility.value]
                            if visibility is Visibility.USER_PRIVATE:
                                sql+="AND r.user_id=%s "
                                args.append(scope.subject_user_id)
                            if project is None:
                                sql+="AND r.project_id IS NULL "
                            else:
                                sql+="AND r.project_id=%s "
                                args.append(project)
                            sql+="ORDER BY r.created_at DESC,r.revision_id DESC LIMIT %s"
                            args.append(limit)
                            await cur.execute(sql,tuple(args))
                            for row in await cur.fetchall():
                                if not isinstance(row,Mapping): raise TypeError('B9 reader requires mapping cursor rows')
                                candidates.append(row)
                    rows=sorted(candidates,key=lambda row:(float(row['revision_created_at']),row['revision_id']),reverse=True)[:limit]
                    envelopes=[]
                    for row in rows:
                        # DictCursor is required by the production pool; the
                        # explicit error avoids accidentally treating tuples as
                        # trusted field layouts.
                        if not isinstance(row, Mapping):
                            raise TypeError("B9 reader requires mapping cursor rows")
                        await cur.execute(
                            "SELECT provenance_id,source_revisions,transformation_id,transformation_version,"
                            "actor_principal,actor_type,subject_user_id,provider,model,UNIX_TIMESTAMP(created_at) AS created_at,limitations "
                            "FROM memory_provenance WHERE revision_id=%s ORDER BY created_at,provenance_id",
                            (row["revision_id"],),
                        )
                        provenance_rows=await cur.fetchall()
                        obj=MemoryObject(row["memory_id"],ObjectKind(row["object_kind"]),row["tenant_id"],float(row["object_created_at"]))
                        payload=row["payload"]
                        if isinstance(payload, bytes): payload=payload.decode("utf-8", "replace")
                        revision=MemoryRevision(row["revision_id"],row["memory_id"],row["content_digest"],
                                                Visibility(row["visibility"]),row["user_id"],row["project_id"],
                                                Lifecycle(row["lifecycle_state"]),float(row["revision_created_at"]),payload,
                                                row["provenance_status"],row["prior_revision_id"])
                        prov=[]
                        for p in provenance_rows:
                            if not isinstance(p, Mapping): raise TypeError("B9 reader requires mapping cursor rows")
                            source=p["source_revisions"]
                            if isinstance(source, bytes): source=source.decode("utf-8")
                            if isinstance(source, str): source=json.loads(source)
                            prov.append(MemoryProvenance(p["provenance_id"],revision.revision_id,tuple(source or ()),
                                                         p["transformation_id"],p["transformation_version"],p["actor_principal"],
                                                         p["actor_type"],p["subject_user_id"],p["provider"],p["model"],
                                                         float(p["created_at"]),p["limitations"]))
                        if revision.payload is not None:
                            envelopes.append(MemoryEnvelope(obj,revision,tuple(prov),(),{"store":"B9_MARIADB"}))
                # Explicitly end the read-only snapshot; no state can be
                # committed by this read boundary.
                await conn.rollback()
                return tuple(envelopes)
            except BaseException:
                await conn.rollback()
                raise

    async def prompt_context(self, scope: ScopeContext, *, limit: int = 20) -> PromptMemoryContext:
        return PromptMemoryContext(await self.retrieve(scope, limit=limit))
