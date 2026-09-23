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
    _derive_projection, _digest, _uuid7,
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
                        "(revision_id,memory_id,content_digest,visibility,user_id,project_id,lifecycle_state,created_at,payload,provenance_status,prior_revision_id) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s)",
                        (revision.revision_id,revision.memory_id,revision.content_digest,revision.visibility.value,
                         revision.user_id,revision.project_id,revision.lifecycle.value,revision.created_at,
                        revision.payload,revision.provenance_status,revision.prior_revision_id),
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
            except Exception:
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
            except Exception:
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
        await cur.execute("INSERT INTO memory_revisions (revision_id,memory_id,content_digest,visibility,user_id,project_id,lifecycle_state,created_at,payload,provenance_status,prior_revision_id) VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s)",
                          (revision.revision_id,revision.memory_id,revision.content_digest,revision.visibility.value,revision.user_id,revision.project_id,revision.lifecycle.value,revision.created_at,revision.payload,revision.provenance_status,revision.prior_revision_id))
        await cur.execute("INSERT INTO memory_revision_payloads (revision_id,payload) VALUES (%s,%s)", (revision.revision_id,revision.payload))
        await cur.execute("INSERT INTO memory_provenance (provenance_id,revision_id,source_revisions,transformation_id,transformation_version,actor_principal,actor_type,subject_user_id,provider,model,created_at,limitations) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s)",
                          (provenance.provenance_id,provenance.revision_id,json.dumps(provenance.source_revisions),provenance.transformation_id,provenance.transformation_version,provenance.actor_principal,provenance.actor_type,provenance.subject_user_id,provenance.provider,provenance.model,provenance.created_at,provenance.limitations))
        await cur.execute("INSERT INTO memory_events (event_id,memory_id,revision_id,event_kind,actor_principal,subject_user_id,authority_source,occurred_at,details,compensates_event_id,actor_type,delegation,calling_component,request_id,trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s,%s,%s,%s)",
                          (event.event_id,event.memory_id,event.revision_id,event.kind.value,event.actor_principal,event.subject_user_id,event.authority_source,event.occurred_at,json.dumps(dict(event.details)),event.compensates_event_id,event.actor_type,event.delegation,event.calling_component,event.request_id,event.trace_id))
        await cur.execute("INSERT INTO memory_projections (memory_id,current_revision_id,current_lifecycle_state,current_verification_state,canonical_history_digest,reconciliation_required) VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE current_revision_id=VALUES(current_revision_id),current_lifecycle_state=VALUES(current_lifecycle_state),current_verification_state=VALUES(current_verification_state),canonical_history_digest=VALUES(canonical_history_digest),reconciliation_required=VALUES(reconciliation_required)",
                          (projection.memory_id,projection.current_revision_id,projection.current_lifecycle.value if projection.current_lifecycle else None,projection.current_verification,projection.canonical_history_digest,projection.reconciliation_required))
        if binding:
            await cur.execute("INSERT INTO memory_legacy_bindings (legacy_source_type,legacy_source_namespace,legacy_source_key,memory_id,binding_state,created_at) VALUES (%s,%s,%s,%s,'ACTIVE',FROM_UNIXTIME(%s))", (*binding,obj.memory_id,obj.created_at))

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

    async def append_conversation_message(self, auth: MutationAuthorizationRequest, content: str, *, project_id: str | None=None) -> str:
        return await self.create_memory(auth,ObjectKind.MESSAGE,content,Visibility.USER_PRIVATE,user_id=auth.scope.subject_user_id,project_id=project_id,transformation_id="conversation-message")

    async def import_legacy_memory(self, auth: MutationAuthorizationRequest, legacy_type: str, legacy_namespace: str,
                                   legacy_key: str, kind: ObjectKind, content: str | None) -> str:
        binding=(legacy_type,legacy_namespace,legacy_key)
        async def op(cur: Any) -> str:
            resolved=await self._auth(cur,auth,"IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL); scope=resolved.scope
            await cur.execute("SELECT memory_id,binding_state FROM memory_legacy_bindings WHERE legacy_source_type=%s AND legacy_source_namespace=%s AND legacy_source_key=%s FOR UPDATE", binding)
            existing=await cur.fetchone()
            if existing:
                return existing["memory_id"] if isinstance(existing,Mapping) else existing[0]
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,kind,scope.tenant_id,now,binding)
            rev=MemoryRevision(rid,mid,_digest(content or ""),Visibility.SYSTEM_INTERNAL,None,None,Lifecycle.ACTIVE,now,content,"LEGACY_PROVENANCE_INCOMPLETE")
            prov=MemoryProvenance(_uuid7(),rid,(),"legacy-import","1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,"LEGACY_PROVENANCE_INCOMPLETE")
            event=self._event(scope,resolved,mid,rid,EventKind.IMPORT_LEGACY,now)
            await self._write(cur,obj,rev,prov,event,_derive_projection(mid,[rev],[event]),binding=binding); return mid
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
    async def synthesize_memory(self, auth: MutationAuthorizationRequest, content: str, source_revision_ids: tuple[str,...], *, provider: str | None, model: str | None, transformation_version: str) -> str:
        if not source_revision_ids: raise B9Error("synthesis needs source revisions")
        async def op(cur: Any) -> str:
            resolved=await self._auth(cur,auth,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL); scope=resolved.scope
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,ObjectKind.SYNTHESIS,scope.tenant_id,now)
            # System-internal worker outputs remain in the exact project
            # namespace when their authorized source scope is project-bound.
            rev=MemoryRevision(rid,mid,_digest(content),Visibility.SYSTEM_INTERNAL,None,scope.project_id,Lifecycle.ACTIVE,now,content,"COMPLETE")
            self._assert_project_binding(scope, rev)
            prov=MemoryProvenance(_uuid7(),rid,source_revision_ids,"synthesis",transformation_version,scope.actor_principal,scope.actor_type,scope.subject_user_id,provider,model,now)
            event=self._event(scope,resolved,mid,rid,EventKind.SYNTHESIZE,now,{"derivation_depth":1})
            await self._write(cur,obj,rev,prov,event,_derive_projection(mid,[rev],[event])); return mid
        return await self._store.mutation(op)

    async def reembed_memory(self, auth: MutationAuthorizationRequest, memory_id: str,
                             identity: EmbeddingSpaceIdentity, vector: tuple[float,...] | None=None) -> str:
        async def op(cur: Any) -> str:
            obj,revision=await self._current(cur,memory_id); resolved=await self._auth(cur,auth,"RE_EMBED",revision.visibility); scope=resolved.scope
            revisions,events=await self._assert_reconciled(cur,memory_id)
            if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
            self._assert_project_binding(scope, revision)
            if vector is not None and len(vector) != identity.dimension: raise B9Error("embedding dimension mismatch")
            space_id=identity.embedding_space_id; generation_id=_uuid7(); now=time.time()
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
            except Exception:
                await conn.rollback()
                raise
        return await self.retrieve(decision.scope, limit=limit)

    async def retrieve(self, scope: ScopeContext, *, limit: int = 20) -> tuple[MemoryEnvelope, ...]:
        if scope.project_id:
            raise AuthorizationDenied("project retrieval requires retrieve_authorized")
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
                    await cur.execute(
                        "SELECT o.memory_id,o.object_kind,o.tenant_id,UNIX_TIMESTAMP(o.created_at) AS object_created_at,"
                        "r.revision_id,r.content_digest,r.visibility,r.user_id,r.project_id,r.lifecycle_state,"
                        "UNIX_TIMESTAMP(r.created_at) AS revision_created_at,p.payload,r.provenance_status,r.prior_revision_id "
                        "FROM memory_objects o "
                        "JOIN memory_projections pr ON pr.memory_id=o.memory_id "
                        "JOIN memory_revisions r ON r.revision_id=pr.current_revision_id "
                        "LEFT JOIN memory_revision_payloads p ON p.revision_id=r.revision_id "
                        "WHERE o.tenant_id=%s AND pr.reconciliation_required=FALSE "
                        "AND r.lifecycle_state NOT IN ('TOMBSTONED','PURGED','EXPIRED') "
                        "AND (r.visibility <> 'USER_PRIVATE' OR r.user_id=%s) "
                        "AND (r.visibility <> 'PROJECT_SHARED' OR r.project_id=%s) "
                        "ORDER BY r.created_at DESC LIMIT %s",
                        (scope.tenant_id, scope.subject_user_id, scope.project_id, limit),
                    )
                    rows = await cur.fetchall()
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
            except Exception:
                await conn.rollback()
                raise

    async def prompt_context(self, scope: ScopeContext, *, limit: int = 20) -> PromptMemoryContext:
        return PromptMemoryContext(await self.retrieve(scope, limit=limit))
