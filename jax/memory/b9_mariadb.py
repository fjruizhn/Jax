"""MariaDB persistence adapter for B9 atomic memory bundles.

It is deliberately not coupled to legacy ``MemoryDB`` writes: callers migrate
through the B9 API/adoption path.  The adapter never applies migrations and
never falls back to partial autocommit writes.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .b9 import (
    Lifecycle, MemoryEnvelope, MemoryEvent, MemoryObject, MemoryProjection,
    MemoryProvenance, MemoryRevision, ObjectKind, PromptMemoryContext,
    ScopeContext, ScopeDenied, Visibility,
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
    def __init__(self, pool: Any):
        self._pool = pool

    async def retrieve(self, scope: ScopeContext, *, limit: int = 20) -> tuple[MemoryEnvelope, ...]:
        if not scope.tenant_id:
            raise ScopeDenied("tenant scope is required")
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
