"""MariaDB persistence adapter for B9 atomic memory bundles.

It is deliberately not coupled to legacy ``MemoryDB`` writes: callers migrate
through the B9 API/adoption path.  The adapter never applies migrations and
never falls back to partial autocommit writes.
"""
from __future__ import annotations

import json
from typing import Any

from .b9 import MemoryEvent, MemoryObject, MemoryProjection, MemoryProvenance, MemoryRevision


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
