"""B9 lifecycle and reconciliation job owned by systemd.

This worker never repairs state. It derives projections from immutable B9
revisions/events, flags divergence, and exits unsuccessfully for controlled
reconciliation. A scheduled job must not hide a history/projection split.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
import json
import logging
import os
from typing import Any, Iterable, Mapping

import aiomysql

from jax.core.db_connect_config import db_connect_timeout_seconds
from jax.memory.b9 import EventKind, Lifecycle, MemoryEvent, MemoryRevision, Visibility, _derive_projection

logger = logging.getLogger("jax.memory.lifecycle_worker")


def _timestamp(value: Any) -> float:
    return value.timestamp() if isinstance(value, datetime) else float(value)


def _details(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def projection_mismatches(revision_rows: Iterable[Mapping[str, Any]],
                          event_rows: Iterable[Mapping[str, Any]],
                          projection_rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return IDs whose stored projection differs from canonical B9 reduction."""
    revisions: dict[str, list[MemoryRevision]] = defaultdict(list)
    events: dict[str, list[MemoryEvent]] = defaultdict(list)
    projections = {str(row["memory_id"]): row for row in projection_rows}
    memory_ids = set(projections)
    for row in revision_rows:
        memory_id = str(row["memory_id"])
        memory_ids.add(memory_id)
        revisions[memory_id].append(MemoryRevision(
            str(row["revision_id"]), memory_id, str(row["content_digest"]),
            Visibility(row["visibility"]), row.get("user_id"), row.get("project_id"),
            Lifecycle(row["lifecycle_state"]), _timestamp(row["created_at"]),
            row.get("payload"), str(row["provenance_status"]), row.get("prior_revision_id"),
        ))
    for row in event_rows:
        memory_id = str(row["memory_id"])
        memory_ids.add(memory_id)
        events[memory_id].append(MemoryEvent(
            str(row["event_id"]), memory_id, row.get("revision_id"),
            EventKind(row["event_kind"]), str(row["actor_principal"]),
            row.get("subject_user_id"), str(row["authority_source"]),
            _timestamp(row["occurred_at"]), _details(row.get("details")),
            row.get("compensates_event_id"), row.get("actor_type"), row.get("delegation"),
            row.get("calling_component"), row.get("request_id"), row.get("trace_id"),
        ))
    mismatches: list[str] = []
    for memory_id in sorted(memory_ids):
        canonical = _derive_projection(memory_id, revisions[memory_id], events[memory_id])
        stored = projections.get(memory_id)
        if stored is None or (
            stored.get("current_revision_id") != canonical.current_revision_id
            or stored.get("current_lifecycle_state") != (canonical.current_lifecycle.value if canonical.current_lifecycle else None)
            or bool(stored.get("current_verification_state")) != canonical.current_verification
            or stored.get("canonical_history_digest") != canonical.canonical_history_digest
        ):
            mismatches.append(memory_id)
    return tuple(mismatches)


async def scan_and_mark(pool: Any) -> tuple[str, ...]:
    """Lock objects before projections; reread history in the same transaction.

    Keyset batches bound memory and lock duration. Flags are sticky: this
    scanner never clears reconciliation or rewrites canonical state.
    """
    mismatches = []
    last = ""
    batch_size = int(os.getenv("JAX_MEMORY_LIFECYCLE_BATCH_SIZE", "100"))
    if not 1 <= batch_size <= 1000:
        raise ValueError("invalid lifecycle batch size")
    while True:
        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute("SELECT memory_id FROM memory_objects WHERE memory_id>%s ORDER BY memory_id LIMIT %s FOR UPDATE", (last, batch_size))
                    objects = await cur.fetchall()
                    if not objects:
                        await conn.commit()
                        break
                    for obj in objects:
                        mid = obj["memory_id"]
                        await cur.execute("SELECT memory_id,current_revision_id,current_lifecycle_state,current_verification_state,canonical_history_digest FROM memory_projections WHERE memory_id=%s FOR UPDATE", (mid,))
                        projections = await cur.fetchall()
                        await cur.execute("SELECT revision_id,memory_id,content_digest,visibility,user_id,project_id,lifecycle_state,created_at,payload,provenance_status,prior_revision_id FROM memory_revisions WHERE memory_id=%s ORDER BY created_at,revision_id FOR UPDATE", (mid,))
                        revisions = await cur.fetchall()
                        await cur.execute("SELECT event_id,memory_id,revision_id,event_kind,actor_principal,subject_user_id,authority_source,occurred_at,details,compensates_event_id,actor_type,delegation,calling_component,request_id,trace_id FROM memory_events WHERE memory_id=%s ORDER BY occurred_at,event_id FOR UPDATE", (mid,))
                        events = await cur.fetchall()
                        if not projections or projection_mismatches(revisions, events, projections):
                            mismatches.append(str(mid))
                            await cur.execute("UPDATE memory_projections SET reconciliation_required=TRUE WHERE memory_id=%s", (mid,))
                    last = str(objects[-1]["memory_id"])
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
    return tuple(mismatches)


async def _pool_from_environment() -> Any:
    host, port = os.environ.get("JAX_DB_HOST"), os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError("JAX_DB_HOST and JAX_DB_PORT are required for B9 lifecycle reconciliation")
    return await aiomysql.create_pool(
        host=host, port=int(port), user=os.environ.get("JAX_DB_USER", ""), password=os.environ.get("JAX_DB_PASSWORD", ""),
        db=os.environ.get("JAX_DB_NAME", "jax_memory"), charset="utf8mb4", autocommit=False, minsize=1, maxsize=1,
        connect_timeout=db_connect_timeout_seconds(),
    )


async def main() -> int:
    pool = await _pool_from_environment()
    try:
        mismatches = await scan_and_mark(pool)
    finally:
        pool.close()
        await pool.wait_closed()
    if mismatches:
        logger.error("B9 reconciliation required for %s memory object(s): %s", len(mismatches), ", ".join(mismatches))
        return 2
    logger.info("B9 lifecycle/reconciliation scan completed; all projections canonical")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(asyncio.wait_for(main(), float(os.getenv("JAX_MEMORY_RUN_TIMEOUT_SECONDS", "840")))))
