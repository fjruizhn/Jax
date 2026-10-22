"""Production-shaped B9 embedding composition regression.

The raw aiomysql pool deliberately has its default tuple cursor.  Only the
B9 composition under test may turn it into mappings through ``MappingPool``.
"""
from __future__ import annotations

import os

import pytest

from jax.memory.b9 import B9Error, EmbeddingSpaceIdentity, ObjectKind, ScopeContext, ScopeDenied, Visibility
from jax.memory.b9_mariadb import MariaDBB9Store, PersistentMemoryAPI
from jax.memory.embedding_worker import build_persistent_embedding_writer
from jax.memory.mapping_pool import MappingPool
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver


async def _raw_test_pool():
    import aiomysql
    from base_de_test import exigir_base_de_test

    if not os.getenv("JAX_DB_HOST"):
        pytest.skip("requires an isolated CI test database")
    return await aiomysql.create_pool(
        host=os.environ["JAX_DB_HOST"], port=int(os.getenv("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", "root"), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=exigir_base_de_test(), minsize=1, maxsize=1, autocommit=False,
        connect_timeout=5,
    )


async def _prepare_schema(raw):
    # Temporary tables bind to the one physical connection of the raw pool.
    tables = (
        "CREATE TEMPORARY TABLE memory_objects (memory_id CHAR(36) PRIMARY KEY, object_kind VARCHAR(32), tenant_id VARCHAR(128), created_at DATETIME(6), legacy_source_type VARCHAR(64), legacy_source_namespace VARCHAR(255), legacy_source_key VARCHAR(255))",
        "CREATE TEMPORARY TABLE memory_revisions (revision_id CHAR(36) PRIMARY KEY, memory_id CHAR(36), content_digest CHAR(71), visibility VARCHAR(32), user_id VARCHAR(128), project_id VARCHAR(128), lifecycle_state VARCHAR(32), created_at DATETIME(6), payload LONGBLOB, provenance_status VARCHAR(64), prior_revision_id CHAR(36), tenant_id VARCHAR(128))",
        "CREATE TEMPORARY TABLE memory_revision_payloads (revision_id CHAR(36) PRIMARY KEY, payload LONGBLOB)",
        "CREATE TEMPORARY TABLE memory_provenance (provenance_id CHAR(36) PRIMARY KEY, revision_id CHAR(36), source_revisions JSON, transformation_id VARCHAR(128), transformation_version VARCHAR(64), actor_principal VARCHAR(255), actor_type VARCHAR(64), subject_user_id VARCHAR(128), provider VARCHAR(128), model VARCHAR(255), created_at DATETIME(6), limitations TEXT)",
        "CREATE TEMPORARY TABLE memory_events (event_id CHAR(36) PRIMARY KEY, memory_id CHAR(36), revision_id CHAR(36), event_kind VARCHAR(32), actor_principal VARCHAR(255), subject_user_id VARCHAR(128), authority_source VARCHAR(255), occurred_at DATETIME(6), details JSON, compensates_event_id CHAR(36), actor_type VARCHAR(64), delegation VARCHAR(255), calling_component VARCHAR(255), request_id VARCHAR(255), trace_id VARCHAR(255))",
        "CREATE TEMPORARY TABLE memory_projections (memory_id CHAR(36) PRIMARY KEY, current_revision_id CHAR(36), current_lifecycle_state VARCHAR(32), current_verification_state BOOLEAN, canonical_history_digest CHAR(71), reconciliation_required BOOLEAN)",
        "CREATE TEMPORARY TABLE embedding_spaces (embedding_space_id CHAR(71) PRIMARY KEY, schema_version VARCHAR(64), provider_runtime_class VARCHAR(128), model_identifier VARCHAR(255), model_version_or_digest VARCHAR(255), dimension INT, normalization VARCHAR(64), distance_semantics VARCHAR(64), created_at DATETIME(6))",
        "CREATE TEMPORARY TABLE embedding_generations (generation_id CHAR(36) PRIMARY KEY, revision_id CHAR(36), embedding_space_id CHAR(71), generated_at DATETIME(6), embedding_payload LONGBLOB)",
        "CREATE TEMPORARY TABLE jax_users (user_id VARCHAR(128), tenant_id VARCHAR(128), status VARCHAR(16), role VARCHAR(32), PRIMARY KEY (user_id,tenant_id))",
    )
    async with raw.acquire() as conn:
        async with conn.cursor() as cur:
            for ddl in tables:
                await cur.execute(ddl)
            await cur.execute("INSERT INTO jax_users VALUES ('user-1','tenant-1','ACTIVE','member')")
        await conn.commit()


@pytest.mark.asyncio
async def test_raw_tuple_pool_is_normalized_only_at_embedding_b9_boundary():
    raw = await _raw_test_pool()
    try:
        await _prepare_schema(raw)
        # Prove the underlying runtime condition before B9 composition: bare
        # aiomysql rows are tuples, not mappings.
        async with raw.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT user_id,tenant_id FROM jax_users")
                assert await cur.fetchone() == ("user-1", "tenant-1")

        mapped = MappingPool(raw)
        resolver = MariaDBScopeAuthorityResolver(mapped)
        api = PersistentMemoryAPI(MariaDBB9Store(mapped), resolver)
        request = __import__("jax.memory.b9", fromlist=["MutationAuthorizationRequest"]).MutationAuthorizationRequest(
            ScopeContext("user-1", "USER", "user-1", "tenant-1"), "CREATE", Visibility.USER_PRIVATE
        )
        memory_id = await api.create_memory(request, ObjectKind.FACT, "tuple boundary fixture",
                                            Visibility.USER_PRIVATE, user_id="user-1")
        async with mapped.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT current_revision_id FROM memory_projections WHERE memory_id=%s", (memory_id,))
                revision_id = (await cur.fetchone())["current_revision_id"]

        writer = build_persistent_embedding_writer(raw)
        identity = EmbeddingSpaceIdentity("b9-v1", "ollama", "bge-m3", "digest-fixture", 2, "unit", "cosine")
        # Validation failure occurs inside the B9 mutation transaction and
        # must leave neither generation nor space behind.
        with pytest.raises(B9Error, match="dimension mismatch"):
            await writer.persist(memory_id, identity, (0.1,), Visibility.USER_PRIVATE,
                                 expected_revision_id=revision_id)
        async with mapped.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) AS n FROM embedding_generations")
                assert (await cur.fetchone())["n"] == 0
                await cur.execute("SELECT COUNT(*) AS n FROM embedding_spaces")
                assert (await cur.fetchone())["n"] == 0

        generation_id = await writer.persist(memory_id, identity, (0.1, 0.2), Visibility.USER_PRIVATE,
                                             expected_revision_id=revision_id)
        # Same immutable identity is idempotent: no second space, generation,
        # or RE_EMBED event is created.
        assert await writer.persist(memory_id, identity, (0.1, 0.2), Visibility.USER_PRIVATE,
                                    expected_revision_id=revision_id) == generation_id
        async with mapped.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT revision_id,embedding_space_id FROM embedding_generations")
                generation = await cur.fetchone()
                assert generation == {"revision_id": revision_id, "embedding_space_id": identity.embedding_space_id}
                await cur.execute("SELECT COUNT(*) AS n FROM embedding_spaces")
                assert (await cur.fetchone())["n"] == 1
                await cur.execute("SELECT event_kind,actor_principal,subject_user_id,authority_source FROM memory_events ORDER BY occurred_at,event_id")
                events = await cur.fetchall()
                assert [row["event_kind"] for row in events] == ["CREATE", "RE_EMBED"]
                assert events[-1] == {"event_kind": "RE_EMBED", "actor_principal": "service:embedding", "subject_user_id": "user-1", "authority_source": "jax-service-operation-policy:v1"}
                await cur.execute("SELECT current_revision_id,reconciliation_required FROM memory_projections WHERE memory_id=%s", (memory_id,))
                assert await cur.fetchone() == {"current_revision_id": revision_id, "reconciliation_required": 0}
                await cur.execute("UPDATE jax_users SET status='INACTIVE' WHERE user_id='user-1'")
            await conn.commit()
        # The resolver is invoked in the re-embedding transaction, not trusted
        # from the already-built request scope.
        with pytest.raises(ScopeDenied):
            await writer.persist(memory_id, identity, (0.1, 0.2), Visibility.USER_PRIVATE,
                                 expected_revision_id=revision_id)
    finally:
        raw.close()
        await raw.wait_closed()
