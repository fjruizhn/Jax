import asyncio

from jax.memory.b9 import EmbeddingSpaceIdentity, MutationAuthorizationContext, ScopeContext, Visibility
from jax.memory.embedding_worker import PersistentEmbeddingWriter
from jax.memory.synthesis_worker import PersistentSynthesisWriter
from jax.memory.worker import PersistentExtractionWriter


def _auth(operation, visibility):
    return MutationAuthorizationContext(
        ScopeContext("service:memory-worker", "SERVICE", "u1", "tenant-a", "project-a",
                     delegation="scheduled-worker", calling_component="memory-worker"),
        operation, visibility, frozenset({"tenant_member"}), frozenset(), "resolved-membership",
    )


class _API:
    def __init__(self): self.calls = []
    async def create_memory(self, *args, **kwargs): self.calls.append(("create", args, kwargs)); return "m1"
    async def synthesize_memory(self, *args, **kwargs): self.calls.append(("synthesis", args, kwargs)); return "m2"
    async def reembed_memory(self, *args, **kwargs): self.calls.append(("embed", args, kwargs)); return "g1"


def test_extraction_composition_uses_resolved_auth_not_conversation_claims():
    async def run():
        api = _API()
        async def resolve(_row, op, visibility): return _auth(op, visibility)
        writer = PersistentExtractionWriter(api, resolve)
        await writer.persist({"tenant_id": "forged", "project_id": "project-a", "user_id": "other"}, __import__("jax.memory.b9", fromlist=["ObjectKind"]).ObjectKind.FACT, "fact")
        _, args, _ = api.calls[0]
        assert args[0].scope.tenant_id == "tenant-a"
        assert args[0].authority_source == "resolved-membership"
    asyncio.run(run())


def test_synthesis_requires_composition_auth_and_preserves_lineage():
    async def run():
        api = _API()
        async def resolve(_user, _project, op, visibility): return _auth(op, visibility)
        await PersistentSynthesisWriter(api, resolve).persist(1, 2, "insight", ("r1", "r2"))
        _, args, kwargs = api.calls[0]
        assert args[0].operation == "SYNTHESIZE"
        assert args[2] == ("r1", "r2")
        assert kwargs["transformation_version"] == "b9-worker-v1"
    asyncio.run(run())


def test_embedding_composition_uses_reembed_boundary():
    async def run():
        api = _API()
        async def resolve(_memory, op, visibility): return _auth(op, visibility)
        identity = EmbeddingSpaceIdentity("1", "runtime", "model", "digest", 2, "unit", "cosine")
        await PersistentEmbeddingWriter(api, resolve).persist("m1", identity, (0.1, 0.2), Visibility.USER_PRIVATE)
        _, args, _ = api.calls[0]
        assert args[0].operation == "RE_EMBED"
        assert args[1] == "m1"
    asyncio.run(run())
