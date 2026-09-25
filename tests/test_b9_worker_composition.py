import asyncio

from jax.memory.b9 import EmbeddingSpaceIdentity, MutationAuthorizationRequest, ScopeContext, ScopeDenied, Visibility
from jax.memory.embedding_worker import PersistentEmbeddingWriter
from jax.memory.synthesis_worker import PersistentSynthesisWriter
from jax.memory.worker import PersistentExtractionWriter
def _scope(actor="service:memory-extraction", component="memory-extraction", project="project-a"):
    return ScopeContext(actor, "SERVICE", "u1", "tenant-a", project,
                        calling_component=component)


class _API:
    def __init__(self): self.calls = []
    async def create_memory(self, *args, **kwargs): self.calls.append(("create", args, kwargs)); return "m1"
    async def synthesize_memory(self, *args, **kwargs): self.calls.append(("synthesis", args, kwargs)); return "m2"
    async def reembed_memory(self, *args, **kwargs): self.calls.append(("embed", args, kwargs)); return "g1"


def test_extraction_composition_uses_service_request_and_exact_source_scope():
    async def run():
        api = _API()
        async def build(_row, _op, _visibility): return _scope()
        writer = PersistentExtractionWriter(api, build)
        await writer.persist({"tenant_id": "tenant-a", "project_id": "project-a", "user_id": "u1"}, __import__("jax.memory.b9", fromlist=["ObjectKind"]).ObjectKind.FACT, "fact")
        _, args, _ = api.calls[0]
        assert isinstance(args[0], MutationAuthorizationRequest)
        assert args[0].scope.actor_principal == "service:memory-extraction"
        assert args[0].scope.subject_user_id == "u1"
    asyncio.run(run())


def test_synthesis_requires_composition_auth_and_preserves_lineage():
    async def run():
        api = _API()
        async def build(_user, _project, _sources, _op, _visibility):
            return _scope("service:memory-synthesis", "memory-synthesis", "2")
        await PersistentSynthesisWriter(api, build).persist("u1", "2", "insight", ("r1", "r2"))
        _, args, kwargs = api.calls[0]
        assert args[0].operation == "SYNTHESIZE"
        assert args[2] == ("r1", "r2")
        assert kwargs["transformation_version"] == "b9-worker-v1"
    asyncio.run(run())


def test_embedding_composition_uses_reembed_boundary():
    async def run():
        api = _API()
        async def build(_memory, _op, _visibility): return _scope("service:embedding", "embedding", "project-a")
        identity = EmbeddingSpaceIdentity("1", "runtime", "model", "digest", 2, "unit", "cosine")
        await PersistentEmbeddingWriter(api, build).persist("m1", identity, (0.1, 0.2), Visibility.USER_PRIVATE)
        _, args, _ = api.calls[0]
        assert args[0].operation == "RE_EMBED"
        assert args[1] == "m1"
    asyncio.run(run())


def test_worker_cannot_copy_subject_membership_or_widen_project_scope():
    async def run():
        api = _API()
        async def copied_subject(_row, _op, _visibility):
            # A subject is provenance, not the service principal.  The writer
            # rejects this before persistence ever sees a request.
            return ScopeContext("service:memory-extraction", "SERVICE", "u1", "tenant-a", "other-project",
                                calling_component="memory-extraction")
        with __import__("pytest").raises(ScopeDenied, match="source scope"):
            await PersistentExtractionWriter(api, copied_subject).persist(
                {"tenant_id":"tenant-a", "user_id":"u1", "project_id":"project-a"},
                __import__("jax.memory.b9", fromlist=["ObjectKind"]).ObjectKind.FACT, "fact")
    asyncio.run(run())
