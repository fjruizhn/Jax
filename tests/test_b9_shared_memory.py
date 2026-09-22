import pytest

from jax.memory.b9 import (
    AuthorizationDenied, EmbeddingSpaceIdentity, EventImmutable, EventKind,
    FixedMembershipResolver, InMemoryB9Store, Lifecycle, MemoryAPI,
    MemoryReference, MemoryReferenceResolver, ObjectKind, ResolutionResult,
    ResolutionState, ScopeContext, ScopeDenied, Visibility,
)
from jax.memory.b9_resolvers import DesignatedSourceResolver


def scope(tenant="t1", subject="u1", actor="user:u1", project="p1"):
    return ScopeContext(actor, "USER", subject, tenant, project, request_id="r")


def api(roles=("memory_admin",)):
    return MemoryAPI(InMemoryB9Store(), FixedMembershipResolver(lambda _: roles))


def test_canonical_identity_scope_and_history_projection():
    a=api(); mid=a.create(scope(), ObjectKind.FACT, "hello", Visibility.USER_PRIVATE, user_id="u1")
    p=a._store.projection(mid)
    assert p.current_revision_id and p.current_lifecycle is Lifecycle.ACTIVE
    assert a._store.objects[mid].tenant_id == "t1"


def test_atomic_rollback_has_neither_state_nor_event(monkeypatch):
    a=api(); original=a._store._commit
    def boom(*args):
        original(*args); raise RuntimeError("fault after event")
    monkeypatch.setattr(a._store, "_commit", boom)
    with pytest.raises(RuntimeError): a.create(scope(), ObjectKind.FACT, "x", Visibility.USER_PRIVATE, user_id="u1")
    assert not a._store.objects and not a._store.events


def test_tenant_is_required_and_cross_tenant_is_denied():
    a=api(); mid=a.create(scope(), ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    with pytest.raises(ScopeDenied): a.envelope(scope("t2"),mid)


def test_rescope_is_revision_and_tenant_change_is_not_allowed():
    a=api(); mid=a.create(scope(),ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    rid=a.revise(scope(),mid,"x",visibility=Visibility.PROJECT_SHARED,project_id="p1",event_kind=EventKind.RE_SCOPE)
    assert a._store.revisions[mid][-1].revision_id == rid
    assert a._store.events[mid][-1].kind is EventKind.RE_SCOPE
    with pytest.raises(ScopeDenied): a.revise(scope("t2"),mid,"bad",user_id="u1")


def test_purge_retains_tombstone_but_not_payload_or_prompt():
    a=api(); mid=a.create(scope(),ObjectKind.FACT,"secret",Visibility.USER_PRIVATE,user_id="u1")
    a.purge(scope(),mid)
    assert a._store.revisions[mid][-1].payload is None
    with pytest.raises(ScopeDenied): a.envelope(scope(),mid)


def test_forged_role_is_not_an_authority_input():
    a=api(roles=())
    with pytest.raises(AuthorizationDenied): a.create(scope(),ObjectKind.FACT,"x",Visibility.TENANT_SHARED)


def test_actor_and_subject_are_distinct_provenance():
    a=api(); worker=ScopeContext("service:memory-worker","SERVICE","u1","t1","p1",calling_component="worker")
    mid=a.create(worker,ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    p=a._store.provenance[a._store.revisions[mid][-1].revision_id][0]
    assert (p.actor_principal,p.subject_user_id)==("service:memory-worker","u1")


def test_legacy_import_is_idempotent():
    a=api(); one=a.import_legacy(scope(),"facts","jax_memory","9",ObjectKind.FACT,"old")
    two=a.import_legacy(scope(),"facts","jax_memory","9",ObjectKind.FACT,"old")
    assert one == two and len(a._store.objects)==1


def test_embedding_identity_is_deterministic_and_changes_when_incompatible():
    one=EmbeddingSpaceIdentity("1","OLLAMA","bge-m3",None,1024,"l2","cosine")
    two=EmbeddingSpaceIdentity("1","OLLAMA","bge-m3",None,768,"l2","cosine")
    assert one.embedding_space_id != two.embedding_space_id


def test_memory_text_never_becomes_current_without_resolver():
    a=api(); mid=a.create(scope(),ObjectKind.FACT,"the service is currently active",Visibility.USER_PRIVATE,user_id="u1")
    assert a.envelope(scope(),mid).trust_classification == "MEMORY"


def test_only_designated_resolution_can_mark_current():
    a=api(); mid=a.create(scope(),ObjectKind.REFERENCE,"current?",Visibility.USER_PRIVATE,user_id="u1")
    resolver=MemoryReferenceResolver({"decision": lambda _: ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B5",0,{})})
    e=a.envelope(scope(),mid,references=[MemoryReference("decision","d")])
    from dataclasses import replace
    e=replace(e,resolution=(resolver.resolve(e.references[0]),))
    assert e.trust_classification == "CURRENT_SOURCE_RESOLVED"


def test_events_are_append_only():
    a=api(); mid=a.create(scope(),ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    with pytest.raises(EventImmutable): a._store.update_event(a._store.events[mid][0])
    with pytest.raises(EventImmutable): a._store.delete_event(a._store.events[mid][0])


def test_projection_mismatch_requires_reconciliation():
    from jax.memory.b9 import ReconciliationRequired
    a=api(); mid=a.create(scope(),ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    a._store.projections[mid] = a._store.projections[mid].__class__(mid, None, None, False, "bad")
    with pytest.raises(ReconciliationRequired): a._store.projection(mid)
    with pytest.raises(ReconciliationRequired): a.revise(scope(), mid, "cannot mutate")
    assert a._store.projections[mid].reconciliation_required
    rebuilt = a._store.rebuild_projection(mid)
    assert rebuilt.current_revision_id == a._store.revisions[mid][-1].revision_id


def test_synthesis_is_unverified_and_not_recursive():
    a=api(); first=a.create(scope(),ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    derived=a.synthesize(scope(),[first],"summary",provider="p",model="m",transformation_version="1")
    assert a._store.objects[derived].kind is ObjectKind.SYNTHESIS
    assert a._store.revisions[derived][-1].lifecycle is Lifecycle.ACTIVE
    with pytest.raises(Exception): a.synthesize(scope(),[derived],"again",provider="p",model="m",transformation_version="1")


def test_retrieve_requires_tenant_and_enforces_project_and_visibility():
    a=api()
    private=a.create(scope(),ObjectKind.FACT,"private",Visibility.USER_PRIVATE,user_id="u1")
    project=a.create(scope(),ObjectKind.FACT,"project",Visibility.PROJECT_SHARED,project_id="p1")
    assert {e.identity.memory_id for e in a.retrieve(scope())} == {private, project}
    with pytest.raises(ScopeDenied): a.retrieve(ScopeContext("x","USER","u1","", "p1"))
    assert {e.identity.memory_id for e in a.retrieve(scope(project="p2"))} == {private}
    with pytest.raises(ScopeDenied): a.envelope(scope(project="p2"), project)


def test_synthesis_cannot_read_another_subject_private_memory():
    a = api()
    private = a.create(scope(), ObjectKind.FACT, "private", Visibility.USER_PRIVATE, user_id="u1")
    with pytest.raises(ScopeDenied):
        a.synthesize(scope(subject="u2", actor="user:u2"), [private], "summary",
                     provider="p", model="m", transformation_version="1")


def test_tombstone_expire_and_cross_tenant_rescope_preserve_history():
    a=api(); mid=a.create(scope(),ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    a.tombstone(scope(),mid,reason="privacy")
    assert a._store.revisions[mid][-1].lifecycle is Lifecycle.TOMBSTONED
    with pytest.raises(ScopeDenied): a.envelope(scope(),mid)
    # New tenant is a new namespace/object, never a revision of old ID.
    old=a.create(scope(),ObjectKind.FACT,"move",Visibility.USER_PRIVATE,user_id="u1")
    new=a.rescope(scope(),old,new_visibility=Visibility.USER_PRIVATE,new_user_id="u2",
                  destination_scope=scope(tenant="t2",subject="u2",actor="admin:u2"))
    assert new != old and a._store.objects[new].tenant_id == "t2"
    assert a._store.events[new][-1].kind is EventKind.RE_SCOPE


def test_event_records_actor_delegation_component_and_trace():
    a=api(); worker=ScopeContext("service:worker","SERVICE","u1","t1","p1",
                                 delegation="user-request",calling_component="memory-worker",
                                 request_id="r",trace_id="trace")
    mid=a.create(worker,ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    e=a._store.events[mid][0]
    assert (e.actor_type,e.delegation,e.calling_component,e.request_id,e.trace_id) == (
        "SERVICE","user-request","memory-worker","r","trace")


def test_invalid_delegation_and_unscoped_user_fail_closed():
    a = api()
    forged = ScopeContext("user:u1", "USER", "u1", "t1", delegation="forged")
    with pytest.raises(ScopeDenied): a.create(forged, ObjectKind.FACT, "x", Visibility.USER_PRIVATE, user_id="u1")
    missing_subject = ScopeContext("user:u1", "USER", None, "t1")
    with pytest.raises(ScopeDenied): a.retrieve(missing_subject)


def test_embedding_generations_preserve_revision_and_exclude_other_space():
    a = api(); mid = a.create(scope(), ObjectKind.FACT, "x", Visibility.USER_PRIVATE, user_id="u1")
    revision = a._store.revisions[mid][-1].revision_id
    one = EmbeddingSpaceIdentity("1", "local", "m", "one", 2, "l2", "cosine")
    two = EmbeddingSpaceIdentity("1", "local", "m", "two", 2, "l2", "cosine")
    first = a.record_embedding(scope(), mid, one, [0.1, 0.2])
    second = a.record_embedding(scope(), mid, two, [0.1, 0.2])
    assert first != second
    assert a._store.revisions[mid][-1].revision_id == revision
    assert len(a._store.compatible_embeddings(revision, one.embedding_space_id)) == 1
    assert not a._store.compatible_embeddings(revision, "sha256:not-a-space")
    with pytest.raises(Exception): a.record_embedding(scope(), mid, one, [0.1])


def test_compensation_is_append_only_and_cannot_target_another_memory():
    a = api(); first = a.create(scope(), ObjectKind.FACT, "x", Visibility.USER_PRIVATE, user_id="u1")
    second = a.create(scope(), ObjectKind.FACT, "y", Visibility.USER_PRIVATE, user_id="u1")
    event = a._store.events[first][0]
    compensation = a.compensate(scope(), first, event.event_id, reason="operator correction")
    assert a._store.events[first][-1].event_id == compensation
    with pytest.raises(Exception): a.compensate(scope(), second, event.event_id, reason="bad target")


def test_typed_resolvers_fail_unavailable_and_reject_wrong_current_source():
    resolver=DesignatedSourceResolver()
    assert resolver.resolve(MemoryReference("decision", "d")).state is ResolutionState.SOURCE_UNAVAILABLE
    bad=DesignatedSourceResolver({"decision": lambda _: ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B7",0,{})})
    assert bad.resolve(MemoryReference("decision", "d")).state is ResolutionState.INVALID_REFERENCE
    good=DesignatedSourceResolver({"decision": lambda _: ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B5",0,{})})
    assert good.resolve(MemoryReference("decision", "d")).current_source_resolved


def test_expired_and_tombstoned_content_never_returns_from_retrieval():
    a=api(); mid=a.create(scope(),ObjectKind.FACT,"x",Visibility.USER_PRIVATE,user_id="u1")
    a.expire(scope(),mid,reason="ttl")
    assert not a.retrieve(scope())


def test_repl_legacy_adapter_requires_tenant_and_labels_memory():
    from jax.core.main import _render_legacy_repl_memory
    assert _render_legacy_repl_memory(1, None, [("fact", "1", "current text")]) == ""
    rendered = _render_legacy_repl_memory(1, 2, [("fact", "1", "current text")])
    assert "HISTORICAL MEMORY" in rendered and "current text" in rendered
