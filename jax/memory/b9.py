"""B9 owned shared-memory boundary.

This module deliberately models memory as operational, scoped history.  It is
not an authority ledger, evidence store, or a source of current truth.  The
small in-memory store is also the reference transactional implementation used
by unit tests; database adapters must preserve the same transaction contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol


def _uuid7() -> str:
    """Return UUIDv7 where supported, retaining sortable UUID identity."""
    if hasattr(uuid, "uuid7"):
        return str(uuid.uuid7())
    # RFC 9562 layout; random low bits are sufficient for the compatibility
    # fallback used only on older interpreters.
    ms = int(time.time() * 1000)
    value = (ms << 80) | (0x7 << 76) | (0x2 << 62) | (uuid.uuid4().int & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   default=str).encode("utf-8")
    ).hexdigest()


class B9Error(RuntimeError): pass
class ScopeDenied(B9Error): pass
class AuthorizationDenied(B9Error): pass
class ReconciliationRequired(B9Error): pass
class EventImmutable(B9Error): pass
class CurrentTruthDenied(B9Error): pass


class ObjectKind(str, Enum):
    CONVERSATION="CONVERSATION"; MESSAGE="MESSAGE"; FACT="FACT"; DECISION_MEMORY="DECISION_MEMORY"
    ACTION_ITEM="ACTION_ITEM"; PERSON_PREFERENCE="PERSON_PREFERENCE"; SYNTHESIS="SYNTHESIS"; REFERENCE="REFERENCE"

class Visibility(str, Enum):
    USER_PRIVATE="USER_PRIVATE"; PROJECT_SHARED="PROJECT_SHARED"; TENANT_SHARED="TENANT_SHARED"; SYSTEM_INTERNAL="SYSTEM_INTERNAL"

class Lifecycle(str, Enum):
    ACTIVE="ACTIVE"; VERIFIED="VERIFIED"; SUPERSEDED="SUPERSEDED"; EXPIRED="EXPIRED"; TOMBSTONED="TOMBSTONED"; PURGED="PURGED"

class EventKind(str, Enum):
    CREATE="CREATE"; VERIFY="VERIFY"; CORRECT="CORRECT"; SUPERSEDE="SUPERSEDE"; EXPIRE="EXPIRE"; TOMBSTONE="TOMBSTONE"; CONTENT_PURGE="CONTENT_PURGE"; RE_SCOPE="RE_SCOPE"; SYNTHESIZE="SYNTHESIZE"; RE_EMBED="RE_EMBED"; IMPORT_LEGACY="IMPORT_LEGACY"; COMPENSATE="COMPENSATE"

class ResolutionState(str, Enum):
    RESOLVED_CURRENT="RESOLVED_CURRENT"; RESOLVED_HISTORICAL="RESOLVED_HISTORICAL"; STALE_REFERENCE="STALE_REFERENCE"; INVALID_REFERENCE="INVALID_REFERENCE"; SOURCE_DELETED="SOURCE_DELETED"; SOURCE_PAYLOAD_PURGED="SOURCE_PAYLOAD_PURGED"; SOURCE_UNAVAILABLE="SOURCE_UNAVAILABLE"; UNRESOLVED="UNRESOLVED"


@dataclass(frozen=True)
class ScopeContext:
    actor_principal: str
    actor_type: str
    subject_user_id: str | None
    tenant_id: str
    project_id: str | None = None
    delegation: str | None = None
    calling_component: str | None = None
    request_id: str | None = None
    trace_id: str | None = None

    def namespace(self, visibility: Visibility, user_id: str | None, project_id: str | None) -> tuple[str, ...]:
        return (self.tenant_id, visibility.value, user_id or "", project_id or "")


@dataclass(frozen=True)
class MutationAuthorizationContext:
    scope: ScopeContext
    operation: str
    target_visibility: Visibility
    resolved_roles: frozenset[str]
    resolved_capabilities: frozenset[str]
    authority_source: str


class MembershipResolver(Protocol):
    def resolve(self, scope: ScopeContext, operation: str, target_visibility: Visibility) -> MutationAuthorizationContext: ...


class FixedMembershipResolver:
    """Small JAX-owned resolver for adapters/tests; it never trusts caller flags."""
    def __init__(self, resolve_roles: Callable[[ScopeContext], Iterable[str]], source: str="jax-membership"):
        self._resolve_roles, self._source = resolve_roles, source

    def resolve(self, scope: ScopeContext, operation: str, target_visibility: Visibility) -> MutationAuthorizationContext:
        if not scope.tenant_id:
            raise ScopeDenied("tenant scope is required")
        roles = frozenset(self._resolve_roles(scope))
        # private users may act on their subject; shared mutation needs a
        # resolver-derived role, never a boolean in a request payload.
        if target_visibility != Visibility.USER_PRIVATE and not roles.intersection({"memory_reviewer", "memory_admin", "tenant_member", "project_member"}):
            raise AuthorizationDenied("resolved membership does not permit shared memory mutation")
        return MutationAuthorizationContext(scope, operation, target_visibility, roles, frozenset(), self._source)


@dataclass(frozen=True)
class MemoryObject:
    memory_id: str
    kind: ObjectKind
    tenant_id: str                    # immutable namespace identity
    created_at: float
    legacy_binding: tuple[str, str, str] | None = None

@dataclass(frozen=True)
class MemoryRevision:
    revision_id: str
    memory_id: str
    content_digest: str
    visibility: Visibility
    user_id: str | None
    project_id: str | None
    lifecycle: Lifecycle
    created_at: float
    payload: str | None
    provenance_status: str = "COMPLETE"
    prior_revision_id: str | None = None

@dataclass(frozen=True)
class MemoryProvenance:
    provenance_id: str
    revision_id: str
    source_revisions: tuple[str, ...]
    transformation_id: str
    transformation_version: str
    actor_principal: str
    actor_type: str
    subject_user_id: str | None
    provider: str | None
    model: str | None
    created_at: float
    limitations: str | None = None

@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    memory_id: str
    revision_id: str | None
    kind: EventKind
    actor_principal: str
    subject_user_id: str | None
    authority_source: str
    occurred_at: float
    details: Mapping[str, Any] = field(default_factory=dict)
    compensates_event_id: str | None = None

@dataclass(frozen=True)
class MemoryProjection:
    memory_id: str
    current_revision_id: str | None
    current_lifecycle: Lifecycle | None
    current_verification: bool
    canonical_history_digest: str
    reconciliation_required: bool = False

@dataclass(frozen=True)
class MemoryReference:
    reference_type: str
    reference_value: str

@dataclass(frozen=True)
class ResolutionResult:
    state: ResolutionState
    source: str
    observed_at: float
    value: Mapping[str, Any] | None = None

    @property
    def current_source_resolved(self) -> bool:
        return self.state is ResolutionState.RESOLVED_CURRENT

@dataclass(frozen=True)
class EmbeddingSpaceIdentity:
    schema_version: str
    provider_runtime_class: str
    model_identifier: str
    model_version_or_digest: str | None
    dimension: int
    normalization: str
    distance_semantics: str

    @property
    def embedding_space_id(self) -> str:
        return _digest({"schema": self.schema_version, "provider": self.provider_runtime_class,
                        "model": self.model_identifier, "version": self.model_version_or_digest or "UNKNOWN",
                        "dimension": self.dimension, "normalization": self.normalization,
                        "distance": self.distance_semantics})

@dataclass(frozen=True)
class EmbeddingGeneration:
    generation_id: str
    revision_id: str
    embedding_space_id: str
    generated_at: float
    vector: tuple[float, ...] | None = None

@dataclass(frozen=True)
class MemoryEnvelope:
    identity: MemoryObject
    revision: MemoryRevision
    provenance: tuple[MemoryProvenance, ...]
    references: tuple[MemoryReference, ...]
    retrieval_metadata: Mapping[str, Any]
    resolution: tuple[ResolutionResult, ...] = ()

    @property
    def trust_classification(self) -> str:
        # Crucially this is not derived from content/score/verification.
        return "CURRENT_SOURCE_RESOLVED" if any(r.current_source_resolved for r in self.resolution) else "MEMORY"

@dataclass(frozen=True)
class PromptMemoryContext:
    entries: tuple[MemoryEnvelope, ...]

    def render(self) -> str:
        sections: list[str] = []
        for e in self.entries:
            if e.revision.payload is None: continue
            if e.trust_classification == "CURRENT_SOURCE_RESOLVED": label="CURRENT-SOURCE-RESOLVED REFERENCE"
            elif e.revision.lifecycle is Lifecycle.VERIFIED: label="VERIFIED MEMORY"
            elif e.identity.kind is ObjectKind.SYNTHESIS: label="SYNTHESIZED MEMORY"
            else: label="UNVERIFIED MEMORY"
            sections.append(f"[{label} id={e.identity.memory_id} revision={e.revision.revision_id}]\n{e.revision.payload}")
        return "\n\n".join(sections)


def legacy_prompt_context(scope: ScopeContext, entries: Iterable[tuple[str, str, str]], *, label: str="LEGACY") -> PromptMemoryContext:
    """Compatibility adapter for existing DB rows during B9 adoption.

    It deliberately does not grant verification/current truth.  Callers need a
    tenant-bearing ScopeContext before legacy content can reach a model.
    """
    if not scope.tenant_id:
        raise ScopeDenied("legacy prompt memory requires tenant scope")
    envelopes=[]
    for kind, source_key, content in entries:
        now=time.time(); mid="legacy:" + _digest((kind, source_key))
        obj=MemoryObject(mid, ObjectKind.FACT if kind == "fact" else ObjectKind.MESSAGE, scope.tenant_id, now,
                         (kind, "jax_memory", source_key))
        rev=MemoryRevision("legacy-revision:" + _digest((kind,source_key,content)),mid,_digest(content),Visibility.USER_PRIVATE,
                           scope.subject_user_id,scope.project_id,Lifecycle.ACTIVE,now,content,"LEGACY_PROVENANCE_INCOMPLETE")
        prov=MemoryProvenance("legacy-provenance:" + _digest((kind,source_key)),rev.revision_id,(),"legacy-prompt-adapter","1",
                              scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,label)
        envelopes.append(MemoryEnvelope(obj,rev,(prov,),(),{"legacy":True}))
    return PromptMemoryContext(tuple(envelopes))


def _derive_projection(memory_id: str, revisions: list[MemoryRevision], events: list[MemoryEvent]) -> MemoryProjection:
    current = revisions[-1] if revisions else None
    history = [{"r": r.revision_id, "l": r.lifecycle.value, "d": r.content_digest} for r in revisions]
    history += [{"e": e.event_id, "k": e.kind.value, "r": e.revision_id} for e in events]
    return MemoryProjection(memory_id, current.revision_id if current else None,
                            current.lifecycle if current else None,
                            bool(current and current.lifecycle is Lifecycle.VERIFIED), _digest(history))


class InMemoryB9Store:
    """Reference atomic store.  A database adapter must expose identical semantics."""
    def __init__(self):
        self.objects: dict[str, MemoryObject] = {}; self.revisions: dict[str, list[MemoryRevision]] = {}
        self.events: dict[str, list[MemoryEvent]] = {}; self.provenance: dict[str, list[MemoryProvenance]] = {}
        self.projections: dict[str, MemoryProjection] = {}; self.bindings: dict[tuple[str,str,str], str] = {}
        self.embeddings: dict[str, list[EmbeddingGeneration]] = {}; self._lock = threading.RLock()

    def transaction(self, operation: Callable[["InMemoryB9Store"], Any]) -> Any:
        with self._lock:
            snapshot = copy.deepcopy((self.objects,self.revisions,self.events,self.provenance,self.projections,self.bindings,self.embeddings))
            try: return operation(self)
            except Exception:
                self.objects,self.revisions,self.events,self.provenance,self.projections,self.bindings,self.embeddings = snapshot
                raise

    def _commit(self, obj: MemoryObject, revision: MemoryRevision, provenance: MemoryProvenance, event: MemoryEvent) -> None:
        if event.kind not in EventKind: raise B9Error("invalid event")
        self.objects.setdefault(obj.memory_id, obj); self.revisions.setdefault(obj.memory_id, []).append(revision)
        self.provenance.setdefault(revision.revision_id, []).append(provenance); self.events.setdefault(obj.memory_id, []).append(event)
        self.projections[obj.memory_id] = _derive_projection(obj.memory_id, self.revisions[obj.memory_id], self.events[obj.memory_id])

    def projection(self, memory_id: str) -> MemoryProjection:
        p = self.projections[memory_id]; expected = _derive_projection(memory_id, self.revisions[memory_id], self.events[memory_id])
        if p != expected: raise ReconciliationRequired(memory_id)
        return p

    def update_event(self, *_: Any, **__: Any) -> None: raise EventImmutable("MemoryEvent is append-only")
    def delete_event(self, *_: Any, **__: Any) -> None: raise EventImmutable("MemoryEvent is append-only")


class MemoryAPI:
    """Supported B9 API; caller claims never construct authority or scope."""
    def __init__(self, store: InMemoryB9Store, authorizer: MembershipResolver): self._store, self._authorizer = store, authorizer

    def _authorize(self, scope: ScopeContext, operation: str, visibility: Visibility) -> MutationAuthorizationContext:
        return self._authorizer.resolve(scope, operation, visibility)

    def create(self, scope: ScopeContext, kind: ObjectKind, content: str, visibility: Visibility,
               *, user_id: str | None=None, project_id: str | None=None, transformation_id: str="application",
               provider: str | None=None, model: str | None=None, provenance_status: str="COMPLETE") -> str:
        auth = self._authorize(scope, "CREATE", visibility)
        if visibility is Visibility.USER_PRIVATE and user_id != scope.subject_user_id: raise ScopeDenied("private memory subject mismatch")
        if project_id and project_id != scope.project_id: raise ScopeDenied("project scope mismatch")
        def work(s: InMemoryB9Store) -> str:
            mid, rid, now = _uuid7(), _uuid7(), time.time()
            obj = MemoryObject(mid, kind, scope.tenant_id, now)
            rev = MemoryRevision(rid, mid, _digest(content), visibility, user_id, project_id, Lifecycle.ACTIVE, now, content, provenance_status)
            prov = MemoryProvenance(_uuid7(), rid, (), transformation_id, "1", scope.actor_principal, scope.actor_type, scope.subject_user_id, provider, model, now)
            event = MemoryEvent(_uuid7(), mid, rid, EventKind.CREATE, scope.actor_principal, scope.subject_user_id, auth.authority_source, now)
            s._commit(obj, rev, prov, event); return mid
        return self._store.transaction(work)

    def import_legacy(self, scope: ScopeContext, legacy_type: str, legacy_namespace: str, legacy_key: str,
                      kind: ObjectKind, content: str | None) -> str:
        binding=(legacy_type, legacy_namespace, legacy_key); auth=self._authorize(scope,"IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL)
        def work(s: InMemoryB9Store) -> str:
            if binding in s.bindings: return s.bindings[binding]
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,kind,scope.tenant_id,now,binding)
            payload=content; rev=MemoryRevision(rid,mid,_digest(content or ""),Visibility.SYSTEM_INTERNAL,None,None,Lifecycle.ACTIVE,now,payload,"LEGACY_PROVENANCE_INCOMPLETE")
            prov=MemoryProvenance(_uuid7(),rid,(),"legacy-import","1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,"LEGACY_PROVENANCE_INCOMPLETE")
            event=MemoryEvent(_uuid7(),mid,rid,EventKind.IMPORT_LEGACY,scope.actor_principal,scope.subject_user_id,auth.authority_source,now)
            s._commit(obj,rev,prov,event); s.bindings[binding]=mid; return mid
        return self._store.transaction(work)

    def revise(self, scope: ScopeContext, memory_id: str, content: str, *, visibility: Visibility | None=None,
               user_id: str | None=None, project_id: str | None=None, event_kind: EventKind=EventKind.CORRECT) -> str:
        obj=self._store.objects[memory_id]; old=self._store.projection(memory_id)
        if old.reconciliation_required: raise ReconciliationRequired(memory_id)
        previous=self._store.revisions[memory_id][-1]; vis=visibility or previous.visibility
        auth=self._authorize(scope,event_kind.value,vis)
        if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant cannot change under same memory id")
        if vis is Visibility.USER_PRIVATE and user_id != scope.subject_user_id: raise ScopeDenied("private subject mismatch")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7(); rev=MemoryRevision(rid,memory_id,_digest(content),vis,user_id,project_id,Lifecycle.ACTIVE,now,content,previous.provenance_status,previous.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(previous.revision_id,),"revision","1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now)
            event=MemoryEvent(_uuid7(),memory_id,rid,event_kind,scope.actor_principal,scope.subject_user_id,auth.authority_source,now,{"old_scope":previous.visibility.value,"new_scope":vis.value})
            s._commit(obj,rev,prov,event); return rid
        return self._store.transaction(work)

    def rescope(self, scope: ScopeContext, memory_id: str, *, new_visibility: Visibility,
                new_user_id: str | None=None, new_project_id: str | None=None,
                destination_scope: ScopeContext | None=None) -> str:
        """Create a same-tenant successor revision or a new object across tenants."""
        obj=self._store.objects[memory_id]; prior=self._store.revisions[memory_id][-1]
        destination_scope = destination_scope or scope
        self._authorize(scope, "RE_SCOPE", prior.visibility)
        self._authorize(destination_scope, "RE_SCOPE", new_visibility)
        if destination_scope.tenant_id != obj.tenant_id:
            # Tenant is namespace identity: transfer is explicitly a new object,
            # never a revision pretending to remain in the old namespace.
            if prior.payload is None: raise ScopeDenied("purged payload cannot be re-scoped")
            return self.create(destination_scope, obj.kind, prior.payload, new_visibility,
                               user_id=new_user_id, project_id=new_project_id,
                               transformation_id="tenant-re-scope", provenance_status=prior.provenance_status)
        return self.revise(scope, memory_id, prior.payload or "", visibility=new_visibility,
                           user_id=new_user_id, project_id=new_project_id, event_kind=EventKind.RE_SCOPE)

    def verify(self, scope: ScopeContext, memory_id: str, *, method: str, limitations: str | None=None) -> str:
        obj=self._store.objects[memory_id]; old=self._store.revisions[memory_id][-1]
        auth=self._authorize(scope,"VERIFY",old.visibility)
        if obj.tenant_id != scope.tenant_id or not auth.resolved_roles.intersection({"memory_reviewer","memory_admin"}):
            raise AuthorizationDenied("verification requires resolved reviewer authority")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7(); rev=replace(old,revision_id=rid,lifecycle=Lifecycle.VERIFIED,created_at=now,prior_revision_id=old.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(old.revision_id,),"human-verification",method,scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,limitations)
            event=MemoryEvent(_uuid7(),memory_id,rid,EventKind.VERIFY,scope.actor_principal,scope.subject_user_id,auth.authority_source,now,{"method":method,"limitations":limitations})
            s._commit(obj,rev,prov,event); return rid
        return self._store.transaction(work)

    def synthesize(self, scope: ScopeContext, source_ids: Iterable[str], content: str, *, provider: str | None, model: str | None, transformation_version: str) -> str:
        source_ids=tuple(source_ids)
        if not source_ids: raise B9Error("synthesis needs source memory")
        sources=[self._store.revisions[i][-1] for i in source_ids]
        if any(self._store.objects[i].tenant_id != scope.tenant_id for i in source_ids): raise ScopeDenied("cross-tenant synthesis")
        if any(self._store.objects[i].kind is ObjectKind.SYNTHESIS for i in source_ids): raise B9Error("recursive automated synthesis prohibited")
        # Synthesis cannot self-verify and begins unverified even from verified inputs.
        auth=self._authorize(scope,"SYNTHESIZE",Visibility.SYSTEM_INTERNAL)
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,ObjectKind.SYNTHESIS,scope.tenant_id,now)
            rev=MemoryRevision(rid,mid,_digest(content),Visibility.SYSTEM_INTERNAL,None,None,Lifecycle.ACTIVE,now,content,"COMPLETE")
            prov=MemoryProvenance(_uuid7(),rid,tuple(x.revision_id for x in sources),"synthesis",transformation_version,scope.actor_principal,scope.actor_type,scope.subject_user_id,provider,model,now)
            event=MemoryEvent(_uuid7(),mid,rid,EventKind.SYNTHESIZE,scope.actor_principal,scope.subject_user_id,auth.authority_source,now,{"derivation_depth":1})
            s._commit(obj,rev,prov,event); return mid
        return self._store.transaction(work)

    def purge(self, scope: ScopeContext, memory_id: str) -> str:
        obj=self._store.objects[memory_id]; old=self._store.revisions[memory_id][-1]; auth=self._authorize(scope,"CONTENT_PURGE",old.visibility)
        if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7(); rev=replace(old,revision_id=rid,lifecycle=Lifecycle.PURGED,created_at=now,payload=None,prior_revision_id=old.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(old.revision_id,),"content-purge","1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now)
            event=MemoryEvent(_uuid7(),memory_id,rid,EventKind.CONTENT_PURGE,scope.actor_principal,scope.subject_user_id,auth.authority_source,now)
            s._commit(obj,rev,prov,event); return rid
        return self._store.transaction(work)

    def envelope(self, scope: ScopeContext, memory_id: str, *, references: Iterable[MemoryReference]=()) -> MemoryEnvelope:
        obj=self._store.objects[memory_id]
        if obj.tenant_id != scope.tenant_id: raise ScopeDenied("cross-tenant retrieval")
        rev=self._store.revisions[memory_id][-1]
        if rev.lifecycle in {Lifecycle.TOMBSTONED,Lifecycle.PURGED} or rev.payload is None: raise ScopeDenied("memory payload unavailable")
        if rev.visibility is Visibility.USER_PRIVATE and rev.user_id != scope.subject_user_id: raise ScopeDenied("private retrieval denied")
        return MemoryEnvelope(obj,rev,tuple(self._store.provenance.get(rev.revision_id,())),tuple(references),{})


class MemoryReferenceResolver:
    """Read-only typed resolver.  Memory text cannot manufacture current truth."""
    def __init__(self, resolvers: Mapping[str, Callable[[str], ResolutionResult]]): self._resolvers=dict(resolvers)
    def resolve(self, reference: MemoryReference) -> ResolutionResult:
        resolver=self._resolvers.get(reference.reference_type)
        if resolver is None: return ResolutionResult(ResolutionState.UNRESOLVED,"none",time.time())
        result=resolver(reference.reference_value)
        if not isinstance(result,ResolutionResult): raise CurrentTruthDenied("resolver must return ResolutionResult")
        return result
