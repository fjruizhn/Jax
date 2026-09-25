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

SYNTHESIS_SOURCE_KINDS=frozenset({ObjectKind.MESSAGE,ObjectKind.FACT,ObjectKind.DECISION_MEMORY,
                                  ObjectKind.ACTION_ITEM,ObjectKind.PERSON_PREFERENCE,ObjectKind.REFERENCE})

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
    # Resolver output is retained as decision provenance.  It is *not* a
    # capability: every sensitive execution boundary must resolve/revalidate
    # the request against its designated authority source again.
    project_authorization: Any | None = None

    def validate(self) -> None:
        """Reject incomplete or self-asserted acting contexts before use.

        This is deliberately structural validation only.  Membership and any
        delegation grant are resolved by ``MembershipResolver``; a request
        cannot make itself authoritative by filling in these fields.
        """
        if not self.tenant_id:
            raise ScopeDenied("tenant scope is required")
        if not self.actor_principal or not self.actor_type:
            raise ScopeDenied("authenticated actor is required")
        if self.project_id:
            authorization = self.project_authorization
            # This is deliberately only an internal consistency check.  A
            # Python object (including an object with an ``is_resolved``
            # attribute) cannot prove authorization.  The database-backed
            # resolver is the authority boundary.
            if authorization is None:
                raise ScopeDenied("project scope requires resolver decision provenance")
            if (str(getattr(authorization, "project_id", "")) != str(self.project_id)
                    or str(getattr(authorization, "tenant_id", "")) != str(self.tenant_id)
                    or str(getattr(authorization, "subject_user_id", "")) != str(self.subject_user_id)
                    or getattr(authorization, "project_status", None) != "ACTIVE"
                    or (getattr(authorization, "membership_status", None) != "ACTIVE" and not (
                        self.actor_type == "SERVICE" and self.actor_principal == "service:memory-synthesis"
                        and getattr(authorization, "membership_status", None) == "SERVICE_POLICY"))):
                raise ScopeDenied("project authorization does not match scope")
        if self.actor_type == "USER":
            if not self.subject_user_id:
                raise ScopeDenied("user actor requires subject")
            if self.delegation:
                raise ScopeDenied("user actor cannot assert delegation")
        elif self.delegation and not self.subject_user_id:
            raise ScopeDenied("delegation requires a subject")

    def namespace(self, visibility: Visibility, user_id: str | None, project_id: str | None) -> tuple[str, ...]:
        return (self.tenant_id, visibility.value, user_id or "", project_id or "")


@dataclass(frozen=True)
class MutationAuthorizationContext:
    """Resolver output and audit provenance, never a bearer capability.

    Sensitive stores must call their configured resolver while executing the
    request/transaction.  In particular, callers must not be authorized just
    because they can instantiate this dataclass.
    """
    scope: ScopeContext
    operation: str
    target_visibility: Visibility
    resolved_roles: frozenset[str]
    resolved_capabilities: frozenset[str]
    authority_source: str


@dataclass(frozen=True)
class MutationAuthorizationRequest:
    """Untrusted mutation inputs submitted to the designated resolver.

    This deliberately contains no roles, capabilities or prior authorization
    result.  It is the only value a caller may carry into a persistent B9
    mutation boundary.
    """
    scope: ScopeContext
    operation: str
    target_visibility: Visibility


class MembershipResolver(Protocol):
    def resolve(self, scope: ScopeContext, operation: str, target_visibility: Visibility) -> MutationAuthorizationContext: ...


class FixedMembershipResolver:
    """Small JAX-owned resolver for adapters/tests; it never trusts caller flags."""
    def __init__(self, resolve_roles: Callable[[ScopeContext], Iterable[str]], source: str="jax-membership"):
        self._resolve_roles, self._source = resolve_roles, source

    def resolve(self, scope: ScopeContext, operation: str, target_visibility: Visibility) -> MutationAuthorizationContext:
        scope.validate()
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
    actor_type: str | None = None
    delegation: str | None = None
    calling_component: str | None = None
    request_id: str | None = None
    trace_id: str | None = None

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
            if e.trust_classification == "CURRENT_SOURCE_RESOLVED":
                label="CURRENT-SOURCE-RESOLVED REFERENCE"
            elif any(r.state in {ResolutionState.UNRESOLVED, ResolutionState.SOURCE_UNAVAILABLE,
                                  ResolutionState.INVALID_REFERENCE, ResolutionState.SOURCE_DELETED,
                                  ResolutionState.SOURCE_PAYLOAD_PURGED} for r in e.resolution):
                label="UNRESOLVED REFERENCE"
            elif e.revision.lifecycle is Lifecycle.VERIFIED:
                label="VERIFIED MEMORY"
            elif e.identity.kind is ObjectKind.SYNTHESIS:
                label="SYNTHESIZED MEMORY"
            elif e.identity.kind is ObjectKind.MESSAGE and any(p.actor_type == "USER" for p in e.provenance):
                label="USER-PROVIDED MEMORY"
            elif e.revision.provenance_status.startswith("LEGACY"):
                label="HISTORICAL MEMORY"
            else:
                label="UNVERIFIED MEMORY"
            # A single JSON string is data, even when its contents contain
            # newlines or text resembling a section header.
            payload=json.dumps(e.revision.payload, ensure_ascii=True).replace("[", "\\u005b").replace("]", "\\u005d")
            sections.append(f"[{label} id={e.identity.memory_id} revision={e.revision.revision_id}]\n{payload}")
        return "\n\n".join(sections)


def legacy_prompt_context(scope: ScopeContext, entries: Iterable[tuple[str, str, str]], *, label: str="LEGACY") -> PromptMemoryContext:
    """Compatibility adapter for existing DB rows during B9 adoption.

    It deliberately does not grant verification/current truth.  Callers need a
    tenant-bearing ScopeContext before legacy content can reach a model.
    """
    try:
        scope.validate()
    except ScopeDenied as exc:
        raise ScopeDenied("legacy prompt memory requires validated scope") from exc
    envelopes=[]
    for kind, source_key, content in entries:
        now=time.time(); mid="legacy:" + _digest((scope.tenant_id,kind, source_key))
        obj=MemoryObject(mid, ObjectKind.FACT if kind == "fact" else ObjectKind.MESSAGE, scope.tenant_id, now,
                         (kind, "jax_memory", source_key))
        rev=MemoryRevision("legacy-revision:" + _digest((scope.tenant_id,kind,source_key,content)),mid,_digest(content),Visibility.USER_PRIVATE,
                           scope.subject_user_id,scope.project_id,Lifecycle.ACTIVE,now,content,"LEGACY_PROVENANCE_INCOMPLETE")
        prov=MemoryProvenance("legacy-provenance:" + _digest((scope.tenant_id,kind,source_key)),rev.revision_id,(),"legacy-prompt-adapter","1",
                              scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,label)
        envelopes.append(MemoryEnvelope(obj,rev,(prov,),(),{"legacy":True}))
    return PromptMemoryContext(tuple(envelopes))


def _derive_projection(memory_id: str, revisions: list[MemoryRevision], events: list[MemoryEvent]) -> MemoryProjection:
    """Reduce immutable history into an efficiency projection.

    A revision is immutable content/scope state; events decide which state is
    presently usable.  Keeping that rule here prevents a projection from
    becoming an alternative source of lifecycle truth.
    """
    current = revisions[-1] if revisions else None
    if current is not None:
        by_id = {revision.revision_id: revision for revision in revisions}
        state = current.lifecycle
        for event in events:
            if event.revision_id and event.revision_id in by_id:
                state = by_id[event.revision_id].lifecycle
            elif event.kind is EventKind.TOMBSTONE:
                state = Lifecycle.TOMBSTONED
            elif event.kind is EventKind.CONTENT_PURGE:
                state = Lifecycle.PURGED
        current = replace(current, lifecycle=state)
    # Include every state-affecting immutable field.  Hashing only event IDs
    # would let a modified authorization/detail record look canonical.
    history = [{"r": r.revision_id, "m": r.memory_id, "l": r.lifecycle.value,
                "d": r.content_digest, "v": r.visibility.value, "u": r.user_id,
                "p": r.project_id, "prior": r.prior_revision_id,
                "provenance": r.provenance_status} for r in revisions]
    history += [{"e": e.event_id, "m": e.memory_id, "k": e.kind.value,
                 "r": e.revision_id, "a": e.actor_principal, "s": e.subject_user_id,
                 "authority": e.authority_source, "at": e.actor_type,
                 "delegation": e.delegation, "details": dict(e.details),
                 "compensates": e.compensates_event_id} for e in events]
    return MemoryProjection(memory_id, current.revision_id if current else None,
                            current.lifecycle if current else None,
                            bool(current and current.lifecycle is Lifecycle.VERIFIED), _digest(history))


class InMemoryB9Store:
    """Reference atomic store.  A database adapter must expose identical semantics."""
    def __init__(self):
        self.objects: dict[str, MemoryObject] = {}; self.revisions: dict[str, list[MemoryRevision]] = {}
        self.events: dict[str, list[MemoryEvent]] = {}; self.provenance: dict[str, list[MemoryProvenance]] = {}
        self.projections: dict[str, MemoryProjection] = {}; self.bindings: dict[tuple[str,str,str,str], str] = {}
        self.embeddings: dict[str, list[EmbeddingGeneration]] = {}
        self.embedding_spaces: dict[str, EmbeddingSpaceIdentity] = {}
        self._lock = threading.RLock()

    def transaction(self, operation: Callable[["InMemoryB9Store"], Any]) -> Any:
        with self._lock:
            snapshot = copy.deepcopy((self.objects,self.revisions,self.events,self.provenance,self.projections,self.bindings,self.embeddings,self.embedding_spaces))
            try: return operation(self)
            except Exception:
                self.objects,self.revisions,self.events,self.provenance,self.projections,self.bindings,self.embeddings,self.embedding_spaces = snapshot
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

    def rebuild_projection(self, memory_id: str) -> MemoryProjection:
        """Reconstruct the derived projection from immutable canonical rows."""
        with self._lock:
            if memory_id not in self.objects:
                raise KeyError(memory_id)
            return _derive_projection(memory_id, self.revisions[memory_id], self.events[memory_id])

    def detect_reconciliation(self, memory_id: str) -> bool:
        """Mark and report a projection mismatch without silently repairing it."""
        with self._lock:
            expected = self.rebuild_projection(memory_id)
            actual = self.projections.get(memory_id)
            if actual != expected:
                self.projections[memory_id] = replace(expected, reconciliation_required=True)
                return True
            return actual.reconciliation_required

    def register_embedding_space(self, identity: EmbeddingSpaceIdentity) -> str:
        space_id = identity.embedding_space_id
        with self._lock:
            known = self.embedding_spaces.get(space_id)
            if known is not None and known != identity:
                raise B9Error("embedding space ID cannot identify incompatible metadata")
            self.embedding_spaces[space_id] = identity
        return space_id

    def add_embedding_generation(self, generation: EmbeddingGeneration) -> None:
        with self._lock:
            identity = self.embedding_spaces.get(generation.embedding_space_id)
            if identity is None:
                raise B9Error("embedding generation requires registered space")
            if generation.revision_id not in {r.revision_id for rs in self.revisions.values() for r in rs}:
                raise KeyError(generation.revision_id)
            if generation.vector is not None and len(generation.vector) != identity.dimension:
                raise B9Error("embedding vector dimension does not match space")
            generations = self.embeddings.setdefault(generation.revision_id, [])
            if any(g.generation_id == generation.generation_id for g in generations):
                raise B9Error("embedding generation is immutable")
            generations.append(generation)

    def compatible_embeddings(self, revision_id: str, space_id: str) -> tuple[EmbeddingGeneration, ...]:
        """Never return vectors from a different compatibility identity."""
        return tuple(g for g in self.embeddings.get(revision_id, ()) if g.embedding_space_id == space_id)

    def update_event(self, *_: Any, **__: Any) -> None: raise EventImmutable("MemoryEvent is append-only")
    def delete_event(self, *_: Any, **__: Any) -> None: raise EventImmutable("MemoryEvent is append-only")


class MemoryAPI:
    """Supported B9 API; caller claims never construct authority or scope."""
    def __init__(self, store: InMemoryB9Store, authorizer: MembershipResolver): self._store, self._authorizer = store, authorizer

    def _authorize(self, scope: ScopeContext, operation: str, visibility: Visibility) -> MutationAuthorizationContext:
        return self._authorizer.resolve(scope, operation, visibility)

    def _canonical_for_mutation(self, memory_id: str) -> None:
        # Detection records a durable repair-needed signal; mutation then fails
        # closed rather than overwriting a projection discrepancy.
        if self._store.detect_reconciliation(memory_id):
            raise ReconciliationRequired(memory_id)

    @staticmethod
    def _assert_read_scope(scope: ScopeContext, obj: MemoryObject, revision: MemoryRevision) -> None:
        if obj.tenant_id != scope.tenant_id:
            raise ScopeDenied("cross-tenant retrieval")
        if revision.visibility is Visibility.USER_PRIVATE and revision.user_id != scope.subject_user_id:
            raise ScopeDenied("private retrieval denied")
        if revision.project_id is not None and (
            not scope.project_id or revision.project_id != scope.project_id
        ):
            raise ScopeDenied("project retrieval denied")

    @staticmethod
    def _event(scope: ScopeContext, auth: MutationAuthorizationContext, memory_id: str,
               revision_id: str | None, kind: EventKind, now: float,
               details: Mapping[str, Any] | None = None) -> MemoryEvent:
        return MemoryEvent(
            _uuid7(), memory_id, revision_id, kind, scope.actor_principal,
            scope.subject_user_id, auth.authority_source, now, details or {},
            actor_type=scope.actor_type, delegation=scope.delegation,
            calling_component=scope.calling_component, request_id=scope.request_id,
            trace_id=scope.trace_id,
        )

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
            event = self._event(scope, auth, mid, rid, EventKind.CREATE, now)
            s._commit(obj, rev, prov, event); return mid
        return self._store.transaction(work)

    def import_legacy(self, scope: ScopeContext, legacy_type: str, legacy_namespace: str, legacy_key: str,
                      kind: ObjectKind, content: str | None) -> str:
        binding=(legacy_type, legacy_namespace, legacy_key); key=(scope.tenant_id,*binding)
        auth=self._authorize(scope,"IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL)
        def work(s: InMemoryB9Store) -> str:
            if key in s.bindings: return s.bindings[key]
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,kind,scope.tenant_id,now,binding)
            payload=content; rev=MemoryRevision(rid,mid,_digest(content or ""),Visibility.SYSTEM_INTERNAL,None,None,Lifecycle.ACTIVE,now,payload,"LEGACY_PROVENANCE_INCOMPLETE")
            prov=MemoryProvenance(_uuid7(),rid,(),"legacy-import","1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,"LEGACY_PROVENANCE_INCOMPLETE")
            event=self._event(scope,auth,mid,rid,EventKind.IMPORT_LEGACY,now)
            s._commit(obj,rev,prov,event); s.bindings[key]=mid; return mid
        return self._store.transaction(work)

    def revise(self, scope: ScopeContext, memory_id: str, content: str, *, visibility: Visibility | None=None,
               user_id: str | None=None, project_id: str | None=None, event_kind: EventKind=EventKind.CORRECT) -> str:
        obj=self._store.objects[memory_id]
        # Always compare the stored projection with canonical history before a
        # sensitive mutation.  Reading a boolean from the projection alone
        # would let a tampered/stale projection escape detection.
        self._canonical_for_mutation(memory_id)
        previous=self._store.revisions[memory_id][-1]; vis=visibility or previous.visibility
        auth=self._authorize(scope,event_kind.value,vis)
        if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant cannot change under same memory id")
        if vis is Visibility.USER_PRIVATE and user_id != scope.subject_user_id: raise ScopeDenied("private subject mismatch")
        if project_id is not None and project_id != scope.project_id:
            raise ScopeDenied("project scope mismatch")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7(); rev=MemoryRevision(rid,memory_id,_digest(content),vis,user_id,project_id,Lifecycle.ACTIVE,now,content,previous.provenance_status,previous.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(previous.revision_id,),"revision","1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now)
            event=self._event(scope,auth,memory_id,rid,event_kind,now,{"old_scope":previous.visibility.value,"new_scope":vis.value})
            s._commit(obj,rev,prov,event); return rid
        return self._store.transaction(work)

    def rescope(self, scope: ScopeContext, memory_id: str, *, new_visibility: Visibility,
                new_user_id: str | None=None, new_project_id: str | None=None,
                destination_scope: ScopeContext | None=None) -> str:
        """Create a same-tenant successor revision or a new object across tenants."""
        obj=self._store.objects[memory_id]; prior=self._store.revisions[memory_id][-1]
        self._canonical_for_mutation(memory_id)
        destination_scope = destination_scope or scope
        self._authorize(scope, "RE_SCOPE", prior.visibility)
        self._authorize(destination_scope, "RE_SCOPE", new_visibility)
        if destination_scope.tenant_id != obj.tenant_id:
            # Tenant is namespace identity: transfer is explicitly a new object,
            # never a revision pretending to remain in the old namespace.
            if prior.payload is None: raise ScopeDenied("purged payload cannot be re-scoped")
            # Tenant is part of namespace identity.  The destination is a
            # distinct object, but the provenance records the predecessor so
            # that this is never a silent copy across tenants.
            return self._create_cross_tenant_successor(
                destination_scope, obj, prior, new_visibility, new_user_id,
                new_project_id,
            )
        return self.revise(scope, memory_id, prior.payload or "", visibility=new_visibility,
                           user_id=new_user_id, project_id=new_project_id, event_kind=EventKind.RE_SCOPE)

    def _create_cross_tenant_successor(self, destination: ScopeContext, old_object: MemoryObject,
                                       prior: MemoryRevision, visibility: Visibility,
                                       user_id: str | None, project_id: str | None) -> str:
        auth = self._authorize(destination, "RE_SCOPE", visibility)
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); mid,rid=_uuid7(),_uuid7()
            obj=MemoryObject(mid,old_object.kind,destination.tenant_id,now)
            rev=MemoryRevision(rid,mid,_digest(prior.payload or ""),visibility,user_id,project_id,
                               Lifecycle.ACTIVE,now,prior.payload,prior.provenance_status)
            prov=MemoryProvenance(_uuid7(),rid,(prior.revision_id,),"tenant-re-scope","1",
                                  destination.actor_principal,destination.actor_type,
                                  destination.subject_user_id,None,None,now)
            event=self._event(destination,auth,mid,rid,EventKind.RE_SCOPE,now,
                              {"old_memory_id":old_object.memory_id,"old_tenant_id":old_object.tenant_id,
                               "new_tenant_id":destination.tenant_id})
            s._commit(obj,rev,prov,event); return mid
        return self._store.transaction(work)

    def verify(self, scope: ScopeContext, memory_id: str, *, method: str, limitations: str | None=None) -> str:
        obj=self._store.objects[memory_id]; old=self._store.revisions[memory_id][-1]
        self._canonical_for_mutation(memory_id)
        auth=self._authorize(scope,"VERIFY",old.visibility)
        if obj.tenant_id != scope.tenant_id or not auth.resolved_roles.intersection({"memory_reviewer","memory_admin"}):
            raise AuthorizationDenied("verification requires resolved reviewer authority")
        if old.lifecycle not in {Lifecycle.ACTIVE, Lifecycle.VERIFIED}:
            raise ScopeDenied("memory lifecycle is not eligible for verification")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7(); rev=replace(old,revision_id=rid,lifecycle=Lifecycle.VERIFIED,created_at=now,prior_revision_id=old.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(old.revision_id,),"human-verification",method,scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,limitations)
            event=self._event(scope,auth,memory_id,rid,EventKind.VERIFY,now,{"method":method,"limitations":limitations})
            s._commit(obj,rev,prov,event); return rid
        return self._store.transaction(work)

    def synthesize(self, scope: ScopeContext, source_ids: Iterable[str], content: str, *, provider: str | None, model: str | None, transformation_version: str) -> str:
        source_ids=tuple(source_ids)
        if not source_ids: raise B9Error("synthesis needs source memory")
        def work(s: InMemoryB9Store) -> str:
            sources=[]
            for memory_id in source_ids:
                self._canonical_for_mutation(memory_id)
                obj=s.objects[memory_id]; revision=s.revisions[memory_id][-1]
                self._assert_read_scope(scope,obj,revision)
                if obj.kind not in SYNTHESIS_SOURCE_KINDS or revision.lifecycle not in {Lifecycle.ACTIVE,Lifecycle.VERIFIED} or revision.payload is None:
                    raise ScopeDenied("source revision is ineligible for synthesis")
                sources.append(revision)
            effective_scope={(r.visibility,r.user_id,r.project_id) for r in sources}
            if len(effective_scope) != 1: raise ScopeDenied("synthesis sources have different scopes")
            visibility,user_id,project_id=effective_scope.pop()
            auth=self._authorize(scope,"SYNTHESIZE",visibility)
            now=time.time(); mid,rid=_uuid7(),_uuid7(); obj=MemoryObject(mid,ObjectKind.SYNTHESIS,scope.tenant_id,now)
            rev=MemoryRevision(rid,mid,_digest(content),visibility,user_id,project_id,Lifecycle.ACTIVE,now,content,"COMPLETE")
            prov=MemoryProvenance(_uuid7(),rid,tuple(x.revision_id for x in sources),"synthesis",transformation_version,scope.actor_principal,scope.actor_type,scope.subject_user_id,provider,model,now)
            event=self._event(scope,auth,mid,rid,EventKind.SYNTHESIZE,now,{"derivation_depth":1})
            s._commit(obj,rev,prov,event); return mid
        return self._store.transaction(work)

    def purge(self, scope: ScopeContext, memory_id: str) -> str:
        obj=self._store.objects[memory_id]; old=self._store.revisions[memory_id][-1]; auth=self._authorize(scope,"CONTENT_PURGE",old.visibility)
        self._canonical_for_mutation(memory_id)
        if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7(); rev=replace(old,revision_id=rid,lifecycle=Lifecycle.PURGED,created_at=now,payload=None,prior_revision_id=old.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(old.revision_id,),"content-purge","1",scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now)
            event=self._event(scope,auth,memory_id,rid,EventKind.CONTENT_PURGE,now)
            s._commit(obj,rev,prov,event)
            s.revisions[memory_id]=[replace(r,payload=None) for r in s.revisions[memory_id]]
            for r in s.revisions[memory_id]: s.embeddings.pop(r.revision_id,None)
            return rid
        return self._store.transaction(work)

    def tombstone(self, scope: ScopeContext, memory_id: str, *, reason: str) -> str:
        """Withdraw content from retrieval while preserving privacy-safe identity."""
        obj=self._store.objects[memory_id]; old=self._store.revisions[memory_id][-1]
        self._canonical_for_mutation(memory_id)
        auth=self._authorize(scope,"TOMBSTONE",old.visibility)
        if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7()
            rev=replace(old, revision_id=rid, lifecycle=Lifecycle.TOMBSTONED,
                        created_at=now, payload=None, prior_revision_id=old.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(old.revision_id,),"tombstone","1",
                                  scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,reason)
            event=self._event(scope,auth,memory_id,rid,EventKind.TOMBSTONE,now,{"reason":reason})
            s._commit(obj,rev,prov,event); return rid
        return self._store.transaction(work)

    def expire(self, scope: ScopeContext, memory_id: str, *, reason: str) -> str:
        return self._lifecycle_revision(scope, memory_id, Lifecycle.EXPIRED, EventKind.EXPIRE, reason)

    def supersede(self, scope: ScopeContext, memory_id: str, content: str) -> str:
        return self.revise(scope, memory_id, content, event_kind=EventKind.SUPERSEDE)

    def _lifecycle_revision(self, scope: ScopeContext, memory_id: str, lifecycle: Lifecycle,
                            event_kind: EventKind, reason: str) -> str:
        obj=self._store.objects[memory_id]; old=self._store.revisions[memory_id][-1]
        self._canonical_for_mutation(memory_id)
        auth=self._authorize(scope,event_kind.value,old.visibility)
        if obj.tenant_id != scope.tenant_id: raise ScopeDenied("tenant mismatch")
        def work(s: InMemoryB9Store) -> str:
            now=time.time(); rid=_uuid7()
            rev=replace(old,revision_id=rid,lifecycle=lifecycle,created_at=now,prior_revision_id=old.revision_id)
            prov=MemoryProvenance(_uuid7(),rid,(old.revision_id,),event_kind.value.lower(),"1",
                                  scope.actor_principal,scope.actor_type,scope.subject_user_id,None,None,now,reason)
            event=self._event(scope,auth,memory_id,rid,event_kind,now,{"reason":reason})
            s._commit(obj,rev,prov,event); return rid
        return self._store.transaction(work)

    def envelope(self, scope: ScopeContext, memory_id: str, *, references: Iterable[MemoryReference]=()) -> MemoryEnvelope:
        scope.validate()
        obj=self._store.objects[memory_id]
        rev=self._store.revisions[memory_id][-1]
        self._assert_read_scope(scope, obj, rev)
        if rev.lifecycle in {Lifecycle.TOMBSTONED,Lifecycle.PURGED} or rev.payload is None: raise ScopeDenied("memory payload unavailable")
        return MemoryEnvelope(obj,rev,tuple(self._store.provenance.get(rev.revision_id,())),tuple(references),{})

    def retrieve(self, scope: ScopeContext, *, visibility: Visibility | None=None,
                 project_id: str | None=None) -> tuple[MemoryEnvelope, ...]:
        """Scope-first retrieval reference implementation.

        Ranking is deliberately outside this method: selection never mutates
        verification, lifecycle, or current-source classification.
        """
        scope.validate()
        if project_id is not None and project_id != scope.project_id:
            raise ScopeDenied("project scope mismatch")
        results=[]
        for memory_id, obj in self._store.objects.items():
            if obj.tenant_id != scope.tenant_id: continue
            rev=self._store.revisions[memory_id][-1]
            if visibility is not None and rev.visibility is not visibility: continue
            if rev.visibility is Visibility.USER_PRIVATE and rev.user_id != scope.subject_user_id: continue
            if rev.project_id is not None and (not scope.project_id or rev.project_id != scope.project_id): continue
            if rev.lifecycle in {Lifecycle.TOMBSTONED,Lifecycle.PURGED,Lifecycle.EXPIRED} or rev.payload is None: continue
            results.append(self.envelope(scope,memory_id))
        return tuple(results)

    def record_embedding(self, scope: ScopeContext, memory_id: str, identity: EmbeddingSpaceIdentity,
                         vector: Iterable[float] | None = None) -> str:
        """Record a new immutable generation without changing memory identity.

        The caller can re-embed a revision in a changed compatible space, but
        consumers must explicitly request that exact space to compare it.
        """
        self._canonical_for_mutation(memory_id)
        obj = self._store.objects[memory_id]
        revision = self._store.revisions[memory_id][-1]
        auth = self._authorize(scope, "RE_EMBED", revision.visibility)
        if obj.tenant_id != scope.tenant_id:
            raise ScopeDenied("tenant mismatch")
        space_id = identity.embedding_space_id
        generation = EmbeddingGeneration(_uuid7(), revision.revision_id, space_id, time.time(),
                                         tuple(vector) if vector is not None else None)
        def work(s: InMemoryB9Store) -> str:
            s.register_embedding_space(identity)
            s.add_embedding_generation(generation)
            # Re-embedding is operational history but does not create a
            # content revision or alter verification/lifecycle state.
            event = self._event(scope, auth, memory_id, revision.revision_id,
                                EventKind.RE_EMBED, time.time(), {"embedding_space_id": space_id,
                                                                    "generation_id": generation.generation_id})
            s.events.setdefault(memory_id, []).append(event)
            s.projections[memory_id] = s.rebuild_projection(memory_id)
            return generation.generation_id
        return self._store.transaction(work)

    def compensate(self, scope: ScopeContext, memory_id: str, compensates_event_id: str, *, reason: str) -> str:
        """Append an auditable compensation marker; it never rewrites history."""
        self._canonical_for_mutation(memory_id)
        obj = self._store.objects[memory_id]; revision = self._store.revisions[memory_id][-1]
        auth = self._authorize(scope, "COMPENSATE", revision.visibility)
        if obj.tenant_id != scope.tenant_id:
            raise ScopeDenied("tenant mismatch")
        if compensates_event_id not in {e.event_id for e in self._store.events[memory_id]}:
            raise B9Error("compensation target is not an event of this memory")
        def work(s: InMemoryB9Store) -> str:
            event = replace(self._event(scope, auth, memory_id, revision.revision_id,
                                        EventKind.COMPENSATE, time.time(), {"reason": reason}),
                            compensates_event_id=compensates_event_id)
            s.events[memory_id].append(event)
            s.projections[memory_id] = s.rebuild_projection(memory_id)
            return event.event_id
        return self._store.transaction(work)


class MemoryReferenceResolver:
    """Read-only typed resolver.  Memory text cannot manufacture current truth."""
    def __init__(self, resolvers: Mapping[str, Callable[[str], ResolutionResult]]): self._resolvers=dict(resolvers)
    def resolve(self, reference: MemoryReference) -> ResolutionResult:
        resolver=self._resolvers.get(reference.reference_type)
        if resolver is None: return ResolutionResult(ResolutionState.UNRESOLVED,"none",time.time())
        result=resolver(reference.reference_value)
        if not isinstance(result,ResolutionResult): raise CurrentTruthDenied("resolver must return ResolutionResult")
        return result
