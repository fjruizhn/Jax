"""F2-B accredited, side-effect-free current-source resolution contracts.

This module deliberately does not render, emit, or register itself at import
time.  A registry is an explicit server-owned object: possessing a resolver
callable is not accreditation to make a current-state assertion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Mapping

from .response import (
    ExistenceState, GovernanceContractError, ReferenceRef, ReferenceType,
    ResponseScope, TemporalClass, _freeze, _plain, _text,
)


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"
    CONFLICT = "CONFLICT"
    WRONG_SCOPE = "WRONG_SCOPE"
    WRONG_ENVIRONMENT = "WRONG_ENVIRONMENT"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    UNSUPPORTED = "UNSUPPORTED"
    UNACCREDITED = "UNACCREDITED"
    CONFIGURATION_MISMATCH = "CONFIGURATION_MISMATCH"
    VERSION_MISMATCH = "VERSION_MISMATCH"


class ReferenceValidationStatus(str, Enum):
    VALID = "VALID"
    DANGLING = "DANGLING"
    TOMBSTONED = "TOMBSTONED"
    WRONG_TYPE = "WRONG_TYPE"
    WRONG_SCOPE = "WRONG_SCOPE"
    WRONG_REVISION = "WRONG_REVISION"
    HISTORICAL = "HISTORICAL"
    ACCESS_DENIED = "ACCESS_DENIED"


def _digest(value: Any) -> str:
    raw = json.dumps(_plain(_freeze(value)), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _time(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise GovernanceContractError(f"{name} must be timezone-aware datetime")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ScopeRule:
    """Exact scope policy, including explicit ``None`` values.

    F2-B deliberately has no wildcard or widening relation.  A later block
    may add a separately accredited relation type, but an omitted value can
    never silently mean "any tenant/user/project".
    """
    environment: str
    tenant_id: str | None
    project_id: str | None
    subject_id: str | None
    actor_id: str | None
    audience: str | None
    component_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", _text(self.environment, "environment"))
        for name in ("tenant_id", "project_id", "subject_id", "actor_id", "audience", "component_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))

    def matches(self, scope: ResponseScope) -> ResolutionStatus | None:
        if scope.environment != self.environment:
            return ResolutionStatus.WRONG_ENVIRONMENT
        for name in ("tenant_id", "project_id", "subject_id", "actor_id", "audience", "component_id"):
            expected = getattr(self, name)
            if getattr(scope, name) != expected:
                return ResolutionStatus.WRONG_SCOPE
        return None

    def projection(self) -> dict[str, str | None]:
        return {name: getattr(self, name) for name in ("environment", "tenant_id", "project_id", "subject_id", "actor_id", "audience", "component_id")}


@dataclass(frozen=True)
class PredicateAuthorityBinding:
    predicate: str
    predicate_version: str
    designated_source_identity: str
    source_owner_authority_ref: str
    environment: str
    tenant_project_scope_rule: ScopeRule
    subject_audience_scope_rule: ScopeRule
    freshness_sla_seconds: int
    conflict_policy: str
    resolver_implementation_identity: str
    resolver_version: str
    source_configuration_digest: str | None
    binding_version: str
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("predicate", "predicate_version", "designated_source_identity", "source_owner_authority_ref", "environment", "conflict_policy", "resolver_implementation_identity", "resolver_version", "binding_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.tenant_project_scope_rule, ScopeRule) or not isinstance(self.subject_audience_scope_rule, ScopeRule):
            raise GovernanceContractError("binding scope rules must be ScopeRule")
        if self.environment != self.tenant_project_scope_rule.environment or self.environment != self.subject_audience_scope_rule.environment:
            raise GovernanceContractError("binding environment must match all scope rules")
        if not isinstance(self.freshness_sla_seconds, int) or self.freshness_sla_seconds <= 0:
            raise GovernanceContractError("freshness_sla_seconds must be positive int")
        if self.source_configuration_digest is not None:
            object.__setattr__(self, "source_configuration_digest", _text(self.source_configuration_digest, "source_configuration_digest"))

    def projection(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "predicate_version": self.predicate_version,
                "designated_source_identity": self.designated_source_identity,
                "source_owner_authority_ref": self.source_owner_authority_ref,
                "environment": self.environment,
                "tenant_project_scope_rule": self.tenant_project_scope_rule.projection(),
                "subject_audience_scope_rule": self.subject_audience_scope_rule.projection(),
                "freshness_sla_seconds": self.freshness_sla_seconds, "conflict_policy": self.conflict_policy,
                "resolver_implementation_identity": self.resolver_implementation_identity,
                "resolver_version": self.resolver_version, "source_configuration_digest": self.source_configuration_digest,
                "binding_version": self.binding_version, "enabled": self.enabled}

    @property
    def digest(self) -> str:
        return _digest(self.projection())


@dataclass(frozen=True)
class ResolutionObservation:
    """Untrusted resolver output; accreditation is performed by the registry."""
    actual_source_identity: str
    source_configuration_digest: str | None
    observation_scope: ResponseScope
    observed_at: datetime
    status: ResolutionStatus
    provenance_ref: str
    result: Mapping[str, Any] = field(default_factory=dict)
    resolver_id: str = ""
    resolver_version: str = ""

    def __post_init__(self) -> None:
        for name in ("actual_source_identity", "provenance_ref", "resolver_id", "resolver_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.source_configuration_digest is not None:
            object.__setattr__(self, "source_configuration_digest", _text(self.source_configuration_digest, "source_configuration_digest"))
        if not isinstance(self.observation_scope, ResponseScope): raise GovernanceContractError("observation_scope must be ResponseScope")
        object.__setattr__(self, "observed_at", _time(self.observed_at, "observed_at"))
        if not isinstance(self.status, ResolutionStatus): raise GovernanceContractError("resolution status must be ResolutionStatus")
        if not isinstance(self.result, Mapping): raise GovernanceContractError("result must be mapping")
        object.__setattr__(self, "result", _freeze(self.result, "result"))


_RECEIPT_TOKEN = object()
@dataclass(frozen=True, init=False)
class GovernedResolutionReceipt:
    receipt_id: str; predicate: str; arguments_digest: str; binding_version: str; registry_snapshot_digest: str
    actual_source_identity: str; source_configuration_digest: str | None; observation_scope_digest: str
    observed_at: datetime; not_after: datetime; status: ResolutionStatus; provenance_ref: str
    resolver_id: str; resolver_version: str; result_digest: str
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise GovernanceContractError("receipts are server-minted by ResolverRegistry")
    @classmethod
    def _mint(cls, token: object, **values: Any) -> "GovernedResolutionReceipt":
        if token is not _RECEIPT_TOKEN: raise GovernanceContractError("receipt requires server minting pathway")
        if not isinstance(values["status"], ResolutionStatus): raise GovernanceContractError("invalid receipt status")
        values["observed_at"] = _time(values["observed_at"], "observed_at")
        values["not_after"] = _time(values["not_after"], "not_after")
        if values["not_after"] < values["observed_at"]: raise GovernanceContractError("not_after precedes observed_at")
        for key in ("receipt_id", "predicate", "arguments_digest", "binding_version", "registry_snapshot_digest", "actual_source_identity", "observation_scope_digest", "provenance_ref", "resolver_id", "resolver_version", "result_digest"):
            values[key] = _text(values[key], key)
        if values["source_configuration_digest"] is not None:
            values["source_configuration_digest"] = _text(values["source_configuration_digest"], "source_configuration_digest")
        instance = object.__new__(cls)
        for key, value in values.items(): object.__setattr__(instance, key, value)
        return instance
    def is_current_at(self, now: datetime) -> bool:
        now = _time(now, "validation_time")
        return self.status is ResolutionStatus.RESOLVED and self.observed_at <= now <= self.not_after
    def projection(self) -> dict[str, Any]:
        return {key: (getattr(self, key).isoformat() if key in {"observed_at", "not_after"} else _plain(getattr(self, key))) for key in self.__annotations__}


Resolver = Callable[[Mapping[str, Any], ResponseScope], ResolutionObservation]

@dataclass(frozen=True)
class RegistryEntry:
    binding: PredicateAuthorityBinding
    resolver: Resolver | None
    argument_keys: tuple[str, ...]
    template_contract_ref: str | None = None
    def __post_init__(self) -> None:
        if self.resolver is not None and not callable(self.resolver): raise GovernanceContractError("resolver must be callable")
        if not isinstance(self.argument_keys, tuple) or not all(isinstance(x, str) and x for x in self.argument_keys): raise GovernanceContractError("argument_keys must be nonempty strings tuple")
        if len(set(self.argument_keys)) != len(self.argument_keys): raise GovernanceContractError("duplicate argument key")
        if self.template_contract_ref is not None: object.__setattr__(self, "template_contract_ref", _text(self.template_contract_ref, "template_contract_ref"))


class ResolverRegistry:
    """Fixed registry constructed from approved bindings; no dynamic registration."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise GovernanceContractError("registries require the server-owned approved construction pathway")
    @classmethod
    def _from_approved_entries(cls, token: object, entries: tuple[RegistryEntry, ...]) -> "ResolverRegistry":
        if token is not _REGISTRY_TOKEN: raise GovernanceContractError("registry entries require server approval")
        if not isinstance(entries, tuple) or not all(isinstance(x, RegistryEntry) for x in entries):
            raise GovernanceContractError("registry entries must be tuple[RegistryEntry, ...]")
        if len({x.binding.predicate for x in entries}) != len(entries): raise GovernanceContractError("duplicate registry predicate")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_entries", {x.binding.predicate: x for x in entries})
        object.__setattr__(instance, "_snapshot_digest", _digest({p: e.binding.projection() | {"argument_keys": list(e.argument_keys), "template_contract_ref": e.template_contract_ref} for p, e in sorted(instance._entries.items())}))
        return instance
    @property
    def snapshot_digest(self) -> str: return self._snapshot_digest
    def status_table(self) -> tuple[dict[str, Any], ...]:
        return tuple({"predicate": p, "enabled": e.binding.enabled and e.resolver is not None,
                      "source_owner": e.binding.source_owner_authority_ref,
                      "freshness_sla_seconds": e.binding.freshness_sla_seconds,
                      "human_decision_required": not (e.binding.enabled and e.resolver is not None)} for p, e in sorted(self._entries.items()))
    def resolve(self, predicate: str, arguments: Mapping[str, Any], scope: ResponseScope, *, validation_time: datetime) -> GovernedResolutionReceipt:
        predicate = _text(predicate, "predicate")
        if not isinstance(arguments, Mapping) or not isinstance(scope, ResponseScope): raise GovernanceContractError("arguments and scope must be typed")
        now = _time(validation_time, "validation_time")
        entry = self._entries.get(predicate)
        if entry is None: return self._failure(predicate, arguments, scope, now, ResolutionStatus.UNSUPPORTED)
        binding = entry.binding
        if not binding.enabled or entry.resolver is None: return self._failure(predicate, arguments, scope, now, ResolutionStatus.UNACCREDITED, binding)
        if set(arguments) != set(entry.argument_keys): return self._failure(predicate, arguments, scope, now, ResolutionStatus.UNAVAILABLE, binding)
        mismatch = binding.tenant_project_scope_rule.matches(scope) or binding.subject_audience_scope_rule.matches(scope)
        if mismatch: return self._failure(predicate, arguments, scope, now, mismatch, binding)
        try: observation = entry.resolver(_freeze(arguments, "arguments"), scope)
        except Exception:  # fail-soft: resolver failure is sealed as UNAVAILABLE, never a current observation.
            return self._failure(predicate, arguments, scope, now, ResolutionStatus.UNAVAILABLE, binding)
        if not isinstance(observation, ResolutionObservation): return self._failure(predicate, arguments, scope, now, ResolutionStatus.UNAVAILABLE, binding)
        status = observation.status
        # Existing resolver outcomes (notably SOURCE_CONFLICT) remain their
        # own deterministic failure rather than being masked as a mismatch.
        if status is ResolutionStatus.RESOLVED:
            if observation.actual_source_identity != binding.designated_source_identity: status = ResolutionStatus.SOURCE_MISMATCH
            elif observation.resolver_id != binding.resolver_implementation_identity or observation.resolver_version != binding.resolver_version: status = ResolutionStatus.VERSION_MISMATCH
            elif observation.observation_scope.scope_digest != scope.scope_digest: status = ResolutionStatus.WRONG_SCOPE if observation.observation_scope.environment == scope.environment else ResolutionStatus.WRONG_ENVIRONMENT
            elif binding.source_configuration_digest is not None and observation.source_configuration_digest != binding.source_configuration_digest: status = ResolutionStatus.CONFIGURATION_MISMATCH
            elif observation.observed_at > now or now - observation.observed_at > timedelta(seconds=binding.freshness_sla_seconds): status = ResolutionStatus.STALE
        not_after = min(observation.observed_at + timedelta(seconds=binding.freshness_sla_seconds), now + timedelta(seconds=binding.freshness_sla_seconds))
        return self._mint(predicate, arguments, scope, binding, observation, status, not_after)
    def _failure(self, predicate: str, args: Mapping[str, Any], scope: ResponseScope, now: datetime, status: ResolutionStatus, binding: PredicateAuthorityBinding | None = None) -> GovernedResolutionReceipt:
        b = binding
        return self._mint(predicate, args, scope, b, ResolutionObservation(b.designated_source_identity if b else "unaccredited", b.source_configuration_digest if b else None, scope, now, status, "none", {}, b.resolver_implementation_identity if b else "none", b.resolver_version if b else "none"), status, now)
    def _mint(self, predicate: str, arguments: Mapping[str, Any], scope: ResponseScope, binding: PredicateAuthorityBinding | None, observation: ResolutionObservation, status: ResolutionStatus, not_after: datetime) -> GovernedResolutionReceipt:
        bid = binding.binding_version if binding else "none"
        seed = {"predicate": predicate, "arguments": _plain(_freeze(arguments)), "binding": bid, "registry": self.snapshot_digest, "source": observation.actual_source_identity, "scope": scope.scope_digest, "observed_at": observation.observed_at.isoformat(), "status": status.value, "result": _plain(observation.result)}
        return GovernedResolutionReceipt._mint(_RECEIPT_TOKEN, receipt_id="receipt:" + _digest(seed)[7:], predicate=predicate,
            arguments_digest=_digest(arguments), binding_version=bid, registry_snapshot_digest=self.snapshot_digest,
            actual_source_identity=observation.actual_source_identity, source_configuration_digest=observation.source_configuration_digest,
            observation_scope_digest=scope.scope_digest, observed_at=observation.observed_at, not_after=not_after, status=status,
            provenance_ref=observation.provenance_ref, resolver_id=observation.resolver_id, resolver_version=observation.resolver_version,
            result_digest=_digest(observation.result))


_REGISTRY_TOKEN = object()
def _build_approved_registry_for_server(entries: tuple[RegistryEntry, ...]) -> ResolverRegistry:
    """Module-private, server-owned registry construction pathway.

    Runtime resolvers, clients, tools, and plugins have no public registry
    mutation API. Deployment configuration is responsible for invoking this
    boundary with reviewed bindings.
    """
    return ResolverRegistry._from_approved_entries(_REGISTRY_TOKEN, entries)


@dataclass(frozen=True)
class ReferenceLookupRecord:
    """Injected immutable lookup/access result for a reference dereference."""
    ref_type: ReferenceType; immutable_identity: str; revision_or_digest: str
    scope_digest: str; temporal_class: TemporalClass; existence_state: ExistenceState
    accessible: bool
    def __post_init__(self) -> None:
        if not isinstance(self.ref_type, ReferenceType) or not isinstance(self.temporal_class, TemporalClass) or not isinstance(self.existence_state, ExistenceState):
            raise GovernanceContractError("lookup record must use typed governance enums")
        for name in ("immutable_identity", "revision_or_digest", "scope_digest"):
            object.__setattr__(self, name, _text(getattr(self, name), name))


def validate_reference(ref: ReferenceRef, scope: ResponseScope, *, lookup: ReferenceLookupRecord | None, expected_type: ReferenceType | None = None, expected_revision: str | None = None) -> ReferenceValidationStatus:
    """F2-B dereference guard using injected deterministic access metadata.

    It validates an actual immutable lookup record; it never upgrades a B7,
    B8, or B9 historical reference to current semantics.
    """
    if not isinstance(ref, ReferenceRef) or not isinstance(scope, ResponseScope): raise GovernanceContractError("typed reference and scope required")
    if lookup is None: return ReferenceValidationStatus.DANGLING
    if lookup.existence_state is ExistenceState.TOMBSTONED: return ReferenceValidationStatus.TOMBSTONED
    if lookup.existence_state is not ExistenceState.PRESENT: return ReferenceValidationStatus.DANGLING
    if not lookup.accessible: return ReferenceValidationStatus.ACCESS_DENIED
    if expected_type is not None and ref.ref_type is not expected_type: return ReferenceValidationStatus.WRONG_TYPE
    if ref.ref_type is not lookup.ref_type: return ReferenceValidationStatus.WRONG_TYPE
    if expected_revision is not None and ref.revision_or_digest != expected_revision: return ReferenceValidationStatus.WRONG_REVISION
    if ref.immutable_identity != lookup.immutable_identity or ref.revision_or_digest != lookup.revision_or_digest: return ReferenceValidationStatus.WRONG_REVISION
    if ref.scope_digest != scope.scope_digest or lookup.scope_digest != scope.scope_digest: return ReferenceValidationStatus.WRONG_SCOPE
    if ref.temporal_class is not lookup.temporal_class: return ReferenceValidationStatus.HISTORICAL
    if ref.temporal_class is TemporalClass.HISTORICAL: return ReferenceValidationStatus.HISTORICAL
    return ReferenceValidationStatus.VALID


# Existing resolver adapters.  They are intentionally dependency-injected and
# do no I/O on import.  A caller still needs an accredited binding to place
# any of these adapters in a registry.
class CapabilityAvailableAdapter:
    resolver_id = "policy.governance.validator:CAPABILITY_AVAILABLE"
    resolver_version = "existing-v1"
    def __init__(self, validation_context: Any, *, ops_source_identity: str, catalog_source_identity: str,
                 ops_configuration_digest: str, catalog_configuration_digest: str,
                 verdict_resolver: Callable[[Mapping[str, Any], Any], str] | None = None,
                 clock: Callable[[], datetime] | None = None):
        self._context = validation_context
        self._ops_source = _text(ops_source_identity, "ops_source_identity")
        self._catalog_source = _text(catalog_source_identity, "catalog_source_identity")
        self._ops_config = _text(ops_configuration_digest, "ops_configuration_digest")
        self._catalog_config = _text(catalog_configuration_digest, "catalog_configuration_digest")
        self._verdict = verdict_resolver
        self._clock = clock or (lambda: datetime.now(timezone.utc))
    def __call__(self, arguments: Mapping[str, Any], scope: ResponseScope) -> ResolutionObservation:
        # The deployed adapter injects the existing validator callable.  Core
        # import remains pure and does not import its legacy absolute modules.
        verdict_status = self._verdict(arguments, self._context) if self._verdict else "UNAVAILABLE"
        name = arguments.get("name")
        in_ops = name in self._context.ops
        in_catalog = self._context.catalog.get_capability(name) is not None
        if in_ops and in_catalog:
            source, config, status = "capability:conflict", None, ResolutionStatus.CONFLICT
        elif in_ops:
            source, config, status = self._ops_source, self._ops_config, ResolutionStatus.RESOLVED if verdict_status == "VALID" else ResolutionStatus.UNAVAILABLE
        elif in_catalog:
            source, config, status = self._catalog_source, self._catalog_config, ResolutionStatus.RESOLVED if verdict_status == "VALID" else ResolutionStatus.UNAVAILABLE
        else:
            source, config, status = "capability:unavailable", None, ResolutionStatus.UNAVAILABLE
        return ResolutionObservation(source, config, scope, self._clock(), status, "governance:capability-validator", dict(arguments), self.resolver_id, self.resolver_version)


@dataclass(frozen=True)
class _LegacyClaimInput:
    """Minimal legacy validator input; no model/client claim is trusted here."""
    predicate: str
    args: Mapping[str, Any]


def capability_adapter_from_legacy(validation_context: Any, legacy_resolver: Callable[[Any, Any], Any], *,
                                   ops_source_identity: str, catalog_source_identity: str,
                                   ops_configuration_digest: str, catalog_configuration_digest: str,
                                   clock: Callable[[], datetime] | None = None) -> CapabilityAvailableAdapter:
    """Wrap the existing ``_resolve_capability_available`` callable safely.

    The legacy module is supplied by the composition root, so importing the
    F2-B core has no legacy import, filesystem, database, or network effect.
    """
    if not callable(legacy_resolver): raise GovernanceContractError("legacy capability resolver must be callable")
    def verdict(arguments: Mapping[str, Any], context: Any) -> str:
        result = legacy_resolver(_LegacyClaimInput("CAPABILITY_AVAILABLE", dict(arguments)), context)
        return getattr(result, "status", "UNAVAILABLE")
    return CapabilityAvailableAdapter(validation_context, ops_source_identity=ops_source_identity,
        catalog_source_identity=catalog_source_identity, ops_configuration_digest=ops_configuration_digest,
        catalog_configuration_digest=catalog_configuration_digest, verdict_resolver=verdict, clock=clock)


class FileExistsAdapter:
    resolver_id = "policy.governance.validator:FILE_EXISTS"
    resolver_version = "existing-v1"
    def __init__(self, validation_context: Any, *, source_identity: str, configuration_digest: str,
                 verdict_resolver: Callable[[Mapping[str, Any], Any], str] | None = None,
                 clock: Callable[[], datetime] | None = None):
        self._context = validation_context; self._source = _text(source_identity, "source_identity"); self._config = _text(configuration_digest, "configuration_digest"); self._verdict=verdict_resolver; self._clock=clock or (lambda: datetime.now(timezone.utc))
    def __call__(self, arguments: Mapping[str, Any], scope: ResponseScope) -> ResolutionObservation:
        verdict_status = self._verdict(arguments, self._context) if self._verdict else "UNAVAILABLE"
        status = ResolutionStatus.RESOLVED if verdict_status == "VALID" else ResolutionStatus.UNAVAILABLE
        return ResolutionObservation(self._source, self._config, scope, self._clock(), status, "governance:file-validator", dict(arguments), self.resolver_id, self.resolver_version)


def file_exists_adapter_from_legacy(validation_context: Any, legacy_resolver: Callable[[Any, Any], Any], *,
                                    source_identity: str, configuration_digest: str,
                                    clock: Callable[[], datetime] | None = None) -> FileExistsAdapter:
    """Wrap the existing ``_resolve_file_exists`` callable at composition time."""
    if not callable(legacy_resolver): raise GovernanceContractError("legacy file resolver must be callable")
    def verdict(arguments: Mapping[str, Any], context: Any) -> str:
        result = legacy_resolver(_LegacyClaimInput("FILE_EXISTS", dict(arguments)), context)
        return getattr(result, "status", "UNAVAILABLE")
    return FileExistsAdapter(validation_context, source_identity=source_identity,
        configuration_digest=configuration_digest, verdict_resolver=verdict, clock=clock)


class B9DesignatedSourceAdapter:
    resolver_id = "jax.memory.b9_resolvers:DesignatedSourceResolver"
    resolver_version = "existing-v1"
    def __init__(self, resolver: Any, *, source_identity: str, clock: Callable[[], datetime] | None = None):
        self._resolver = resolver; self._source = _text(source_identity, "source_identity"); self._clock=clock or (lambda: datetime.now(timezone.utc))
    def __call__(self, arguments: Mapping[str, Any], scope: ResponseScope) -> ResolutionObservation:
        from jax.memory.b9 import MemoryReference, ResolutionState
        kind, value = arguments.get("reference_type"), arguments.get("reference_value")
        if not isinstance(kind, str) or not isinstance(value, str):
            return ResolutionObservation(self._source, None, scope, self._clock(), ResolutionStatus.UNAVAILABLE, "b9:none", {}, self.resolver_id, self.resolver_version)
        result = self._resolver.resolve(MemoryReference(kind, value))
        status = ResolutionStatus.RESOLVED if result.state is ResolutionState.RESOLVED_CURRENT else ResolutionStatus.UNAVAILABLE
        # The underlying adapter refuses source relabelling; it is still bound
        # again by ResolverRegistry before a receipt can be current.
        return ResolutionObservation(self._source, None, scope, self._clock(), status, "b9:" + result.source, {"state": result.state.value}, self.resolver_id, self.resolver_version)
