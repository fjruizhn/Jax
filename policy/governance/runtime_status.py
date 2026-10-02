"""F2-E closed runtime-status resolver composition.

No request selects a source, callable, HTTP endpoint, SQL statement, or store.
Platform may only pass its fixed typed snapshot to the two platform-owned kinds.

Freshness is measured from the source transition/probe timestamp. A job or
pipeline that remains unchanged longer than its 60-second SLA is stale even
when its stored status remains readable; terminal states receive no exception.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Mapping

from .response import GovernanceContractError, ResponseScope, _freeze, _plain, _text
from .resolution import (AdapterKind, ResolutionObservation, ResolutionStatus,
                         RuntimeStatusEvidence, _runtime_status_evidence_from_server,
                         PredicateAuthorityBinding, ScopeRule, ConflictPolicy,
                         TrustedAdapterRegistration, RegistryEntry,
                         _build_approved_registry_for_server, SourceScopeClass)

_PLATFORM_KINDS = {
    "FACET_RUNTIME_STATUS": (AdapterKind.FACET_RUNTIME_STATUS, "platform:facet-state"),
    "ENGINE_STATUS": (AdapterKind.ENGINE_STATUS, "platform:las-manos-health"),
}
RUNTIME_STATUS_API_VERSION = "f2-e.runtime-status.1"
_BINDING_VERSION = "f2-e.runtime-status.1"
_RUNTIME_SPECS = {
    "JOB_STATUS": (AdapterKind.MOTOR_JOB_STATUS, "motor:job-store", "authority:motor-registry", 60, SourceScopeClass.EXACT_RESPONSE_SCOPE, "MotorJobStatusResolver"),
    "PIPELINE_STATUS": (AdapterKind.JACOBS_PIPELINE_STATUS, "jacobs:canonical-store", "authority:jacobs", 60, SourceScopeClass.EXACT_RESPONSE_SCOPE, "JacobsPipelineStatusResolver"),
    "FACET_RUNTIME_STATUS": (AdapterKind.FACET_RUNTIME_STATUS, "platform:facet-state", "authority:jax-platform", 15, SourceScopeClass.INSTALLATION_GLOBAL, "PlatformFacetRuntimeStatusResolver"),
    "ENGINE_STATUS": (AdapterKind.ENGINE_STATUS, "platform:las-manos-health", "authority:jax-platform", 60, SourceScopeClass.INSTALLATION_GLOBAL, "PlatformEngineStatusResolver"),
}


def _source_configuration_digest(predicate: str, source: str, resolver: str,
                                 source_scope: SourceScopeClass, sla: int) -> str:
    """Hash the closed server composition that defines a status source."""
    body = json.dumps({
        "api_version": RUNTIME_STATUS_API_VERSION,
        "predicate": predicate,
        "source": source,
        "resolver": resolver,
        "source_scope_class": source_scope.value,
        "freshness_sla_seconds": sla,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()

def build_runtime_status_registry(scope: ResponseScope, *, authenticator):
    """Build exactly the four human-authorized F2-E entries from fixed constants."""
    if not isinstance(scope, ResponseScope):
        raise GovernanceContractError("runtime registry requires ResponseScope")
    rule = ScopeRule(scope.environment, scope.tenant_id, scope.project_id, scope.subject_id,
        scope.actor_id, scope.audience, scope.component_id)
    entries = []
    for predicate, (kind, source, owner, sla, source_scope, resolver_name) in _RUNTIME_SPECS.items():
        identity = f"policy.governance.runtime_status:{resolver_name}"
        config_digest = _source_configuration_digest(predicate, source, resolver_name, source_scope, sla)
        binding = PredicateAuthorityBinding(predicate, _BINDING_VERSION, source, owner,
            scope.environment, rule, rule, sla, ConflictPolicy.SINGLE_SOURCE_REQUIRED,
            identity, _BINDING_VERSION, config_digest, _BINDING_VERSION,
            source_scope_class=source_scope)
        adapter = TrustedAdapterRegistration(kind, identity, _BINDING_VERSION, source,
            config_digest)
        keys = ("job_id", "status") if predicate == "JOB_STATUS" else (("pipeline_id", "status") if predicate == "PIPELINE_STATUS" else ("name", "status"))
        entries.append(RegistryEntry(binding, adapter, keys, f"{predicate}@{_BINDING_VERSION}:es"))
    return _build_approved_registry_for_server(tuple(entries), authenticator=authenticator)

@dataclass(frozen=True)
class PlatformRuntimeStatusSnapshot:
    """Fixed server-produced observation; this is not a generic resolver input."""
    predicate: str
    arguments: Mapping[str, object]
    observed_at: datetime
    provenance_ref: str
    def __post_init__(self):
        predicate = _text(self.predicate, "predicate")
        if predicate not in _PLATFORM_KINDS:
            raise GovernanceContractError("platform runtime predicate unsupported")
        if not isinstance(self.arguments, Mapping):
            raise GovernanceContractError("platform runtime arguments must be mapping")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "arguments", _freeze(self.arguments, "platform runtime arguments"))
        _text(self.provenance_ref, "platform runtime provenance_ref")

def platform_runtime_status_evidence(snapshot: PlatformRuntimeStatusSnapshot, arguments: Mapping[str, object], scope: ResponseScope) -> RuntimeStatusEvidence:
    """Narrow platform bridge. Predicate/source/kind are constants above."""
    if not isinstance(snapshot, PlatformRuntimeStatusSnapshot) or not isinstance(scope, ResponseScope):
        raise GovernanceContractError("typed platform snapshot and scope required")
    if not isinstance(arguments, Mapping) or _plain(snapshot.arguments) != _plain(arguments):
        raise GovernanceContractError("platform observation must equal canonical predicate arguments")
    kind, source = _PLATFORM_KINDS[snapshot.predicate]
    observation = ResolutionObservation(ResolutionStatus.RESOLVED, snapshot.observed_at,
        f"{source}:{snapshot.provenance_ref}", snapshot.arguments)
    return _runtime_status_evidence_from_server(kind, observation, scope)

class MotorJobStatusResolver:
    """Canonical Motor JSONL reader. Ownerless legacy jobs fail closed."""
    def __init__(self):
        # Select the singleton configured by LAS MANOS server composition.
        # No API lets a request, model, or arbitrary caller choose a store/path.
        from motor_registry import routes
        from motor_registry.job_store import JobStore
        if type(routes._STORE) is not JobStore:
            raise GovernanceContractError("canonical Motor Registry JobStore unavailable")
        self._store = routes._STORE
    def evidence(self, arguments: Mapping[str, object], scope: ResponseScope) -> RuntimeStatusEvidence:
        if not isinstance(scope, ResponseScope) or scope.project_id is not None:
            raise GovernanceContractError("JOB_STATUS project scope unsupported")
        if not isinstance(arguments, Mapping) or set(arguments) != {"job_id", "status"}:
            raise GovernanceContractError("JOB_STATUS arguments invalid")
        job_id, status = arguments["job_id"], arguments["status"]
        if not isinstance(job_id, str) or not isinstance(status, str):
            raise GovernanceContractError("JOB_STATUS arguments invalid")
        try:
            view = self._store.get(job_id)
        except Exception:  # fail-soft: una lectura fallida queda UNAVAILABLE, nunca se acredita.
            view = None
        observed_at = _job_transition_time(view)
        if view is None or observed_at is None or view.tenant_id is None or view.user_id is None:
            obs = ResolutionObservation(ResolutionStatus.UNAVAILABLE, datetime.now(timezone.utc), "motor-job:unavailable", {})
        elif (view.tenant_id, view.user_id) != (scope.tenant_id, scope.subject_id):
            obs = ResolutionObservation(ResolutionStatus.WRONG_SCOPE, observed_at, f"motor-job:{job_id}", {})
        else:
            obs = ResolutionObservation(ResolutionStatus.RESOLVED, observed_at, f"motor-job:{job_id}",
                {"job_id": job_id, "status": view.status.value})
        return _runtime_status_evidence_from_server(AdapterKind.MOTOR_JOB_STATUS, obs, scope)

class JacobsPipelineStatusResolver:
    """Canonical Jacobs store only; projections and caller stores are excluded."""
    async def evidence(self, arguments: Mapping[str, object], scope: ResponseScope) -> RuntimeStatusEvidence:
        if not isinstance(scope, ResponseScope) or scope.project_id is not None:
            raise GovernanceContractError("PIPELINE_STATUS project scope unsupported")
        if not isinstance(arguments, Mapping) or set(arguments) != {"pipeline_id", "status"}:
            raise GovernanceContractError("PIPELINE_STATUS arguments invalid")
        pipeline_id, status = arguments["pipeline_id"], arguments["status"]
        if not isinstance(pipeline_id, str) or not isinstance(status, str):
            raise GovernanceContractError("PIPELINE_STATUS arguments invalid")
        from jacobs import store as jacobs_store
        try:
            pipeline = await jacobs_store.pipeline_get(pipeline_id)
        except Exception:  # fail-soft: Jacobs inaccesible queda UNAVAILABLE, nunca se acredita.
            pipeline = None
        observed_at = _timestamp(getattr(pipeline, "updated_at", None))
        if pipeline is None or observed_at is None or pipeline.tenant_id is None or pipeline.user_id is None:
            obs = ResolutionObservation(ResolutionStatus.UNAVAILABLE, datetime.now(timezone.utc), "jacobs-pipeline:unavailable", {})
        elif (pipeline.tenant_id, pipeline.user_id) != (scope.tenant_id, scope.subject_id):
            obs = ResolutionObservation(ResolutionStatus.WRONG_SCOPE, observed_at, f"jacobs-pipeline:{pipeline_id}", {})
        else:
            status_value = getattr(getattr(pipeline, "status", None), "value", None)
            if not isinstance(status_value, str):
                obs = ResolutionObservation(ResolutionStatus.UNAVAILABLE, datetime.now(timezone.utc), "jacobs-pipeline:unavailable", {})
            else:
                obs = ResolutionObservation(ResolutionStatus.RESOLVED, observed_at,
                    f"jacobs-pipeline:{pipeline_id}", {"pipeline_id": pipeline_id, "status": status_value})
        return _runtime_status_evidence_from_server(AdapterKind.JACOBS_PIPELINE_STATUS, obs, scope)

def _timestamp(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return datetime.fromtimestamp(value, timezone.utc)
    return None

def _job_transition_time(view):
    if view is None:
        return None
    if view.status.value == "pending":
        return _timestamp(view.created_at)
    if view.status.value in {"completed", "failed", "cancelled", "rejected"}:
        return _timestamp(view.finished_at)
    if view.status.value in {"cancelling", "tools_requested"}:
        return _timestamp(view.status_updated_at)
    return _timestamp(view.started_at)
