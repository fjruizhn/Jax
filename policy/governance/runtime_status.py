"""F2-E closed runtime-status resolver composition.

No request selects a source, callable, HTTP endpoint, SQL statement, or store.
Platform may only pass its fixed typed snapshot to the two platform-owned kinds.

Freshness records when the designated authority was observed, separately from
when its state changed. A trusted JobStore read, canonical Jacobs persistence
read, or server-owned FacetState read may produce a fresh observation of an
unchanged state without rewriting its transition/change timestamp. ENGINE_STATUS
instead uses the completed health-probe timestamp because its resolver consumes
that independent observation rather than performing the probe itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import sys
from typing import Mapping

import aiomysql

from .response import GovernanceContractError, ResponseScope, _freeze, _plain, _text
from .resolution import (AdapterKind, ResolutionObservation, ResolutionStatus,
                         RuntimeStatusEvidence, _runtime_status_evidence_from_server,
                         PredicateAuthorityBinding, ScopeRule, ConflictPolicy,
                         TrustedAdapterRegistration, RegistryEntry,
                         _build_approved_registry_for_server, SourceScopeClass)

logger = logging.getLogger(__name__)

_PLATFORM_KINDS = {
    "FACET_RUNTIME_STATUS": (AdapterKind.FACET_RUNTIME_STATUS, "platform:facet-state"),
    "ENGINE_STATUS": (AdapterKind.ENGINE_STATUS, "platform:las-manos-health"),
}
RUNTIME_STATUS_API_VERSION = "f2-e.runtime-status.4"
_BINDING_VERSION = "f2-e.runtime-status.4"
_RESOLVER_VERSION = "f2-e.runtime-status-resolver.4"
_RUNTIME_SPECS = {
    "JOB_STATUS": (AdapterKind.MOTOR_JOB_STATUS, "motor:job-store", "authority:motor-registry", 60, SourceScopeClass.EXACT_RESPONSE_SCOPE, "MotorJobStatusResolver"),
    "PIPELINE_STATUS": (AdapterKind.JACOBS_PIPELINE_STATUS, "jacobs:canonical-store", "authority:jacobs", 60, SourceScopeClass.EXACT_RESPONSE_SCOPE, "JacobsPipelineStatusResolver"),
    "STEP_STATUS": (AdapterKind.JACOBS_STEP_STATUS, "jacobs:canonical-step-store", "authority:jacobs", 60, SourceScopeClass.EXACT_RESPONSE_SCOPE, "JacobsStepStatusResolver"),
    "FACET_RUNTIME_STATUS": (AdapterKind.FACET_RUNTIME_STATUS, "platform:facet-state", "authority:jax-platform", 15, SourceScopeClass.INSTALLATION_GLOBAL, "PlatformFacetRuntimeStatusResolver"),
    "ENGINE_STATUS": (AdapterKind.ENGINE_STATUS, "platform:las-manos-health", "authority:jax-platform", 60, SourceScopeClass.INSTALLATION_GLOBAL, "PlatformEngineStatusResolver"),
    "PROCESSING_JOB_STATUS": (AdapterKind.LAS_MANOS_PROCESSING_JOB_STATUS, "las-manos:processing-job-store", "authority:las-manos", 60, SourceScopeClass.EXACT_RESPONSE_SCOPE, "ProcessingJobStatusResolver"),
}


def _source_configuration_digest(predicate: str, source: str, resolver: str,
                                 source_scope: SourceScopeClass, sla: int,
                                 source_configuration: Mapping[str, object] | None = None) -> str:
    """Hash the closed server composition that defines a status source."""
    body = json.dumps({
        "api_version": RUNTIME_STATUS_API_VERSION,
        "predicate": predicate,
        "source": source,
        "resolver": resolver,
        "source_scope_class": source_scope.value,
        "freshness_sla_seconds": sla,
        "source_configuration": _plain(source_configuration or {}),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _jacobs_source_configuration(*, require_config: bool = False) -> dict[str, object]:
    """Read only non-secret identity from the fixed Jacobs DB composition."""
    from jacobs.store import _db_cfg
    try:
        config = _db_cfg()
    except Exception:
        if require_config:
            raise
        # Keep unrelated predicates registrable in tests/minimal composition;
        # this sentinel can never match a usable DB source observation.
        config = {"host": "unconfigured", "port": 1, "db": "unconfigured"}
    return {"database_engine": "mariadb", "host": config["host"],
        "port": config["port"], "database": config["db"],
        "store_contract": "jacobs-pipeline-store-v1", "table": "jacobs_pipelines"}


def _jacobs_step_source_configuration(*, require_config: bool = False) -> dict[str, object]:
    """Bind STEP_STATUS to the joined step+owner source contract."""
    from jacobs.store import _db_cfg
    try:
        config = _db_cfg()
    except Exception:
        if require_config:
            raise
        config = {"host": "unconfigured", "port": 1, "db": "unconfigured"}
    from jacobs.models import StepStatus
    return {
        "database_engine": "mariadb", "host": config["host"], "port": config["port"],
        "database": config["db"], "store_contract": "jacobs-step-owner-snapshot-v1",
        "step_table": "jacobs_steps", "pipeline_table": "jacobs_pipelines",
        "owner_contract": "pipeline-tenant-user-owner-ack-v1",
        "visibility_contract": "jacobs-pipeline-visible-v1",
        "allowed_statuses": sorted(status.value for status in StepStatus),
    }


def _job_store_source_configuration() -> dict[str, str]:
    """Use the loaded server singleton, or its fixed source identity without importing routes."""
    routes = sys.modules.get("motor_registry.routes")
    store = getattr(routes, "_STORE", None) if routes is not None else None
    source_config = getattr(store, "source_configuration", None)
    if callable(source_config):
        return source_config()
    # The platform bridge imports this module without loading LAS MANOS routes.
    # Reconstruct only the constant server composition path; never import a
    # caller-selected store or the route module (which imports Motor workers).
    from pathlib import Path
    repository = Path(__file__).resolve().parents[2]
    return {
        "store_contract": "motor-job-store-v1",
        "source_id": str((repository / "las_manos" / "logs" / "motor_jobs.jsonl").resolve()),
        "event_format": "motor-job-event-v1",
        "durability": "append-flush-fsync-v1",
    }


def _processing_job_store_source_configuration() -> dict[str, object]:
    """Read only the fixed LAS MANOS Processing source identity."""
    routes = sys.modules.get("procesamiento_routes")
    store = getattr(routes, "_STORE", None) if routes is not None else None
    source_config = getattr(store, "source_configuration", None)
    if callable(source_config):
        return source_config()
    from pathlib import Path
    repository = Path(__file__).resolve().parents[2]
    return {
        "store_contract": "processing-job-store-v1",
        "source_id": str((repository / "las_manos" / "logs" / "procesamiento_jobs.jsonl").resolve()),
        "event_format": "processing-job-event-v1",
        "durability": "append-flush-fsync-v1",
        "ownership_contract": "platform-authenticated-processing-owner.1",
        "source_role": "las-manos-processing-jobs",
        "allowed_statuses": ["pending", "running", "cancelling", "completed", "failed", "cancelled"],
    }

def runtime_status_source_configuration_digest(predicate: str, source_configuration: Mapping[str, object]) -> str:
    """Digest a closed, server-owned non-secret runtime source identity."""
    if predicate not in _RUNTIME_SPECS or not isinstance(source_configuration, Mapping):
        raise GovernanceContractError("runtime source configuration invalid")
    if predicate == "FACET_RUNTIME_STATUS":
        expected = {"state_contract": "JAXEngineState.FacetState", "status_field": "status",
            "observed_at_field": "resolver_read_time", "allowed_statuses": ["idle", "thinking", "error", "offline"]}
        if _plain(source_configuration) != expected:
            raise GovernanceContractError("facet runtime source configuration mismatch")
    elif predicate == "ENGINE_STATUS":
        config = _plain(source_configuration)
        if set(config) != {"endpoint_sha256", "method", "path", "timeout_seconds", "poll_interval_seconds", "success_status_code"}:
            raise GovernanceContractError("engine health source configuration shape mismatch")
        endpoint_digest = config["endpoint_sha256"]
        if (not isinstance(endpoint_digest, str) or len(endpoint_digest) != 71
                or not endpoint_digest.startswith("sha256:")
                or any(ch not in "0123456789abcdef" for ch in endpoint_digest[7:])
                or config["method"] != "GET" or config["path"] != "/health"
                or config["timeout_seconds"] != 5 or config["poll_interval_seconds"] != 30
                or config["success_status_code"] != 200):
            raise GovernanceContractError("engine health source configuration mismatch")
    elif predicate == "JOB_STATUS":
        config = _plain(source_configuration)
        if (set(config) != {"store_contract", "source_id", "event_format", "durability"}
                or config["store_contract"] != "motor-job-store-v1"
                or not isinstance(config["source_id"], str) or not config["source_id"]
                or config["event_format"] != "motor-job-event-v1"
                or config["durability"] != "append-flush-fsync-v1"):
            raise GovernanceContractError("Motor JobStore source configuration mismatch")
    elif predicate == "PROCESSING_JOB_STATUS":
        config = _plain(source_configuration)
        if (set(config) != {"store_contract", "source_id", "event_format", "durability", "ownership_contract", "source_role", "allowed_statuses"}
                or config["store_contract"] != "processing-job-store-v1"
                or not isinstance(config["source_id"], str) or not config["source_id"]
                or config["event_format"] != "processing-job-event-v1"
                or config["durability"] != "append-flush-fsync-v1"
                or config["ownership_contract"] != "platform-authenticated-processing-owner.1"
                or config["source_role"] != "las-manos-processing-jobs"
                or config["allowed_statuses"] != ["pending", "running", "cancelling", "completed", "failed", "cancelled"]):
            raise GovernanceContractError("Processing JobStore source configuration mismatch")
    elif predicate == "STEP_STATUS":
        from jacobs.models import StepStatus
        config = _plain(source_configuration)
        if (set(config) != {"database_engine", "host", "port", "database", "store_contract",
                            "step_table", "pipeline_table", "owner_contract", "visibility_contract",
                            "allowed_statuses"}
                or config["database_engine"] != "mariadb"
                or not isinstance(config["host"], str) or not config["host"]
                or not isinstance(config["port"], int) or isinstance(config["port"], bool)
                or not 1 <= config["port"] <= 65535
                or not isinstance(config["database"], str) or not config["database"]
                or config["store_contract"] != "jacobs-step-owner-snapshot-v1"
                or config["step_table"] != "jacobs_steps"
                or config["pipeline_table"] != "jacobs_pipelines"
                or config["owner_contract"] != "pipeline-tenant-user-owner-ack-v1"
                or config["visibility_contract"] != "jacobs-pipeline-visible-v1"
                or config["allowed_statuses"] != sorted(status.value for status in StepStatus)):
            raise GovernanceContractError("Jacobs STEP_STATUS source configuration mismatch")
    else:
        config = _plain(source_configuration)
        if (set(config) != {"database_engine", "host", "port", "database", "store_contract", "table"}
                or config["database_engine"] != "mariadb"
                or not isinstance(config["host"], str) or not config["host"]
                or not isinstance(config["port"], int) or isinstance(config["port"], bool)
                or not 1 <= config["port"] <= 65535
                or not isinstance(config["database"], str) or not config["database"]
                or config["store_contract"] != "jacobs-pipeline-store-v1"
                or config["table"] != "jacobs_pipelines"):
            raise GovernanceContractError("Jacobs persistence source configuration mismatch")
    _, source, _, sla, source_scope, resolver_name = _RUNTIME_SPECS[predicate]
    return _source_configuration_digest(predicate, source, resolver_name, source_scope, sla, source_configuration)

def build_runtime_status_registry(scope: ResponseScope, *, authenticator,
                                  platform_source_configuration: Mapping[str, Mapping[str, object]]):
    """Build exactly the six human-authorized F2-E status entries from fixed constants."""
    if not isinstance(scope, ResponseScope):
        raise GovernanceContractError("runtime registry requires ResponseScope")
    if not isinstance(platform_source_configuration, Mapping) or set(platform_source_configuration) != {"FACET_RUNTIME_STATUS", "ENGINE_STATUS"}:
        raise GovernanceContractError("Platform runtime source configuration required")
    job_config = _job_store_source_configuration()
    processing_job_config = _processing_job_store_source_configuration()
    jacobs_config = _jacobs_source_configuration()
    jacobs_step_config = _jacobs_step_source_configuration()
    rule = ScopeRule(scope.environment, scope.tenant_id, scope.project_id, scope.subject_id,
        scope.actor_id, scope.audience, scope.component_id)
    entries = []
    for predicate, (kind, source, owner, sla, source_scope, resolver_name) in _RUNTIME_SPECS.items():
        identity = f"policy.governance.runtime_status:{resolver_name}"
        source_config = (platform_source_configuration[predicate] if predicate in _PLATFORM_KINDS
            else job_config if predicate == "JOB_STATUS"
            else processing_job_config if predicate == "PROCESSING_JOB_STATUS"
            else jacobs_step_config if predicate == "STEP_STATUS" else jacobs_config)
        config_digest = runtime_status_source_configuration_digest(predicate, source_config)
        binding = PredicateAuthorityBinding(predicate, _BINDING_VERSION, source, owner,
            scope.environment, rule, rule, sla, ConflictPolicy.SINGLE_SOURCE_REQUIRED,
            identity, _RESOLVER_VERSION, config_digest, _BINDING_VERSION,
            source_scope_class=source_scope)
        adapter = TrustedAdapterRegistration(kind, identity, _RESOLVER_VERSION, source,
            config_digest)
        keys = (("job_id", "status") if predicate == "JOB_STATUS"
            else ("processing_job_id", "status") if predicate == "PROCESSING_JOB_STATUS"
            else ("step_id", "status") if predicate == "STEP_STATUS"
            else ("pipeline_id", "status") if predicate == "PIPELINE_STATUS" else ("name", "status"))
        entries.append(RegistryEntry(binding, adapter, keys, f"{predicate}@{_BINDING_VERSION}:es"))
    return _build_approved_registry_for_server(tuple(entries), authenticator=authenticator)

@dataclass(frozen=True)
class PlatformRuntimeStatusSnapshot:
    """Fixed server-produced observation; this is not a generic resolver input."""
    predicate: str
    arguments: Mapping[str, object]
    observed_at: datetime
    provenance_ref: str
    source_configuration: Mapping[str, object]
    source_configuration_digest: str = field(init=False)
    def __post_init__(self):
        predicate = _text(self.predicate, "predicate")
        if predicate not in _PLATFORM_KINDS:
            raise GovernanceContractError("platform runtime predicate unsupported")
        if not isinstance(self.arguments, Mapping):
            raise GovernanceContractError("platform runtime arguments must be mapping")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "arguments", _freeze(self.arguments, "platform runtime arguments"))
        _text(self.provenance_ref, "platform runtime provenance_ref")
        if not isinstance(self.source_configuration, Mapping):
            raise GovernanceContractError("Platform runtime source configuration invalid")
        configuration = _freeze(self.source_configuration, "Platform runtime source configuration")
        object.__setattr__(self, "source_configuration", configuration)
        object.__setattr__(self, "source_configuration_digest",
            runtime_status_source_configuration_digest(predicate, configuration))

def platform_runtime_status_evidence(snapshot: PlatformRuntimeStatusSnapshot, arguments: Mapping[str, object], scope: ResponseScope) -> RuntimeStatusEvidence:
    """Narrow platform bridge. Predicate/source/kind are constants above."""
    if not isinstance(snapshot, PlatformRuntimeStatusSnapshot) or not isinstance(scope, ResponseScope):
        raise GovernanceContractError("typed platform snapshot and scope required")
    if not isinstance(arguments, Mapping) or _plain(snapshot.arguments) != _plain(arguments):
        raise GovernanceContractError("platform observation must equal canonical predicate arguments")
    kind, source = _PLATFORM_KINDS[snapshot.predicate]
    observation = ResolutionObservation(ResolutionStatus.RESOLVED, snapshot.observed_at,
        f"{source}:{snapshot.provenance_ref}", snapshot.arguments)
    if snapshot.source_configuration_digest != runtime_status_source_configuration_digest(snapshot.predicate, snapshot.source_configuration):
        raise GovernanceContractError("Platform source configuration digest mismatch")
    return _runtime_status_evidence_from_server(kind, observation, scope,
        snapshot.source_configuration_digest)

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
            snapshot = self._store.authoritative_snapshot(job_id)
        except Exception:  # fail-soft: una lectura fallida queda UNAVAILABLE, nunca se acredita.
            snapshot = None
        view = snapshot.view if snapshot is not None else None
        observed_at = snapshot.observed_at if snapshot is not None else None
        if view is None or observed_at is None or view.tenant_id is None or view.user_id is None:
            obs = ResolutionObservation(ResolutionStatus.UNAVAILABLE, datetime.now(timezone.utc), "motor-job:unavailable", {})
        elif (view.tenant_id, view.user_id) != (scope.tenant_id, scope.subject_id):
            obs = ResolutionObservation(ResolutionStatus.WRONG_SCOPE, observed_at, f"motor-job:{job_id}", {})
        else:
            obs = ResolutionObservation(ResolutionStatus.RESOLVED, observed_at, f"motor-job:{job_id}",
                {"job_id": job_id, "status": view.status.value})
        return _runtime_status_evidence_from_server(AdapterKind.MOTOR_JOB_STATUS, obs, scope,
            runtime_status_source_configuration_digest("JOB_STATUS", self._store.source_configuration()))


class ProcessingJobStatusResolver:
    """Resolve only from the server singleton and immutable Processing owner."""
    def __init__(self):
        import procesamiento_routes
        from processing_job_store import ProcessingJobStore
        if type(procesamiento_routes._STORE) is not ProcessingJobStore:
            raise GovernanceContractError("canonical ProcessingJobStore unavailable")
        self._store = procesamiento_routes._STORE

    def evidence(self, arguments: Mapping[str, object], scope: ResponseScope) -> RuntimeStatusEvidence:
        if (not isinstance(scope, ResponseScope) or scope.project_id is None
                or scope.subject_id is None or scope.actor_id is None):
            raise GovernanceContractError("PROCESSING_JOB_STATUS exact scope required")
        if not isinstance(arguments, Mapping) or set(arguments) != {"processing_job_id", "status"}:
            raise GovernanceContractError("PROCESSING_JOB_STATUS arguments invalid")
        job_id, requested_status = arguments["processing_job_id"], arguments["status"]
        if not isinstance(job_id, str) or not isinstance(requested_status, str):
            raise GovernanceContractError("PROCESSING_JOB_STATUS arguments invalid")
        from processing_job_store import ProcessingJobStatus
        try:
            ProcessingJobStatus(requested_status)
        except ValueError:
            snapshot = None
        else:
            try:
                snapshot = self._store.authoritative_snapshot(job_id)
            except (OSError, TypeError, ValueError) as exc:  # fail-soft: expected authoritative-store read failure is UNAVAILABLE.
                logger.warning("Processing authoritative source read unavailable: %s", type(exc).__name__)
                snapshot = None
        if snapshot is None:
            observation = ResolutionObservation(ResolutionStatus.UNAVAILABLE, datetime.now(timezone.utc),
                "las-manos-processing-job:unavailable", {})
        elif (snapshot.owner.tenant_id, snapshot.owner.user_id, snapshot.owner.project_id) != (
                scope.tenant_id, scope.subject_id, scope.project_id):
            observation = ResolutionObservation(ResolutionStatus.WRONG_SCOPE, snapshot.observed_at,
                f"las-manos-processing-job:{job_id}", {})
        else:
            actual_status = snapshot.view.status.value
            if actual_status not in {item.value for item in ProcessingJobStatus}:
                observation = ResolutionObservation(ResolutionStatus.UNAVAILABLE, datetime.now(timezone.utc),
                    "las-manos-processing-job:unavailable", {})
            else:
                observation = ResolutionObservation(ResolutionStatus.RESOLVED, snapshot.observed_at,
                    f"las-manos-processing-job:{job_id}",
                    {"processing_job_id": job_id, "status": actual_status})
        return _runtime_status_evidence_from_server(AdapterKind.LAS_MANOS_PROCESSING_JOB_STATUS,
            observation, scope, runtime_status_source_configuration_digest(
                "PROCESSING_JOB_STATUS", self._store.source_configuration()))

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
        observed_at = datetime.now(timezone.utc) if pipeline is not None else None
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
        try:
            source_config = _jacobs_source_configuration(require_config=True)
        except Exception:  # fail-soft: missing Jacobs DB config must leave PIPELINE_STATUS unavailable.
            source_config = _jacobs_source_configuration()
            obs = ResolutionObservation(ResolutionStatus.UNAVAILABLE, datetime.now(timezone.utc),
                "jacobs-pipeline:source-configuration-unavailable", {})
        return _runtime_status_evidence_from_server(AdapterKind.JACOBS_PIPELINE_STATUS, obs, scope,
            runtime_status_source_configuration_digest("PIPELINE_STATUS", source_config))


class JacobsStepStatusResolver:
    """Resolve STEP_STATUS from one canonical step+pipeline-owner snapshot."""

    async def evidence(self, arguments: Mapping[str, object], scope: ResponseScope) -> RuntimeStatusEvidence:
        if (not isinstance(scope, ResponseScope) or scope.project_id is not None
                or scope.subject_id is None):
            raise GovernanceContractError("STEP_STATUS exact user scope unsupported")
        if not isinstance(arguments, Mapping) or set(arguments) != {"step_id", "status"}:
            raise GovernanceContractError("STEP_STATUS arguments invalid")
        step_id, requested_status = arguments["step_id"], arguments["status"]
        if (not isinstance(step_id, str) or not step_id
                or not isinstance(requested_status, str)):
            raise GovernanceContractError("STEP_STATUS arguments invalid")

        from jacobs.models import PipelineStatus, StepStatus
        try:
            StepStatus(requested_status)
        except ValueError:
            raise GovernanceContractError("STEP_STATUS status is not canonical") from None

        from jacobs import store as jacobs_store
        unavailable = ResolutionObservation(ResolutionStatus.UNAVAILABLE,
            datetime.now(timezone.utc), "jacobs-step:unavailable", {})
        try:
            source_config = _jacobs_step_source_configuration(require_config=True)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Jacobs STEP_STATUS source configuration unavailable: %s", type(exc).__name__)
            source_config = _jacobs_step_source_configuration()
            observation = unavailable
        else:
            try:
                snapshot = await jacobs_store.step_status_snapshot(step_id)
            except (aiomysql.Error, OSError, RuntimeError, ValueError) as exc:
                logger.warning("Jacobs step authoritative snapshot unavailable: %s", type(exc).__name__)
                snapshot = None

            if snapshot is None:
                observation = unavailable
            else:
                try:
                    actual_status = StepStatus(snapshot.status).value
                    pipeline_status = PipelineStatus(snapshot.pipeline_status)
                    owner_is_valid = (
                        snapshot.step_id == step_id
                        and isinstance(snapshot.pipeline_id, str) and bool(snapshot.pipeline_id)
                        and isinstance(snapshot.tenant_id, str) and bool(snapshot.tenant_id)
                        and isinstance(snapshot.user_id, str) and bool(snapshot.user_id)
                        and snapshot.owner_ack_at is not None
                    )
                    if (not owner_is_valid or pipeline_status in {
                            PipelineStatus.hidden, PipelineStatus.discarded}):
                        observation = unavailable
                    elif (snapshot.tenant_id, snapshot.user_id) != (scope.tenant_id, scope.subject_id):
                        observation = ResolutionObservation(ResolutionStatus.WRONG_SCOPE,
                            snapshot.observed_at, f"jacobs-step:{step_id}", {})
                    else:
                        observation = ResolutionObservation(ResolutionStatus.RESOLVED,
                            snapshot.observed_at, f"jacobs-step:{step_id}",
                            {"step_id": step_id, "status": actual_status})
                except (TypeError, ValueError, AttributeError):
                    observation = unavailable

        return _runtime_status_evidence_from_server(AdapterKind.JACOBS_STEP_STATUS,
            observation, scope, runtime_status_source_configuration_digest("STEP_STATUS", source_config))

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
