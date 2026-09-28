"""Privileged, local composition root for the governed Ariadna PM runtime.

This module is deliberately a host boundary, not an Ariadna capability.  It
constructs the authority engine and its verifiers itself; proposals are data
only and cannot select a verifier, scheduler, or effect executor.  Production
effects are disabled unless the host configuration explicitly enables them.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol


PROJECT_ID = "las-voces"
_AUTHORITY_SOURCE = "projects/las-voces/authority/ariadna_authority.py"
_RUNTIME_SOURCE = "projects/las-voces/authority/ariadna_runtime.py"


class HostLifecycle(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    DISABLED = "DISABLED"
    KILLED = "KILLED"


class HostMode(str, Enum):
    DRY_RUN = "dry-run"
    PRODUCTION = "production"


class GitHubFetcher(Protocol):
    def __call__(self, path: str) -> dict[str, Any]: ...


def _load_module(name: str, source: Path) -> Any:
    """Load a sibling source file without making ``las-voces`` a Python package."""
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load governed module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GitHubActionsVerifier:
    """Fail-closed GitHub Actions verifier bound to repository, run, job and SHA.

    A transport error deliberately returns ``None``: the LV-003 engine maps an
    unavailable independent verifier to HUMAN_REQUIRED rather than trusting a
    caller-supplied manifest.  A response that is present but inconsistent is
    rejected with ``False``.
    """
    def __init__(self, repository_id: str, token: str | None, *, fetcher: GitHubFetcher | None = None):
        self._repository_id = repository_id
        self._token = token
        self._fetcher = fetcher

    def _get(self, path: str) -> dict[str, Any]:
        if self._fetcher is not None:
            return self._fetcher(path)
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(
            "https://api.github.com" + path,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise OSError("invalid GitHub API response")
        return value

    def execution_record(self, manifest: dict[str, Any]) -> bool | None:
        if manifest.get("repository_id") != self._repository_id:
            return False
        run_id, job_id, commit = manifest.get("run_id"), manifest.get("job_id"), manifest.get("commit_sha")
        if not all(isinstance(value, str) and value for value in (run_id, job_id, commit)):
            return False
        if not run_id.isdecimal() or not job_id.isdecimal() or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            return False
        try:
            run = self._get(f"/repos/{self._repository_id}/actions/runs/{run_id}")
            jobs = self._get(f"/repos/{self._repository_id}/actions/runs/{run_id}/jobs?per_page=100")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
            return None
        if run.get("head_sha") != commit or run.get("status") != "completed" or run.get("conclusion") != "success":
            return False
        workflow = manifest.get("workflow")
        if not isinstance(workflow, str) or not workflow or not str(run.get("path", "")).endswith("/" + workflow):
            return False
        values = jobs.get("jobs")
        if not isinstance(values, list):
            return False
        for job in values:
            if not isinstance(job, dict):
                continue
            identity_matches = str(job.get("id")) == job_id or job.get("name") == job_id
            if identity_matches and job.get("status") == "completed" and job.get("conclusion") == "success":
                return True
        return False


class CanonicalHumanAuthorityVerifier:
    """Resolve acceptance only to a matching append-only Human Authority row.

    Actor strings are not proof.  A completion acceptance must name one
    ``activity:<event_id>`` record whose task, decision, and commit binding
    agree with the acceptance evidence.  In the absence of such a record the
    adapter returns ``None`` and LV-003 requires a human instead of guessing.
    """
    def __init__(self, root: Path):
        self._root = root

    def acceptance_record(self, record: dict[str, Any]) -> bool | None:
        source = record.get("authority_source")
        if not isinstance(source, dict) or source.get("type") != "human_authority":
            return False
        locator = source.get("record_id")
        if not isinstance(locator, str) or not locator.startswith("activity:"):
            return None
        event_id = locator.removeprefix("activity:")
        try:
            rows = [json.loads(line) for line in (self._root / "projects/las-voces/activity.ndjson").read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            return None
        matches = [row for row in rows if isinstance(row, dict) and row.get("event_id") == event_id]
        if not matches:
            return None
        if len(matches) != 1:
            return False
        event = matches[0]
        if event.get("event_type") != "HUMAN_AUTHORITY_TASK_ACCEPTED" or event.get("status") != "RECORDED":
            return False
        if event.get("project_id") != PROJECT_ID or event.get("actor") != "Fernando / Human Authority":
            return False
        if event.get("task_id") != record.get("task_id") or event.get("decision") != record.get("decision"):
            return False
        commit = record.get("commit_sha")
        if event.get("accepted_commit_sha") != commit:
            return False
        return True


@dataclass(frozen=True)
class HostConfig:
    mode: HostMode = HostMode.DRY_RUN
    production_effects_enabled: bool = False
    tick_seconds: float = 30.0
    github_repository: str = "fjruizhn/Jax"
    github_token_env: str = "ARIADNA_GITHUB_TOKEN"
    kill_switch_path: Path = Path("/etc/jax/ariadna-pm.kill")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "HostConfig":
        if not isinstance(value, dict):
            raise ValueError("host configuration must be an object")
        allowed = {"mode", "production_effects_enabled", "tick_seconds", "github_repository", "github_token_env", "kill_switch_path"}
        if set(value) - allowed:
            raise ValueError("unknown host configuration field")
        try:
            mode = HostMode(value.get("mode", HostMode.DRY_RUN.value))
        except ValueError as exc:
            raise ValueError("invalid host mode") from exc
        enabled = value.get("production_effects_enabled", False)
        cadence = value.get("tick_seconds", 30.0)
        repository = value.get("github_repository", "fjruizhn/Jax")
        token_env = value.get("github_token_env", "ARIADNA_GITHUB_TOKEN")
        kill = value.get("kill_switch_path", "/etc/jax/ariadna-pm.kill")
        if not isinstance(enabled, bool) or not isinstance(cadence, (int, float)) or cadence <= 0:
            raise ValueError("invalid production enablement or tick cadence")
        if not all(isinstance(item, str) and item for item in (repository, token_env, kill)):
            raise ValueError("invalid host identity or kill-switch path")
        return cls(mode, enabled, float(cadence), repository, token_env, Path(kill))

    @classmethod
    def from_file(cls, path: Path) -> "HostConfig":
        try:
            return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("unreadable host configuration") from exc


class AriadnaHost:
    """Host-owned scheduler around the LV-004 ``run_once`` core."""
    def __init__(self, root: Path, config: HostConfig, *, github_fetcher: GitHubFetcher | None = None, logger: logging.Logger | None = None):
        self.root, self.config = root.resolve(), config
        authority = _load_module("ariadna_host_authority", self.root / _AUTHORITY_SOURCE)
        runtime = _load_module("ariadna_host_runtime", self.root / _RUNTIME_SOURCE)
        token = os.environ.get(config.github_token_env)
        self._ci_verifier = GitHubActionsVerifier(config.github_repository, token, fetcher=github_fetcher)
        self._human_verifier = CanonicalHumanAuthorityVerifier(self.root)
        self._engine = authority.AuthorityEngine(self.root, ci_verifier=self._ci_verifier, human_authority_verifier=self._human_verifier)
        self._runtime = runtime.AriadnaRuntime(self.root, self._engine)
        self._runtime_module = runtime
        self._state = HostLifecycle.NOT_STARTED
        self._stop = threading.Event()
        self._logger = logger or logging.getLogger("las_voces.ariadna_host")
        self._health_path = runtime._state_dir(self.root) / "host-health.json"

    @property
    def lifecycle(self) -> HostLifecycle:
        return self._state

    def _killed(self) -> bool:
        return self.config.kill_switch_path.exists()

    def _log(self, event: str, **detail: Any) -> None:
        self._logger.info(json.dumps({"event": event, "host_lifecycle": self._state.value, **detail}, sort_keys=True))

    def _persist_health(self) -> None:
        self._health_path.write_text(json.dumps(self.health(), sort_keys=True), encoding="utf-8")

    def start(self) -> HostLifecycle:
        if self._state is not HostLifecycle.NOT_STARTED:
            return self._state
        if self._killed():
            self._state = HostLifecycle.KILLED
        elif self.config.mode is HostMode.PRODUCTION and not self.config.production_effects_enabled:
            self._state = HostLifecycle.DISABLED
        else:
            runtime_state = self._runtime.start()
            if runtime_state.value == "READY":
                self._state = HostLifecycle.READY
            elif runtime_state.value == "RECONCILIATION_REQUIRED":
                self._state = HostLifecycle.RECONCILIATION_REQUIRED
            else:
                self._state = HostLifecycle.STOPPED
        self._log("host_start")
        self._persist_health()
        return self._state

    def request_stop(self) -> HostLifecycle:
        self._stop.set()
        if self._state in {HostLifecycle.READY, HostLifecycle.RECONCILIATION_REQUIRED}:
            self._state = HostLifecycle.STOPPING
            self._runtime.request_stop()
        self._persist_health()
        return self._state

    def shutdown(self) -> HostLifecycle:
        self.request_stop()
        self._runtime.shutdown()
        if self._state is not HostLifecycle.KILLED:
            self._state = HostLifecycle.STOPPED
        self._log("host_shutdown")
        self._persist_health()
        return self._state

    def health(self) -> dict[str, Any]:
        return {
            "lifecycle": self._state.value,
            "ready": self._state is HostLifecycle.READY,
            "mode": self.config.mode.value,
            "production_effects_enabled": self.config.production_effects_enabled,
            "kill_switch_active": self._killed(),
            "runtime": self._runtime.health(),
        }

    def acquire_task(self, lease: Any) -> bool:
        """Host-mediated task ownership; proposals cannot write the lease ledger."""
        return self._state is HostLifecycle.READY and self._runtime.acquire_task(lease)

    def release_task(self, lease: Any) -> bool:
        return self._runtime.release_task(lease)

    def run_once(self, proposal: Any | None = None, *, lease_id: str | None = None) -> str:
        if self._state is HostLifecycle.NOT_STARTED:
            self.start()
        if self._killed():
            self._runtime.request_stop()
            self._runtime.shutdown()
            self._state = HostLifecycle.KILLED
            self._log("kill_switch_observed")
            self._persist_health()
            return "NOOP_KILLED"
        if self._state is not HostLifecycle.READY or self._stop.is_set():
            return "NOOP_NOT_READY"
        if proposal is None:
            return "NOOP_NO_PROPOSAL"
        if self.config.mode is HostMode.DRY_RUN:
            decision = self._engine.evaluate(sender_agent=self._runtime_module.ARIADNA_ID, task_id=proposal.task_id, action=proposal.action, target_status=proposal.target_status, evidence_refs=list(proposal.evidence_refs), handoff=proposal.handoff)
            self._log("dry_run_cycle", task_id=proposal.task_id, action=proposal.action, verdict=decision.verdict.value)
            return "DRY_RUN_" + decision.verdict.value
        result = self._runtime.run_once(proposal, lease_id=lease_id)
        self._log("production_cycle", task_id=proposal.task_id, action=proposal.action, result=result)
        return result

    def run_forever(self, proposal_supplier: Callable[[], Any | None] | None = None) -> HostLifecycle:
        self.start()
        if self._state is not HostLifecycle.READY:
            return self._state
        while self._state is HostLifecycle.READY and not self._stop.is_set():
            self.run_once(proposal_supplier() if proposal_supplier is not None else None)
            if self._stop.wait(self.config.tick_seconds):
                break
        return self.shutdown()
