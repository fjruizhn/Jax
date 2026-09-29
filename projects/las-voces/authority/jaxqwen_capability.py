"""ACK-bound, structured execution capability for the LAS VOCES Qwen builder.

This is deliberately a *broker*, not a general agent runner.  It never accepts
an executable, argv, shell fragment, environment, or caller-selected cwd.  A
privileged host may supply an authenticated model transport separately; model
output is only a request for one of the tools enforced below.

The dispatcher-owned linked worktree is treated as read-only input.  Before a
mission is made available to ``jaxqwen`` it is copied into an independent Git
clone, so the machine identity never needs write access to the source
checkout's shared Git common directory.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import secrets
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

try:  # normal host composition imports this module as part of authority
    from .jaxqwen_trust import JaxQwenTrustBroker, TrustError, ISSUER, VERSION, BOUND
except ImportError:  # direct, file-based authority tests retain no package ambient state
    import importlib.util
    _trust_spec = importlib.util.spec_from_file_location("jaxqwen_trust", Path(__file__).with_name("jaxqwen_trust.py"))
    assert _trust_spec and _trust_spec.loader
    _trust = importlib.util.module_from_spec(_trust_spec); sys.modules.setdefault("jaxqwen_trust", _trust); _trust_spec.loader.exec_module(_trust)
    JaxQwenTrustBroker, TrustError, ISSUER, VERSION, BOUND = _trust.JaxQwenTrustBroker, _trust.TrustError, _trust.ISSUER, _trust.VERSION, _trust.BOUND

PROJECT_ID = "las-voces"
CAPABILITY = "las_voces.builder.qwen.execute"
SERVICE_IDENTITY = "jaxqwen"
_KEY = re.compile(r"^[0-9a-f]{64}$")
_TASK = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
_GIT = "/usr/bin/git"
_FORBIDDEN_PATHS = ("policy/", "projects/las-voces/authority/", "config/systemd/", ".git/")
_TOOLS = frozenset({"read_file", "list_files", "search_text", "write_file", "apply_patch", "run_named_test", "inspect_diff", "request_commit"})
_WRITABLE = frozenset({"write_file", "apply_patch"})


class CapabilityError(ValueError):
    """The machine request is not safe to execute."""


@dataclass(frozen=True)
class Mission:
    mission_id: str
    project_id: str
    task_id: str
    canonical_owner: str
    builder_identity: str
    ack_id: str
    correlation_id: str
    lease_id: str
    project_hash: str
    source_revision: str
    branch: str
    workspace: str
    scope: str
    allowed_tools: tuple[str, ...]
    named_tests: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    evidence_requirements: tuple[str, ...]


@dataclass(frozen=True)
class MissionResult:
    decision: str
    reason: str
    mission_id: str | None = None
    workspace: str | None = None


class ModelTransport(Protocol):
    """Composition-owned local-model transport, never controlled by a mission."""
    def request_tools(self, mission: Mission, messages: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]: ...


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_OPTIONAL_LOCKS": "0"}


def _git(root: Path, *args: str) -> str:
    done = subprocess.run([_GIT, "-C", str(root), *args], text=True, capture_output=True, env=_git_env())
    if done.returncode:
        raise CapabilityError("trusted git operation rejected")
    return done.stdout.strip()


def _project_hash(root: Path) -> str:
    return hashlib.sha256((root / "projects/las-voces/project.json").read_bytes()).hexdigest()


def _repository_identity(root: Path) -> str:
    """Stable repository allowlist identity; filesystem path is not identity."""
    remote = _git(root, "remote", "get-url", "origin").strip()
    normalized = remote.removesuffix(".git").rstrip("/").lower()
    if normalized not in {"git@github.com:fjruizhn/jax", "https://github.com/fjruizhn/jax", "ssh://git@github.com/fjruizhn/jax"}:
        raise CapabilityError("repository identity is not allowlisted")
    return "github.com/fjruizhn/Jax"


def _safe_regular(path: Path, *, required: bool = True) -> None:
    if path.is_symlink() or (required and (not path.exists() or not path.is_file())):
        raise CapabilityError("unsafe or missing capability state")
    if path.exists() and not stat.S_ISREG(path.stat().st_mode):
        raise CapabilityError("capability state is not a regular file")


class JaxQwenCapability:
    """The host-owned capability; ``jaxqwen`` only receives a Mission + tools.

    The caller must be the dedicated machine identity at the authenticated
    transport boundary.  This library verifies that identity as data as well,
    which is useful for adapters and tests but is not a replacement for OS or
    transport authentication.
    """
    def __init__(self, root: Path, handoff_state_dir: Path, workspace_root: Path, *, canonical_root: Path | None = None, source_worktree_root: Path | None = None, trust_broker: JaxQwenTrustBroker | None = None, host_state_dir: Path | None = None, operator_verifier: Callable[[object], bool] | None = None, identity_verifier: Callable[[], str] | None = None):
        self.root, self.handoff_state_dir, self.workspace_root = root.resolve(), handoff_state_dir.resolve(), workspace_root.resolve()
        if canonical_root is None: raise CapabilityError("explicit canonical read-only checkout is required")
        self.canonical_root = Path(canonical_root).resolve()
        if source_worktree_root is None: raise CapabilityError("explicit dispatcher worktree root is required")
        self.source_worktree_root = Path(source_worktree_root).resolve()
        if not (self.root / "projects/las-voces/project.json").is_file():
            raise CapabilityError("invalid canonical checkout")
        if str(self.root).startswith("/srv/jax-prod") or str(self.workspace_root).startswith("/srv/jax-prod"):
            raise CapabilityError("production checkout cannot host jaxqwen mission")
        common = Path(_git(self.root, "rev-parse", "--git-common-dir"))
        common = (self.root / common).resolve() if not common.is_absolute() else common.resolve()
        self.dispatch_dir = common / "ariadna-builder-dispatch"
        canonical_common = Path(_git(self.canonical_root, "rev-parse", "--git-common-dir"))
        canonical_common = (self.canonical_root / canonical_common).resolve() if not canonical_common.is_absolute() else canonical_common.resolve()
        if self.handoff_state_dir != canonical_common / "ariadna-pm-control":
            raise CapabilityError("handoff state is not the canonical producer control plane")
        # Mission and credential state is host state, never shared Git metadata.
        if host_state_dir is None: raise CapabilityError("explicit host state directory is required")
        self.ledger_dir = Path(os.path.abspath(host_state_dir))
        if any(part.is_symlink() for part in (self.ledger_dir, *self.ledger_dir.parents)):
            raise CapabilityError("host state path cannot traverse a symlink")
        for forbidden in (self.root, self.workspace_root, common):
            try: self.ledger_dir.relative_to(forbidden); raise CapabilityError("host state cannot be inside checkout or workspace")
            except ValueError: pass  # fail-soft: outside the forbidden root is the expected safe case
        self.ledger, self.lock_path = self.ledger_dir / "missions.ndjson", self.ledger_dir / "missions.lock"
        self._trust_broker = trust_broker
        self._operator_verifier = operator_verifier
        # Retained only for constructor compatibility. UID/user name is never authority.
        self._identity_verifier = identity_verifier

    @staticmethod
    def _os_identity() -> str:
        import pwd
        return pwd.getpwuid(os.geteuid()).pw_name

    @staticmethod
    def allowed_action(action: str) -> bool:
        return action == CAPABILITY

    def _locked(self):
        self.ledger_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = self.ledger_dir.stat()
        if st.st_uid != os.geteuid() or st.st_mode & 0o077: raise CapabilityError("mission state must be owner-only host state")
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.geteuid() or lock_stat.st_mode & 0o077:
            os.close(fd); raise CapabilityError("unsafe mission lock")
        return os.fdopen(fd, "r+")

    @contextmanager
    def _lease_guard(self):
        """Share the governed lease mutex through validation and each effect."""
        path = self.handoff_state_dir / "task-leases.lock"
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise CapabilityError("governed lease lock is unavailable") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode): raise CapabilityError("governed lease lock is unsafe")
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CapabilityError("governed lease update is in progress") from exc
            yield
        finally:
            try: fcntl.flock(fd, fcntl.LOCK_UN)
            finally: os.close(fd)

    def _append(self, row: dict[str, Any]) -> None:
        self.ledger_dir.mkdir(parents=True, exist_ok=True); _safe_regular(self.ledger, required=False)
        fd = os.open(self.ledger, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode): raise CapabilityError("unsafe mission ledger")
            os.write(fd, (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()); os.fsync(fd)
        finally: os.close(fd)

    def _history(self) -> dict[str, list[dict[str, Any]]]:
        if not self.ledger.exists(): return {}
        _safe_regular(self.ledger); out: dict[str, list[dict[str, Any]]] = {}
        try:
            for raw in self.ledger.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw); mission = row.get("mission_id") if isinstance(row, dict) else None
                if not isinstance(mission, str) or not _KEY.fullmatch(mission): raise CapabilityError("malformed mission history")
                out.setdefault(mission, []).append(row)
        except json.JSONDecodeError as exc: raise CapabilityError("malformed mission history") from exc
        return out

    def _ack(self, key: str) -> dict[str, Any]:
        path = self.dispatch_dir / "acks.ndjson"; _safe_regular(path)
        try: rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        except json.JSONDecodeError as exc: raise CapabilityError("malformed dispatcher acknowledgement") from exc
        rows = [row for row in rows if isinstance(row, dict) and row.get("idempotency_key") == key]
        dispatched = [row for row in rows if row.get("state") == "DISPATCHED"]
        if len(dispatched) != 1 or any(row.get("state") == "REJECTED" for row in rows): raise CapabilityError("ACK is absent or not uniquely dispatched")
        ack = dispatched[0]
        required = {"task_id", "lease_id", "canonical_owner", "builder_identity", "branch", "worktree", "project_hash", "base_commit"}
        if not required <= ack.keys() or ack.get("builder_identity") != "Qwen" or ack.get("canonical_owner") != "Qwen/Infra":
            raise CapabilityError("ACK lacks canonical Qwen binding")
        if ack.get("builder_process_started") is not False: raise CapabilityError("ACK claims an unexpected builder process")
        return ack

    def _handoff(self, key: str, ack: dict[str, Any]) -> dict[str, Any]:
        path = self.handoff_state_dir / "handoffs.ndjson"; _safe_regular(path)
        try: rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        except json.JSONDecodeError as exc: raise CapabilityError("malformed handoff state") from exc
        found = [x for x in rows if isinstance(x, dict) and x.get("idempotency_key") == key]
        if len(found) != 1: raise CapabilityError("handoff is absent or ambiguous")
        row = found[0]; handoff = row.get("handoff")
        if set(row) != {"event_type", "idempotency_key", "task_id", "handoff", "lease_id"} or not isinstance(handoff, dict): raise CapabilityError("handoff record is malformed")
        if row.get("event_type") != "emit_handoff" or row.get("task_id") != ack["task_id"] or row.get("lease_id") != ack["lease_id"]: raise CapabilityError("handoff differs from ACK")
        if handoff.get("recipient_agent") != ack["canonical_owner"] or handoff.get("correlation_id") != ack["lease_id"] or handoff.get("message_id") != ack["lease_id"]: raise CapabilityError("handoff recipient/correlation is invalid")
        return row

    def _handoff_audit(self, key: str, ack: dict[str, Any]) -> dict[str, Any]:
        path = self.handoff_state_dir / "runtime-audit.ndjson"; _safe_regular(path)
        try:
            rows = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw]
        except json.JSONDecodeError as exc: raise CapabilityError("malformed handoff audit") from exc
        matches = [row for row in rows if isinstance(row, dict) and row.get("idempotency_key") == key]
        if len(matches) != 1: raise CapabilityError("handoff audit is absent or ambiguous")
        audit = matches[0]
        expected = {"event_type": "ARIADNA_CYCLE", "task_id": ack["task_id"], "action": "emit_handoff",
                    "verdict": "ALLOW", "result": "EFFECT_EMIT_HANDOFF", "lease_id": ack["lease_id"],
                    "project_hash": ack["project_hash"]}
        if any(audit.get(name) != value for name, value in expected.items()):
            raise CapabilityError("handoff audit is not the current authoritative effect")
        if not isinstance(audit.get("runtime_instance_id"), str) or not audit["runtime_instance_id"]:
            raise CapabilityError("handoff audit lacks runtime identity")
        return audit

    def _runtime_control_is_live(self, lease: dict[str, Any]) -> bool:
        path = self.handoff_state_dir / "control.lock"; _safe_regular(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (value.get("state") != "ACTIVE" or value.get("instance_id") != lease.get("runtime_instance_id")
                    or value.get("pid") != lease.get("pid")):
                return False
            if type(lease.get("pid")) is not int or lease["pid"] <= 0: return False
            try: os.kill(lease["pid"], 0)
            except ProcessLookupError: return False
            except PermissionError: pass  # fail-soft: permission denial proves the process exists
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                try: fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError: return True
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            finally: os.close(fd)
        except (OSError, json.JSONDecodeError, TypeError):  # fail-soft: uncertain runtime ownership denies the mission
            return False

    def _validate_current(self, ack: dict[str, Any], row: dict[str, Any]) -> None:
        if _project_hash(self.root) != ack["project_hash"]: raise CapabilityError("ACK is stale against canonical project")
        if _git(self.root, "rev-parse", "HEAD") != ack["base_commit"]: raise CapabilityError("ACK source revision is stale")
        if (_project_hash(self.canonical_root) != ack["project_hash"]
                or _git(self.canonical_root, "rev-parse", "HEAD") != ack["base_commit"]
                or _repository_identity(self.canonical_root) != _repository_identity(self.root)
                or _git(self.canonical_root, "status", "--porcelain")
                or _git(self.root, "status", "--porcelain")):
            raise CapabilityError("canonical source changed after dispatcher acknowledgement")
        if ack["branch"] != f"las-voces/{ack['task_id'].lower()}": raise CapabilityError("ACK branch is not task-bound")
        project = json.loads((self.root / "projects/las-voces/project.json").read_text(encoding="utf-8"))
        tasks = [x for x in project.get("tasks", []) if isinstance(x, dict) and x.get("id") == ack["task_id"]]
        if len(tasks) != 1 or tasks[0].get("owner") != ack["canonical_owner"] or tasks[0].get("status") != "READY": raise CapabilityError("task is no longer eligible")
        by_id = {x.get("id"): x for x in project.get("tasks", []) if isinstance(x, dict)}
        if any(by_id.get(dep, {}).get("status") != "DONE" for dep in tasks[0].get("depends", [])) or tasks[0].get("blocker") or tasks[0].get("blockers"):
            raise CapabilityError("task dependencies or blockers reject mission")
        leases = self.handoff_state_dir / "task-leases.ndjson"; _safe_regular(leases)
        active: dict[str, dict[str, Any]] = {}
        for raw in leases.read_text(encoding="utf-8").splitlines():
            event = json.loads(raw)
            if event.get("event") == "ACQUIRED": active[event["lease"]["lease_id"]] = event["lease"]
            elif event.get("event") == "RELEASED": active.pop(event.get("lease_id"), None)
            else: raise CapabilityError("invalid task lease record")
        lease = active.get(ack["lease_id"])
        if not lease or lease.get("task_id") != ack["task_id"] or lease.get("owner") != ack["canonical_owner"]: raise CapabilityError("ACK lease is inactive or mismatched")
        audit = self._handoff_audit(ack["idempotency_key"], ack)
        if (lease.get("runtime_instance_id") != audit["runtime_instance_id"]
                or not isinstance(lease.get("pid"), int) or lease["pid"] <= 0
                or not self._runtime_control_is_live(lease)):
            raise CapabilityError("ACK lease runtime is not live")
        expected_worktree = self.source_worktree_root / f"las-voces-{ack['task_id'].lower()}"
        observed_worktree = Path(ack.get("worktree", ""))
        if observed_worktree.is_symlink() or observed_worktree != expected_worktree or not observed_worktree.is_dir():
            raise CapabilityError("ACK worktree is outside the configured dispatcher worktree root")
        if _git(observed_worktree, "rev-parse", "--show-toplevel") != str(expected_worktree):
            raise CapabilityError("ACK worktree identity is invalid")
        if (_repository_identity(observed_worktree) != _repository_identity(self.root)
                or _git(observed_worktree, "branch", "--show-current") != ack["branch"]
                or _git(observed_worktree, "rev-parse", "HEAD") != ack["base_commit"]
                or _git(observed_worktree, "status", "--porcelain")):
            raise CapabilityError("ACK worktree repository, revision or cleanliness is invalid")
        scope = row["handoff"].get("authority_context", {}).get("handoff", {}).get("scope")
        if not isinstance(scope, str) or not scope.startswith(f"LAS VOCES {ack['task_id']}:"):
            raise CapabilityError("handoff scope is not bounded")

    def _workspace(self, mission_id: str, ack: dict[str, Any]) -> Path:
        # The ACK worktree may share metadata; clone it into per-mission storage.
        expected_source = (self.source_worktree_root / f"las-voces-{ack['task_id'].lower()}").resolve()
        source, target = Path(ack["worktree"]), self.workspace_root / ack["task_id"].lower() / mission_id
        if source != expected_source:
            raise CapabilityError("ACK worktree is outside the configured dispatcher worktree root")
        if source.is_symlink() or not source.is_dir() or str(source).startswith("/srv/jax-prod"): raise CapabilityError("ACK worktree is unsafe")
        try: target.relative_to(self.workspace_root)
        except ValueError as exc: raise CapabilityError("mission workspace escapes root") from exc
        if target.exists():
            if target.is_symlink() or _git(target, "branch", "--show-current") != ack["branch"]: raise CapabilityError("existing mission workspace conflicts")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        # Clone from the *already validated* worktree only.  --no-local avoids a
        # shared object/reference shortcut; hooks are disabled on the clone.
        done = subprocess.run([_GIT, "clone", "--no-local", "--no-checkout", str(source), str(target)], text=True, capture_output=True, env=_git_env())
        if done.returncode: raise CapabilityError("isolated mission clone failed")
        _git(target, "config", "core.hooksPath", "/dev/null")
        _git(target, "checkout", "-B", ack["branch"], ack["base_commit"])
        if _git(target, "rev-parse", "--git-common-dir") == ".git": return target
        # A standalone clone reports .git relative to its own root.  Anything
        # else could be another linked worktree and is rejected.
        common = Path(_git(target, "rev-parse", "--git-common-dir")); resolved = (target / common).resolve() if not common.is_absolute() else common.resolve()
        if resolved != (target / ".git").resolve(): raise CapabilityError("mission clone shares Git metadata")
        return target

    def start(self, *, idempotency_key: str) -> MissionResult:
        # This is a dispatcher-side preparation operation; it does not start a
        # model or execute tools. Transport authentication happens below.
        if self._trust_broker is None: return MissionResult("REJECTED", "host trust broker is required")
        if not isinstance(idempotency_key, str) or not _KEY.fullmatch(idempotency_key): return MissionResult("REJECTED", "invalid handoff key")
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with self._lease_guard():
                    ack = self._ack(idempotency_key); row = self._handoff(idempotency_key, ack); self._validate_current(ack, row)
                    mission_id = _digest({"capability": CAPABILITY, "ack": idempotency_key, "lease": ack["lease_id"], "revision": ack["base_commit"]})
                    old = self._history().get(mission_id, [])
                    if old: return MissionResult("NOOP", "mission already recorded", mission_id, old[0].get("workspace"))
                    workspace = self._workspace(mission_id, ack)
                    mission = Mission(mission_id, PROJECT_ID, ack["task_id"], ack["canonical_owner"], ack["builder_identity"], idempotency_key, ack["lease_id"], ack["lease_id"], ack["project_hash"], ack["base_commit"], ack["branch"], str(workspace), ".", tuple(sorted(_TOOLS)), ("pytest_las_voces",), tuple(row["handoff"].get("acceptance_criteria", [])), tuple(row["handoff"].get("evidence_requirements", [])))
                    self._append({"event_type": "JAXQWEN_MISSION", "state": "ACCEPTED", "mission_id": mission_id,
                                  "task_id": mission.task_id, "ack_id": mission.ack_id, "lease_id": mission.lease_id,
                                  "project_hash": mission.project_hash, "capability": CAPABILITY,
                                  "machine_identity": SERVICE_IDENTITY, "workspace": str(workspace),
                                  "mission": asdict(mission), "observed_at": int(time.time())})
                    return MissionResult("ACCEPTED", "bounded mission prepared; model transport not started", mission_id, str(workspace))
            except (CapabilityError, KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
                return MissionResult("REJECTED", str(exc))

    def _mission(self, mission_id: str) -> Mission:
        rows = self._history().get(mission_id, [])
        if not rows or not isinstance(rows[0].get("mission"), dict): raise CapabilityError("unknown mission")
        return Mission(**rows[0]["mission"])

    def provision_credential(self, mission_id: str, *, ttl_seconds: int = 300) -> str:
        """Host-composition only issuance interface.

        The IPC adapter must restrict this method to the trusted dispatcher;
        this library deliberately has no socket, HTTP listener, or caller
        identity shortcut.  The returned opaque token is for one tool effect.
        """
        if self._trust_broker is None: raise CapabilityError("host trust broker is required")
        mission = self._mission(mission_id); self._assert_current(mission)
        history = self._history().get(mission_id, [])
        if not any(row.get("state") == "MISSION_STARTED" for row in history) or any(row.get("state") in {"MISSION_FINISHED", "MISSION_FAILED", "COMPLETED", "CANCELLED"} for row in history):
            raise CapabilityError("mission execution is not active")
        claims = self._claims(mission); claims["nonce"] = secrets.token_hex(32)
        return self._trust_broker.issue(claims, ttl_seconds=ttl_seconds)

    def _claims(self, mission: Mission) -> dict[str, Any]:
        return {"service_identity": SERVICE_IDENTITY, "capability_id": CAPABILITY, "mission_id": mission.mission_id, "project_id": mission.project_id, "task_id": mission.task_id, "dispatcher_ack_id": mission.ack_id, "correlation_id": mission.correlation_id, "lease_id": mission.lease_id, "project_hash": mission.project_hash, "repository": _repository_identity(self.root), "branch": mission.branch, "worktree": mission.workspace, "allowed_tools": list(mission.allowed_tools), "expiry": 0, "nonce": "", "issuer": ISSUER, "issuer_version": VERSION}

    def _assert_current(self, mission: Mission) -> None:
        # Re-read authoritative handoff/ACK/lease and task state before every effect.
        ack = self._ack(mission.ack_id); row = self._handoff(mission.ack_id, ack); self._validate_current(ack, row)
        if (mission.project_id != PROJECT_ID or mission.task_id != ack["task_id"] or mission.lease_id != ack["lease_id"]
                or mission.branch != ack["branch"] or mission.project_hash != ack["project_hash"]
                or mission.source_revision != ack["base_commit"] or mission.canonical_owner != ack["canonical_owner"]
                or mission.ack_id != ack["idempotency_key"] or mission.correlation_id != ack["lease_id"]):
            raise CapabilityError("mission binding differs from current dispatcher ACK")
        if _repository_identity(self.root) != "github.com/fjruizhn/Jax": raise CapabilityError("repository identity changed")
        workspace = Path(mission.workspace)
        if workspace.is_symlink() or not workspace.is_dir(): raise CapabilityError("mission workspace is unavailable")
        resolved = workspace.resolve()
        try: resolved.relative_to(self.workspace_root)
        except ValueError as exc: raise CapabilityError("mission workspace escapes host mission root") from exc
        common = Path(_git(resolved, "rev-parse", "--git-common-dir"))
        common = (resolved / common).resolve() if not common.is_absolute() else common.resolve()
        if common != (resolved / ".git").resolve(): raise CapabilityError("mission workspace shares Git metadata")
        if _git(resolved, "branch", "--show-current") != mission.branch: raise CapabilityError("mission worktree branch changed")
        if _git(resolved, "rev-parse", "HEAD") != mission.source_revision: raise CapabilityError("mission source revision changed")
        if any(x.get("state") in {"CANCELLED", "COMPLETED", "MISSION_FINISHED", "MISSION_FAILED"} for x in self._history().get(mission.mission_id, [])):
            raise CapabilityError("mission is terminal")

    def _authorize(self, mission: Mission, credential: Any, tool: str) -> None:
        if self._trust_broker is None: raise CapabilityError("host trust broker is required")
        if tool not in mission.allowed_tools: raise CapabilityError("tool is not mission-bound")
        expected = {key: value for key, value in self._claims(mission).items() if key in BOUND}
        try:
            self._trust_broker.verify_and_consume(credential, expected)
        except TrustError as exc: raise CapabilityError(str(exc)) from exc

    def _current_for_auth(self, mission: Mission) -> bool:
        try: self._assert_current(mission); return True
        except (CapabilityError, OSError, json.JSONDecodeError): return False

    def _broker_claims_current(self, claims: dict[str, Any]) -> bool:
        """Host-composition verifier target; never supplied by a request."""
        try:
            mission = self._mission(claims["mission_id"])
            expected = {key: value for key, value in self._claims(mission).items() if key in BOUND}
            if any(claims.get(key) != value for key, value in expected.items()): return False
            return self._current_for_auth(mission)
        except (CapabilityError, KeyError, TypeError, OSError, json.JSONDecodeError): return False

    def _claim_execution(self, mission_id: str) -> Mission:
        """Durably claim the sole model/tool execution for a prepared mission."""
        with self._locked() as lock, self._lease_guard():
            fcntl.flock(lock, fcntl.LOCK_EX)
            mission = self._mission(mission_id); self._assert_current(mission)
            if any(row.get("state") in {"MISSION_STARTED", "MISSION_FINISHED", "MISSION_FAILED", "COMPLETED", "CANCELLED"}
                   for row in self._history().get(mission_id, [])):
                raise CapabilityError("mission execution already claimed or terminal")
            self._append({"event_type": "JAXQWEN_MISSION", "state": "MISSION_STARTED", "mission_id": mission_id,
                          "task_id": mission.task_id, "ack_id": mission.ack_id, "lease_id": mission.lease_id,
                          "project_hash": mission.project_hash, "workspace": mission.workspace, "observed_at": int(time.time())})
            return mission

    @staticmethod
    def _path(mission: Mission, relative: str, *, write: bool = False) -> Path:
        if not isinstance(relative, str) or not relative or "\x00" in relative: raise CapabilityError("invalid tool path")
        root = Path(mission.workspace).resolve(); target = (root / relative).resolve()
        try: target.relative_to(root)
        except ValueError as exc: raise CapabilityError("tool path escapes mission workspace") from exc
        rel = target.relative_to(root).as_posix()
        if any(rel == item.rstrip("/") or rel.startswith(item) for item in _FORBIDDEN_PATHS): raise CapabilityError("tool path touches forbidden authority scope")
        if write and target.is_symlink(): raise CapabilityError("symlink write rejected")
        return target

    def tool(self, mission_id: str, name: str, *, credential: Any = None, **request: Any) -> dict[str, Any]:
        if name == "request_commit": return self.commit(mission_id, request.get("message"), credential=credential)
        with self._locked() as lock, self._lease_guard():
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._tool_effect(mission_id, name, credential=credential, **request)

    def _tool_effect(self, mission_id: str, name: str, *, credential: Any = None, **request: Any) -> dict[str, Any]:
        mission = self._mission(mission_id); self._assert_current(mission)
        if name not in _TOOLS: raise CapabilityError("tool is not allowlisted")
        self._authorize(mission, credential, name)
        if name in {"read_file", "list_files", "search_text", "inspect_diff"}:
            if name == "inspect_diff": return {"diff": _git(Path(mission.workspace), "diff", "--no-ext-diff", "--")}
            path = self._path(mission, request.get("path", ""))
            if name == "read_file": return {"content": path.read_text(encoding="utf-8")}
            if name == "list_files": return {"paths": sorted(x.relative_to(path).as_posix() for x in path.rglob("*") if x.is_file() and not x.is_symlink())}
            text = request.get("text");
            if not isinstance(text, str) or not text: raise CapabilityError("search text is invalid")
            return {"matches": [x.relative_to(path).as_posix() for x in path.rglob("*") if x.is_file() and not x.is_symlink() and text in x.read_text(encoding="utf-8", errors="strict")]}
        if name == "write_file":
            path = self._path(mission, request.get("path", ""), write=True); content = request.get("content")
            if not isinstance(content, str): raise CapabilityError("write content is invalid")
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8"); return {"written": path.relative_to(mission.workspace).as_posix()}
        if name == "apply_patch":
            path = self._path(mission, request.get("path", ""), write=True); old, new = request.get("old"), request.get("new")
            if not isinstance(old, str) or not isinstance(new, str) or not path.is_file(): raise CapabilityError("structured patch is invalid")
            content = path.read_text(encoding="utf-8")
            if content.count(old) != 1: raise CapabilityError("patch must match exactly once")
            path.write_text(content.replace(old, new), encoding="utf-8"); return {"patched": path.relative_to(mission.workspace).as_posix()}
        if name == "run_named_test":
            test = request.get("test")
            if test not in mission.named_tests: raise CapabilityError("test is not allowlisted")
            # Repo tests are arbitrary code.  A host-contained sandbox is not
            # implemented here, so executing them in the broker process fails closed.
            raise CapabilityError("named tests require a separate host sandbox")
        raise CapabilityError("unreachable tool")

    def run_tool_loop(self, mission_id: str, transport: ModelTransport, *, credential_provider: Callable[[Mission], str] | None = None, max_iterations: int = 8) -> list[dict[str, Any]]:
        """Run a bounded, data-only model/tool exchange.

        The host owns ``transport``; a mission cannot select an endpoint,
        credentials, model, command, environment or tool catalog.  Model
        output is merely a list of structured requests and each is checked by
        :meth:`tool` before it can have an effect.
        """
        if not isinstance(max_iterations, int) or not 1 <= max_iterations <= 32:
            raise CapabilityError("tool loop bound is invalid")
        mission = self._claim_execution(mission_id)
        history: list[dict[str, Any]] = [{"role": "system", "content": "Use only declared structured tools for the assigned mission."}]
        effects: list[dict[str, Any]] = []
        try:
            for _ in range(max_iterations):
                calls = transport.request_tools(mission, tuple(history))
                if not isinstance(calls, list): raise CapabilityError("model transport returned an invalid tool request")
                if not calls:
                    self._append({"event_type": "JAXQWEN_MISSION", "state": "MISSION_FINISHED", "mission_id": mission_id,
                                  "task_id": mission.task_id, "ack_id": mission.ack_id, "lease_id": mission.lease_id,
                                  "project_hash": mission.project_hash, "workspace": mission.workspace, "observed_at": int(time.time())})
                    return effects
                for call in calls:
                    if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
                        raise CapabilityError("model tool request is malformed")
                    if credential_provider is None: raise CapabilityError("authenticated credential provider is required")
                    result = self.tool(mission_id, call["name"], credential=credential_provider(mission), **call["arguments"])
                    effect = {"name": call["name"], "result": result}; effects.append(effect); history.append({"role": "tool", "content": json.dumps(effect, sort_keys=True)})
                    if call["name"] == "request_commit": return effects
            raise CapabilityError("bounded tool loop exhausted")
        except BaseException:
            with self._locked() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._append({"event_type": "JAXQWEN_MISSION", "state": "MISSION_FAILED", "mission_id": mission_id,
                              "task_id": mission.task_id, "ack_id": mission.ack_id, "lease_id": mission.lease_id,
                              "project_hash": mission.project_hash, "workspace": mission.workspace, "observed_at": int(time.time())})
            raise

    def commit(self, mission_id: str, message: Any, *, credential: Any = None) -> dict[str, Any]:
        with self._locked() as lock, self._lease_guard():
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._commit_locked(mission_id, message, credential=credential)

    def _commit_locked(self, mission_id: str, message: Any, *, credential: Any = None) -> dict[str, Any]:
        mission = self._mission(mission_id); self._assert_current(mission); self._authorize(mission, credential, "request_commit")
        if not isinstance(message, str) or not message or "\n" in message or len(message) > 200: raise CapabilityError("commit message is invalid")
        workspace = Path(mission.workspace)
        changed = _git(workspace, "diff", "--name-only", "--").splitlines()
        for status in _git(workspace, "status", "--porcelain", "--untracked-files=all").splitlines():
            candidate = status[3:]
            if status.startswith("?? ") and candidate not in changed: changed.append(candidate)
        if not changed: return {"commit": None, "reason": "no changes"}
        for relative in changed: self._path(mission, relative, write=True)
        _git(workspace, "add", "--", *changed)
        if sorted(_git(workspace, "diff", "--cached", "--name-only", "--").splitlines()) != sorted(changed): raise CapabilityError("commit staging differs from reviewed diff")
        _git(workspace, "-c", "user.name=jaxqwen", "-c", "user.email=jaxqwen@localhost", "commit", "--no-verify", "-m", message)
        sha = _git(workspace, "rev-parse", "HEAD")
        self._append({"event_type": "JAXQWEN_MISSION", "state": "COMPLETED", "mission_id": mission_id,
                      "task_id": mission.task_id, "ack_id": mission.ack_id, "lease_id": mission.lease_id,
                      "project_hash": mission.project_hash, "commit": sha, "workspace": mission.workspace, "observed_at": int(time.time())})
        return {"commit": sha, "evidence_only": True}

    def cancel(self, mission_id: str, *, operator_authorization: object) -> MissionResult:
        # Only the host's authenticated operator transport can produce this
        # opaque invocation object; actor labels and identity strings are not proof.
        if self._operator_verifier is None or not self._operator_verifier(operator_authorization): return MissionResult("REJECTED", "operator authentication required", mission_id)
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); mission = self._mission(mission_id)
            if any(x.get("state") in {"CANCELLED", "COMPLETED"} for x in self._history()[mission_id]): return MissionResult("NOOP", "mission already terminal", mission_id, mission.workspace)
            if self._trust_broker is None: return MissionResult("REJECTED", "host trust broker is required", mission_id)
            self._trust_broker.revoke_mission(mission_id, task_id=mission.task_id,
                                              dispatcher_ack_id=mission.ack_id, lease_id=mission.lease_id)
            self._append({"event_type": "JAXQWEN_MISSION", "state": "CANCELLED", "mission_id": mission_id,
                          "task_id": mission.task_id, "ack_id": mission.ack_id, "lease_id": mission.lease_id,
                          "project_hash": mission.project_hash, "workspace": mission.workspace, "observed_at": int(time.time())})
            return MissionResult("CANCELLED", "operator cancellation recorded", mission_id, mission.workspace)
