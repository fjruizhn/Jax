"""Fail-closed consumer for governed Ariadna builder handoffs.

This is a privileged *dispatcher* boundary, deliberately separate from the
Ariadna PM runtime.  It has one narrow effect: after independently validating
an audited handoff and its live task lease, it may create the declared Git
worktree.  It never invokes a builder, a shell, a deployer, or an authority
engine supplied by a handoff.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ID = "las-voces"
ARIADNA_ID = "ariadna-project-manager"
FORBIDDEN_ACTIONS = frozenset({
    "merge", "deploy", "production_mutation", "capability_grant",
    "authority_change", "authority_contract_edit", "verifier_replacement",
    "verifier_configuration_change", "policy_gate_disable", "self_activation",
    "self_approval", "bypass_human_required", "arbitrary_command_execution",
    "shell_execution", "runtime_execution", "builder_execution",
})
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
_KEY = re.compile(r"^[0-9a-f]{64}$")
_OUTBOX_KEYS = {"event_type", "idempotency_key", "task_id", "handoff", "lease_id"}
_GIT = "/usr/bin/git"


class DispatchError(ValueError):
    """A handoff cannot safely become a builder assignment."""


@dataclass(frozen=True)
class DispatchResult:
    decision: str
    reason: str
    idempotency_key: str | None = None
    task_id: str | None = None
    worktree: str | None = None


def _load_module(name: str, source: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise DispatchError("cannot load governed authority contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _project_hash(root: Path) -> str:
    return hashlib.sha256((root / "projects/las-voces/project.json").read_bytes()).hexdigest()


def _git_common_dir(root: Path) -> Path:
    common = Path(_git(root, "rev-parse", "--git-common-dir"))
    return ((root / common).resolve() if not common.is_absolute() else common.resolve())


def _git(root: Path, *args: str) -> str:
    result = subprocess.run([_GIT, "-C", str(root), *args], text=True, capture_output=True, env=_git_env())
    if result.returncode:
        raise DispatchError("git worktree operation rejected")
    return result.stdout.strip()


def _git_optional(root: Path, *args: str) -> str | None:
    result = subprocess.run([_GIT, "-C", str(root), *args], text=True, capture_output=True, env=_git_env())
    return result.stdout.strip() if not result.returncode else None


def _git_env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_OPTIONAL_LOCKS": "0"}


def _safe_file(path: Path, *, required: bool = True) -> Path:
    if path.is_symlink() or (required and (not path.exists() or not path.is_file())):
        raise DispatchError("unsafe or missing dispatcher state file")
    if path.exists() and not stat.S_ISREG(path.stat().st_mode):
        raise DispatchError("dispatcher state is not a regular file")
    return path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GovernedHandoffConsumer:
    """Consume one append-only handoff at most once under a local OS lock.

    ``root`` is the development source checkout used to create an isolated
    builder worktree.  ``handoff_state_dir`` is a read-only producer control
    plane (which can be a separately hosted Ariadna runtime checkout).  ACKs
    are written only below this dispatcher's Git common directory.
    """
    def __init__(self, root: Path, handoff_state_dir: Path, worktree_root: Path, *, canonical_root: Path | None = None):
        self.root = root.resolve()
        self.canonical_root = (canonical_root or root).resolve()
        self.handoff_state_dir = handoff_state_dir.resolve()
        self.worktree_root = worktree_root.resolve()
        if not (self.root / "projects/las-voces/project.json").is_file() or not (self.canonical_root / "projects/las-voces/project.json").is_file():
            raise DispatchError("invalid dispatcher or canonical source checkout")
        self._assert_safe_git_environment(self.root)
        if self.canonical_root != self.root:
            self._assert_safe_git_environment(self.canonical_root)
        if _project_hash(self.root) != _project_hash(self.canonical_root):
            raise DispatchError("builder checkout and canonical producer state differ")
        if _git(self.root, "rev-parse", "HEAD") != _git(self.canonical_root, "rev-parse", "HEAD"):
            raise DispatchError("builder checkout and canonical producer revision differ")
        self.base_commit = _git(self.root, "rev-parse", "HEAD")
        self.canonical_commit = _git(self.canonical_root, "rev-parse", "HEAD")
        if _git(self.root, "status", "--porcelain") or _git(self.canonical_root, "status", "--porcelain"):
            raise DispatchError("dispatcher source checkout is not clean")
        if self.root != self.canonical_root:
            left, right = _git_optional(self.root, "remote", "get-url", "origin"), _git_optional(self.canonical_root, "remote", "get-url", "origin")
            if not left or left != right:
                raise DispatchError("builder and canonical checkout repository identities differ")
        common = _git(self.canonical_root, "rev-parse", "--git-common-dir")
        expected_state = ((self.canonical_root / common).resolve() if not Path(common).is_absolute() else Path(common).resolve()) / "ariadna-pm-control"
        if self.handoff_state_dir != expected_state:
            raise DispatchError("handoff state is not the canonical runtime control plane")
        if self.handoff_state_dir == self.worktree_root or self.handoff_state_dir in self.worktree_root.parents:
            raise DispatchError("worktree root cannot contain producer control state")
        common = _git(self.root, "rev-parse", "--git-common-dir")
        self.state_dir = ((self.root / common).resolve() if not Path(common).is_absolute() else Path(common)) / "ariadna-builder-dispatch"
        self.ack_path, self.lock_path = self.state_dir / "acks.ndjson", self.state_dir / "dispatch.lock"
        self.authority = _load_module("ariadna_dispatch_authority", self.canonical_root / "projects/las-voces/authority/ariadna_authority.py")

    @staticmethod
    def action_allowed(action: str) -> bool:
        """The dispatcher exposes only worktree preparation, never execution."""
        return action == "create_isolated_builder_worktree" and action not in FORBIDDEN_ACTIONS

    @staticmethod
    def _assert_safe_git_environment(root: Path) -> None:
        """Reject Git configuration/attributes that can execute helper commands."""
        unsafe = _git_optional(root, "config", "--local", "--get-regexp", r"^(filter\.|core\.fsmonitor$|diff\.external$)")
        if unsafe:
            raise DispatchError("builder repository has an external Git execution driver")
        tracked = _git(root, "ls-files", "-z").split("\0")
        for rel in tracked:
            if rel.endswith(".gitattributes"):
                path = root / rel
                if path.is_symlink() or "filter=" in path.read_text(encoding="utf-8", errors="strict"):
                    raise DispatchError("builder repository declares a Git filter attribute")

    def _append_ack(self, row: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _safe_file(self.ack_path, required=False)
        fd = os.open(self.ack_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise DispatchError("unsafe acknowledgement target")
            os.write(fd, (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def _acks(self) -> dict[str, list[dict[str, Any]]]:
        if not self.ack_path.exists():
            return {}
        _safe_file(self.ack_path)
        records: dict[str, list[dict[str, Any]]] = {}
        try:
            for raw in self.ack_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                key = row.get("idempotency_key") if isinstance(row, dict) else None
                if not isinstance(key, str) or not _KEY.fullmatch(key):
                    raise DispatchError("malformed acknowledgement history")
                records.setdefault(key, []).append(row)
        except json.JSONDecodeError as exc:
            raise DispatchError("malformed acknowledgement history") from exc
        return records

    @staticmethod
    def _active_leases(path: Path) -> dict[str, dict[str, Any]]:
        _safe_file(path)
        active: dict[str, dict[str, Any]] = {}
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise DispatchError("malformed task lease record")
                if row.get("event") == "ACQUIRED" and isinstance(row.get("lease"), dict):
                    lease = row["lease"]
                    identifier = lease.get("lease_id")
                    if not isinstance(identifier, str):
                        raise DispatchError("malformed task lease identifier")
                    active[identifier] = lease
                elif row.get("event") == "RELEASED" and isinstance(row.get("lease_id"), str):
                    active.pop(row["lease_id"], None)
                else:
                    raise DispatchError("unknown task lease record")
        except json.JSONDecodeError as exc:
            raise DispatchError("malformed task lease record") from exc
        return active

    def _audit_for(self, key: str, task_id: str, lease_id: str) -> dict[str, Any]:
        path = _safe_file(self.handoff_state_dir / "runtime-audit.ndjson")
        try:
            matches = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()
                       if raw and json.loads(raw).get("idempotency_key") == key]
        except json.JSONDecodeError as exc:
            raise DispatchError("malformed runtime audit") from exc
        if len(matches) != 1:
            raise DispatchError("handoff has no unique authoritative audit")
        audit = matches[0]
        required = {"event_type": "ARIADNA_CYCLE", "task_id": task_id, "action": "emit_handoff", "verdict": "ALLOW", "result": "EFFECT_EMIT_HANDOFF", "lease_id": lease_id}
        if any(audit.get(name) != value for name, value in required.items()) or not isinstance(audit.get("project_hash"), str):
            raise DispatchError("handoff audit is not an allowed Ariadna effect")
        if not isinstance(audit.get("runtime_instance_id"), str) or not audit["runtime_instance_id"]:
            raise DispatchError("handoff audit has no runtime identity")
        return audit

    def _runtime_control_is_live(self, lease: dict[str, Any]) -> bool:
        path = _safe_file(self.handoff_state_dir / "control.lock")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("state") != "ACTIVE" or value.get("instance_id") != lease.get("runtime_instance_id") or value.get("pid") != lease.get("pid"):
                return False
            fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            finally:
                os.close(fd)
        except (OSError, json.JSONDecodeError, TypeError):
            return False

    def _record(self, key: str) -> dict[str, Any] | None:
        path = _safe_file(self.handoff_state_dir / "handoffs.ndjson")
        try:
            matches = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()
                       if raw and json.loads(raw).get("idempotency_key") == key]
        except json.JSONDecodeError as exc:
            raise DispatchError("malformed handoff outbox") from exc
        if len(matches) != 1:
            raise DispatchError("handoff key is not unique")
        row = matches[0]
        if not isinstance(row, dict) or set(row) != _OUTBOX_KEYS or row.get("event_type") != "emit_handoff":
            raise DispatchError("unsupported handoff record")
        if row.get("idempotency_key") != key or not _KEY.fullmatch(key):
            raise DispatchError("invalid handoff idempotency key")
        if not isinstance(row.get("task_id"), str) or not _TASK_ID.fullmatch(row["task_id"]):
            raise DispatchError("unsafe task id")
        if not isinstance(row.get("lease_id"), str) or not isinstance(row.get("handoff"), dict):
            raise DispatchError("malformed handoff record")
        return row

    @staticmethod
    def _runtime_key(row: dict[str, Any], observed_hash: str) -> str:
        handoff = {name: value for name, value in row["handoff"].items() if name != "created_at"}
        material = {"hash": observed_hash, "task": row["task_id"], "action": "emit_handoff", "target": None,
                    "evidence": {}, "handoff": handoff, "lease": row["lease_id"]}
        return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()

    def _canonical_builder(self, task_id: str, envelope: dict[str, Any]) -> tuple[str, str, str]:
        self.authority.validate_handoff(self.canonical_root, envelope, expected_task_id=task_id)
        project = json.loads((self.canonical_root / "projects/las-voces/project.json").read_text(encoding="utf-8"))
        matches = [task for task in project.get("tasks", []) if isinstance(task, dict) and task.get("id") == task_id]
        if len(matches) != 1 or matches[0].get("status") != "READY":
            raise DispatchError("task is no longer dispatchable")
        dependencies = matches[0].get("depends", [])
        by_id = {task.get("id"): task for task in project.get("tasks", []) if isinstance(task, dict)}
        if not isinstance(dependencies, list) or any(not isinstance(dep, str) or by_id.get(dep, {}).get("status") != "DONE" for dep in dependencies):
            raise DispatchError("task dependencies are not currently satisfied")
        if matches[0].get("blocker") or matches[0].get("blockers"):
            raise DispatchError("task has an unresolved canonical blocker")
        owner = matches[0].get("owner")
        if not isinstance(owner, str) or not owner:
            raise DispatchError("canonical task owner is invalid")
        identity = owner.split("/", 1)[0]
        agents = [agent for agent in project.get("agents", []) if isinstance(agent, dict) and agent.get("name") == identity]
        if len(agents) != 1 or "BUILDER" not in str(agents[0].get("role", "")).upper():
            raise DispatchError("canonical owner has no unique supported builder identity")
        branch = f"las-voces/{task_id.lower()}"
        declared = f"branch:{branch}; worktree:worktrees/las-voces-{task_id.lower()}"
        handoff = envelope["authority_context"]["handoff"]
        if handoff.get("branch_worktree") != declared:
            raise DispatchError("handoff worktree boundary differs from canonical convention")
        # Preserve the owner *and* its canonical builder registry identity.
        # Neither identity authorizes a builder process.
        return owner, identity, branch

    def _ack(self, state: str, *, key: str, task_id: str, lease_id: str | None = None, owner: str | None = None, builder: str | None = None,
             branch: str | None = None, worktree: Path | None = None, project_hash: str | None = None, reason: str | None = None) -> None:
        row = {"event_type": "ARIADNA_HANDOFF_ACK", "state": state, "idempotency_key": key,
               "task_id": task_id, "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
        for name, value in {"lease_id": lease_id, "canonical_owner": owner, "builder_identity": builder, "branch": branch,
                            "worktree": str(worktree) if worktree else None, "project_hash": project_hash, "base_commit": self.base_commit if owner else None, "builder_process_started": False if owner else None, "reason": reason}.items():
            if value is not None: row[name] = value
        self._append_ack(row)

    def _worktree_valid(self, path: Path, branch: str) -> bool:
        try:
            registered = {line.removeprefix("worktree ") for line in _git(self.root, "worktree", "list", "--porcelain").splitlines() if line.startswith("worktree ")}
            return path.is_dir() and not path.is_symlink() and str(path.resolve()) in registered and _git(path, "rev-parse", "--show-toplevel") == str(path.resolve()) and _git_common_dir(path) == _git_common_dir(self.root) and _git(path, "rev-parse", "HEAD") == self.base_commit and _git(path, "branch", "--show-current") == branch and not _git(path, "status", "--porcelain")
        except DispatchError:
            return False

    def _create_worktree(self, path: Path, branch: str) -> None:
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        if self.worktree_root.is_symlink() or not self.worktree_root.is_dir():
            raise DispatchError("unsafe builder worktree root")
        if path.exists() or path.is_symlink():
            raise DispatchError("conflicting builder worktree already exists")
        if _git(self.root, "branch", "--list", branch):
            raise DispatchError("conflicting builder branch already exists")
        self._assert_safe_git_environment(self.root)
        result = subprocess.run([_GIT, "-C", str(self.root), "-c", "core.hooksPath=/dev/null", "worktree", "add", "-b", branch, str(path), self.base_commit], text=True, capture_output=True, env=_git_env())
        if result.returncode or not self._worktree_valid(path, branch):
            raise DispatchError("isolated builder worktree creation failed")

    def _locked(self):
        """Open the mutex without following a path substituted after validation."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd); raise DispatchError("unsafe dispatcher lock target")
        return os.fdopen(fd, "r+")

    def consume(self, key: str) -> DispatchResult:
        """Validate and acknowledge exactly one requested outbox record."""
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            return DispatchResult("REJECTED", "invalid idempotency key")
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            acks = self._acks().get(key, [])
            terminal = {row.get("state") for row in acks}
            if "DISPATCHED" in terminal:
                return DispatchResult("NOOP", "already dispatched", key)
            if "REJECTED" in terminal:
                return DispatchResult("NOOP", "previously rejected", key)
            task_id = "UNKNOWN"
            try:
                row = self._record(key); assert row is not None
                task_id = row["task_id"]
                if row["handoff"].get("message_id") != row["lease_id"] or row["handoff"].get("correlation_id") != row["lease_id"] or row["handoff"].get("intent") != "governed_task_handoff" or row["handoff"].get("status") != "RECORDED":
                    raise DispatchError("handoff correlation or transport status is invalid")
                audit = self._audit_for(key, task_id, row["lease_id"])
                observed = _project_hash(self.canonical_root)
                if audit["project_hash"] != observed:
                    raise DispatchError("handoff is stale against current canonical project")
                if self._runtime_key(row, observed) != key:
                    raise DispatchError("handoff content is not bound to Ariadna audit identity")
                owner, builder, branch = self._canonical_builder(task_id, row["handoff"])
                engine = self.authority.AuthorityEngine(self.canonical_root)
                if not engine.activation_approved():
                    raise DispatchError("Ariadna is not canonically ACTIVE_GOVERNED")
                decision = engine.evaluate(sender_agent=ARIADNA_ID, task_id=task_id, action="emit_handoff", handoff=row["handoff"])
                if decision.verdict.value != "ALLOW":
                    raise DispatchError("current AuthorityEngine rejected handoff")
                leases = self._active_leases(self.handoff_state_dir / "task-leases.ndjson")
                lease = leases.get(row["lease_id"])
                expected_worktree = f"worktrees/las-voces-{task_id.lower()}"
                if not lease or lease.get("task_id") != task_id or lease.get("owner") != owner or lease.get("worktree") != expected_worktree or lease.get("writable_scope") != ".":
                    raise DispatchError("handoff lease is absent, released, or inconsistent")
                if lease.get("runtime_instance_id") != audit["runtime_instance_id"] or not isinstance(lease.get("pid"), int) or lease["pid"] <= 0 or not _pid_alive(lease["pid"]):
                    raise DispatchError("handoff lease is not bound to the audited runtime")
                if not self._runtime_control_is_live(lease):
                    raise DispatchError("handoff runtime control lease is not live")
                active = self._active_leases(self.handoff_state_dir / "task-leases.ndjson").values()
                if any(other.get("lease_id") != row["lease_id"] and (other.get("task_id") == task_id or other.get("worktree") == expected_worktree or other.get("writable_scope") == ".") for other in active):
                    raise DispatchError("conflicting live task lease")
                target = self.worktree_root / f"las-voces-{task_id.lower()}"
                try: target.relative_to(self.worktree_root)
                except ValueError as exc: raise DispatchError("builder worktree escapes configured root") from exc
                accepted = "ACCEPTED" in terminal
                expected_ack = {"lease_id": row["lease_id"], "canonical_owner": owner, "builder_identity": builder,
                                "branch": branch, "worktree": str(target), "project_hash": observed,
                                "base_commit": self.base_commit, "builder_process_started": False}
                for prior in acks:
                    if prior.get("state") in {"RECEIVED", "ACCEPTED", "DISPATCHED"} and any(prior.get(name) != value for name, value in expected_ack.items()):
                        raise DispatchError("acknowledgement does not bind this handoff")
                if target.exists() and not accepted:
                    raise DispatchError("unacknowledged builder worktree already exists")
                if not acks:
                    self._ack("RECEIVED", key=key, task_id=task_id, lease_id=row["lease_id"], owner=owner, builder=builder, branch=branch, worktree=target, project_hash=observed)
                    self._ack("ACCEPTED", key=key, task_id=task_id, lease_id=row["lease_id"], owner=owner, builder=builder, branch=branch, worktree=target, project_hash=observed)
                elif not accepted:
                    self._ack("ACCEPTED", key=key, task_id=task_id, lease_id=row["lease_id"], owner=owner, builder=builder, branch=branch, worktree=target, project_hash=observed)
                if _project_hash(self.canonical_root) != observed or _project_hash(self.root) != observed or _git(self.root, "rev-parse", "HEAD") != self.base_commit or _git(self.canonical_root, "rev-parse", "HEAD") != self.canonical_commit or _git(self.root, "status", "--porcelain") or _git(self.canonical_root, "status", "--porcelain"):
                    raise DispatchError("canonical project changed before worktree effect")
                if self._active_leases(self.handoff_state_dir / "task-leases.ndjson").get(row["lease_id"]) != lease:
                    raise DispatchError("handoff lease changed before worktree effect")
                if self._worktree_valid(target, branch):
                    self._ack("DISPATCHED", key=key, task_id=task_id, lease_id=row["lease_id"], owner=owner, builder=builder, branch=branch, worktree=target, project_hash=observed, reason="worktree prepared; builder execution intentionally not started")
                    return DispatchResult("DISPATCHED", "worktree prepared; no builder execution", key, task_id, str(target))
                self._create_worktree(target, branch)
                self._ack("DISPATCHED", key=key, task_id=task_id, lease_id=row["lease_id"], owner=owner, builder=builder, branch=branch, worktree=target, project_hash=observed, reason="worktree prepared; builder execution intentionally not started")
                return DispatchResult("DISPATCHED", "worktree prepared; no builder execution", key, task_id, str(target))
            except (DispatchError, ValueError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                self._ack("REJECTED", key=key, task_id=task_id, reason=str(exc))
                return DispatchResult("REJECTED", str(exc), key, task_id)
