"""Deterministic local host for the proposed Ariadna PM loop.

This module has no daemon, network endpoint, shell runner, or authority
policy.  A host supplies data-only proposals; LV-003 remains the sole
authorization and canonical-transition boundary.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import stat
import threading
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

PROJECT_ID = "las-voces"
ARIADNA_ID = "ariadna-project-manager"

class Lifecycle(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"

@dataclass(frozen=True)
class Proposal:
    task_id: str
    action: str
    expected_project_hash: str
    target_status: str | None = None
    evidence_refs: tuple[str, ...] = ()
    handoff: dict[str, Any] | None = None

@dataclass(frozen=True)
class TaskLease:
    task_id: str
    owner: str
    worktree: str
    writable_scope: str
    lease_id: str
    runtime_instance_id: str

def project_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _state_dir(root: Path) -> Path:
    """Git-common control plane when available; a test-only local fallback."""
    try:
        common = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--git-common-dir"], text=True, stderr=subprocess.DEVNULL).strip()
        base = (root / common).resolve() if not Path(common).is_absolute() else Path(common)
    except (OSError, subprocess.CalledProcessError):
        base = root / ".ariadna-local-state"
    path = base / "ariadna-pm-control"; path.mkdir(parents=True, exist_ok=True)
    return path

def _safe_open(path: Path, flags: int) -> int:
    """Refuse symlink persistence targets; an unsafe runtime must fail closed."""
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd); raise RuntimeError("runtime state target is not a regular file")
    return fd

class LocalControlLease:
    """An honest local-only OS lock: crash releases flock, metadata exposes it."""
    def __init__(self, root: Path, instance_id: str):
        self.path = _state_dir(root) / "control.lock"
        self.instance_id, self.fd, self.previous_stale = instance_id, None, False
    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = _safe_open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd); return False
        try:
            prior = json.loads(os.read(fd, 65536) or b"{}")
            self.previous_stale = isinstance(prior, dict) and prior.get("state") == "ACTIVE" and isinstance(prior.get("pid"), int) and not _pid_alive(prior["pid"])
        except json.JSONDecodeError:
            self.previous_stale = True
        os.lseek(fd, 0, os.SEEK_SET); os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"instance_id": self.instance_id, "pid": os.getpid(), "state": "ACTIVE"}).encode()); os.fsync(fd)
        self.fd = fd; return True
    def release(self) -> None:
        if self.fd is not None:
            os.lseek(self.fd, 0, os.SEEK_SET); os.ftruncate(self.fd, 0)
            os.write(self.fd, json.dumps({"instance_id": self.instance_id, "pid": os.getpid(), "state": "RELEASED"}).encode()); os.fsync(self.fd)
            fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd); self.fd = None

def _pid_alive(pid: int) -> bool:
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True

class TaskLeaseRegistry:
    """Append-only, local serialized leases for task and writable boundaries."""
    def __init__(self, root: Path):
        self.path = _state_dir(root) / "task-leases.ndjson"
        self.lock = _state_dir(root) / "task-leases.lock"
    def _active(self) -> dict[str, TaskLease]:
        out: dict[str, TaskLease] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] == "ACQUIRED": out[event["lease"]["lease_id"]] = TaskLease(**event["lease"])
                elif event["event"] == "RELEASED": out.pop(event["lease_id"], None)
        return out
    @staticmethod
    def _overlap(a: str, b: str) -> bool:
        x, y = Path(a).parts, Path(b).parts
        return x == y or x[:len(y)] == y or y[:len(x)] == x
    @staticmethod
    def _scope(value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts: raise ValueError("scope must be a nonempty relative boundary")
        return str(path)
    def acquire(self, lease: TaskLease) -> bool:
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            lease = TaskLease(lease.task_id, lease.owner, self._scope(lease.worktree), self._scope(lease.writable_scope), lease.lease_id, lease.runtime_instance_id)
            active = self._active().values()
            if any(x.task_id == lease.task_id or x.worktree == lease.worktree or self._overlap(x.writable_scope, lease.writable_scope) for x in active): return False
            self._append({"event": "ACQUIRED", "lease": asdict(lease)}); return True
    def release(self, lease: TaskLease) -> bool:
        with self.lock.open("a+") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            active = self._active().get(lease.lease_id)
            if active != lease: return False
            self._append({"event": "RELEASED", "lease_id": lease.lease_id}); return True
    def owns(self, lease_id: str | None, task_id: str, instance_id: str) -> bool:
        if lease_id is None: return False
        lease = self._active().get(lease_id)
        return lease is not None and lease.task_id == task_id and lease.runtime_instance_id == instance_id
    def _append(self, value: dict[str, Any]) -> None:
        with self.path.open("a") as log:
            log.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"); log.flush(); os.fsync(log.fileno())

class AriadnaRuntime:
    """Explicit tick host.  It never executes arbitrary commands."""
    def __init__(self, root: Path, authority_engine: Any, *, instance_id: str | None = None):
        self.root, self.engine = root, authority_engine
        self.instance_id = instance_id or str(uuid.uuid4())
        self.lifecycle = Lifecycle.NOT_STARTED
        self.control, self.leases = LocalControlLease(root, self.instance_id), TaskLeaseRegistry(root)
        self.audit_path = _state_dir(root) / "runtime-audit.ndjson"
        self.outbox_path = _state_dir(root) / "handoffs.ndjson"
        self._mutex, self._owned_leases, self._stop = threading.RLock(), {}, threading.Event()
    def start(self) -> Lifecycle:
        if self.lifecycle is not Lifecycle.NOT_STARTED: return self.lifecycle
        if not self.control.acquire():
            self.lifecycle = Lifecycle.STOPPED; return self.lifecycle
        # The supplied LV-003 engine exposes the authoritative journal state.
        if self.engine.interrupted_transitions():
            self.lifecycle = Lifecycle.RECONCILIATION_REQUIRED
        else: self.lifecycle = Lifecycle.READY
        return self.lifecycle
    def request_stop(self) -> None:
        self._stop.set()
        with self._mutex:
            if self.lifecycle in {Lifecycle.READY, Lifecycle.RECONCILIATION_REQUIRED}: self.lifecycle = Lifecycle.STOPPING
    def shutdown(self) -> Lifecycle:
        with self._mutex:
            self.request_stop()
            for lease in tuple(self._owned_leases.values()): self.leases.release(lease)
            self._owned_leases.clear(); self.control.release(); self.lifecycle = Lifecycle.STOPPED; return self.lifecycle
    def readiness(self) -> Lifecycle: return self.lifecycle
    def health(self) -> dict[str, Any]:
        return {"lifecycle": self.lifecycle.value, "instance_id": self.instance_id, "local_only": True, "stale_owner_detected": self.control.previous_stale}
    def acquire_task(self, lease: TaskLease) -> bool:
        with self._mutex:
            if self.lifecycle is not Lifecycle.READY or self._stop.is_set() or lease.runtime_instance_id != self.instance_id: return False
            try:
                project = json.loads((self.root / "projects/las-voces/project.json").read_text())
                if project.get("project", {}).get("id") != PROJECT_ID or not lease.owner or len([x for x in project.get("tasks", []) if x.get("id") == lease.task_id]) != 1: return False
            except (OSError, json.JSONDecodeError):
                return False
            if self.leases.acquire(lease): self._owned_leases[lease.lease_id] = lease; return True
            return False
    def release_task(self, lease: TaskLease) -> bool:
        with self._mutex:
            if self._owned_leases.get(lease.lease_id) != lease: return False
            if self.leases.release(lease): self._owned_leases.pop(lease.lease_id, None); return True
            return False
    def _audit(self, record: dict[str, Any]) -> None:
        key = record["idempotency_key"]
        if self.audit_path.exists() and any(json.loads(line).get("idempotency_key") == key for line in self.audit_path.read_text().splitlines() if line): return
        with self.audit_path.open("a") as log:
            log.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"); log.flush(); os.fsync(log.fileno())
    def run_once(self, proposal: Proposal, *, lease_id: str | None = None) -> str:
      with self._mutex:
        if self.lifecycle is not Lifecycle.READY or self._stop.is_set(): return "NOOP_NOT_READY"
        if self.engine.interrupted_transitions():
            self.lifecycle = Lifecycle.RECONCILIATION_REQUIRED
            return "NOOP_NOT_READY"
        state = self.root / "projects/las-voces/project.json"; observed_hash = project_hash(state)
        project = json.loads(state.read_text()); tasks = [x for x in project.get("tasks", []) if x.get("id") == proposal.task_id]
        if project.get("project", {}).get("id") != PROJECT_ID or len(tasks) != 1 or proposal.expected_project_hash != observed_hash: return "NOOP_STALE_OR_INVALID"
        if proposal.action in {"coordinate_verified_work", "emit_handoff"} and not self.leases.owns(lease_id, proposal.task_id, self.instance_id): return "NOOP_UNOWNED_LEASE"
        try:
            evidence_binding = {}
            for ref in proposal.evidence_refs:
                if ref.startswith("commit:"):
                    evidence_binding[ref] = hashlib.sha256(ref.encode()).hexdigest(); continue
                if not ref or ref.startswith(("/", "~")): raise ValueError("unsafe evidence reference")
                path = (self.root / ref).resolve()
                path.relative_to(self.root.resolve())
                if path.is_symlink() or not path.is_file(): raise ValueError("unsafe evidence reference")
                evidence_binding[ref] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError):
            return "NOOP_INVALID_EVIDENCE"
        material = {"hash": observed_hash, "task": proposal.task_id, "action": proposal.action, "target": proposal.target_status, "evidence": evidence_binding, "handoff": proposal.handoff, "lease": lease_id}
        key = hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()
        if self.audit_path.exists() and any(json.loads(line).get("idempotency_key") == key for line in self.audit_path.read_text().splitlines() if line): return "NOOP_IDEMPOTENT"
        decision = self.engine.evaluate(sender_agent=ARIADNA_ID, task_id=proposal.task_id, action=proposal.action, target_status=proposal.target_status, evidence_refs=list(proposal.evidence_refs), handoff=proposal.handoff)
        verdict = decision.verdict.value
        result = "NOOP_" + verdict
        if self._stop.is_set(): return "NOOP_STOPPING"
        if verdict == "ALLOW" and proposal.action == "transition_status":
            final = self.engine.transition(sender_agent=ARIADNA_ID, task_id=proposal.task_id, target_status=proposal.target_status, evidence_refs=list(proposal.evidence_refs), expected_project_hash=observed_hash)
            verdict = final.verdict.value
            result = "EFFECT_ALLOW" if verdict == "ALLOW" else "NOOP_" + verdict
        elif verdict == "ALLOW" and proposal.action in {"coordinate_verified_work", "emit_handoff"}:
            if project_hash(state) != observed_hash: return "NOOP_STALE_OR_INVALID"
            with self.outbox_path.open("a") as outbox:
                outbox.write(json.dumps({"event_type": proposal.action, "idempotency_key": key, "task_id": proposal.task_id, "handoff": proposal.handoff, "lease_id": lease_id}, sort_keys=True) + "\n"); outbox.flush(); os.fsync(outbox.fileno())
            result = "EFFECT_" + proposal.action.upper()
        self._audit({"event_type": "ARIADNA_CYCLE", "runtime_instance_id": self.instance_id, "cycle_id": key, "idempotency_key": key, "project_hash": observed_hash, "task_id": proposal.task_id, "action": proposal.action, "verdict": verdict, "result": result, "lease_id": lease_id, "evidence_refs": list(proposal.evidence_refs)})
        return result
