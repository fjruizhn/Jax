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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    def __init__(self, root: Path, handoff_state_dir: Path, workspace_root: Path):
        self.root, self.handoff_state_dir, self.workspace_root = root.resolve(), handoff_state_dir.resolve(), workspace_root.resolve()
        if not (self.root / "projects/las-voces/project.json").is_file():
            raise CapabilityError("invalid canonical checkout")
        if str(self.root).startswith("/srv/jax-prod") or str(self.workspace_root).startswith("/srv/jax-prod"):
            raise CapabilityError("production checkout cannot host jaxqwen mission")
        common = Path(_git(self.root, "rev-parse", "--git-common-dir"))
        common = (self.root / common).resolve() if not common.is_absolute() else common.resolve()
        self.dispatch_dir = common / "ariadna-builder-dispatch"
        self.ledger_dir = common / "jaxqwen-capability"
        self.ledger, self.lock_path = self.ledger_dir / "missions.ndjson", self.ledger_dir / "missions.lock"

    @staticmethod
    def allowed_action(action: str) -> bool:
        return action == CAPABILITY

    def _locked(self):
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd); raise CapabilityError("unsafe mission lock")
        return os.fdopen(fd, "r+")

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

    def _validate_current(self, ack: dict[str, Any], row: dict[str, Any]) -> None:
        if _project_hash(self.root) != ack["project_hash"]: raise CapabilityError("ACK is stale against canonical project")
        if _git(self.root, "rev-parse", "HEAD") != ack["base_commit"]: raise CapabilityError("ACK source revision is stale")
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
        scope = row["handoff"].get("authority_context", {}).get("handoff", {}).get("scope")
        if not isinstance(scope, str) or not scope.startswith(f"LAS VOCES {ack['task_id']}:"):
            raise CapabilityError("handoff scope is not bounded")

    def _workspace(self, mission_id: str, ack: dict[str, Any]) -> Path:
        # The ACK worktree may share metadata; clone it into per-mission storage.
        source, target = Path(ack["worktree"]), self.workspace_root / ack["task_id"].lower() / mission_id
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

    def start(self, *, machine_identity: str, idempotency_key: str) -> MissionResult:
        if machine_identity != SERVICE_IDENTITY: return MissionResult("REJECTED", "unauthenticated machine identity")
        if not isinstance(idempotency_key, str) or not _KEY.fullmatch(idempotency_key): return MissionResult("REJECTED", "invalid handoff key")
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                ack = self._ack(idempotency_key); row = self._handoff(idempotency_key, ack); self._validate_current(ack, row)
                mission_id = _digest({"capability": CAPABILITY, "ack": idempotency_key, "lease": ack["lease_id"], "revision": ack["base_commit"]})
                old = self._history().get(mission_id, [])
                if old: return MissionResult("NOOP", "mission already recorded", mission_id, old[0].get("workspace"))
                workspace = self._workspace(mission_id, ack)
                mission = Mission(mission_id, PROJECT_ID, ack["task_id"], ack["canonical_owner"], ack["builder_identity"], idempotency_key, ack["lease_id"], ack["lease_id"], ack["project_hash"], ack["base_commit"], ack["branch"], str(workspace), ".", tuple(sorted(_TOOLS)), ("pytest_las_voces",), tuple(row["handoff"].get("acceptance_criteria", [])), tuple(row["handoff"].get("evidence_requirements", [])))
                self._append({"event_type": "JAXQWEN_MISSION", "state": "ACCEPTED", "mission_id": mission_id, "capability": CAPABILITY, "machine_identity": SERVICE_IDENTITY, "workspace": str(workspace), "mission": asdict(mission)})
                return MissionResult("ACCEPTED", "bounded mission prepared; model transport not started", mission_id, str(workspace))
            except (CapabilityError, KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
                return MissionResult("REJECTED", str(exc))

    def _mission(self, mission_id: str) -> Mission:
        rows = self._history().get(mission_id, [])
        if not rows or not isinstance(rows[0].get("mission"), dict): raise CapabilityError("unknown mission")
        return Mission(**rows[0]["mission"])

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

    def tool(self, mission_id: str, name: str, **request: Any) -> dict[str, Any]:
        mission = self._mission(mission_id)
        if name not in _TOOLS: raise CapabilityError("tool is not allowlisted")
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
            done = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=mission.workspace, text=True, capture_output=True, env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}, timeout=120)
            return {"test": test, "returncode": done.returncode, "stdout": done.stdout[-4000:], "stderr": done.stderr[-4000:]}
        if name == "request_commit": return self.commit(mission_id, request.get("message"))
        raise CapabilityError("unreachable tool")

    def commit(self, mission_id: str, message: Any) -> dict[str, Any]:
        mission = self._mission(mission_id)
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
        self._append({"event_type": "JAXQWEN_MISSION", "state": "COMPLETED", "mission_id": mission_id, "commit": sha, "task_id": mission.task_id, "lease_id": mission.lease_id, "workspace": mission.workspace})
        return {"commit": sha, "evidence_only": True}

    def cancel(self, mission_id: str, *, operator_identity: str) -> MissionResult:
        if operator_identity not in {"operator", "host"}: return MissionResult("REJECTED", "operator authentication required", mission_id)
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); mission = self._mission(mission_id)
            if any(x.get("state") in {"CANCELLED", "COMPLETED"} for x in self._history()[mission_id]): return MissionResult("NOOP", "mission already terminal", mission_id, mission.workspace)
            self._append({"event_type": "JAXQWEN_MISSION", "state": "CANCELLED", "mission_id": mission_id, "task_id": mission.task_id, "lease_id": mission.lease_id, "workspace": mission.workspace})
            return MissionResult("CANCELLED", "operator cancellation recorded", mission_id, mission.workspace)
