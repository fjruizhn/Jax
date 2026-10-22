"""Deterministic, read-only task proposal source for the Ariadna host.

This is deliberately not an LLM and does not own an AuthorityEngine, a
filesystem effect, or a builder process.  It reads the canonical LAS VOCES
backlog and returns data for the privileged host to validate, lease and submit
through the existing runtime contract.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ID = "las-voces"
ARIADNA_ID = "ariadna-project-manager"


@dataclass(frozen=True)
class PlannedHandoff:
    """A data-only request; the host alone may turn it into an effect."""

    task_id: str
    owner: str
    expected_project_hash: str
    worktree: str
    writable_scope: str
    lease_id: str
    envelope: dict[str, Any]


@dataclass(frozen=True)
class PlanningResult:
    project_hash: str
    eligible_count: int
    selected_task_id: str | None
    reason: str
    handoff: PlannedHandoff | None = None


def _project_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _priority(value: Any) -> tuple[int, str]:
    """P0 first; malformed priorities sort last but remain deterministic."""
    if isinstance(value, str) and len(value) >= 2 and value[0] == "P" and value[1:].isdigit():
        return (int(value[1:]), value)
    return (9999, str(value))


def _has_unresolved_blocker(task: dict[str, Any]) -> bool:
    """Unknown blocker shape is conservatively treated as a blocker."""
    return bool(task.get("blocker") or task.get("blockers"))


def _overlap(left: str, right: str) -> bool:
    if not left or not right:
        return True
    a, b = Path(left).parts, Path(right).parts
    return a == b or a[:len(b)] == b or b[:len(a)] == a


class DeterministicTaskPlanner:
    """Canonical READY/dependency selector with stable ownership boundaries."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _load(self) -> tuple[dict[str, Any], str]:
        path = self.root / "projects/las-voces/project.json"
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("project", {}).get("id") != PROJECT_ID:
            raise ValueError("invalid LAS VOCES canonical project")
        if not isinstance(value.get("tasks"), list):
            raise ValueError("canonical tasks must be a list")
        return value, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _boundary(task_id: str) -> tuple[str, str]:
        slug = task_id.lower()
        # The canonical backlog does not declare granular writable paths.  A
        # task-specific worktree is an expectation for the builder, while the
        # local lease uses the repository boundary until canonical scope data
        # exists.  This deliberately serializes assignments rather than
        # inventing safe parallel filesystem ownership.
        return (f"worktrees/las-voces-{slug}", ".")

    def eligible_tasks(self, project: dict[str, Any], active_leases: Iterable[Any] = ()) -> list[dict[str, Any]]:
        tasks = project["tasks"]
        by_id = {task.get("id"): task for task in tasks if isinstance(task, dict) and isinstance(task.get("id"), str)}
        if len(by_id) != len(tasks):
            return []
        active = tuple(active_leases)
        eligible: list[dict[str, Any]] = []
        for task in tasks:
            if not isinstance(task, dict) or task.get("status") != "READY":
                continue
            task_id, owner = task.get("id"), task.get("owner")
            dependencies = task.get("depends", [])
            if not isinstance(task_id, str) or not task_id or not isinstance(owner, str) or not owner:
                continue
            if _priority(task.get("priority"))[0] == 9999:
                continue
            if not isinstance(dependencies, list) or any(not isinstance(dep, str) or by_id.get(dep, {}).get("status") != "DONE" for dep in dependencies):
                continue
            if _has_unresolved_blocker(task):
                continue
            worktree, scope = self._boundary(task_id)
            if any(getattr(lease, "task_id", None) == task_id or getattr(lease, "worktree", None) == worktree or _overlap(str(getattr(lease, "writable_scope", "")), scope) for lease in active):
                continue
            eligible.append(task)
        return sorted(eligible, key=lambda task: (_priority(task.get("priority")), task["id"]))

    def plan(self, active_leases: Iterable[Any] = ()) -> PlanningResult:
        project, observed_hash = self._load()
        eligible = self.eligible_tasks(project, active_leases)
        if not eligible:
            return PlanningResult(observed_hash, 0, None, "NO_ELIGIBLE_TASK")
        task = eligible[0]
        task_id, owner = task["id"], task["owner"]
        worktree, scope = self._boundary(task_id)
        # Stable identities make retries converge even though created_at is a
        # factual clock value. Runtime idempotency deliberately excludes those
        # transport-only envelope fields while retaining canonical state.
        correlation = hashlib.sha256(f"{PROJECT_ID}:{task_id}:{observed_hash}:emit_handoff".encode()).hexdigest()
        envelope = {
            "message_id": correlation,
            "project_id": PROJECT_ID,
            "task_id": task_id,
            "sender_agent": ARIADNA_ID,
            "recipient_agent": owner,
            "intent": "governed_task_handoff",
            "evidence_refs": [],
            "authority_context": {"handoff": {
                "owner": owner,
                "branch_worktree": f"branch:las-voces/{task_id.lower()}; worktree:{worktree}",
                "scope": f"LAS VOCES {task_id}: {task.get('title', '')}",
                "acceptance_criteria": "Implement only the canonical task scope; provide tests, commit/PR, and acceptance evidence before DONE.",
                "commit_pr": "Builder must provide a commit and pull request; Ariadna cannot merge.",
                "test_evidence": "Builder must provide controlled test evidence required by the existing DONE contract.",
                "blockers": "Report canonical or operational blockers through a governed handoff; do not self-approve.",
                "next_action": f"Acquire the declared isolated worktree and begin {task_id} within the canonical scope.",
            }},
            "correlation_id": correlation,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "RECORDED",
        }
        return PlanningResult(
            observed_hash, len(eligible), task_id, "ELIGIBLE_TASK_SELECTED",
            PlannedHandoff(task_id, owner, observed_hash, worktree, scope, correlation, envelope),
        )
