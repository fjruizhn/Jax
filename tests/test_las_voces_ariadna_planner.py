"""Deterministic proposal-source tests; every effect is confined to a temp copy."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


planner = load("lv004_planner", REPO / "projects/las-voces/authority/ariadna_planner.py")
runtime = load("lv004_planner_runtime", REPO / "projects/las-voces/authority/ariadna_runtime.py")
authority = load("lv004_planner_authority", REPO / "projects/las-voces/authority/ariadna_authority.py")
hostmod = load("lv004_planner_host", REPO / "projects/las-voces/authority/ariadna_host.py")


@pytest.fixture()
def root(tmp_path):
    shutil.copytree(REPO / "projects/las-voces", tmp_path / "projects/las-voces")
    return tmp_path


def host_config(tmp_path, **extra):
    return hostmod.HostConfig.from_mapping({
        "mode": "production", "production_effects_enabled": True,
        "kill_switch_path": str(tmp_path / "kill"), "tick_seconds": 0.001,
        **extra,
    })


def canonical(root):
    return root / "projects/las-voces/project.json"


def alter(root, mutate):
    path = canonical(root)
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_current_fixture_discovers_lv001_and_validates_canonical_handoff(root):
    before = canonical(root).read_bytes()
    result = planner.DeterministicTaskPlanner(root).plan()
    assert result.reason == "ELIGIBLE_TASK_SELECTED"
    assert result.selected_task_id == "LV-001"
    assert result.eligible_count == 1
    assert result.handoff is not None
    assert result.handoff.owner == "Qwen/Infra"
    assert result.handoff.worktree == "worktrees/las-voces-lv-001"
    assert result.handoff.writable_scope == "."
    authority.validate_handoff(root, result.handoff.envelope)
    assert canonical(root).read_bytes() == before


def test_ready_dependency_blocker_and_done_tasks_are_not_improperly_selected(root):
    def mutate(value):
        tasks = {task["id"]: task for task in value["tasks"]}
        tasks["LV-001"]["blockers"] = ["pending infrastructure fact"]
        tasks["LV-010"]["status"] = "READY"  # LV-001 is not DONE.
    alter(root, mutate)
    result = planner.DeterministicTaskPlanner(root).plan()
    assert result.selected_task_id is None
    assert result.reason == "NO_ELIGIBLE_TASK"


def test_priority_ties_and_live_task_or_worktree_leases_fail_closed(root):
    def add_ready(value, task_id, priority):
        value["tasks"].append({"id": task_id, "title": task_id, "owner": "Qwen", "status": "READY", "priority": priority, "depends": ["LV-000"]})
    alter(root, lambda value: (add_ready(value, "LV-099", "P1"), add_ready(value, "LV-098", "P0")))
    source = planner.DeterministicTaskPlanner(root)
    assert source.plan().selected_task_id == "LV-001"  # P0 then stable task id.
    alter(root, lambda value: next(task for task in value["tasks"] if task["id"] == "LV-001").update(status="PLANNED"))
    assert source.plan().selected_task_id == "LV-098"  # P0 tie resolves by task id.
    active = runtime.TaskLease("LV-001", "Qwen/Infra", "other", "other", "held", "other-host", 1)
    # Canonical writable scopes do not exist, so the planner intentionally
    # serializes ownership at the repository boundary instead of guessing.
    assert source.plan([active]).selected_task_id is None
    worktree_collision = runtime.TaskLease("LV-999", "other", "worktrees/las-voces-lv-001", "unrelated", "held-2", "other-host", 1)
    assert source.plan([worktree_collision]).selected_task_id is None


def test_host_emits_one_valid_handoff_then_converges_until_canonical_change(root, tmp_path, caplog):
    caplog.set_level("INFO", logger="las_voces.ariadna_host")
    host = hostmod.AriadnaHost(root, host_config(tmp_path))
    assert host.start() is hostmod.HostLifecycle.READY
    assert host.run_once() == "EFFECT_EMIT_HANDOFF"
    observations = [json.loads(record.message) for record in caplog.records if record.name == "las_voces.ariadna_host" and '"planner_cycle"' in record.message]
    assert observations[-1]["project_hash"] == runtime.project_hash(canonical(root))
    assert observations[-1]["eligible_task_count"] == 1
    assert observations[-1]["selected_task_id"] == "LV-001"
    assert observations[-1]["authority_verdict"] == "ALLOW"
    assert observations[-1]["result"] == "EFFECT_EMIT_HANDOFF"
    rows = [json.loads(line) for line in host._runtime.outbox_path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["task_id"] == "LV-001"
    assert host.run_once() == "NOOP_NO_ELIGIBLE_TASK"
    assert len(host._runtime.outbox_path.read_text().splitlines()) == 1
    lease = next(iter(host._runtime._owned_leases.values()))
    assert host.release_task(lease)
    # Same canonical state has the same semantic idempotency key despite a new
    # factual envelope timestamp, so it cannot create a second assignment.
    assert host.run_once() == "NOOP_IDEMPOTENT"
    assert len(host._runtime.outbox_path.read_text().splitlines()) == 1
    alter(root, lambda value: next(task for task in value["tasks"] if task["id"] == "LV-001").update(title="changed canonical task fact"))
    assert host.run_once() == "EFFECT_EMIT_HANDOFF"
    assert len(host._runtime.outbox_path.read_text().splitlines()) == 2
    host.shutdown()


def test_stale_planned_observation_releases_lease_without_handoff(root, tmp_path):
    host = hostmod.AriadnaHost(root, host_config(tmp_path))
    assert host.start() is hostmod.HostLifecycle.READY
    original = host._planner.plan
    def stale_plan(*args):
        result = original(*args)
        alter(root, lambda value: next(task for task in value["tasks"] if task["id"] == "LV-001").update(title="newer canonical fact"))
        return result
    host._planner.plan = stale_plan
    assert host.run_once() == "NOOP_STALE_OR_INVALID"
    assert not host._runtime._owned_leases
    assert not host._runtime.outbox_path.exists()
    host.shutdown()


def test_planner_is_read_only_cannot_inject_verifiers_and_forever_ticks_are_bounded(root, tmp_path):
    with pytest.raises(TypeError):
        planner.DeterministicTaskPlanner(root, verifier=object())
    host = hostmod.AriadnaHost(root, host_config(tmp_path))
    calls = []
    def supplier():
        calls.append(1)
        if len(calls) >= 2:
            host.request_stop()
        return runtime.Proposal("LV-001", "runtime_execution", runtime.project_hash(canonical(root)))
    assert host.run_forever(supplier) is hostmod.HostLifecycle.STOPPED
    assert len(calls) == 2  # explicit wait/event scheduling; no busy loop.
    assert not host._runtime.outbox_path.exists()


def test_planner_output_cannot_mutate_or_expand_forbidden_authority(root, tmp_path):
    before = canonical(root).read_bytes()
    result = planner.DeterministicTaskPlanner(root).plan()
    assert result.handoff is not None and canonical(root).read_bytes() == before
    host = hostmod.AriadnaHost(root, host_config(tmp_path))
    assert host.start() is hostmod.HostLifecycle.READY
    for action in authority.FORBIDDEN:
        proposal = runtime.Proposal("LV-001", action, runtime.project_hash(canonical(root)))
        assert host.run_once(proposal) == "NOOP_DENY"
    assert canonical(root).read_bytes() == before
    host.shutdown()


def test_handoff_task_and_owner_spoofing_are_denied(root):
    handoff = planner.DeterministicTaskPlanner(root).plan().handoff
    assert handoff is not None
    engine = authority.AuthorityEngine(root)
    forged = json.loads(json.dumps(handoff.envelope))
    forged["recipient_agent"] = "attacker"
    forged["authority_context"]["handoff"]["owner"] = "attacker"
    assert engine.evaluate(sender_agent=authority.ARIADNA_ID, task_id="LV-001", action="emit_handoff", handoff=forged).verdict is authority.Verdict.DENY
    assert engine.evaluate(sender_agent=authority.ARIADNA_ID, task_id="LV-002", action="emit_handoff", handoff=handoff.envelope).verdict is authority.Verdict.DENY
