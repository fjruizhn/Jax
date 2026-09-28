"""LV-004 adversarial runtime tests; mutations use only temporary repositories."""
from __future__ import annotations
import importlib.util
import json
import shutil
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module
runtime = load("lv004_runtime", REPO / "projects/las-voces/authority/ariadna_runtime.py")
authority = load("lv004_authority", REPO / "projects/las-voces/authority/ariadna_authority.py")

@pytest.fixture()
def root(tmp_path):
    shutil.copytree(REPO / "projects/las-voces", tmp_path / "projects/las-voces")
    return tmp_path
def engine(root): return authority.AuthorityEngine(root)
def proposal(root, action="runtime_execution", **kwargs):
    return runtime.Proposal(task_id="LV-004", action=action, expected_project_hash=runtime.project_hash(root / "projects/las-voces/project.json"), **kwargs)

def test_single_instance_stale_owner_and_graceful_release(root):
    one = runtime.AriadnaRuntime(root, engine(root), instance_id="one"); assert one.start() is runtime.Lifecycle.READY
    assert runtime.AriadnaRuntime(root, engine(root), instance_id="two").start() is runtime.Lifecycle.STOPPED
    assert one.shutdown() is runtime.Lifecycle.STOPPED
    lock = runtime._state_dir(root) / "control.lock"; lock.write_text('{"pid":999999,"state":"ACTIVE"}')
    recovered = runtime.AriadnaRuntime(root, engine(root)); assert recovered.start() is runtime.Lifecycle.READY
    assert recovered.health()["stale_owner_detected"]

def test_stale_deny_human_and_forbidden_have_no_canonical_effect(root):
    host = runtime.AriadnaRuntime(root, engine(root)); host.start()
    stale = proposal(root); state = root / "projects/las-voces/project.json"; before = state.read_bytes(); state.write_bytes(before + b" ")
    assert host.run_once(stale) == "NOOP_STALE_OR_INVALID"
    assert host.run_once(proposal(root, "runtime_execution")) == "NOOP_DENY"
    assert host.run_once(proposal(root, "invent_scope")) == "NOOP_HUMAN_REQUIRED"
    assert state.read_bytes() == before + b" "

def test_idempotency_audit_and_planner_cannot_fake_outcome(root):
    host = runtime.AriadnaRuntime(root, engine(root)); host.start()
    lease = runtime.TaskLease("LV-004", "builder", "worktree-a", "projects/las-voces", "lease", host.instance_id)
    assert host.acquire_task(lease)
    p = proposal(root, "coordinate_verified_work", evidence_refs=("projects/las-voces/agents/ariadna.json",))
    assert host.run_once(p, lease_id="lease") == "EFFECT_COORDINATE_VERIFIED_WORK"
    assert host.run_once(p, lease_id="lease") == "NOOP_IDEMPOTENT"
    rows = [json.loads(x) for x in host.audit_path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["verdict"] == "ALLOW" and rows[0]["project_hash"] == p.expected_project_hash
    assert "outcome" not in runtime.Proposal.__dataclass_fields__

def test_kill_switch_and_unresolved_transition_prevent_work(root):
    host = runtime.AriadnaRuntime(root, engine(root)); host.start(); host.request_stop()
    assert host.run_once(proposal(root)) == "NOOP_NOT_READY"
    host.shutdown()
    history = root / "projects/las-voces/activity.ndjson"
    history.write_text(history.read_text() + json.dumps({"event_type":"TRANSITION_INTENT","transition_id":"unresolved"}) + "\n")
    restarted = runtime.AriadnaRuntime(root, engine(root)); assert restarted.start() is runtime.Lifecycle.RECONCILIATION_REQUIRED
    assert restarted.run_once(proposal(root)) == "NOOP_NOT_READY"

def test_task_and_worktree_leases_conflict_then_reacquire(root):
    host = runtime.AriadnaRuntime(root, engine(root)); host.start()
    first = runtime.TaskLease("LV-004", "builder-a", "work-a", "projects/las-voces", "a", host.instance_id)
    assert host.acquire_task(first)
    assert not host.acquire_task(runtime.TaskLease("LV-004", "builder-b", "work-b", "other", "b", host.instance_id))
    assert not host.acquire_task(runtime.TaskLease("LV-010", "builder-b", "work-a", "other", "c", host.instance_id))
    assert not host.acquire_task(runtime.TaskLease("LV-010", "builder-b", "work-b", "projects/las-voces/authority", "d", host.instance_id))
    assert host.release_task(first)
    assert host.acquire_task(runtime.TaskLease("LV-004", "builder-b", "work-b", "projects/las-voces/authority", "b", host.instance_id))

def test_done_remains_governed_and_expected_hash_reaches_lv003(root):
    host = runtime.AriadnaRuntime(root, engine(root)); host.start()
    assert host.run_once(proposal(root, "merge")) == "NOOP_DENY"
    assert host.run_once(proposal(root, "transition_status", target_status="DONE")) in {"NOOP_DENY", "NOOP_HUMAN_REQUIRED"}
    assert "proposed and not active" in json.loads((root / "projects/las-voces/agents/ariadna.json").read_text())["authority"].lower()
    # Direct LV-003 caller can also bind an observed hash without allowing stale mutation.
    assert authority.transition(root, sender_agent=authority.ARIADNA_ID, task_id="LV-004", target_status="BLOCKED", evidence_refs=[], expected_project_hash="0" * 64).verdict is authority.Verdict.DENY
