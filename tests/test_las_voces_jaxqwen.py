"""Adversarial tests for the ACK-bound jaxqwen structured capability."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FDS: dict[str, list[object]] = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


dispatcher = load("qwen_dispatcher", REPO / "projects/las-voces/authority/ariadna_dispatcher.py")
planner = load("qwen_planner", REPO / "projects/las-voces/authority/ariadna_planner.py")
capability = load("qwen_capability", REPO / "projects/las-voces/authority/jaxqwen_capability.py")


def git(root, *args): return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture()
def fixture(tmp_path):
    root = tmp_path / "jax"; shutil.copytree(REPO / "projects/las-voces", root / "projects/las-voces", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run(["git", "init", "-q", str(root)], check=True); git(root, "config", "user.email", "test@example.invalid"); git(root, "config", "user.name", "test")
    git(root, "add", "projects"); git(root, "commit", "-qm", "fixture"); FDS[str(root)] = []
    producer = root / ".git/ariadna-pm-control"; producer.mkdir(parents=True)
    plan = planner.DeterministicTaskPlanner(root).plan(); assert plan.handoff
    handoff = json.loads(json.dumps(plan.handoff.envelope)); row = {"event_type": "emit_handoff", "idempotency_key": "0" * 64, "task_id": plan.handoff.task_id, "handoff": handoff, "lease_id": plan.handoff.lease_id}
    key = dispatcher.GovernedHandoffConsumer._runtime_key(row, plan.handoff.expected_project_hash); row["idempotency_key"] = key
    audit = {"event_type": "ARIADNA_CYCLE", "idempotency_key": key, "project_hash": plan.handoff.expected_project_hash, "task_id": "LV-001", "action": "emit_handoff", "verdict": "ALLOW", "result": "EFFECT_EMIT_HANDOFF", "lease_id": plan.handoff.lease_id, "runtime_instance_id": "host"}
    lease = {"task_id": "LV-001", "owner": "Qwen/Infra", "worktree": "worktrees/las-voces-lv-001", "writable_scope": ".", "lease_id": plan.handoff.lease_id, "runtime_instance_id": "host", "pid": 1}
    (producer / "handoffs.ndjson").write_text(json.dumps(row) + "\n"); (producer / "runtime-audit.ndjson").write_text(json.dumps(audit) + "\n"); (producer / "task-leases.ndjson").write_text(json.dumps({"event": "ACQUIRED", "lease": lease}) + "\n")
    control = producer / "control.lock"; control.write_text(json.dumps({"state": "ACTIVE", "instance_id": "host", "pid": 1})); fd = control.open("r+"); fcntl.flock(fd, fcntl.LOCK_EX); FDS[str(root)].append(fd)
    builder = tmp_path / "builder-worktrees/las-voces-lv-001"
    subprocess.run(["git", "clone", "-q", str(root), str(builder)], check=True)
    git(builder, "checkout", "-qB", "las-voces/lv-001")
    dispatch = root / ".git/ariadna-builder-dispatch"; dispatch.mkdir()
    ack = {"event_type": "ARIADNA_HANDOFF_ACK", "state": "DISPATCHED", "idempotency_key": key, "task_id": "LV-001", "lease_id": plan.handoff.lease_id, "canonical_owner": "Qwen/Infra", "builder_identity": "Qwen", "branch": "las-voces/lv-001", "worktree": str(builder), "project_hash": plan.handoff.expected_project_hash, "base_commit": git(root, "rev-parse", "HEAD"), "builder_process_started": False}
    (dispatch / "acks.ndjson").write_text(json.dumps(ack) + "\n")
    yield root, producer, key, tmp_path / "missions"
    for fd in FDS.pop(str(root)): fd.close()


def make(fixture):
    root, producer, key, workspaces = fixture
    return capability.JaxQwenCapability(root, producer, workspaces, identity_verifier=lambda: "jaxqwen"), key


def test_current_lv001_ack_creates_one_independent_bounded_qwen_mission(fixture):
    item, key = make(fixture); result = item.start(idempotency_key=key)
    assert result.decision == "ACCEPTED" and result.mission_id
    mission = item._mission(result.mission_id)
    assert mission.task_id == "LV-001" and mission.canonical_owner == "Qwen/Infra" and mission.branch == "las-voces/lv-001"
    assert Path(mission.workspace).is_relative_to(item.workspace_root)
    assert (Path(mission.workspace) / ".git").is_dir()
    assert item.start(idempotency_key=key).decision == "NOOP"


@pytest.mark.parametrize("mutate", [
    lambda ack: ack.update({"canonical_owner": "attacker"}),
    lambda ack: ack.update({"branch": "evil"}),
    lambda ack: ack.update({"project_hash": "0" * 64}),
])
def test_forged_ack_or_stale_binding_is_rejected(fixture, mutate):
    item, key = make(fixture); path = item.dispatch_dir / "acks.ndjson"; rows = [json.loads(x) for x in path.read_text().splitlines()]
    mutate(rows[-1]); path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert item.start(idempotency_key=key).decision == "REJECTED"


def test_identity_stale_lease_and_non_ready_task_fail_closed(fixture):
    item, key = make(fixture); root, producer, _, workspaces = fixture; bad = capability.JaxQwenCapability(root, producer, workspaces, identity_verifier=lambda: "ariadna-project-manager"); assert bad.start(idempotency_key=key).decision == "REJECTED"
    with (item.handoff_state_dir / "task-leases.ndjson").open("a") as out: out.write(json.dumps({"event": "RELEASED", "lease_id": json.loads((item.dispatch_dir / "acks.ndjson").read_text().splitlines()[-1])["lease_id"]}) + "\n")
    assert item.start(idempotency_key=key).decision == "REJECTED"


def test_concurrent_start_is_single_winner_and_restart_is_idempotent(fixture):
    item, key = make(fixture); values = []
    threads = [threading.Thread(target=lambda: values.append(item.start(idempotency_key=key).decision)) for _ in range(2)]
    [x.start() for x in threads]; [x.join() for x in threads]
    assert sorted(values) == ["ACCEPTED", "NOOP"]
    replay = capability.JaxQwenCapability(item.root, item.handoff_state_dir, item.workspace_root, identity_verifier=lambda: "jaxqwen")
    assert replay.start(idempotency_key=key).decision == "NOOP"


def test_structured_tools_reject_path_shell_and_test_injection(fixture):
    item, key = make(fixture); result = item.start(idempotency_key=key); mission = result.mission_id; assert mission
    assert item.tool(mission, "write_file", path="notes.txt", content="safe")["written"] == "notes.txt"
    with pytest.raises(capability.CapabilityError): item.tool(mission, "write_file", path="../../escape", content="x")
    with pytest.raises(capability.CapabilityError): item.tool(mission, "write_file", path="projects/las-voces/authority/x.py", content="x")
    with pytest.raises(capability.CapabilityError): item.tool(mission, "shell", command="id")
    with pytest.raises(capability.CapabilityError): item.tool(mission, "run_named_test", test="pytest; touch /tmp/x")


def test_symlink_escape_commit_scope_and_completion_do_not_mark_done(fixture):
    item, key = make(fixture); result = item.start(idempotency_key=key); mission = item._mission(result.mission_id); workspace = Path(mission.workspace)
    (workspace / "outside").symlink_to("/tmp")
    with pytest.raises(capability.CapabilityError): item.tool(mission.mission_id, "write_file", path="outside/no", content="x")
    (workspace / "outside").unlink()
    item.tool(mission.mission_id, "write_file", path="README.mission", content="evidence")
    commit = item.tool(mission.mission_id, "request_commit", message="feat(las-voces): bounded evidence")
    assert commit["evidence_only"] and commit["commit"]
    project = json.loads((item.root / "projects/las-voces/project.json").read_text())
    assert next(x for x in project["tasks"] if x["id"] == "LV-001")["status"] == "READY"


def test_operator_cancel_is_exact_and_machine_cannot_cancel(fixture):
    item, key = make(fixture); result = item.start(idempotency_key=key); assert result.mission_id
    assert item.cancel(result.mission_id, operator_identity="jaxqwen").decision == "REJECTED"
    assert item.cancel(result.mission_id, operator_identity="operator").decision == "CANCELLED"
    assert item.cancel(result.mission_id, operator_identity="operator").decision == "NOOP"


def test_capability_has_no_forbidden_actions_or_shell_true():
    assert capability.JaxQwenCapability.allowed_action(capability.CAPABILITY)
    for name in ("merge", "deploy", "production_mutation", "capability_grant", "runtime_execution", "shell_execution"):
        assert not capability.JaxQwenCapability.allowed_action(name)
    source = (REPO / "projects/las-voces/authority/jaxqwen_capability.py").read_text()
    assert "shell=True" not in source and "Popen(" not in source


def test_bounded_model_loop_only_executes_structured_allowlisted_tools(fixture):
    item, key = make(fixture); mission = item.start(idempotency_key=key).mission_id; assert mission
    class Transport:
        def __init__(self): self.calls = 0
        def request_tools(self, _mission, _history):
            self.calls += 1
            return [{"name": "write_file", "arguments": {"path": "model.txt", "content": "bounded"}}] if self.calls == 1 else []
    assert item.run_tool_loop(mission, Transport())[-1]["result"]["written"] == "model.txt"
    class Escape:
        def request_tools(self, _mission, _history): return [{"name": "shell", "arguments": {"command": "id"}}]
    with pytest.raises(capability.CapabilityError): item.run_tool_loop(mission, Escape())
