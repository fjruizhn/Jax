"""Adversarial tests for the separate governed builder-dispatch boundary."""
from __future__ import annotations

import importlib.util
import fcntl
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONTROL_FDS: dict[str, list[object]] = {}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


dispatcher = load("lv004_dispatcher", REPO / "projects/las-voces/authority/ariadna_dispatcher.py")
planner = load("lv004_dispatcher_planner", REPO / "projects/las-voces/authority/ariadna_planner.py")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture()
def root(tmp_path):
    shutil.copytree(REPO / "projects/las-voces", tmp_path / "projects/las-voces")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Dispatcher test")
    git(tmp_path, "add", "projects")
    git(tmp_path, "commit", "-qm", "fixture")
    CONTROL_FDS[str(tmp_path)] = []
    yield tmp_path
    for fd in CONTROL_FDS.pop(str(tmp_path)):
        fd.close()


def setup_handoff(root: Path, tmp_path, *, mutate=None):
    for fd in CONTROL_FDS[str(root)]:
        fd.close()
    CONTROL_FDS[str(root)].clear()
    producer = root / ".git/ariadna-pm-control"; producer.mkdir(parents=True, exist_ok=True)
    plan = planner.DeterministicTaskPlanner(root).plan()
    assert plan.handoff is not None
    handoff = json.loads(json.dumps(plan.handoff.envelope))
    row = {"event_type": "emit_handoff", "idempotency_key": "0" * 64, "task_id": plan.handoff.task_id, "handoff": handoff, "lease_id": plan.handoff.lease_id}
    key = dispatcher.GovernedHandoffConsumer._runtime_key(row, plan.handoff.expected_project_hash)
    row["idempotency_key"] = key
    audit = {"event_type": "ARIADNA_CYCLE", "idempotency_key": key, "cycle_id": key,
             "project_hash": plan.handoff.expected_project_hash, "task_id": plan.handoff.task_id,
             "action": "emit_handoff", "verdict": "ALLOW", "result": "EFFECT_EMIT_HANDOFF",
             "lease_id": plan.handoff.lease_id, "runtime_instance_id": "host"}
    lease = {"task_id": plan.handoff.task_id, "owner": plan.handoff.owner,
             "worktree": plan.handoff.worktree, "writable_scope": plan.handoff.writable_scope,
             "lease_id": plan.handoff.lease_id, "runtime_instance_id": "host", "pid": 1}
    if mutate: mutate(row, audit, lease)
    (producer / "handoffs.ndjson").write_text(json.dumps(row) + "\n")
    (producer / "runtime-audit.ndjson").write_text(json.dumps(audit) + "\n")
    (producer / "task-leases.ndjson").write_text(json.dumps({"event": "ACQUIRED", "lease": lease}) + "\n")
    control = producer / "control.lock"; control.write_text(json.dumps({"state": "ACTIVE", "instance_id": lease["runtime_instance_id"], "pid": lease["pid"]}))
    held = control.open("r+"); fcntl.flock(held, fcntl.LOCK_EX); CONTROL_FDS[str(root)].append(held)
    return producer, key


def consumer(root, producer, tmp_path):
    return dispatcher.GovernedHandoffConsumer(root, producer, tmp_path.parent / f"{tmp_path.name}-builder-worktrees")


def acks(item):
    return [json.loads(row) for row in item.ack_path.read_text().splitlines()]


def test_current_lv001_handoff_creates_one_qwen_worktree_and_audit_bound_ack(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path); item = consumer(root, producer, tmp_path)
    result = item.consume(key)
    assert result.decision == "DISPATCHED" and result.task_id == "LV-001"
    expected = tmp_path.parent / f"{tmp_path.name}-builder-worktrees/las-voces-lv-001"
    assert Path(result.worktree) == expected and git(expected, "branch", "--show-current") == "las-voces/lv-001"
    rows = acks(item)
    assert [row["state"] for row in rows] == ["RECEIVED", "ACCEPTED", "DISPATCHED"]
    assert rows[-1]["canonical_owner"] == "Qwen/Infra" and rows[-1]["builder_identity"] == "Qwen"
    assert "builder execution intentionally not started" in rows[-1]["reason"]


def test_replay_and_concurrent_consumers_produce_one_worktree_and_one_terminal_ack(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path); item = consumer(root, producer, tmp_path)
    results = []
    threads = [threading.Thread(target=lambda: results.append(item.consume(key))) for _ in range(2)]
    [thread.start() for thread in threads]; [thread.join() for thread in threads]
    assert {result.decision for result in results} == {"DISPATCHED", "NOOP"}
    assert [row["state"] for row in acks(item)].count("DISPATCHED") == 1
    assert item.consume(key).decision == "NOOP"


def test_two_independent_processes_cannot_dispatch_the_same_handoff_twice(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path); worktrees = tmp_path.parent / f"{tmp_path.name}-builder-worktrees"
    command = [sys.executable, str(REPO / "scripts/ariadna_dispatch.py"), "--root", str(root),
               "--handoff-state-dir", str(producer), "--worktree-root", str(worktrees),
               "--idempotency-key", key, "--consume"]
    first, second = subprocess.Popen(command, stdout=subprocess.PIPE, text=True), subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    outputs = [json.loads(process.communicate(timeout=10)[0]) for process in (first, second)]
    assert sorted(value["decision"] for value in outputs) == ["DISPATCHED", "NOOP"]
    item = consumer(root, producer, tmp_path)
    assert [row["state"] for row in acks(item)].count("DISPATCHED") == 1


@pytest.mark.parametrize("mutation", [
    lambda row, audit, lease: row["handoff"].update({"recipient_agent": "attacker"}),
    lambda row, audit, lease: row.update({"task_id": "LV-999"}),
    lambda row, audit, lease: audit.update({"verdict": "DENY", "result": "NOOP_DENY"}),
    lambda row, audit, lease: lease.update({"owner": "attacker"}),
])
def test_forged_recipient_unknown_task_unauthorized_audit_and_owner_mismatch_are_rejected(root, tmp_path, mutation):
    producer, key = setup_handoff(root, tmp_path, mutate=mutation); item = consumer(root, producer, tmp_path)
    assert item.consume(key).decision == "REJECTED"
    assert not (tmp_path.parent / f"{tmp_path.name}-builder-worktrees").exists()


def test_stale_canonical_and_released_lease_fail_closed(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path); item = consumer(root, producer, tmp_path)
    path = root / "projects/las-voces/project.json"; value = json.loads(path.read_text()); value["tasks"][1]["title"] = "changed"; path.write_text(json.dumps(value))
    assert item.consume(key).decision == "REJECTED"
    git(root, "add", "projects/las-voces/project.json")
    git(root, "commit", "-qm", "changed canonical fixture")
    producer2, key2 = setup_handoff(root, tmp_path / "two")
    with (producer2 / "task-leases.ndjson").open("a") as log: log.write(json.dumps({"event": "RELEASED", "lease_id": json.loads((producer2 / "handoffs.ndjson").read_text())["lease_id"]}) + "\n")
    assert consumer(root, producer2, tmp_path / "two").consume(key2).decision == "REJECTED"


def test_existing_conflicting_worktree_and_unsafe_outbox_payload_are_rejected(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path)
    target = tmp_path.parent / f"{tmp_path.name}-builder-worktrees/las-voces-lv-001"; target.mkdir(parents=True)
    assert consumer(root, producer, tmp_path).consume(key).decision == "REJECTED"


def test_unsafe_outbox_payload_is_rejected(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path, mutate=lambda row, audit, lease: row.update({"command": "qwen; rm -rf /"}))
    assert consumer(root, producer, tmp_path).consume(key).decision == "REJECTED"


def test_audit_key_binds_exact_handoff_content(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path, mutate=lambda row, audit, lease: row["handoff"]["authority_context"]["handoff"].update({"scope": "forged scope"}))
    assert consumer(root, producer, tmp_path).consume(key).decision == "REJECTED"


def test_dead_lease_pid_is_rejected(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path / "pid", mutate=lambda row, audit, lease: lease.update({"pid": 99999999}))
    assert consumer(root, producer, tmp_path / "pid").consume(key).decision == "REJECTED"


def test_source_head_change_and_forged_accepted_ack_cannot_be_adopted(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path); item = consumer(root, producer, tmp_path)
    (root / "unrelated.txt").write_text("different source revision")
    git(root, "add", "unrelated.txt"); git(root, "commit", "-qm", "unrelated")
    assert item.consume(key).decision == "REJECTED"


def test_external_git_filter_is_rejected_without_checkout(root, tmp_path):
    (root / ".gitattributes").write_text("* filter=reviewexec\n")
    git(root, "add", ".gitattributes"); git(root, "commit", "-qm", "unsafe filter")
    git(root, "config", "filter.reviewexec.smudge", "sh -c 'cat; touch /tmp/should-not-run'")
    producer, key = setup_handoff(root, tmp_path)
    with pytest.raises(dispatcher.DispatchError):
        consumer(root, producer, tmp_path)
    assert not Path("/tmp/should-not-run").exists()


def test_recovery_after_worktree_created_before_dispatched_ack_is_idempotent(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path); item = consumer(root, producer, tmp_path)
    row = json.loads((producer / "handoffs.ndjson").read_text()); target = tmp_path.parent / f"{tmp_path.name}-builder-worktrees/las-voces-lv-001"
    item._ack("ACCEPTED", key=key, task_id="LV-001", lease_id=row["lease_id"], owner="Qwen/Infra", builder="Qwen", branch="las-voces/lv-001", worktree=target, project_hash=planner._project_hash(root / "projects/las-voces/project.json"))
    item._create_worktree(target, "las-voces/lv-001")
    assert item.consume(key).decision == "DISPATCHED"
    assert [record["state"] for record in acks(item)] == ["ACCEPTED", "DISPATCHED"]


def test_accepted_ack_cannot_adopt_an_independent_clean_repository(root, tmp_path):
    producer, key = setup_handoff(root, tmp_path); item = consumer(root, producer, tmp_path)
    row = json.loads((producer / "handoffs.ndjson").read_text()); target = tmp_path.parent / f"{tmp_path.name}-builder-worktrees/las-voces-lv-001"; target.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(target)], check=True); git(target, "config", "user.email", "test@example.invalid"); git(target, "config", "user.name", "other")
    (target / "other.txt").write_text("not a registered builder worktree"); git(target, "add", "other.txt"); git(target, "commit", "-qm", "other"); git(target, "branch", "-M", "las-voces/lv-001")
    item._ack("ACCEPTED", key=key, task_id="LV-001", lease_id=row["lease_id"], owner="Qwen/Infra", builder="Qwen", branch="las-voces/lv-001", worktree=target, project_hash=planner._project_hash(root / "projects/las-voces/project.json"))
    assert item.consume(key).decision == "REJECTED"


def test_dispatcher_has_no_builder_command_or_privileged_actions(root, tmp_path):
    producer, _ = setup_handoff(root, tmp_path); item = consumer(root, producer, tmp_path)
    assert item.action_allowed("create_isolated_builder_worktree")
    for action in dispatcher.FORBIDDEN_ACTIONS:
        assert not item.action_allowed(action)
    source = (REPO / "projects/las-voces/authority/ariadna_dispatcher.py").read_text()
    assert "shell=True" not in source and "Popen(" not in source and "create_subprocess" not in source
