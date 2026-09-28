"""Host integration tests: all effects are confined to copied temp repositories."""
from __future__ import annotations
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
hostmod = load("lv004_host", REPO / "projects/las-voces/authority/ariadna_host.py")
runtime = load("lv004_host_runtime", REPO / "projects/las-voces/authority/ariadna_runtime.py")
authority = load("lv004_host_authority", REPO / "projects/las-voces/authority/ariadna_authority.py")

@pytest.fixture()
def root(tmp_path):
    shutil.copytree(REPO / "projects/las-voces", tmp_path / "projects/las-voces")
    return tmp_path

def config(tmp_path, **extra):
    return hostmod.HostConfig.from_mapping({"kill_switch_path": str(tmp_path / "kill"), **extra})

def test_composition_owns_fixed_verifiers_and_github_binds_exact_run_job_sha(root, tmp_path):
    sha = "a" * 40
    seen = []
    def fetch(path):
        seen.append(path)
        if path.endswith("/jobs?per_page=100"):
            return {"jobs": [{"id": 9, "name": "policy", "status": "completed", "conclusion": "success"}]}
        return {"head_sha": sha, "status": "completed", "conclusion": "success", "path": ".github/workflows/policy.yml"}
    verifier = hostmod.GitHubActionsVerifier("fjruizhn/Jax", None, fetcher=fetch)
    manifest = {"repository_id": "fjruizhn/Jax", "run_id": "7", "job_id": "9", "commit_sha": sha, "workflow": "policy.yml"}
    assert verifier.execution_record(manifest) is True and len(seen) == 2
    assert verifier.execution_record({**manifest, "commit_sha": "b" * 40}) is False
    assert hostmod.GitHubActionsVerifier("fjruizhn/Jax", None, fetcher=lambda _path: (_ for _ in ()).throw(OSError("offline"))).execution_record(manifest) is None
    host = hostmod.AriadnaHost(root, config(tmp_path), github_fetcher=fetch)
    assert type(host._ci_verifier) is hostmod.GitHubActionsVerifier
    with pytest.raises(TypeError): host.run_once(None, verifier=object())

def test_human_acceptance_needs_exact_canonical_record_not_actor(root):
    verifier = hostmod.CanonicalHumanAuthorityVerifier(root); sha = "c" * 40
    record = {"task_id": "LV-004", "decision": "ACCEPTED", "commit_sha": sha, "authority_source": {"type": "human_authority"}, "acceptance_actor": "Fernando"}
    assert verifier.acceptance_record(record) is None
    record["authority_source"]["record_id"] = "activity:accept"
    assert verifier.acceptance_record(record) is None
    event = {"event_id": "accept", "event_type": "HUMAN_AUTHORITY_TASK_ACCEPTED", "status": "RECORDED", "project_id": "las-voces", "actor": "Fernando / Human Authority", "task_id": "LV-004", "decision": "ACCEPTED", "accepted_commit_sha": sha}
    with (root / "projects/las-voces/activity.ndjson").open("a") as handle: handle.write(json.dumps(event) + "\n")
    assert verifier.acceptance_record(record) is True

def test_default_production_disabled_dry_run_is_non_mutating_and_cli_health(root, tmp_path):
    state = root / "projects/las-voces/project.json"; before = state.read_bytes()
    disabled = hostmod.AriadnaHost(root, config(tmp_path, mode="production")); assert disabled.start() is hostmod.HostLifecycle.DISABLED
    host = hostmod.AriadnaHost(root, config(tmp_path))
    assert host.start() is hostmod.HostLifecycle.READY
    proposal = runtime.Proposal("LV-004", "transition_status", runtime.project_hash(state), target_status="BLOCKED")
    assert host.run_once(proposal) in {"DRY_RUN_ALLOW", "DRY_RUN_DENY", "DRY_RUN_HUMAN_REQUIRED"}
    assert state.read_bytes() == before
    cfg = tmp_path / "host.json"; cfg.write_text(json.dumps({"kill_switch_path": str(tmp_path / "kill")}))
    result = subprocess.run([sys.executable, str(REPO / "scripts/ariadna_host.py"), "--root", str(root), "--config", str(cfg), "--health"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["lifecycle"] == "READY"
    host.shutdown()

def test_single_instance_kill_switch_stop_and_reconciliation_readiness(root, tmp_path):
    first = hostmod.AriadnaHost(root, config(tmp_path)); assert first.start() is hostmod.HostLifecycle.READY
    second = hostmod.AriadnaHost(root, config(tmp_path)); assert second.start() is hostmod.HostLifecycle.STOPPED
    (tmp_path / "kill").write_text("stop")
    before = (root / "projects/las-voces/project.json").read_bytes()
    proposal = runtime.Proposal("LV-004", "runtime_execution", runtime.project_hash(root / "projects/las-voces/project.json"))
    assert first.run_once(proposal) == "NOOP_KILLED" and first.health()["kill_switch_active"]
    assert (root / "projects/las-voces/project.json").read_bytes() == before
    (tmp_path / "kill").unlink()
    history = root / "projects/las-voces/activity.ndjson"
    history.write_text(history.read_text() + json.dumps({"event_type": "TRANSITION_INTENT", "transition_id": "unresolved"}) + "\n")
    blocked = hostmod.AriadnaHost(root, config(tmp_path)); assert blocked.start() is hostmod.HostLifecycle.RECONCILIATION_REQUIRED
    assert blocked.health()["ready"] is False and blocked.run_once(proposal) == "NOOP_NOT_READY"

def test_host_preserves_forbidden_boundary_and_permitted_sandbox_cycle(root, tmp_path):
    host = hostmod.AriadnaHost(root, config(tmp_path, mode="production", production_effects_enabled=True)); assert host.start() is hostmod.HostLifecycle.READY
    state = root / "projects/las-voces/project.json"
    for action in authority.FORBIDDEN:
        proposal = runtime.Proposal("LV-004", action, runtime.project_hash(state))
        assert host.run_once(proposal) == "NOOP_DENY"
    lease = runtime.TaskLease("LV-004", "builder", "sandbox-worktree", "projects/las-voces", "lease", host._runtime.instance_id)
    assert host.acquire_task(lease)
    proposal = runtime.Proposal("LV-004", "coordinate_verified_work", runtime.project_hash(state), evidence_refs=("projects/las-voces/agents/ariadna.json",))
    assert host.run_once(proposal, lease_id="lease") == "EFFECT_COORDINATE_VERIFIED_WORK"
    assert host.run_once(proposal, lease_id="lease") == "NOOP_IDEMPOTENT"
    assert host.request_stop() is hostmod.HostLifecycle.STOPPING
    assert host.shutdown() is hostmod.HostLifecycle.STOPPED

def test_proposed_state_stale_task_lease_and_no_shell_path_fail_closed(root, tmp_path):
    project = root / "projects/las-voces/project.json"; value = json.loads(project.read_text()); next(x for x in value["agents"] if x["name"] == "Ariadna")["lifecycle_status"] = "PROPOSED_NOT_ACTIVE"; project.write_text(json.dumps(value))
    agent = root / "projects/las-voces/agents/ariadna.json"; value = json.loads(agent.read_text()); value["lifecycle_status"] = "PROPOSED_NOT_ACTIVE"; agent.write_text(json.dumps(value))
    assert hostmod.AriadnaHost(root, config(tmp_path)).start() is hostmod.HostLifecycle.STOPPED
    text = (REPO / "projects/las-voces/authority/ariadna_host.py").read_text()
    assert "subprocess" not in text and "os.system" not in text
    assert "ConditionPathExists=!/etc/jax/ariadna-pm.kill" in (REPO / "config/systemd/jax-ariadna-pm.service").read_text()

def test_crashed_task_lease_requires_reconciliation_not_silent_reacquisition(root, tmp_path):
    control = runtime._state_dir(root)
    stale = {"event": "ACQUIRED", "lease": {"task_id": "LV-004", "owner": "builder", "worktree": "dead-worktree", "writable_scope": "projects/las-voces", "lease_id": "dead", "runtime_instance_id": "dead-host", "pid": 999999}}
    (control / "task-leases.ndjson").write_text(json.dumps(stale) + "\n")
    host = hostmod.AriadnaHost(root, config(tmp_path))
    assert host.start() is hostmod.HostLifecycle.RECONCILIATION_REQUIRED
    assert host.run_once() == "NOOP_NOT_READY"
