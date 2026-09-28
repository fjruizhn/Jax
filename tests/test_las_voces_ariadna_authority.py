"""LV-003 adversarial authority tests; mutations stay in temporary repositories."""
from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys, threading
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ariadna_authority", REPO / "projects/las-voces/authority/ariadna_authority.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = mod; SPEC.loader.exec_module(mod)

@pytest.fixture()
def root(tmp_path: Path) -> Path:
    shutil.copytree(REPO / "projects/las-voces", tmp_path / "projects/las-voces")

    # These tests exercise the LV-003 authority transition contract from its
    # pre-closure lifecycle state. Do not depend on canonical master remaining
    # READY after human acceptance closes LV-003.
    project_path = tmp_path / "projects/las-voces/project.json"
    project = json.loads(project_path.read_text())
    next(item for item in project["tasks"] if item["id"] == "LV-003")["status"] = "READY"
    project_path.write_text(json.dumps(project), encoding="utf-8")

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key, value in (("user.email", "test@example.invalid"), ("user.name", "Test")):
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "unrelated commit mentioning LV-003"], check=True)
    return tmp_path

class Trusted:
    def __init__(self, execution=True, acceptance=True): self.execution, self.acceptance = execution, acceptance
    def execution_record(self, _manifest): return self.execution
    def acceptance_record(self, _record): return self.acceptance

def sha(root): return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
def commit(root): return "commit:" + sha(root)
def in_progress(root):
    p = root / "projects/las-voces/project.json"; d = json.loads(p.read_text())
    next(x for x in d["tasks"] if x["id"] == "LV-003")["status"] = "IN_PROGRESS"; p.write_text(json.dumps(d))
def write(root, name, value):
    p = root / "evidence" / name; p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(value)); return str(p.relative_to(root))
def manifest(root, task="LV-003", commit_sha=None):
    return write(root, "controlled-run.json", {"schema_version":"1.0", "kind":mod.TEST_MANIFEST_KIND, "task_id":task, "provider":"github-actions", "repository_id":"fjruizhn/Jax", "commit_sha":commit_sha or sha(root), "workflow":"policy.yml", "run_id":"17", "job_id":"policy", "environment":"CI", "tests":["tests/test_las_voces_ariadna_authority.py"], "counts":{"total":1,"failed":0,"error":0,"passed":1}, "raw_output_blob_hash":"sha256:" + "a" * 64})
def acceptance(root, commit_sha=None, actor="Fernando"):
    return write(root, "acceptance.json", {"schema_version":"1.0", "kind":"acceptance", "task_id":"LV-003", "acceptance_actor":actor, "authority_source":{"type":"human_authority", "record_id":"external-42"}, "commit_sha":commit_sha or sha(root), "criteria":[{"id":"AC-1", "result":"PASS"}], "evidence_refs":["controlled-run.json"], "decision":"ACCEPTED"})
def pr(root, head=None): return write(root, "pr.json", {"kind":"pull_request", "task_id":"LV-003", "provider":"github-actions", "repository_id":"fjruizhn/Jax", "number":286, "head_commit_sha":head or sha(root), "execution_manifest":"evidence/controlled-run.json"})
def blocker(root): return write(root, "blocker.json", {"kind":"blocker", "task_id":"LV-003", "state":"OPEN", "blocker":"waiting for dependency"})
def engine(root, verifier=None):
    """Tests install doubles only at the privileged composition boundary."""
    return mod.AuthorityEngine(root, ci_verifier=verifier, human_authority_verifier=verifier)
def decide(root, status, refs, verifier=None): return engine(root, verifier).evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status=status, evidence_refs=refs)

@pytest.mark.parametrize("action", sorted(mod.FORBIDDEN))
def test_forbidden_actions_are_deny(root, action): assert mod.evaluate(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", action=action).verdict is mod.Verdict.DENY

def test_unknown_action_is_human_required_and_nonactive_ariadna_cannot_run(root):
    assert mod.evaluate(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="invent_scope").verdict is mod.Verdict.HUMAN_REQUIRED
    assert mod.evaluate(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="runtime_execution").verdict is mod.Verdict.DENY
    assert "proposed and not active" in json.loads((root / "projects/las-voces/agents/ariadna.json").read_text())["authority"].lower()

def test_done_missing_evidence_classes_and_nonexistent_refs_deny(root):
    in_progress(root); tests, accepted = manifest(root), acceptance(root)
    for refs in ([], [tests], [accepted], [tests, accepted], [tests, commit(root)]): assert decide(root, "DONE", refs).verdict is not mod.Verdict.ALLOW
    assert decide(root, "DONE", [tests, accepted, commit(root), "evidence/nope.json"]).verdict is mod.Verdict.DENY

def test_self_issued_junit_and_acceptance_cannot_satisfy_done(root):
    in_progress(root)
    fake_junit = write(root, "junit.json", {"kind":"tests", "task_id":"LV-003", "result":"PASSED", "junit_report":"x", "sha256":"a" * 64})
    fake_acceptance = write(root, "fake-acceptance.json", {"kind":"acceptance", "task_id":"LV-003", "result":"ACCEPTED"})
    assert decide(root, "DONE", [fake_junit, fake_acceptance, commit(root)]).verdict is mod.Verdict.DENY
    # Even a correctly shaped local controlled-manifest file cannot become CI
    # evidence without the external controlled execution record.
    assert decide(root, "DONE", [manifest(root), acceptance(root), commit(root)]).verdict is mod.Verdict.HUMAN_REQUIRED
    assert decide(root, "DONE", [manifest(root), acceptance(root, actor=mod.ARIADNA_ID), commit(root)], Trusted()).verdict is mod.Verdict.DENY

def test_done_requires_controlled_trust_and_exact_same_commit(root):
    in_progress(root); refs = [manifest(root), acceptance(root), commit(root)]
    assert decide(root, "DONE", refs, Trusted()).verdict is mod.Verdict.ALLOW
    assert decide(root, "DONE", refs, Trusted(execution=False)).verdict is mod.Verdict.DENY
    assert decide(root, "DONE", refs, Trusted(acceptance=False)).verdict is mod.Verdict.DENY
    other = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD~0"], text=True).strip()
    assert decide(root, "DONE", [manifest(root, commit_sha=other), acceptance(root, commit_sha="b" * 40), commit(root)], Trusted()).verdict is mod.Verdict.DENY

def test_request_cannot_supply_or_replace_privileged_verifier(root):
    in_progress(root); refs = [manifest(root), acceptance(root), commit(root)]
    # A request has no trust/verifier field.  Supplying one is an API error,
    # while a preconfigured engine remains governed by its composition root.
    with pytest.raises(TypeError):
        mod.evaluate(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=refs, trust=Trusted())
    configured = engine(root, Trusted())
    assert configured.evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=refs).verdict is mod.Verdict.ALLOW

def test_pr_evidence_requires_controlled_manifest_with_exact_head(root):
    in_progress(root); tests, pull, accepted = manifest(root), pr(root), acceptance(root)
    assert decide(root, "DONE", [tests, pull, accepted], Trusted()).verdict is mod.Verdict.ALLOW
    assert decide(root, "DONE", [tests, pr(root, "b" * 40), accepted], Trusted()).verdict is mod.Verdict.DENY

def test_unrelated_commit_message_with_lv003_is_not_authorization(root):
    in_progress(root); refs = [manifest(root), acceptance(root), commit(root)]
    # The fixture commit message contains LV-003. Authorization succeeds only
    # due to structural exact bindings and controlled trust, never that text.
    assert decide(root, "DONE", refs).verdict is mod.Verdict.HUMAN_REQUIRED

def test_blocked_requires_real_blocker_and_valid_transition_allows(root):
    assert decide(root, "BLOCKED", []).verdict is mod.Verdict.DENY
    assert decide(root, "BLOCKED", [blocker(root)]).verdict is mod.Verdict.ALLOW
    assert decide(root, "RETIRED", [blocker(root)]).verdict is mod.Verdict.DENY

def test_transition_internal_audit_binding_and_append_only(root):
    before = (root / "projects/las-voces/activity.ndjson").read_bytes(); ref = blocker(root)
    assert mod.transition(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="BLOCKED", evidence_refs=[ref]).verdict is mod.Verdict.ALLOW
    lines = (root / "projects/las-voces/activity.ndjson").read_bytes(); assert lines.startswith(before)
    events = [json.loads(x) for x in lines.decode().splitlines() if "TRANSITION_" in x]
    assert len(events) == 2 and events[0]["transition_id"] == events[1]["transition_id"]
    assert events[0]["task_id"] == "LV-003" and events[1]["decision"] == "ALLOW"

def test_expected_project_hash_rejects_stale_plan_before_journal_intent(root):
    before = (root / "projects/las-voces/activity.ndjson").read_bytes()
    ref = blocker(root)
    decision = mod.transition(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="BLOCKED", evidence_refs=[ref], expected_project_hash="0" * 64)
    assert decision.verdict is mod.Verdict.DENY
    assert (root / "projects/las-voces/activity.ndjson").read_bytes() == before

def test_state_change_during_authorization_leaves_no_stale_intent(root):
    controlled = engine(root)
    original = controlled._evaluate
    def raced(**request):
        result = original(**request)
        path = root / "projects/las-voces/project.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result
    controlled._evaluate = raced
    before = (root / "projects/las-voces/activity.ndjson").read_bytes()
    assert controlled.transition(sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="BLOCKED", evidence_refs=[blocker(root)]).verdict is mod.Verdict.DENY
    assert (root / "projects/las-voces/activity.ndjson").read_bytes() == before

def test_validated_evidence_binds_exact_contents_not_reference_names(root):
    in_progress(root); tests, accepted = manifest(root), acceptance(root)
    controlled = engine(root, Trusted())
    _, first = controlled._evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=[tests, accepted, commit(root)])
    assert first and first.implementation_commit_sha == sha(root)
    test_path = root / tests; changed = json.loads(test_path.read_text()); changed["run_id"] = "18"; test_path.write_text(json.dumps(changed))
    _, changed_test = controlled._evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=[tests, accepted, commit(root)])
    assert changed_test and first.test_manifest_digest != changed_test.test_manifest_digest
    assert first.normalized_evidence_digest != changed_test.normalized_evidence_digest
    acceptance_path = root / accepted; changed = json.loads(acceptance_path.read_text()); changed["criteria"][0]["result"] = "PASS_WITH_NOTE"; acceptance_path.write_text(json.dumps(changed))
    _, changed_acceptance = controlled._evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=[tests, accepted, commit(root)])
    assert changed_acceptance and changed_test.acceptance_record_digest != changed_acceptance.acceptance_record_digest
    assert changed_test.normalized_evidence_digest != changed_acceptance.normalized_evidence_digest

def test_done_intent_and_commit_share_exact_validated_evidence_binding(root):
    in_progress(root); tests, accepted = manifest(root), acceptance(root)
    controlled = engine(root, Trusted())
    assert controlled.transition(sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="DONE", evidence_refs=[tests, accepted, commit(root)]).verdict is mod.Verdict.ALLOW
    events = [json.loads(x) for x in (root / "projects/las-voces/activity.ndjson").read_text().splitlines() if "TRANSITION_" in x]
    intent, committed = events[-2:]
    assert intent["evidence_binding_hash"] == committed["evidence_binding_hash"]
    assert intent["implementation_commit_sha"] == committed["implementation_commit_sha"] == sha(root)
    assert intent["test_manifest_digest"] and intent["acceptance_record_digest"]

def test_done_evidence_closure_reads_each_local_object_once(root, monkeypatch):
    in_progress(root); tests, pull, accepted = manifest(root), pr(root), acceptance(root)
    calls = []
    original = mod._evidence_document
    def counted(snapshot_root, ref):
        calls.append(ref)
        return original(snapshot_root, ref)
    monkeypatch.setattr(mod, "_evidence_document", counted)
    assert decide(root, "DONE", [tests, pull, accepted], Trusted()).verdict is mod.Verdict.ALLOW
    assert calls.count(tests) == calls.count(pull) == calls.count(accepted) == 1

def test_verifier_receives_immutable_snapshot_not_later_mutated_file(root):
    in_progress(root); tests, accepted = manifest(root), acceptance(root)
    original = json.loads((root / tests).read_text())
    class MutatingTrusted(Trusted):
        def __init__(self): super().__init__(); self.seen = None
        def execution_record(self, record):
            self.seen = json.loads(json.dumps(record))
            changed = json.loads((root / tests).read_text()); changed["run_id"] = "mutated-after-snapshot"
            (root / tests).write_text(json.dumps(changed))
            return True
    trusted = MutatingTrusted()
    decision, validated = engine(root, trusted)._evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=[tests, accepted, commit(root)])
    assert decision.verdict is mod.Verdict.ALLOW and validated
    assert trusted.seen == original
    assert validated.test_manifest_digest == mod._sha(json.dumps(original).encode())

def test_normalized_digest_is_exact_evidence_snapshot_used_by_verifier(root):
    in_progress(root); tests, accepted = manifest(root), acceptance(root)
    trusted = Trusted()
    decision, validated = engine(root, trusted)._evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=[tests, accepted, commit(root)])
    assert decision.verdict is mod.Verdict.ALLOW and validated
    expected = [{"ref": tests, "sha256": mod._sha((root / tests).read_bytes())}, {"ref": accepted, "sha256": mod._sha((root / accepted).read_bytes())}, {"ref": commit(root), "sha256": mod._sha(commit(root).encode())}]
    assert validated.normalized_evidence_digest == mod._sha(mod._canonical({"task_id":"LV-003", "evidence":sorted(expected, key=lambda x: x["ref"])}))

def test_pr_nested_execution_manifest_is_transitively_bound(root):
    in_progress(root); tests, accepted = manifest(root), acceptance(root)
    nested = write(root, "nested-run.json", json.loads((root / tests).read_text()))
    pull = pr(root)
    data = json.loads((root / pull).read_text()); data["execution_manifest"] = nested; (root / pull).write_text(json.dumps(data))
    controlled = engine(root, Trusted())
    _, first = controlled._evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=[tests, pull, accepted])
    assert first and first.normalized_evidence_digest
    changed = json.loads((root / nested).read_text()); changed["run_id"] = "nested-changed"; (root / nested).write_text(json.dumps(changed))
    _, second = controlled._evaluate(sender_agent=mod.ARIADNA_ID, task_id="LV-003", action="transition_status", target_status="DONE", evidence_refs=[tests, pull, accepted])
    assert second and first.normalized_evidence_digest != second.normalized_evidence_digest

@pytest.mark.parametrize("boundary, state_changed", [("before_intent_append", False), ("before_project_write", False), ("before_commit_append", True)])
def test_persistence_boundary_failures_are_recoverable(root, boundary, state_changed):
    ref = blocker(root)
    def fail(name):
        if name == boundary: raise OSError("injected")
    with pytest.raises(mod.AuthorityError): mod.transition(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="BLOCKED", evidence_refs=[ref], failure_hook=fail)
    status = next(x for x in json.loads((root / "projects/las-voces/project.json").read_text())["tasks"] if x["id"] == "LV-003")["status"]
    assert (status == "BLOCKED") is state_changed
    if boundary != "before_intent_append": assert len(mod.interrupted_transitions(root / "projects/las-voces/activity.ndjson")) == 1

def test_stale_cas_and_interrupted_intent_fail_closed(root):
    ref = blocker(root)
    def race(name):
        if name == "before_project_cas":
            p = root / "projects/las-voces/project.json"; p.write_text(p.read_text() + " ")
    with pytest.raises(mod.AuthorityError): mod.transition(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="BLOCKED", evidence_refs=[ref], failure_hook=race)
    assert mod.interrupted_transitions(root / "projects/las-voces/activity.ndjson")
    assert mod.transition(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="BLOCKED", evidence_refs=[ref]).verdict is mod.Verdict.DENY

def test_concurrent_incompatible_transition_has_one_winner(root):
    ref, results, start = blocker(root), [], threading.Barrier(2)
    def worker():
        start.wait(); results.append(mod.transition(root, sender_agent=mod.ARIADNA_ID, task_id="LV-003", target_status="BLOCKED", evidence_refs=[ref]).verdict)
    a, b = threading.Thread(target=worker), threading.Thread(target=worker); a.start(); b.start(); a.join(); b.join()
    assert results.count(mod.Verdict.ALLOW) == 1 and results.count(mod.Verdict.DENY) == 1

def envelope(): return {"message_id":"m", "project_id":"las-voces", "task_id":"LV-003", "sender_agent":mod.ARIADNA_ID, "recipient_agent":"Thot", "intent":"handoff", "evidence_refs":[], "authority_context":{"handoff":{"owner":"Thot","branch_worktree":"x","scope":"x","acceptance_criteria":"x","commit_pr":"x","test_evidence":"x","blockers":"x","next_action":"x"}}, "correlation_id":"c", "created_at":"2026-09-28T00:00:00Z", "status":"RECORDED"}
def test_handoff_schema_unknown_sender_task_and_project_fail(root):
    mod.validate_handoff(root, envelope())
    bad = envelope(); bad.pop("message_id")
    with pytest.raises(mod.AuthorityError): mod.validate_handoff(root, bad)
    bad = envelope(); bad["sender_agent"] = "unknown"
    with pytest.raises(mod.AuthorityError): mod.validate_handoff(root, bad)
    bad = envelope(); bad["task_id"] = "NOPE"
    with pytest.raises(mod.AuthorityError): mod.validate_handoff(root, bad)
    bad = envelope(); bad["project_id"] = "wrong"
    with pytest.raises(mod.AuthorityError): mod.validate_handoff(root, bad)
