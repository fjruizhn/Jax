from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parents[2]
def text(path): return (ROOT/path).read_text(encoding="utf-8")

def test_root_adapter_has_required_boundaries_without_policy_copy():
    value=text("AGENTS.md")
    for token in ("ADAPTER != AUTHORITY", "policy/**", "Memory may", "capability confers no JAX authority", "Before claiming a fact"):
        assert token in value
    assert "CTL." not in value

def test_operations_inventory_and_runbooks_are_complete():
    for name in ("query-guide.md","service-map.md","database-map.md","trusted-files.md","composition-map.md","health-and-diagnostics.md","document-precedence.md","secret-handling.md","generated-docs.md","operational-manual.md"):
        assert (ROOT/"docs/operations"/name).is_file()
    required=("service-lifecycle","database-migrations","authority-root-recovery","trusted-approver-recovery","implementation-identity","decision-replay","execution-reconstruction","evidence-recovery","kill-switch","mariadb-outage","authoritative-backup-restore","ci-pr-recovery","shared-workspace-branch-drift","worker-timeout-cancel","deployment-rollback")
    headings=("Purpose","Scope","Preconditions","Authority impact","Safe procedure","Verification","Fail-closed condition","Recovery / escalation","Prohibited actions")
    for name in required:
        value=text(Path("docs/runbooks")/(name+".md"))
        assert all("## "+heading in value for heading in headings)

def test_generated_manual_is_marked_and_reproducible():
    subprocess.run([sys.executable,"scripts/generate_operational_manual.py","--check"],cwd=ROOT,check=True)
    value=text("docs/operations/generated/jax-operational-docs.html")
    assert "NON-AUTHORITATIVE GENERATED PRESENTATION" in value
    assert "size: Letter" in value and "operational-manual.md" in value

def test_jaxctl_is_readonly_command_surface():
    result=subprocess.run([sys.executable,"-m","jaxctl","--help"],cwd=ROOT,capture_output=True,text=True,check=True)
    assert all(word not in result.stdout for word in ("dispatch", "grant", "approve", "cancel"))

def test_readonly_b7_composition_is_not_a_public_dependency_injection_factory():
    from policy.enforcement_evidence.status_engine import EnforcementStatusService, _runtime_readonly_status_service
    import inspect
    assert not hasattr(EnforcementStatusService, "for_readonly_query")
    assert tuple(inspect.signature(EnforcementStatusService.query_control_status).parameters) == (
        "self", "control_id", "control_version", "claim_level", "scope", "subjects", "as_of_utc")
    assert tuple(inspect.signature(_runtime_readonly_status_service).parameters) == ()

def test_jaxctl_control_routes_only_to_fixed_readonly_composition(monkeypatch):
    import jaxctl.runtime as runtime
    from jaxctl.commands import run
    called={}
    class Reader:
        def query_control_status(self, **kwargs):
            called.update(kwargs); return {"persisted":False,"classification":"AUTHORITATIVE_READONLY_DERIVATION"}
    monkeypatch.setattr(runtime,"_runtime_readonly_status_service",lambda: Reader())
    assert run(["control","CTL.B6.GOVERNED_DISPATCH","--version","1","--claim","ENFORCED","--scope",'{"environment":"SANDBOX_RUNTIME"}',"--subjects",'[{"subject_type":"EXECUTION","identity":"x"}]',"--json"]) == 0
    assert called["control_id"] == "CTL.B6.GOVERNED_DISPATCH"

def test_jaxctl_replay_invokes_block5_replay_not_load_only(monkeypatch):
    import jaxctl.runtime as runtime
    import policy.decision_record.service as decision_service
    import policy.decision_record.replay as decision_replay
    from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
    record={"decision_id":"d-1"}; called={}
    monkeypatch.setattr(runtime,"_connection_factory",lambda: (lambda: object()))
    monkeypatch.setattr(decision_service,"load_decision",lambda store, decision_id: record)
    monkeypatch.setattr(TrustedAuthorityRoot,"load",classmethod(lambda cls: object()))
    result=type("Replay",(),{"status":type("Status",(),{"value":"REPLAY_MATCH"})()})()
    def replay(value, authority_store, trusted_root, checkpoint_store, **kwargs):
        called.update(value=value,authority_store=authority_store,trusted_root=trusted_root,checkpoint_store=checkpoint_store)
        return result
    monkeypatch.setattr(decision_replay,"replay_decision",replay)
    result_value=runtime.decision("d-1", replay=True)
    assert result_value["operation"] == "REPLAY"
    assert called["value"] is record
