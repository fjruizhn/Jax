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
    subprocess.run([sys.executable,"scripts/generate_operational_manual.py"],cwd=ROOT,check=True)
    value=text("docs/operations/generated/jax-operational-docs.html")
    assert "NON-AUTHORITATIVE GENERATED PRESENTATION" in value
    assert "size: Letter" in value and "operational-manual.md" in value

def test_jaxctl_is_readonly_command_surface():
    result=subprocess.run([sys.executable,"-m","jaxctl","--help"],cwd=ROOT,capture_output=True,text=True,check=True)
    assert all(word not in result.stdout for word in ("dispatch", "grant", "approve", "cancel"))
