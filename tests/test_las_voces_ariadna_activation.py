"""ACTIVE_GOVERNED lifecycle is human-approved, fail-closed, and bounded."""
from __future__ import annotations
import importlib.util, json, shutil, sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
auth=load("activation_auth", REPO/"projects/las-voces/authority/ariadna_authority.py")
runtime=load("activation_runtime", REPO/"projects/las-voces/authority/ariadna_runtime.py")

@pytest.fixture()
def root(tmp_path):
    shutil.copytree(REPO/"projects/las-voces", tmp_path/"projects/las-voces")
    project=tmp_path/"projects/las-voces/project.json"; data=json.loads(project.read_text()); next(x for x in data["agents"] if x["name"]=="Ariadna")["lifecycle_status"]="PROPOSED_NOT_ACTIVE"; project.write_text(json.dumps(data))
    agent=tmp_path/"projects/las-voces/agents/ariadna.json"; data=json.loads(agent.read_text()); data["lifecycle_status"]="PROPOSED_NOT_ACTIVE"; agent.write_text(json.dumps(data))
    history=tmp_path/"projects/las-voces/activity.ndjson"; history.write_text("\n".join(x for x in history.read_text().splitlines() if "HUMAN_AUTHORITY_ARIADNA_ACTIVATION_APPROVED" not in x)+"\n")
    return tmp_path
def activate(root):
    project=root/"projects/las-voces/project.json"; data=json.loads(project.read_text()); next(x for x in data["agents"] if x["name"]=="Ariadna")["lifecycle_status"]="ACTIVE_GOVERNED"; project.write_text(json.dumps(data))
    agent=root/"projects/las-voces/agents/ariadna.json"; data=json.loads(agent.read_text()); data["lifecycle_status"]="ACTIVE_GOVERNED"; agent.write_text(json.dumps(data))
    event={"event_id":"lv-004-002","event_type":"HUMAN_AUTHORITY_ARIADNA_ACTIVATION_APPROVED","project_id":"las-voces","actor":"Fernando / Human Authority","decision":"ACTIVATE ARIADNA AS GOVERNED AUTONOMOUS PROJECT MANAGER","target_lifecycle":"ACTIVE_GOVERNED","scope":"LAS VOCES PM runtime","status":"RECORDED","evidence_refs":["authority/CONTRACT.md","authority/ariadna_authority.py","authority/ariadna_runtime.py","project.json","git:cd0905dbe7aca9f146a255f0eca566aecaea6215","git:36ce608425aab759200d2c277df43c00c1958744"]}
    with (root/"projects/las-voces/activity.ndjson").open("a") as f: f.write(json.dumps(event)+"\n")
def test_active_lifecycle_requires_one_valid_human_event(root):
    assert not auth.activation_approved(root)
    activate(root); assert auth.activation_approved(root)
    with (root/"projects/las-voces/activity.ndjson").open("a") as f: f.write('{"event_id":"lv-004-002"}\n')
    assert not auth.activation_approved(root)
def test_active_does_not_expand_authority_or_verifier_injection(root):
    activate(root); engine=auth.AuthorityEngine(root); assert engine.activation_approved()
    for action in auth.FORBIDDEN: assert engine.evaluate(sender_agent=auth.ARIADNA_ID,task_id="LV-004",action=action).verdict is auth.Verdict.DENY
    with pytest.raises(TypeError): engine.evaluate(sender_agent=auth.ARIADNA_ID,task_id="LV-004",action="transition_status",trust=object())
def test_runtime_requires_active_approval_and_stop_remains_host_controlled(root):
    stopped=runtime.AriadnaRuntime(root,auth.AuthorityEngine(root)); assert stopped.start() is runtime.Lifecycle.STOPPED
    activate(root); host=runtime.AriadnaRuntime(root,auth.AuthorityEngine(root)); assert host.start() is runtime.Lifecycle.READY
    host.request_stop(); proposal=runtime.Proposal("LV-004","runtime_execution",runtime.project_hash(root/"projects/las-voces/project.json"))
    assert host.run_once(proposal) == "NOOP_NOT_READY"
