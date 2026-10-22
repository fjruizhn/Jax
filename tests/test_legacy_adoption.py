"""Pure adoption gates, temporary private artifacts, no production config."""
import asyncio
from datetime import datetime
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from jax.memory import legacy_adoption as adoption


def row(**changes):
    return dict(id=1,user_id=1,project_id=None,fact_text="private text",superseded_by=None,expires_at=None,**changes)


def owner(**changes):
    base={"user_id":1,"tenant_id":7,"status":"active"};base.update(changes);return {"1":base}


def classify(source=None, users=None, refs=()):
    return adoption.classify("facts",source or row(),owner() if users is None else users,{}, {},refs,observed_at=datetime(2026,9,26))


def test_owner_from_user_database_never_inferred_from_content():
    reason,dest=classify()
    assert reason is None and dest["tenant_id"]=="7" and dest["user_id"]=="1"
    assert classify(users=owner(status="disabled"))[0]=="OWNER_UNAVAILABLE"
    source=row();source["user_id"]=2
    assert classify(source=source)[0]=="OWNER_UNAVAILABLE"


def test_conflicting_reference_or_unbound_project_quarantined():
    assert classify(refs=[{"user_id":2,"tenant_id":7,"project_id":None}])[0]=="REFERENCE_SCOPE_CONFLICT"
    assert classify(refs=[None])[0]=="REFERENCE_MISSING"
    source=row();source["project_id"]=9001
    assert classify(source=source)[0]=="PROJECT_SCOPE_UNBOUND"


def test_superseded_expired_and_completed_never_revived():
    source=row();source["expires_at"]="2000-01-01T00:00:00"
    assert classify(source=source)[0]=="EXPIRED"
    assert not adoption.source_eligible("facts",source)
    source=row();source["superseded_by"]=3
    assert classify(source=source)[0]=="SUPERSEDED"
    assert not adoption.source_eligible("action_items",{"description":"x","status":"done"})


def test_artifacts_private_no_contents_or_digests_stdout(tmp_path,capsys):
    path=tmp_path/"plan.json"
    adoption.private_write(path,{"secret":"private"})
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert adoption.private_read(path)=={"secret":"private"}
    assert capsys.readouterr().out==""
    path.chmod(0o644)
    with pytest.raises(ValueError): adoption.private_read(path)


def test_plan_change_rejected_and_row_digest_ignores_only_vectors():
    source=row();snapshot=adoption.snapshot_row(source)
    item={"table":"facts","legacy_id":"1","source_snapshot":snapshot,"source_digest":adoption.digest(snapshot)}
    plan={"schema_version":1,"items":[item]};plan["plan_digest"]=adoption.digest(plan)
    adoption.validate_plan(plan)
    item["source_snapshot"]["user_id"]=2
    with pytest.raises(ValueError): adoption.validate_plan(plan)
    assert adoption.row_digest(dict(source,embedding=b"vector"))==adoption.row_digest(source)
    assert adoption.row_digest(dict(source,user_id=2))!=adoption.row_digest(source)


def test_invalid_backup_proof_rejected(tmp_path):
    with pytest.raises(ValueError): adoption.validate_backup({"restoration_verified":False})


def test_quarantine_performs_no_api_calls_and_stale_snapshot_fails(tmp_path,monkeypatch):
    async def run():
        source=row();item={"table":"facts","legacy_id":"1","source_snapshot":source,"source_digest":adoption.row_digest(source),"quarantine_reason":"PROJECT_SCOPE_UNBOUND","destination":None}
        plan={"schema_version":1,"namespace":"legacy","items":[item]};plan["plan_digest"]=adoption.digest(plan)
        pp=tmp_path/"plan.json";mp=tmp_path/"backup.json"
        adoption.private_write(pp,plan);adoption.private_write(mp,{})
        monkeypatch.setattr(adoption,"validate_backup",lambda _:None)
        api=SimpleNamespace(import_legacy_memory=AsyncMock())
        assert await adoption.apply_plan(api,pp,mp,actor_user_id=1)=={"adopted":0,"quarantined":1}
        api.import_legacy_memory.assert_not_called()
        item["quarantine_reason"]=None;item["destination"]={"tenant_id":"7","user_id":"1","project_id":None,"visibility":"USER_PRIVATE"}
        plan.pop("plan_digest");plan["plan_digest"]=adoption.digest(plan)
        pp2=tmp_path/"plan2.json";adoption.private_write(pp2,plan)
        api.import_legacy_memory.side_effect=RuntimeError("source snapshot changed")
        with pytest.raises(RuntimeError):await adoption.apply_plan(api,pp2,mp,actor_user_id=1)
        assert api.import_legacy_memory.call_args.kwargs["expected_source_digest"]==adoption.row_digest(source)
    asyncio.run(run())


def test_backup_requires_restored_table_names_counts_and_identical_artifact(tmp_path):
    path=tmp_path/"db.sql";path.write_bytes(b"restorable backup")
    tables=["facts","decisions","action_items","jax_users","messages","conversations","memory_objects","memory_revisions","memory_events","memory_projections","memory_legacy_bindings"]
    proof={"restoration_verified":True,"restored_at":"2026-09-26T01:00:00","restored_tables":tables,"restored_counts":dict.fromkeys(tables,0),"backup_path":str(path),"backup_size":path.stat().st_size,"backup_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    adoption.validate_backup(proof)
    missing=dict(proof,restored_counts={})
    with pytest.raises(ValueError,match="counts"):adoption.validate_backup(missing)
    path.write_bytes(b"altered")
    with pytest.raises(ValueError):adoption.validate_backup(proof)


def test_partial_supersession_metadata_remains_quarantined():
    source=row();source["superseded_at"]="2026-09-25T01:00:00"
    assert classify(source=source)[0]=="SUPERSEDED"
    assert not adoption.source_eligible("facts",source)
    source=row();source["superseded_by_user"]=1
    assert classify(source=source)[0]=="SUPERSEDED"
