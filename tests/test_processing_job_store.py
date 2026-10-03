import json
import pytest
from processing_job_store import ProcessingJobStatus, ProcessingJobStore
from processing_ownership import ProcessingOwnershipContext


OWNER = ProcessingOwnershipContext("processing-owner.1", "1", "2", "3")


def test_processing_store_persists_owner_and_fresh_observation(tmp_path):
    store = ProcessingJobStore(str(tmp_path / "jobs.jsonl"))
    job_id = store.create(ownership=OWNER, caller="jax-platform:proyectos-documentos", capability="ingesta_archivos", motor="n/a", trace_id="t", prompt="", recursion_depth=0)
    first, second = store.authoritative_snapshot(job_id), store.authoritative_snapshot(job_id)
    assert first is not None and first.owner == OWNER and first.view.status.value == "pending"
    assert (first.view.tenant_id, first.view.user_id, first.view.project_id) == ("1", "2", "3")
    assert second is not None and second.observed_at >= first.observed_at
    store.update(job_id, status=ProcessingJobStatus.COMPLETED.value)
    assert store.authoritative_snapshot(job_id).view.status.value == "completed"


def test_ownerless_and_owner_drift_events_are_not_authoritative(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"job_id":"legacy","status":"completed","motor":"n/a","capability":"x","caller":"x","trace_id":"t","created_at":1,"recursion_depth":0}) + "\n")
    store = ProcessingJobStore(str(path))
    assert store.authoritative_snapshot("legacy") is None
    # A complete legacy record is quarantined by job ID.  It cannot poison
    # unrelated owner-bound records, but can never later be promoted.
    assert store.authoritative_history_intact


def test_ownerless_job_stays_quarantined_while_owned_job_remains_available(tmp_path):
    path = tmp_path / "jobs.jsonl"
    legacy = {"job_id":"legacy","status":"completed","motor":"n/a","capability":"x","caller":"x","trace_id":"t","created_at":1,"recursion_depth":0}
    owned = {**legacy, "job_id": "owned", "processing_ownership": OWNER.__dict__}
    # A later owner-bearing event for the legacy id cannot promote it.
    promoted = {**owned, "job_id": "legacy"}
    path.write_text("".join(json.dumps(event) + "\n" for event in (legacy, owned, promoted)))
    store = ProcessingJobStore(str(path))
    assert store.authoritative_history_intact
    assert store.authoritative_snapshot("legacy") is None
    assert store.authoritative_snapshot("owned").owner == OWNER


def test_corrupt_history_and_invalid_requested_status_fail_closed(tmp_path):
    path = tmp_path / "jobs.jsonl"
    store = ProcessingJobStore(str(path))
    job_id = store.create(ownership=OWNER, caller="x", capability="x", motor="n/a", trace_id="t", prompt="", recursion_depth=0)
    with pytest.raises(ValueError):
        store.update(job_id, status="invented")
    path.write_text(path.read_text() + '{"job_id":"bad","job_id":"duplicate"}\n')
    reloaded = ProcessingJobStore(str(path))
    assert not reloaded.authoritative_history_intact
    assert reloaded.authoritative_snapshot(job_id) is None


def test_closed_statuses_and_source_configuration(tmp_path):
    store = ProcessingJobStore(str(tmp_path / "jobs.jsonl"))
    assert [item.value for item in ProcessingJobStatus] == ["pending", "running", "cancelling", "completed", "failed", "cancelled"]
    assert set(store.source_configuration()) == {"store_contract", "source_id", "event_format", "durability", "ownership_contract", "source_role", "allowed_statuses"}
