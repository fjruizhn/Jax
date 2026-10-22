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
    assert first.view.caller == "user:2"
    assert second is not None and second.observed_at >= first.observed_at
    store.update(job_id, status=ProcessingJobStatus.COMPLETED.value)
    assert store.authoritative_snapshot(job_id).view.status.value == "completed"


def test_ownerless_events_are_not_authoritative(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"job_id":"legacy","status":"completed","motor":"n/a","capability":"x","caller":"x","trace_id":"t","created_at":1,"recursion_depth":0}) + "\n")
    store = ProcessingJobStore(str(path))
    assert store.authoritative_snapshot("legacy") is None
    # A complete legacy record is quarantined by job ID.  It cannot poison
    # unrelated owner-bound records, but can never later be promoted.
    assert store.authoritative_history_intact


@pytest.mark.parametrize("changed", ("tenant_id", "user_id", "project_id"))
def test_replay_owner_drift_fails_closed_for_every_job(tmp_path, changed):
    path = tmp_path / "jobs.jsonl"
    writer = ProcessingJobStore(str(path))
    drifted_job = writer.create(ownership=OWNER, caller="x", capability="x", motor="n/a", trace_id="t", prompt="", recursion_depth=0)
    unrelated_job = writer.create(ownership=OWNER, caller="x", capability="x", motor="n/a", trace_id="u", prompt="", recursion_depth=0)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    owner_b = {**OWNER.__dict__, changed: "9"}
    drifted_event = {
        **events[0], "status": "completed", changed: "9",
        "processing_ownership": owner_b,
    }
    if changed == "user_id":
        drifted_event["caller"] = "user:9"
    path.write_text("".join(json.dumps(event) + "\n" for event in (*events, drifted_event)))

    reloaded = ProcessingJobStore(str(path))
    assert not reloaded.authoritative_history_intact
    assert reloaded.authoritative_snapshot(drifted_job) is None
    assert reloaded.authoritative_snapshot(unrelated_job) is None


@pytest.mark.parametrize("changed", ("tenant_id", "user_id", "project_id"))
@pytest.mark.parametrize("bad_value", ("9", 9, None), ids=("different-string", "numeric-not-string", "missing"))
def test_replay_flat_owner_mismatch_fails_closed_for_every_job(tmp_path, changed, bad_value):
    path = tmp_path / "jobs.jsonl"
    writer = ProcessingJobStore(str(path))
    contradictory_job = writer.create(ownership=OWNER, caller="x", capability="x", motor="n/a", trace_id="t", prompt="", recursion_depth=0)
    unrelated_job = writer.create(ownership=OWNER, caller="x", capability="x", motor="n/a", trace_id="u", prompt="", recursion_depth=0)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    contradictory = dict(events[0])
    if bad_value is None:
        contradictory.pop(changed)
    else:
        contradictory[changed] = bad_value
    path.write_text("".join(json.dumps(event) + "\n" for event in (contradictory, events[1])))

    reloaded = ProcessingJobStore(str(path))
    assert not reloaded.authoritative_history_intact
    assert reloaded.authoritative_snapshot(contradictory_job) is None
    assert reloaded.authoritative_snapshot(unrelated_job) is None


def test_ownerless_job_stays_quarantined_while_owned_job_remains_available(tmp_path):
    path = tmp_path / "jobs.jsonl"
    legacy = {"job_id":"legacy","status":"completed","motor":"n/a","capability":"x","caller":"x","trace_id":"t","created_at":1,"recursion_depth":0}
    owned = {**legacy, "job_id": "owned", "caller": "user:2", "tenant_id": "1", "user_id": "2", "project_id": "3", "processing_ownership": OWNER.__dict__}
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


def test_caller_is_derived_and_cannot_be_forged_or_updated(tmp_path):
    path = tmp_path / "jobs.jsonl"
    store = ProcessingJobStore(str(path))
    job_id = store.create(ownership=OWNER, caller="forged-user", capability="x", motor="n/a", trace_id="t", prompt="", recursion_depth=0)
    assert store.get(job_id).caller == "user:2"
    lines_before = path.read_text().splitlines()
    with pytest.raises(ValueError):
        store.update(job_id, caller="forged-user")
    assert path.read_text().splitlines() == lines_before


@pytest.mark.parametrize("bad_value", ("user:9", None, 9), ids=("changed", "missing", "non-string"))
def test_replay_caller_contradiction_fails_closed(tmp_path, bad_value):
    path = tmp_path / "jobs.jsonl"
    writer = ProcessingJobStore(str(path))
    job_id = writer.create(ownership=OWNER, caller="ignored", capability="x", motor="n/a", trace_id="t", prompt="", recursion_depth=0)
    unrelated_job = writer.create(ownership=OWNER, caller="ignored", capability="x", motor="n/a", trace_id="u", prompt="", recursion_depth=0)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    contradictory = dict(events[0])
    if bad_value is None:
        contradictory.pop("caller")
    else:
        contradictory["caller"] = bad_value
    path.write_text("".join(json.dumps(event) + "\n" for event in (contradictory, events[1])))
    reloaded = ProcessingJobStore(str(path))
    assert not reloaded.authoritative_history_intact
    assert reloaded.authoritative_snapshot(job_id) is None
    assert reloaded.authoritative_snapshot(unrelated_job) is None


@pytest.mark.parametrize("tail", ("complete-without-newline", "truncated"))
def test_replay_nonnewline_or_truncated_final_record_fails_closed(tmp_path, tail):
    path = tmp_path / "jobs.jsonl"
    writer = ProcessingJobStore(str(path))
    job_id = writer.create(ownership=OWNER, caller="user:2", capability="x", motor="n/a", trace_id="t", prompt="", recursion_depth=0)
    valid_event = json.loads(path.read_text().splitlines()[0])
    if tail == "complete-without-newline":
        final_record = json.dumps({**valid_event, "job_id": "later"})
    else:
        final_record = '{"job_id":"later"'
    path.write_text(path.read_text() + final_record)

    reloaded = ProcessingJobStore(str(path))
    assert not reloaded.authoritative_history_intact
    assert reloaded.authoritative_snapshot(job_id) is None


def test_closed_statuses_and_source_configuration(tmp_path):
    store = ProcessingJobStore(str(tmp_path / "jobs.jsonl"))
    assert [item.value for item in ProcessingJobStatus] == ["pending", "running", "cancelling", "completed", "failed", "cancelled"]
    assert set(store.source_configuration()) == {"store_contract", "source_id", "event_format", "durability", "ownership_contract", "source_role", "allowed_statuses"}
