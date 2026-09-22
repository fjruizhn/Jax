from jax.memory.b9 import EventKind, Lifecycle, MemoryEvent, MemoryRevision, Visibility, _derive_projection
from jax.memory.lifecycle_worker import projection_mismatches


def _rows():
    revision = {
        "revision_id": "r1", "memory_id": "m1", "content_digest": "sha256:x",
        "visibility": "USER_PRIVATE", "user_id": "u1", "project_id": None,
        "lifecycle_state": "ACTIVE", "created_at": 1.0, "payload": "x",
        "provenance_status": "COMPLETE", "prior_revision_id": None,
    }
    event = {
        "event_id": "e1", "memory_id": "m1", "revision_id": "r1", "event_kind": "CREATE",
        "actor_principal": "service:memory-worker", "subject_user_id": "u1",
        "authority_source": "membership", "occurred_at": 1.0, "details": {},
        "compensates_event_id": None, "actor_type": "SERVICE", "delegation": "conversation:u1",
        "calling_component": "memory-worker", "request_id": "request", "trace_id": "trace",
    }
    canonical = _derive_projection(
        "m1", [MemoryRevision("r1", "m1", "sha256:x", Visibility.USER_PRIVATE, "u1", None,
                                  Lifecycle.ACTIVE, 1.0, "x")],
        [MemoryEvent("e1", "m1", "r1", EventKind.CREATE, "service:memory-worker", "u1", "membership", 1.0,
                     actor_type="SERVICE", delegation="conversation:u1", calling_component="memory-worker",
                     request_id="request", trace_id="trace")],
    )
    projection = {
        "memory_id": "m1", "current_revision_id": canonical.current_revision_id,
        "current_lifecycle_state": canonical.current_lifecycle.value,
        "current_verification_state": canonical.current_verification,
        "canonical_history_digest": canonical.canonical_history_digest,
    }
    return revision, event, projection


def test_canonical_projection_does_not_require_reconciliation():
    revision, event, projection = _rows()
    assert projection_mismatches([revision], [event], [projection]) == ()


def test_projection_divergence_is_detected_without_repairing_it():
    revision, event, projection = _rows()
    projection["current_lifecycle_state"] = "VERIFIED"
    assert projection_mismatches([revision], [event], [projection]) == ("m1",)


def test_missing_projection_is_reconciliation_required():
    revision, event, _ = _rows()
    assert projection_mismatches([revision], [event], []) == ("m1",)
