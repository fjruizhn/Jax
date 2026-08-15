"""REFORMAS-v3 §3.1.4 — forma exacta del rechazo tipado CAPABILITY_UNBOUND."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jacobs.plan import CapabilityUnbound  # noqa: E402


def test_capability_unbound_forma_exacta():
    cu = CapabilityUnbound(
        required=["read_audit_log"],
        candidates=["hyde", "gptoss_120b"],
        task_id="task-123",
    )
    assert cu.to_dict() == {
        "status": "CAPABILITY_UNBOUND",
        "required": ["read_audit_log"],
        "candidates": ["hyde", "gptoss_120b"],
        "task_id": "task-123",
    }


def test_capability_unbound_status_no_es_parametro():
    """status siempre es CAPABILITY_UNBOUND — no se puede pisar por afuera."""
    cu = CapabilityUnbound(required=["x"], candidates=[], task_id="t")
    assert cu.status == "CAPABILITY_UNBOUND"
