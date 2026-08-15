"""
REFORMAS-v3 §3.1.5 (R3.5) — cada motor recibe: quién es, qué capabilities
tiene en esta tarea, qué motores existen y qué puede cada uno, la lista de
predicados emitibles, y el protocolo de rechazo tipado.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
sys.path.insert(0, str(LAS_MANOS))

from motor_registry.identity_context import build_identity_context  # noqa: E402


def test_identity_context_incluye_los_cinco_elementos_de_r35():
    catalog = {
        "kimi": {"allowed_motors_for": ["code_swarm", "bug_hunt"]},
        "ada": {"allowed_motors_for": ["architecture_review"]},
    }
    predicates = ["CAPABILITY_AVAILABLE", "FACET_EXISTS", "ENGINE_STATUS",
                  "CONFIG_VALUE", "FILE_EXISTS", "AUDIT_EVENT_EXISTS",
                  "JOB_STATUS", "MEMORY_ENTRY_EXISTS"]

    ctx = build_identity_context(
        motor_name="kimi",
        capabilities=["code_swarm"],
        catalog=catalog,
        predicates=predicates,
        task_id="task-42",
    )

    assert "kimi" in ctx  # quién es
    assert "code_swarm" in ctx  # qué capabilities tiene en esta tarea
    assert "ada" in ctx  # qué otros motores existen
    assert "architecture_review" in ctx  # qué puede cada uno
    assert "AUDIT_EVENT_EXISTS" in ctx  # predicados emitibles
    assert "CAPABILITY_UNBOUND" in ctx  # protocolo de rechazo tipado
    assert "task-42" in ctx
