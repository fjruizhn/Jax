"""
REFORMAS-v3 Fase 1: read_audit_log se otorga por contrato de tarea, no por
identidad de facet — cualquier facet puede pedirla aunque su config.toml
[facets.<x>] allowed_ops no la incluya explícitamente.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
sys.path.insert(0, str(LAS_MANOS))

from policy import PolicyEngine  # noqa: E402


def _engine_without_audit_log_read_for(facet: str) -> PolicyEngine:
    config = {
        "facets": {
            facet: {
                "allowed_envs": ["local"],
                "allowed_ops": ["read_file"],  # NO incluye audit_log_read
                "can_write_prod": False,
            }
        },
        "ops": {"audit_log_read": {}, "read_file": {}},
        "environments": {"local": ["127.0.0.1"]},
    }
    return PolicyEngine(config)


def test_read_audit_log_otorgada_por_contrato_no_por_facet():
    engine = _engine_without_audit_log_read_for("hipatia")
    result = engine.check(
        facet="hipatia",
        operation="audit_log_read",
        target_host="127.0.0.1",
    )
    assert result.ok, f"debía otorgarse por contrato; razón de rechazo: {result.reason}"


def test_read_file_sigue_gateado_por_allowed_ops():
    """Control: el cambio NO afloja otras operaciones, solo audit_log_read."""
    engine = _engine_without_audit_log_read_for("hipatia")
    result = engine.check(
        facet="hipatia",
        operation="read_file",
        target_host="127.0.0.1",
    )
    assert result.ok  # read_file SÍ está en allowed_ops de este fixture

    engine2 = _engine_without_audit_log_read_for("hipatia")
    result2 = engine2.check(
        facet="hipatia",
        operation="list_dir",  # ni siquiera está en self.ops del fixture
        target_host="127.0.0.1",
    )
    assert not result2.ok
