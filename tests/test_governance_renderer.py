"""
Test de policy/governance/renderer.py — motor de plantillas, puro.
Deliberadamente agnóstico de qué predicados tienen resolver real: prueba
con templates ya cargados y claims sintéticos, sin pasar por validator.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import claims  # noqa: E402
import loaders  # noqa: E402
import renderer  # noqa: E402

TEMPLATES = {
    "CAPABILITY_AVAILABLE": loaders.TemplateSpec(
        status="definida", template="La capability {name} está disponible en modo {mode}."
    ),
    "FACET_EXISTS": loaders.TemplateSpec(status="pendiente", template=None),
}


def _claim(**overrides):
    base = dict(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
        authority="OBSERVADO",
        provenance_ref="test",
        evidence_pointer="test",
        scope="jax",
    )
    base.update(overrides)
    return claims.Claim(**base)


def test_render_known_predicate_with_definida_template():
    claim = _claim()
    text = renderer.render(claim, TEMPLATES)
    assert text == "La capability code_swarm está disponible en modo read_only."


def test_render_raises_for_pendiente_template():
    claim = _claim(predicate="FACET_EXISTS", args={"name": "hyde", "engine": "kimi"})
    with pytest.raises(ValueError, match="plantilla"):
        renderer.render(claim, TEMPLATES)


def test_render_raises_for_predicate_without_template_entry():
    claim = _claim(predicate="JOB_STATUS", args={"job_id": "1", "status": "done"})
    with pytest.raises(ValueError, match="plantilla"):
        renderer.render(claim, TEMPLATES)
