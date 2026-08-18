"""
Test de policy/governance/loaders.py — I/O de config ESTÁTICA de policy
(predicates.yaml, closed_vocabulary.yaml, render_templates.yaml + hash).

El primer test es el fail-closed de load_templates(): si el hash de
render_templates.yaml no coincide con el registrado en VERSION, el
subsistema no debe cargar nada — ni parcial, ni con warning. Va primero
a propósito (ver plan, Task 3): es la prueba que faltó en
backup-hall9000.sh y no debe ser la que se recorta bajo presión.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import loaders  # noqa: E402


def test_load_templates_fails_closed_on_hash_mismatch(tmp_path, monkeypatch):
    templates_file = tmp_path / "render_templates.yaml"
    templates_file.write_text(
        "templates:\n  FOO:\n    status: definida\n    template: 'x'\n",
        encoding="utf-8",
    )
    version_file = tmp_path / "VERSION"
    version_file.write_text(
        "version: 0.1.0\nsha256: deadbeef\ntemplates_sha256: 0000000000\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(loaders, "TEMPLATES_FILE", templates_file)
    monkeypatch.setattr(loaders, "VERSION_FILE", version_file)

    with pytest.raises(RuntimeError, match="Hash de"):
        loaders.load_templates()


def test_load_templates_happy_path_returns_real_specs():
    templates = loaders.load_templates()
    assert templates["CAPABILITY_AVAILABLE"].status == "definida"
    assert templates["CAPABILITY_AVAILABLE"].template == (
        "La capability {name} está disponible en modo {mode}."
    )
    assert templates["FACET_EXISTS"].status == "pendiente"
    assert templates["FACET_EXISTS"].template is None
