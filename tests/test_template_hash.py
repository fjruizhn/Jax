"""
Test de policy/tools/template_hash.py — hash de render_templates.yaml,
mismo algoritmo que corpus_hash.py pero para un solo archivo.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

POLICY_TOOLS = Path(__file__).resolve().parent.parent / "policy" / "tools"
sys.path.insert(0, str(POLICY_TOOLS))

import template_hash  # noqa: E402

TEMPLATES_FILE = Path(__file__).resolve().parent.parent / "policy" / "templates" / "render_templates.yaml"


def test_compute_hash_matches_manual_sha256():
    expected = hashlib.sha256(TEMPLATES_FILE.read_bytes()).hexdigest()
    assert template_hash.compute_hash() == expected
