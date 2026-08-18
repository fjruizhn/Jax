"""
Test de policy/governance/vocab_sweep.py — barrido léxico puro, sin I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import vocab_sweep  # noqa: E402

VOCAB = frozenset({"trae a hyde", "hyde", "code_swarm"})


def test_sweep_no_matches_returns_empty():
    assert vocab_sweep.sweep("un texto sin nada prohibido", VOCAB) == []


def test_sweep_finds_known_term():
    assert vocab_sweep.sweep("invocá a hyde ahora", VOCAB) == ["hyde"]


def test_sweep_finds_multiple_terms_sorted():
    result = vocab_sweep.sweep("code_swarm y también trae a hyde", VOCAB)
    assert result == sorted(result)
    assert "code_swarm" in result
    assert "trae a hyde" in result
