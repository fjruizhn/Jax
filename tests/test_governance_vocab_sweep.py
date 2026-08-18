"""
Test de policy/governance/vocab_sweep.py — barrido léxico puro, sin I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import vocab_sweep  # noqa: E402

TERM_CATEGORIES = {
    "trae a hyde": frozenset({"commands"}),
    "hyde": frozenset({"facets_las_manos", "facets_jax"}),
    "code_swarm": frozenset({"capabilities"}),
    "ada": frozenset({"facets_las_manos", "facets_jax", "motors"}),
}


def test_sweep_no_matches_returns_empty():
    assert vocab_sweep.sweep("un texto sin nada prohibido", TERM_CATEGORIES) == []


def test_sweep_finds_known_term_with_categories():
    result = vocab_sweep.sweep("invocá a hyde ahora", TERM_CATEGORIES)
    assert result == [("hyde", frozenset({"facets_las_manos", "facets_jax"}))]


def test_sweep_finds_multiple_terms_sorted():
    result = vocab_sweep.sweep("code_swarm y también trae a hyde", TERM_CATEGORIES)
    terms = [t for t, _ in result]
    assert terms == sorted(terms)
    assert ("code_swarm", frozenset({"capabilities"})) in result
    assert ("trae a hyde", frozenset({"commands"})) in result


def test_sweep_term_with_multiple_categories():
    result = vocab_sweep.sweep("dale, trae a ada", TERM_CATEGORIES)
    assert result == [("ada", frozenset({"facets_las_manos", "facets_jax", "motors"}))]


def test_sweep_no_false_positive_on_substring_inside_word():
    tc = {"ada": frozenset({"motors"})}
    assert vocab_sweep.sweep("no hay nada que hacer, cada faceta", tc) == []


def test_sweep_case_insensitive_match():
    tc = {"hyde": frozenset({"facets_jax"})}
    result = vocab_sweep.sweep("invocá a HYDE ahora", tc)
    assert result == [("hyde", frozenset({"facets_jax"}))]
