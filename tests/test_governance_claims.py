"""
Test de policy/governance/claims.py — schema estructural del claim.
Dos capas de validación en el subsistema: esta (Pydantic) y la semántica
de validator.py (Task 4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

from pydantic import ValidationError

import claims  # noqa: E402


def _valid_kwargs(**overrides):
    base = dict(
        predicate="FILE_EXISTS",
        args={"path": "las_manos/config.toml", "hash": "a" * 64},
        authority="OBSERVADO",
        provenance_ref="tool_call:read_file:abc123",
        evidence_pointer="las_manos/config.toml",
        scope="jax",
    )
    base.update(overrides)
    return base


def test_claim_valid_construction():
    claim = claims.Claim(**_valid_kwargs())
    assert claim.predicate == "FILE_EXISTS"
    assert claim.authority == "OBSERVADO"


def test_claim_rejects_invalid_authority():
    with pytest.raises(ValidationError):
        claims.Claim(**_valid_kwargs(authority="ADIVINADO"))


def test_claim_rejects_missing_field():
    kwargs = _valid_kwargs()
    del kwargs["provenance_ref"]
    with pytest.raises(ValidationError):
        claims.Claim(**kwargs)


def test_claim_accepts_all_four_authority_values():
    for value in ("EJECUTADO", "OBSERVADO", "RECUPERADO", "INFERIDO"):
        claim = claims.Claim(**_valid_kwargs(authority=value))
        assert claim.authority == value
