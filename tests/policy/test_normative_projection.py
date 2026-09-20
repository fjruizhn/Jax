from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from policy.canonicalization.canonical_json import canonical_json_bytes
from policy.canonicalization.errors import SchemaValidationError
from policy.canonicalization.projection import normative_projection


def _rule() -> dict:
    return {
        "id": "P01",
        "statement": "La regla sustantiva.",
        "scope": "policy",
        "blocking": False,
        "origin": "Documento verificable.",
        "status": "CULTURAL",
        "enforcement": {"mechanism": None, "test": None},
        "version": "0.1",
        "created": "2026-08-15",
        "amended_by": None,
        "notes": "Texto para personas.",
        "history": ["migrada"],
        "document_reference": "docs/historico.md",
    }


def _projection_digest(rule: dict) -> str:
    return sha256(canonical_json_bytes(normative_projection(rule))).hexdigest()


def test_display_y_provenance_no_cambian_hash() -> None:
    left = _rule()
    right = deepcopy(left)
    right.update(
        origin="Otra fuente igualmente presente.",
        version="9.9",
        notes="Otra presentación.",
    )
    assert _projection_digest(left) == _projection_digest(right)


@pytest.mark.parametrize(
    ("field", "value"),
    [("statement", "Otra regla."), ("scope", "runtime"), ("blocking", True)],
)
def test_statement_scope_blocking_cambian_hash(field: str, value: object) -> None:
    left = _rule()
    right = deepcopy(left)
    right[field] = value
    assert _projection_digest(left) != _projection_digest(right)


def test_history_y_referencia_documental_no_cambian_hash() -> None:
    left = _rule()
    right = deepcopy(left)
    right["history"] = ["otro evento"]
    right["document_reference"] = "docs/otro.md"
    assert _projection_digest(left) == _projection_digest(right)


def test_unknown_field_rechaza() -> None:
    rule = _rule()
    rule["campo_inventado"] = "no permitido"
    with pytest.raises(SchemaValidationError):
        normative_projection(rule)


def test_factual_forbidden_invalida_fuente() -> None:
    rule = _rule()
    rule["runtime_fact"] = "producci\u00f3n est\u00e1 sana"
    with pytest.raises(SchemaValidationError):
        normative_projection(rule)


def test_legacy_version_no_participa_en_normative_projection() -> None:
    left = _rule()
    right = deepcopy(left)
    right["version"] = "0.2"
    assert normative_projection(left) == normative_projection(right)
