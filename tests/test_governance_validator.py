"""
Test de policy/governance/validator.py — validación semántica de claims
contra fuentes reales (config.toml, motor_registry, filesystem).

Orden de este archivo, a propósito:
  1. Chequeos estructurales (predicado conocido, args, authority) — antes
     de cualquier resolver.
  2. RESOLVER_NOT_IMPLEMENTED como rechazo real (ENGINE_STATUS nunca
     pasa silenciosamente).
  3. FILE_EXISTS: allowlist antes que filesystem, con test que lo hace
     explotar si el código toca disco para un path no permitido.
  4. CAPABILITY_AVAILABLE contra el config.toml real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOVERNANCE = REPO_ROOT / "policy" / "governance"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GOVERNANCE))

import claims  # noqa: E402
import loaders  # noqa: E402
import validator  # noqa: E402
from las_manos.motor_registry.catalog import MotorCatalog  # noqa: E402

FILE_EXISTS_SPEC = loaders.PredicateSpec(
    name="FILE_EXISTS", args=("path", "hash"), source_of_truth="Sistema de archivos"
)
CAPABILITY_AVAILABLE_SPEC = loaders.PredicateSpec(
    name="CAPABILITY_AVAILABLE", args=("name", "mode"), source_of_truth="Registro de capabilities"
)
ENGINE_STATUS_SPEC = loaders.PredicateSpec(
    name="ENGINE_STATUS", args=("name", "status"), source_of_truth="Health check"
)
PREDICATES = {
    "FILE_EXISTS": FILE_EXISTS_SPEC,
    "CAPABILITY_AVAILABLE": CAPABILITY_AVAILABLE_SPEC,
    "ENGINE_STATUS": ENGINE_STATUS_SPEC,
}


def _claim(**overrides):
    base = dict(
        predicate="FILE_EXISTS",
        args={"path": "las_manos/config.toml", "hash": "a" * 64},
        authority="OBSERVADO",
        provenance_ref="test",
        evidence_pointer="test",
        scope="jax",
    )
    base.update(overrides)
    return claims.Claim(**base)


def _empty_ctx() -> "validator.ValidationContext":
    return validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )


def test_validate_unknown_predicate():
    claim = _claim(predicate="NOT_A_REAL_PREDICATE", args={})
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "UNKNOWN_PREDICATE"


def test_validate_args_mismatch():
    claim = _claim(predicate="FILE_EXISTS", args={"path": "x"})  # falta 'hash'
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "ARGS_MISMATCH"


def test_validate_authority_inferido_rejected():
    claim = _claim(authority="INFERIDO")
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "AUTHORITY_INVALID"
