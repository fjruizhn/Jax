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

import hashlib
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


def test_engine_status_is_resolver_not_implemented_never_valid():
    claim = _claim(
        predicate="ENGINE_STATUS",
        args={"name": "kimi", "status": "healthy"},
    )
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "RESOLVER_NOT_IMPLEMENTED"
    assert verdict.status != "VALID"
    assert "ENGINE_STATUS" in verdict.detail


def test_file_exists_rejects_path_outside_allowlist_without_touching_disk(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "tocó el filesystem antes de chequear la allowlist — "
            "PATH_NOT_ALLOWED debe devolverse sin exists()/read_bytes()"
        )

    monkeypatch.setattr(Path, "exists", _boom)
    monkeypatch.setattr(Path, "read_bytes", _boom)

    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"las_manos/config.toml"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "/etc/shadow", "hash": "0" * 64})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "PATH_NOT_ALLOWED"


def test_file_exists_rejects_dotdot_traversal_without_touching_disk(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "tocó el filesystem para un path con traversal — debía rechazarse por allowlist"
        )

    monkeypatch.setattr(Path, "exists", _boom)
    monkeypatch.setattr(Path, "read_bytes", _boom)

    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"policy/"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "policy/../../../../../../etc/passwd", "hash": "0" * 64})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "PATH_NOT_ALLOWED"


def test_file_exists_allowed_path_valid_hash():
    real_file = REPO_ROOT / "las_manos" / "config.toml"
    actual_hash = hashlib.sha256(real_file.read_bytes()).hexdigest()
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"las_manos/config.toml"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "las_manos/config.toml", "hash": actual_hash})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "VALID"


def test_file_exists_allowed_policy_prefix_valid_hash():
    real_file = REPO_ROOT / "policy" / "VERSION"
    actual_hash = hashlib.sha256(real_file.read_bytes()).hexdigest()
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"policy/"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "policy/VERSION", "hash": actual_hash})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "VALID"


def test_file_exists_allowed_path_wrong_hash():
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"las_manos/config.toml"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "las_manos/config.toml", "hash": "f" * 64})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "FACT_MISMATCH"


def test_file_exists_allowed_directory_prefix_nonexistent_file():
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"policy/"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "policy/no_existe_este_archivo.yaml", "hash": "0" * 64})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "FACT_MISMATCH"


def test_file_exists_directory_entry_returns_verdict_not_exception():
    # "policy/" está en config_paths de closed_vocabulary.yaml — un claim
    # con path="policy" coincide con esa entrada de directorio, pasa
    # _path_allowed(), .exists() es True (el directorio existe), y
    # .read_bytes() explota con IsADirectoryError si no se captura.
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"policy/"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "policy", "hash": "0" * 64})
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"
    assert "directorio" in verdict.detail.lower()


def _real_ctx() -> "validator.ValidationContext":
    vocab = loaders.load_vocabulary()
    return validator.load_validation_context(REPO_ROOT, vocab.config_paths)


def test_capability_available_found_only_in_ops_read_only_mode_matches():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "ssh_exec_readonly", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "VALID"


def test_capability_available_found_only_in_ops_mode_mismatch():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "ssh_exec", "mode": "read_only"},  # ssh_exec ES mutante
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"


def test_capability_available_found_only_in_catalog_mode_unverified_but_accepted():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "VALID"


def test_capability_available_not_found_anywhere():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "totalmente_inventado_xyz", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"


def test_capability_available_source_conflict_when_present_in_both():
    # Sintético a propósito: hoy [ops.*] y [capabilities.*] son disjuntos
    # en config.toml (0 solapamiento, verificado en el spec). Se fabrica
    # el conflicto a mano para probar la rama SOURCE_CONFLICT sin
    # depender de que el config real cambie.
    ctx = validator.ValidationContext(
        ops=frozenset({"code_swarm"}),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({"capabilities": {"code_swarm": {}}}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "SOURCE_CONFLICT"
    assert "code_swarm" in verdict.detail
