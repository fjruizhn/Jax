from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pytest

from policy.canonicalization.bootstrap import verified_bootstrap
from policy.canonicalization.corpus import hash_policy_corpus, load_shadow_manifest
from policy.canonicalization.schemas import validate_shadow_manifest
from policy.canonicalization.errors import SchemaValidationError, StrictYAMLError
from policy.canonicalization.schemas import _validate
from policy.canonicalization.strict_yaml import load_strict_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_shadow_no_autoriza_ni_reemplaza_version() -> None:
    manifest = load_shadow_manifest(ROOT)
    validate_shadow_manifest(manifest, verified_bootstrap())
    assert manifest["mode"] == "SHADOW"
    assert manifest["source_classification"] == "LEGACY_RULE_SHADOW"
    assert manifest["authorizes"] is False
    assert manifest["blocks_production"] is False
    assert manifest["replaces_legacy_identity"] is False
    assert manifest["writes_policy"] is False
    assert manifest["shadow_policy_corpus_hash"] == hash_policy_corpus(ROOT)


def test_const_false_no_acepta_integer_zero() -> None:
    with pytest.raises(SchemaValidationError):
        _validate(0, {"const": False})


def test_const_true_no_acepta_integer_one() -> None:
    with pytest.raises(SchemaValidationError):
        _validate(1, {"const": True})


def test_manifest_false_requiere_boolean_real() -> None:
    bootstrap = verified_bootstrap()
    base = load_shadow_manifest(ROOT)
    for valid in (False,):
        candidate = dict(base, authorizes=valid)
        validate_shadow_manifest(candidate, bootstrap)
    for invalid in ("false", 0):
        candidate = dict(base, authorizes=invalid)
        with pytest.raises(SchemaValidationError):
            validate_shadow_manifest(candidate, bootstrap)
    with pytest.raises(StrictYAMLError):
        load_strict_yaml("authorizes: no\n")


def test_enum_distingue_tipos_json() -> None:
    with pytest.raises(SchemaValidationError):
        _validate(False, {"enum": [0]})
    with pytest.raises(SchemaValidationError):
        _validate(0, {"enum": [False]})
    with pytest.raises(SchemaValidationError):
        _validate(True, {"enum": [1]})


def test_json_object_rechaza_boolean_key() -> None:
    with pytest.raises(SchemaValidationError, match="clave JSON"):
        _validate({False: "x"}, {"type": "object"})


def test_json_object_rechaza_integer_key() -> None:
    with pytest.raises(SchemaValidationError, match="clave JSON"):
        _validate({0: "x"}, {"type": "object"})


def test_json_object_rechaza_non_string_key_nested() -> None:
    with pytest.raises(SchemaValidationError, match="clave JSON"):
        _validate({"outer": {1: "x"}}, {"type": "object"})


def test_json_equal_no_colision_false_zero_en_keys() -> None:
    collision = {False: "false", 0: "zero"}
    with pytest.raises(SchemaValidationError, match="clave JSON"):
        _validate(collision, {"const": {"0": "zero"}})


def test_required_nested_rechaza_solo_campo_faltante() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "required": ["required_field"],
                "properties": {"required_field": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "required": ["outer"],
        "additionalProperties": False,
    }
    with pytest.raises(
        SchemaValidationError,
        match=r"^\$\.outer: faltan campos \['required_field'\]$",
    ):
        _validate({"outer": {}}, schema)

    _validate({"outer": {"required_field": "ok"}}, schema)


def test_additional_properties_nested_rechaza_solo_campo_desconocido() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"known": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "required": ["outer"],
        "additionalProperties": False,
    }
    with pytest.raises(
        SchemaValidationError,
        match=r"^\$\.outer: campos desconocidos \['unexpected'\]$",
    ):
        _validate({"outer": {"known": "ok", "unexpected": "x"}}, schema)

    _validate({"outer": {"known": "ok"}}, schema)


def test_const_nested_rechaza_solo_valor_distinto() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"mode": {"const": "SHADOW"}},
                "required": ["mode"],
                "additionalProperties": False,
            }
        },
        "required": ["outer"],
        "additionalProperties": False,
    }
    with pytest.raises(
        SchemaValidationError,
        match=r"^\$\.outer\.mode: se esperaba 'SHADOW'$",
    ):
        _validate({"outer": {"mode": "PRODUCTION"}}, schema)

    _validate({"outer": {"mode": "SHADOW"}}, schema)


def test_enum_nested_rechaza_solo_valor_fuera_del_enum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"state": {"enum": ["A", "B"]}},
                "required": ["state"],
                "additionalProperties": False,
            }
        },
        "required": ["outer"],
        "additionalProperties": False,
    }
    with pytest.raises(
        SchemaValidationError,
        match=r"^\$\.outer\.state: valor fuera del enum$",
    ):
        _validate({"outer": {"state": "C"}}, schema)

    _validate({"outer": {"state": "A"}}, schema)


def test_launcher_usa_python_sin_bytecode() -> None:
    launcher = ROOT / "policy/canonicalization/jax-policy-c14n"
    assert launcher.stat().st_mode & 0o111
    assert 'exec "${PYTHON:-python3}" -B -m policy.canonicalization.cli "$@"' in (
        launcher.read_text()
    )


def test_cli_launcher_no_crea_pycache() -> None:
    trees = [
        ROOT / "policy/bootstrap",
        ROOT / "policy/canonicalization",
        ROOT / "policy/shadow",
        ROOT / "tests/policy",
    ]
    before = {path for tree in trees for path in tree.rglob("__pycache__")}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [str(ROOT / "policy/canonicalization/jax-policy-c14n")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"mode": "SHADOW"' in completed.stdout
    after = {path for tree in trees for path in tree.rglob("__pycache__")}
    assert after == before


def test_cli_no_modifica_version_manifest_ni_rules() -> None:
    observed = [
        ROOT / "policy/VERSION",
        ROOT / "policy/shadow/manifest.yaml",
        *(ROOT / "policy/rules").glob("*.yaml"),
    ]
    before = {path: path.read_bytes() for path in observed}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [str(ROOT / "policy/canonicalization/jax-policy-c14n")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    assert {path: path.read_bytes() for path in observed} == before


def test_manifest_bootstrap_identity_debe_coincidir_con_root() -> None:
    manifest = load_shadow_manifest(ROOT)
    forged = dict(manifest, bootstrap_bundle_id="sha256:" + "0" * 64)
    with pytest.raises(SchemaValidationError):
        validate_shadow_manifest(forged, verified_bootstrap())
