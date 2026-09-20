from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from policy.canonicalization.bootstrap import (
    BOOTSTRAP_DIR,
    CANONICALIZER_IDENTITY,
    CANONICALIZER_VERSION,
    PINNED_BOOTSTRAP_BUNDLE_ID,
    PINNED_UNICODE_VERSION,
    bootstrap_resource_bytes,
    compute_bootstrap_bundle_id,
    verify_bootstrap_bundle,
)
from policy.canonicalization.errors import BootstrapIntegrityError
from policy.canonicalization import normalize


def test_bootstrap_digest_mismatch_rechaza() -> None:
    resources = bootstrap_resource_bytes(BOOTSTRAP_DIR)
    resources["field-classes.json"] += b" "
    with pytest.raises(BootstrapIntegrityError):
        verify_bootstrap_bundle(resources)


def test_manifest_y_schema_no_pueden_autoautorizarse() -> None:
    resources = bootstrap_resource_bytes(BOOTSTRAP_DIR)

    manifest_schema = json.loads(resources["manifest-bootstrap.schema.json"])
    manifest_schema["properties"]["authorizes"] = {"const": True}
    resources["manifest-bootstrap.schema.json"] = json.dumps(manifest_schema).encode()

    rule_schema = json.loads(resources["legacy-rule-shadow.schema.json"])
    rule_schema["additionalProperties"] = True
    resources["legacy-rule-shadow.schema.json"] = json.dumps(rule_schema).encode()

    classes = json.loads(resources["field-classes.json"])
    classes["fields"]["version"] = "NORMATIVE"
    resources["field-classes.json"] = json.dumps(classes).encode()

    bundle = json.loads(resources["bundle.json"])
    bundle["claim"] = "bundle alterado y autoautorizado"
    resources["bundle.json"] = json.dumps(bundle).encode()

    forged_id = compute_bootstrap_bundle_id(resources)
    resources["BUNDLE.sha256"] = f"{forged_id}\n".encode()
    assert forged_id != PINNED_BOOTSTRAP_BUNDLE_ID
    with pytest.raises(BootstrapIntegrityError):
        verify_bootstrap_bundle(resources)


def test_bundle_de_repositorio_coincide_con_pin() -> None:
    verified = verify_bootstrap_bundle(bootstrap_resource_bytes(BOOTSTRAP_DIR))
    assert verified.bundle_id == PINNED_BOOTSTRAP_BUNDLE_ID
    artifact = (Path(BOOTSTRAP_DIR) / "BUNDLE.sha256").read_text().strip()
    assert artifact == PINNED_BOOTSTRAP_BUNDLE_ID


def test_bootstrap_v2_literal_exacto() -> None:
    assert compute_bootstrap_bundle_id(bootstrap_resource_bytes(BOOTSTRAP_DIR)) == (
        "sha256:a01df4bb87f628202b7ffa5b28e72f50238f5a00dc6bd29b11fbe7241eacf594"
    )


def test_unicode_version_coincide_con_bootstrap() -> None:
    verified = verify_bootstrap_bundle(bootstrap_resource_bytes(BOOTSTRAP_DIR))
    assert verified.bundle["unicode_normalization"] == "NFC"
    assert verified.bundle["unicode_version"] == PINNED_UNICODE_VERSION
    assert unicodedata.unidata_version == PINNED_UNICODE_VERSION


def test_unicode_version_mismatch_rechaza(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(normalize.unicodedata, "unidata_version", "0.0.0")
    with pytest.raises(ValueError, match="UNSUPPORTED_UNICODE_VERSION"):
        normalize.normalize_semantic("Cafe\u0301")


def test_unicode_version_participa_en_bootstrap_digest() -> None:
    resources = bootstrap_resource_bytes(BOOTSTRAP_DIR)
    bundle = json.loads(resources["bundle.json"])
    bundle["unicode_version"] = "0.0.0"
    resources["bundle.json"] = json.dumps(bundle, sort_keys=True).encode()
    assert compute_bootstrap_bundle_id(resources) != PINNED_BOOTSTRAP_BUNDLE_ID
    with pytest.raises(BootstrapIntegrityError):
        verify_bootstrap_bundle(resources)


def test_nfc_solo_se_ejecuta_bajo_version_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    real_normalize = unicodedata.normalize

    def spy(form: str, value: str) -> str:
        calls.append((form, value))
        return real_normalize(form, value)

    monkeypatch.setattr(normalize.unicodedata, "normalize", spy)
    assert normalize.normalize_semantic("Cafe\u0301") == "Caf\u00e9"
    assert calls == [("NFC", "Cafe\u0301")]
    calls.clear()
    monkeypatch.setattr(normalize.unicodedata, "unidata_version", "0.0.0")
    with pytest.raises(ValueError, match="UNSUPPORTED_UNICODE_VERSION"):
        normalize.normalize_semantic("Cafe\u0301")
    assert calls == []


def test_bootstrap_mutation_requires_new_canonicalizer_version() -> None:
    resources = bootstrap_resource_bytes(BOOTSTRAP_DIR)
    resources["field-classes.json"] += b" "
    mutated_same_version = compute_bootstrap_bundle_id(
        resources, canonicalizer_version=CANONICALIZER_VERSION
    )
    mutated_new_version = compute_bootstrap_bundle_id(
        resources, canonicalizer_version="3"
    )
    assert mutated_same_version != PINNED_BOOTSTRAP_BUNDLE_ID
    assert mutated_new_version != mutated_same_version
    with pytest.raises(BootstrapIntegrityError):
        verify_bootstrap_bundle(resources)


def test_canonicalizer_v2_pin_correcto() -> None:
    verified = verify_bootstrap_bundle(bootstrap_resource_bytes(BOOTSTRAP_DIR))
    assert CANONICALIZER_VERSION == "2"
    assert CANONICALIZER_IDENTITY == "JAX-POLICY-C14N/2"
    assert BOOTSTRAP_DIR.name == "v2"
    assert verified.bundle["canonicalizer_version"] == CANONICALIZER_IDENTITY
    assert verified.bundle_id == PINNED_BOOTSTRAP_BUNDLE_ID
