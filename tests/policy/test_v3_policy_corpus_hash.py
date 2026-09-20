from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from policy.canonicalization.bootstrap_v3 import (
    PINNED_BOOTSTRAP_BUNDLE_ID,
    bootstrap_resource_bytes,
    verify_bootstrap_bundle,
)
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError


ROOT = Path(__file__).resolve().parents[2]


def test_empty_candidate_is_reproducible_and_not_active():
    first = validate_candidate_corpus(ROOT)
    second = validate_candidate_corpus(ROOT)
    assert first == second
    assert first["state"] == "VALID_CANDIDATE"
    assert first["bootstrap_bundle_id"] == PINNED_BOOTSTRAP_BUNDLE_ID
    assert "ACTIVE" not in first.values()


@pytest.mark.parametrize("resource", ["bundle.json", "field-classes.json", "authority-meta-contract.schema.json", "authoritative-policy-manifest.schema.json", "normative-policy-document.schema.json"])
def test_each_pinned_bootstrap_resource_is_covered(resource):
    resources = bootstrap_resource_bytes()
    resources[resource] += b"\n"
    with pytest.raises(CanonicalizationError):
        verify_bootstrap_bundle(resources)


def test_bundle_sha_cannot_replace_compiled_pin():
    resources = bootstrap_resource_bytes()
    resources["BUNDLE.sha256"] = b"sha256:" + b"0" * 64 + b"\n"
    with pytest.raises(CanonicalizationError):
        verify_bootstrap_bundle(resources)


def test_provenance_reference_change_does_not_change_hash(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "policy", repo / "policy")
    first = validate_candidate_corpus(repo)["policy_corpus_hash"]
    manifest = repo / "policy" / "manifest.yaml"
    manifest.write_text(manifest.read_text().replace("P10 fail-open prohibido", "presentation changed"))
    assert validate_candidate_corpus(repo)["policy_corpus_hash"] == first


def test_reference_cannot_have_normative_field(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "policy", repo / "policy")
    manifest = repo / "policy" / "manifest.yaml"
    manifest.write_text(manifest.read_text() + "    normative_layer: CONSTITUTIONAL_CORE\n")
    with pytest.raises(CanonicalizationError):
        validate_candidate_corpus(repo)
