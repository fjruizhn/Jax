from pathlib import Path
import shutil

import pytest

from test_authority_resolution_context import ROOT, empty_view, context
from policy.authority_resolution import (
    load_validated_candidate,
    resolve_static_authority,
    to_static_policy_view,
)
from policy.authority_resolution.errors import (
    InvalidValidatedCorpusError,
    ResolverContractError,
)
from policy.authority_resolution.models import ValidatedCandidateCorpus
import policy.authority_resolution.candidate_loader as candidate_loader
from policy.canonicalization.corpus_v3 import validate_candidate_corpus

def test_candidate_result_never_grants_execution():
    result = resolve_static_authority(empty_view(), context())
    assert all(value not in {"ACTIVE", "RATIFIED", "AUTHORIZED"} for value in result.__dict__.values())


def test_public_constructor_rejects_forged_hash_with_valid_content():
    corpus = load_validated_candidate(ROOT)
    with pytest.raises(InvalidValidatedCorpusError):
        ValidatedCandidateCorpus(
            "sha256:" + "f" * 64,
            corpus.canonicalizer_identity,
            corpus.bootstrap_bundle_id,
            corpus.authority,
            corpus.manifest,
            corpus.normative_documents,
        )


def test_real_candidate_loader_keeps_the_c14n3_validated_identity():
    corpus = load_validated_candidate(ROOT)
    assert corpus.policy_corpus_hash == validate_candidate_corpus(ROOT)["policy_corpus_hash"]


def test_adapter_rejects_unsealed_hash_content_mismatch():
    corpus = load_validated_candidate(ROOT)
    forged = object.__new__(ValidatedCandidateCorpus)
    for name, value in (
        ("policy_corpus_hash", "sha256:" + "f" * 64),
        ("canonicalizer_identity", corpus.canonicalizer_identity),
        ("bootstrap_bundle_id", corpus.bootstrap_bundle_id),
        ("authority", corpus.authority),
        ("manifest", corpus.manifest),
        ("normative_documents", corpus.normative_documents),
        ("_loader_seal", None),
    ):
        object.__setattr__(forged, name, value)
    with pytest.raises(ResolverContractError):
        to_static_policy_view(forged)


def test_loader_validates_and_builds_from_the_same_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "candidate"
    shutil.copytree(ROOT / "policy", source / "policy", ignore=shutil.ignore_patterns("__pycache__"))
    before = candidate_loader.validate_candidate_corpus(source)["policy_corpus_hash"]
    authority_path = source / "policy" / "authority.yaml"
    original = authority_path.read_text(encoding="utf-8")

    real_validate = candidate_loader.validate_candidate_corpus

    def transition_after_snapshot(snapshot_root: Path):
        authority_path.write_text(
            original.replace(
                "Constitutional changes require authorized human ratification.",
                "Changed only after the private snapshot was captured.",
            ),
            encoding="utf-8",
        )
        return real_validate(snapshot_root)

    monkeypatch.setattr(candidate_loader, "validate_candidate_corpus", transition_after_snapshot)
    corpus = candidate_loader.load_validated_candidate(source)

    assert corpus.policy_corpus_hash == before
    assert candidate_loader.validate_candidate_corpus(source)["policy_corpus_hash"] != before
    assert any(
        item.statement == "Constitutional changes require authorized human ratification."
        for item in corpus.authority.protected_metanorms
    )


def test_expected_invalid_candidate_is_typed(tmp_path):
    source = tmp_path / "invalid"
    shutil.copytree(ROOT / "policy", source / "policy", ignore=shutil.ignore_patterns("__pycache__"))
    manifest = source / "policy" / "manifest.yaml"
    manifest.write_text("not: a-valid-manifest\n", encoding="utf-8")
    with pytest.raises(InvalidValidatedCorpusError):
        load_validated_candidate(source)


def test_unexpected_internal_type_error_is_not_reclassified(monkeypatch):
    def internal_bug(_repo_root):
        raise TypeError("sentinel internal bug")

    monkeypatch.setattr(candidate_loader, "validate_candidate_corpus", internal_bug)
    with pytest.raises(TypeError, match="sentinel internal bug"):
        candidate_loader.load_validated_candidate(ROOT)
