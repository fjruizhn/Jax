from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from policy.canonicalization import corpus
from policy.canonicalization.cli import _shadow_report
from policy.canonicalization.corpus import (
    _policy_corpus_digest,
    hash_policy_corpus,
)
from policy.canonicalization.errors import DuplicateRuleIdError, SchemaValidationError


ROOT = Path(__file__).resolve().parents[2]


def _rule(rule_id: str = "P01", **changes: str) -> bytes:
    values = {
        "id": rule_id,
        "statement": "Una regla estable.",
        "origin": "Documento A.",
        "status": "CULTURAL",
        "version": "0.1",
        "notes": "Visible.",
    }
    values.update(changes)
    return (
        f"id: {values['id']}\n"
        f"statement: {values['statement']}\n"
        f"origin: {values['origin']}\n"
        f"status: {values['status']}\n"
        "enforcement:\n  mechanism: null\n  test: null\n"
        f"version: {values['version']}\n"
        "created: 2026-08-15\n"
        "amended_by: null\n"
        f"notes: {values['notes']}\n"
    ).encode()


def _shadow_repo(path: Path, rules: dict[str, bytes] | None = None) -> Path:
    manifest = (ROOT / "policy/shadow/manifest.yaml").read_bytes()
    (path / "policy/shadow").mkdir(parents=True)
    (path / "policy/rules").mkdir()
    (path / "policy/shadow/manifest.yaml").write_bytes(manifest)
    for name, source in (rules or {"P01.yaml": _rule()}).items():
        (path / "policy/rules" / name).write_bytes(source)
    return path


def test_whitespace_crlf_comentarios_no_cambian_hash(tmp_path: Path) -> None:
    compact = _rule()
    formatted = compact.replace(b"\n", b"\r\n").replace(
        b"id: P01\r\n", b"# comentario\r\nid:    P01\r\n"
    )
    left = _shadow_repo(tmp_path / "left", {"a.yaml": compact})
    right = _shadow_repo(tmp_path / "right", {"a.yaml": formatted})
    assert hash_policy_corpus(left) == hash_policy_corpus(right)


def test_membership_normativa_cambia_hash(tmp_path: Path) -> None:
    one = _shadow_repo(tmp_path / "one", {"a.yaml": _rule("P01")})
    two = _shadow_repo(
        tmp_path / "two", {"a.yaml": _rule("P01"), "b.yaml": _rule("P02")}
    )
    assert hash_policy_corpus(one) != hash_policy_corpus(two)


def test_mover_source_path_no_cambia_hash(tmp_path: Path) -> None:
    source = _rule()
    left = _shadow_repo(tmp_path / "checkout-a", {"rule.yaml": source})
    right = _shadow_repo(tmp_path / "checkout-b", {"renamed.yaml": source})
    assert hash_policy_corpus(left) == hash_policy_corpus(right)


def test_dos_checkouts_logicamente_equivalentes_producen_mismo_hash(
    tmp_path: Path,
) -> None:
    checkout_a = _shadow_repo(
        tmp_path / "a", {"P01.yaml": _rule("P01"), "P02.yaml": _rule("P02")}
    )
    checkout_b = _shadow_repo(
        tmp_path / "b",
        {
            "z.yaml": _rule("P02").replace(b"\n", b"\r\n"),
            "a.yaml": b"# moved\n" + _rule("P01"),
        },
    )
    assert hash_policy_corpus(checkout_a) == hash_policy_corpus(checkout_b)


def test_legacy_version_no_cambia_shadow_semantic_hash(tmp_path: Path) -> None:
    plain_01 = _rule()
    plain_02 = _rule().replace(b"version: 0.1", b"version: 0.2")
    left = _shadow_repo(tmp_path / "v01", {"a.yaml": plain_01})
    right = _shadow_repo(tmp_path / "v02", {"a.yaml": plain_02})
    assert hash_policy_corpus(left) == hash_policy_corpus(right)


def test_no_public_hash_policy_sources_api() -> None:
    assert not hasattr(corpus, "hash_policy_sources")
    assert corpus.__all__ == ["hash_policy_corpus", "load_shadow_manifest"]


def test_public_corpus_api_acepta_solo_repo_root() -> None:
    signature = inspect.signature(hash_policy_corpus)
    assert tuple(signature.parameters) == ("repo_root",)
    assert signature.parameters["repo_root"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_caller_no_puede_sustituir_rules_dir(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    (arbitrary / "P02.yaml").write_bytes(_rule("P02"))
    with pytest.raises(TypeError):
        hash_policy_corpus(repo, rules_dir=arbitrary)  # type: ignore[call-arg]


def test_manifest_glob_path_traversal_rechaza(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    manifest = repo / "policy/shadow/manifest.yaml"
    manifest.write_text(
        manifest.read_text().replace("policy/rules/*.yaml", "../rules/*.yaml")
    )
    with pytest.raises(SchemaValidationError):
        hash_policy_corpus(repo)


def test_absolute_manifest_glob_rechaza(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    manifest = repo / "policy/shadow/manifest.yaml"
    manifest.write_text(
        manifest.read_text().replace("policy/rules/*.yaml", "/tmp/*.yaml")
    )
    with pytest.raises(SchemaValidationError):
        hash_policy_corpus(repo)


def test_symlink_de_rules_dir_rechaza(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    rules = repo / "policy/rules"
    target = tmp_path / "real-rules"
    rules.rename(target)
    rules.symlink_to(target, target_is_directory=True)
    with pytest.raises(SchemaValidationError, match="symlink"):
        hash_policy_corpus(repo)


def test_symlink_de_rule_member_rechaza(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    member = repo / "policy/rules/P01.yaml"
    target = tmp_path / "real-rule.yaml"
    member.rename(target)
    member.symlink_to(target)
    with pytest.raises(SchemaValidationError, match="symlink"):
        hash_policy_corpus(repo)


def test_symlink_intermedio_rechaza(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    policy = repo / "policy"
    target = repo / "real-policy"
    policy.rename(target)
    policy.symlink_to(target, target_is_directory=True)
    with pytest.raises(SchemaValidationError, match="symlink"):
        hash_policy_corpus(repo)


def test_symlink_de_manifest_rechaza(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    manifest = repo / "policy/shadow/manifest.yaml"
    target = tmp_path / "real-manifest.yaml"
    manifest.rename(target)
    manifest.symlink_to(target)
    with pytest.raises(SchemaValidationError, match="symlink"):
        hash_policy_corpus(repo)


def test_membership_real_deriva_del_manifest(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    outside = repo / "other"
    outside.mkdir()
    before = hash_policy_corpus(repo)
    (outside / "P02.yaml").write_bytes(_rule("P02"))
    assert hash_policy_corpus(repo) == before


def test_membership_no_puede_ser_inyectada_por_caller(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    external = tmp_path / "external.yaml"
    external.write_bytes(_rule("P99"))
    expected = hash_policy_corpus(repo)
    with pytest.raises(TypeError):
        hash_policy_corpus(repo, sources={"P99.yaml": external.read_bytes()})  # type: ignore[call-arg]
    assert hash_policy_corpus(repo) == expected


def test_cli_y_public_api_usan_misma_ruta() -> None:
    assert _shadow_report(ROOT)["shadow_policy_corpus_hash"] == hash_policy_corpus(ROOT)


def test_duplicate_document_id_rechaza(tmp_path: Path) -> None:
    repo = _shadow_repo(
        tmp_path / "repo", {"first.yaml": _rule("P01"), "second.yaml": _rule("P01")}
    )
    with pytest.raises(DuplicateRuleIdError, match="id duplicado"):
        hash_policy_corpus(repo)


def test_rename_dentro_del_mismo_membership_no_cambia_hash(tmp_path: Path) -> None:
    repo = _shadow_repo(tmp_path / "repo")
    before = hash_policy_corpus(repo)
    (repo / "policy/rules/P01.yaml").rename(repo / "policy/rules/renamed.yaml")
    assert hash_policy_corpus(repo) == before


def _projection() -> dict:
    return {
        "canonicalizer_version": "2",
        "bootstrap_bundle_id": "sha256:" + "a" * 64,
        "manifest": {"kind": "JAX_POLICY_SHADOW_MANIFEST"},
        "documents": [{"id": "P01", "statement": "estable"}],
    }


def test_bootstrap_bundle_id_participa_en_policy_corpus_hash() -> None:
    left = _projection()
    right = {**left, "bootstrap_bundle_id": "sha256:" + "b" * 64}
    assert _policy_corpus_digest(left) != _policy_corpus_digest(right)


def test_canonicalizer_version_participa_en_policy_corpus_hash() -> None:
    left = _projection()
    right = {**left, "canonicalizer_version": "3"}
    assert _policy_corpus_digest(left) != _policy_corpus_digest(right)


def test_shadow_hash_v2_no_es_hash_v1() -> None:
    assert hash_policy_corpus(ROOT) != (
        "sha256:bafdab9ef2a0b17f1326b2d08fd0e8d712d8c201bc73b3af539793cf073add15"
    )


def test_shadow_hash_v2_literal_exacto() -> None:
    assert hash_policy_corpus(ROOT) == (
        "sha256:2ff0a5e264bb956ce9c6890f4d379ae7f1d56f57197ceb72c5d95cc383ca8536"
    )


def test_documentos_normativos_iguales_con_mismo_v2_son_deterministas(
    tmp_path: Path,
) -> None:
    repo = _shadow_repo(tmp_path / "repo", {"P01.yaml": _rule()})
    assert hash_policy_corpus(repo) == hash_policy_corpus(repo)
