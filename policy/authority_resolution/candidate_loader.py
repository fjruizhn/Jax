"""I/O boundary: load and validate one atomic C14N/3 candidate corpus."""
from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tempfile

from policy.canonicalization.bootstrap_v3 import verified_bootstrap_v3
from policy.canonicalization.corpus_v3 import _safe, validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError
from policy.canonicalization.schemas_v3 import validate_authority, validate_manifest, validate_document
from policy.canonicalization.strict_yaml import load_strict_yaml

from .errors import InvalidValidatedCorpusError
from .models import (
    FrozenAuthorityMetaContract, FrozenExternalConstraints, FrozenFailClosed,
    FrozenNormativeDocument, FrozenNormativeSources, FrozenPrecedence,
    FrozenRelationships, FrozenScope, ProtectedConstraint,
    ValidatedCandidateCorpus, ValidatedManifestBinding, ValidatedMember,
)

def _authority(value, bootstrap) -> FrozenAuthorityMetaContract:
    validate_authority(value, bootstrap)
    p = value["precedence"]
    n = value["normative_sources"]
    e = value["external_constraints"]
    f = value["fail_closed"]
    metas = tuple(ProtectedConstraint(x["id"], x["statement"], x["non_waivable"]) for x in value["protected_metanorms"])
    return FrozenAuthorityMetaContract(
        scope_jurisdiction=value["scope"]["jurisdiction"],
        precedence=FrozenPrecedence(
            p["authority_meta_contract"], tuple(p["ordered_document_layers"]),
            p["overlay_position"], p["external_constraints"],
            p["equal_rank_conflict"], p["unresolved_conflict"],
        ),
        normative_sources=FrozenNormativeSources(
            tuple(sorted(n["permitted_document_classes"])),
            n["manifest_classification_required"],
            n["document_self_classification_authoritative"],
            n["legacy_status_is_normative_force"],
            n["unlisted_documents_have_normative_force"],
        ),
        protected_metanorms=tuple(sorted(metas, key=lambda x: x.id)),
        external_constraints=FrozenExternalConstraints(
            e["jax_normative"], e["effect"], e["may_grant_authority"],
            e["provenance_required_at_evaluation"],
        ),
        fail_closed=FrozenFailClosed(*(f[k] for k in (
            "default", "missing_or_invalid_root_pair", "invalid_membership",
            "unknown_document_class", "use_of_non_active_candidate",
            "invalid_or_expired_overlay", "unresolved_precedence",
        ))),
    )

def _document(value, bootstrap) -> FrozenNormativeDocument:
    validate_document(value, bootstrap)
    scope = value["scope"]
    rel = value["relationships"]
    return FrozenNormativeDocument(
        value["id"], value["document_class"], value["normative_layer"],
        "ACTIVE_WHEN_CORPUS_ACTIVE",
        FrozenScope(
            scope["jurisdiction"],
            tuple(sorted(scope["subjects"])), tuple(sorted(scope["actions"])),
            tuple(sorted(scope["conditions_all"])),
        ),
        FrozenRelationships(tuple(sorted(rel["supersedes"])), tuple(sorted(rel["superseded_by"]))),
    )


def _copy_snapshot_locator(source_root: Path, snapshot_root: Path, locator: str) -> None:
    """Copy one C14N/3-safe locator into a private candidate snapshot."""
    source = _safe(source_root, locator)
    target = snapshot_root / PurePosixPath(locator)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _snapshot_candidate(source_root: Path, snapshot_root: Path, bootstrap) -> None:
    """Capture every corpus input before C14N/3 validates that snapshot."""
    fixed = bootstrap.bundle["fixed_root_locators"]
    authority_locator = fixed["authority_meta_contract"]
    manifest_locator = fixed["authoritative_manifest"]
    _copy_snapshot_locator(source_root, snapshot_root, authority_locator)
    _copy_snapshot_locator(source_root, snapshot_root, manifest_locator)

    manifest = load_strict_yaml(snapshot_root / PurePosixPath(manifest_locator))
    if not isinstance(manifest, dict):
        # C14N/3 will classify malformed content; this guards the snapshot walk.
        raise InvalidValidatedCorpusError("manifest snapshot inválido")
    for member in manifest.get("normative_documents", ()):
        if not isinstance(member, dict) or not isinstance(member.get("path"), str):
            raise InvalidValidatedCorpusError("member snapshot inválido")
        _copy_snapshot_locator(source_root, snapshot_root, member["path"])
    for reference in manifest.get("reference_documents", ()):
        if not isinstance(reference, dict) or not isinstance(reference.get("source_locator"), str):
            raise InvalidValidatedCorpusError("reference snapshot inválido")
        _copy_snapshot_locator(source_root, snapshot_root, reference["source_locator"])

def load_validated_candidate(repo_root: Path) -> ValidatedCandidateCorpus:
    if not isinstance(repo_root, Path):
        raise InvalidValidatedCorpusError("repo_root debe ser Path")
    try:
        bootstrap = verified_bootstrap_v3()
        root = repo_root.resolve()
        with tempfile.TemporaryDirectory(prefix="jax-authority-candidate-") as temporary:
            snapshot_root = Path(temporary)
            _snapshot_candidate(root, snapshot_root, bootstrap)
            report = validate_candidate_corpus(snapshot_root)
            fixed = bootstrap.bundle["fixed_root_locators"]
            authority = load_strict_yaml(snapshot_root / PurePosixPath(fixed["authority_meta_contract"]))
            manifest = load_strict_yaml(snapshot_root / PurePosixPath(fixed["authoritative_manifest"]))
            validate_authority(authority, bootstrap)
            validate_manifest(manifest, bootstrap)
            docs = []
            members = []
            for member in manifest["normative_documents"]:
                members.append(ValidatedMember(member["id"], member["path"], member["document_class"], member["normative_layer"], member["normative_effect"]))
                doc = load_strict_yaml(snapshot_root / PurePosixPath(member["path"]))
                docs.append(_document(doc, bootstrap))
            docs.sort(key=lambda x: x.id)
            return ValidatedCandidateCorpus._from_validated_snapshot(
                report["policy_corpus_hash"], report["canonicalizer_identity"],
                report["bootstrap_bundle_id"], _authority(authority, bootstrap),
                ValidatedManifestBinding(manifest["id"], authority["id"], tuple(sorted(members, key=lambda x: x.id))),
                tuple(docs),
            )
    except (CanonicalizationError, OSError) as exc:
        raise InvalidValidatedCorpusError("no se pudo cargar corpus candidato validado") from exc
