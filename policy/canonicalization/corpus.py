"""Identidad semántica SHADOW y membership ligado al manifest."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Mapping

from .bootstrap import (
    CANONICALIZER_IDENTITY,
    CANONICALIZER_VERSION,
    VerifiedBootstrap,
    verified_bootstrap,
)
from .canonical_json import canonical_json_bytes
from .errors import DuplicateRuleIdError, SchemaValidationError
from .legacy import LEGACY_RULE_SHADOW, load_legacy_rule_shadow
from .projection import normative_projection
from .schemas import validate_shadow_manifest
from .strict_yaml import load_strict_yaml

__all__ = ["hash_policy_corpus", "load_shadow_manifest"]

POLICY_CORPUS_DOMAIN = b"JAX-POLICY-CORPUS\0"
_MANIFEST_RELATIVE = PurePosixPath("policy/shadow/manifest.yaml")
_GLOB_MAGIC = frozenset("*?[")
_MANIFEST_NORMATIVE_FIELDS = (
    "schema_version",
    "kind",
    "mode",
    "canonicalizer_version",
    "bootstrap_bundle_id",
    "source_classification",
    "legacy_rule_glob",
    "authorizes",
    "blocks_production",
    "replaces_legacy_identity",
    "writes_policy",
    "legacy_identity_mode",
)


def _absolute_without_resolving(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_path(root: Path, relative: PurePosixPath) -> Path:
    if root.is_symlink():
        raise SchemaValidationError(f"repo_root es symlink: {root}")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise SchemaValidationError(f"path requerido ausente: {current}") from exc
        if stat.S_ISLNK(mode):
            raise SchemaValidationError(f"symlink no permitido: {current}")
    return current


def load_shadow_manifest(repo_root: Path | str) -> dict:
    root = _absolute_without_resolving(repo_root)
    manifest_path = _reject_symlink_path(root, _MANIFEST_RELATIVE)
    if not manifest_path.is_file():
        raise SchemaValidationError("manifest shadow no es archivo regular")
    manifest = load_strict_yaml(manifest_path)
    bootstrap = verified_bootstrap()
    validate_shadow_manifest(manifest, bootstrap)
    return manifest


def _validated_glob(value: object) -> tuple[PurePosixPath, str]:
    if not isinstance(value, str) or "\\" in value:
        raise SchemaValidationError("legacy_rule_glob debe usar formato POSIX")
    locator = PurePosixPath(value)
    if locator.is_absolute() or value.startswith("/"):
        raise SchemaValidationError("legacy_rule_glob no puede ser absoluto")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise SchemaValidationError("legacy_rule_glob contiene traversal o segmento vacío")
    if len(locator.parts) < 2:
        raise SchemaValidationError("legacy_rule_glob debe estar ligado al repo root")
    directory_parts = locator.parts[:-1]
    if any(any(char in part for char in _GLOB_MAGIC) for part in directory_parts):
        raise SchemaValidationError("globs en directorios no están permitidos")
    return PurePosixPath(*directory_parts), locator.name


def _manifest_sources(repo_root: Path, manifest: dict) -> dict[str, bytes]:
    relative_dir, filename_glob = _validated_glob(manifest["legacy_rule_glob"])
    rules_dir = _reject_symlink_path(repo_root, relative_dir)
    if not rules_dir.is_dir():
        raise SchemaValidationError("directorio miembro no es directorio regular")

    sources: dict[str, bytes] = {}
    for path in rules_dir.glob(filename_glob):
        relative = PurePosixPath(path.relative_to(repo_root).as_posix())
        member = _reject_symlink_path(repo_root, relative)
        if not member.is_file():
            raise SchemaValidationError(f"miembro no es archivo regular: {relative}")
        sources[relative.as_posix()] = member.read_bytes()
    if not sources:
        raise SchemaValidationError("corpus shadow vacío")
    return sources


def _project_sources(
    sources: Mapping[str, bytes], bootstrap: VerifiedBootstrap
) -> list[dict]:
    projections: list[dict] = []
    seen: set[str] = set()
    for source_bytes in sources.values():
        rule = load_legacy_rule_shadow(
            source_bytes, source_classification=LEGACY_RULE_SHADOW
        )
        projection = normative_projection(rule, bootstrap)
        rule_id = projection["id"]
        if rule_id in seen:
            raise DuplicateRuleIdError(f"id duplicado: {rule_id}")
        seen.add(rule_id)
        projections.append(projection)
    return sorted(projections, key=lambda rule: rule["id"])


def _manifest_normative_projection(
    manifest: dict, bootstrap: VerifiedBootstrap
) -> dict:
    validate_shadow_manifest(manifest, bootstrap)
    if manifest["canonicalizer_version"] != CANONICALIZER_IDENTITY:
        raise SchemaValidationError("manifest declara otro canonicalizer")
    if manifest["bootstrap_bundle_id"] != bootstrap.bundle_id:
        raise SchemaValidationError("manifest bootstrap difiere de la root of trust")
    return {field: manifest[field] for field in _MANIFEST_NORMATIVE_FIELDS}


def _policy_corpus_digest(corpus_projection: dict) -> str:
    version = corpus_projection.get("canonicalizer_version")
    if not isinstance(version, str):
        raise SchemaValidationError("corpus_projection requiere canonicalizer_version")
    try:
        version_bytes = version.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise SchemaValidationError("canonicalizer_version no es ASCII") from exc
    digest = hashlib.sha256(POLICY_CORPUS_DOMAIN)
    digest.update(version_bytes)
    digest.update(b"\0")
    digest.update(canonical_json_bytes(corpus_projection))
    return f"sha256:{digest.hexdigest()}"


def hash_policy_corpus(repo_root: Path | str) -> str:
    """Hash del membership declarado por el manifest real del repositorio."""

    root = _absolute_without_resolving(repo_root)
    manifest = load_shadow_manifest(root)
    bootstrap = verified_bootstrap()
    if manifest["source_classification"] != LEGACY_RULE_SHADOW:
        raise SchemaValidationError("manifest no clasifica fuentes como legacy shadow")
    projected_documents = _project_sources(_manifest_sources(root, manifest), bootstrap)
    corpus_projection = {
        "canonicalizer_version": CANONICALIZER_VERSION,
        "bootstrap_bundle_id": bootstrap.bundle_id,
        "manifest": _manifest_normative_projection(manifest, bootstrap),
        "documents": projected_documents,
    }
    return _policy_corpus_digest(corpus_projection)
