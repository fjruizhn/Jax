"""Compatibilidad de INGESTIÓN SHADOW para el corpus legacy.

Esta capa no es normativa y no debe importarse desde resolvers de autoridad.
La clasificación se entrega desde el manifest shadow ya validado; nunca se
lee del documento legacy.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import StrictYAMLError
from .strict_yaml import load_strict_yaml

LEGACY_RULE_SHADOW = "LEGACY_RULE_SHADOW"
_PLAIN_VERSION_LINE = re.compile(
    r"^(version:[ \t]*)([0-9]+\.[0-9]+)([ \t]*(?:#.*)?)$"
)


def _rewrite_legacy_version(text: str) -> str:
    rewritten: list[str] = []
    for line_with_end in text.splitlines(keepends=True):
        if line_with_end.endswith("\r\n"):
            line, ending = line_with_end[:-2], "\r\n"
        elif line_with_end.endswith("\n") or line_with_end.endswith("\r"):
            line, ending = line_with_end[:-1], line_with_end[-1]
        else:
            line, ending = line_with_end, ""
        match = _PLAIN_VERSION_LINE.fullmatch(line)
        if match:
            line = f'{match.group(1)}"{match.group(2)}"{match.group(3)}'
        rewritten.append(line + ending)
    return "".join(rewritten)


def load_legacy_rule_shadow(
    source: str | bytes | Path, *, source_classification: str | None = None
) -> dict:
    if source_classification != LEGACY_RULE_SHADOW:
        raise StrictYAMLError(
            "compatibilidad legacy requiere clasificación externa LEGACY_RULE_SHADOW"
        )
    if isinstance(source, Path):
        raw = source.read_bytes()
    elif isinstance(source, bytes):
        raw = source
    elif isinstance(source, str):
        raw = source.encode("utf-8")
    else:
        raise TypeError("source debe ser str, bytes o Path")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrictYAMLError("UTF-8 inválido en fuente legacy") from exc
    value = load_strict_yaml(_rewrite_legacy_version(text))
    if not isinstance(value, dict):
        raise StrictYAMLError("una regla legacy debe ser un mapping")
    return value


@dataclass(frozen=True)
class LegacyIdentityComparison:
    identities: dict[str, str]
    divergent: bool


def _declared_hash(version_file: Path) -> str:
    match = re.search(
        r"^sha256:\s*([0-9a-f]{64})\s*$",
        version_file.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("policy/VERSION no contiene sha256 legacy")
    return match.group(1)


def _computed_raw_rules_hash(rules_dir: Path) -> str:
    files: list[tuple[str, Path]] = []
    for path in rules_dir.glob("*.yaml"):
        rule_id = path.stem.split("-", 1)[0]
        files.append((rule_id, path))
    digest = hashlib.sha256()
    for _, path in sorted(files, key=lambda pair: pair[0]):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _generated_embedded_hash(generated_file: Path) -> str:
    match = re.search(
        r"^\*\*SHA256:\*\*\s*`([0-9a-f]{64})`\s*$",
        generated_file.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("CORPUS.md no contiene identidad legacy")
    return match.group(1)


def compare_legacy_identities(policy_dir: Path | str) -> LegacyIdentityComparison:
    root = Path(policy_dir)
    identities = {
        "version_declared_rules_sha256": _declared_hash(root / "VERSION"),
        "computed_raw_rules_sha256": _computed_raw_rules_hash(root / "rules"),
        "generated_corpus_embedded_sha256": _generated_embedded_hash(
            root / "generated" / "CORPUS.md"
        ),
    }
    return LegacyIdentityComparison(
        identities=identities,
        divergent=len(set(identities.values())) != 1,
    )
