"""
policy/governance — I/O de config ESTÁTICA de policy: predicates.yaml,
closed_vocabulary.yaml, render_templates.yaml + hash.

Config versionada por commit. Si `load_templates()` no puede verificar el
hash contra VERSION, el subsistema NO ARRANCA — no hay carga parcial ni
degradación silenciosa. Mismo patrón de falla que ya causó tres
incidentes independientes en este ecosistema (retención de backups,
_HTTP_FACETS, output_validator.py de motor_registry): este módulo existe
específicamente para no ser el cuarto.

Distinto de validator.py: ese lee ESTADO VIVO (config.toml, filesystem),
que cambia sin commits — si un resolver individual falla ahí, es un
rechazo de ese claim, no una falla de arranque del subsistema entero.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

GOVERNANCE_DIR = Path(__file__).resolve().parent
POLICY_DIR = GOVERNANCE_DIR.parent
PREDICATES_FILE = POLICY_DIR / "vocabulary" / "predicates.yaml"
VOCABULARY_FILE = POLICY_DIR / "vocabulary" / "closed_vocabulary.yaml"
TEMPLATES_FILE = POLICY_DIR / "templates" / "render_templates.yaml"
VERSION_FILE = POLICY_DIR / "VERSION"


@dataclass(frozen=True)
class TemplateSpec:
    status: str
    template: str | None


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    args: tuple[str, ...]
    source_of_truth: str


@dataclass(frozen=True)
class ClosedVocabulary:
    flattened: frozenset[str]
    config_paths: frozenset[str]


def _current_templates_hash() -> str:
    return hashlib.sha256(TEMPLATES_FILE.read_bytes()).hexdigest()


def _recorded_templates_hash() -> str:
    content = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"^templates_sha256:\s*(\S+)\s*$", content, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(
            f"{VERSION_FILE} no tiene línea 'templates_sha256:' — correr "
            "policy/tools/template_hash.py --write"
        )
    return match.group(1)


def load_templates() -> dict[str, TemplateSpec]:
    current = _current_templates_hash()
    recorded = _recorded_templates_hash()
    if current != recorded:
        raise RuntimeError(
            f"Hash de {TEMPLATES_FILE} no coincide con VERSION (esperado "
            f"{recorded}, calculado {current}) — el subsistema de "
            "gobernanza no arranca con templates sin verificar."
        )
    data = yaml.safe_load(TEMPLATES_FILE.read_text(encoding="utf-8"))
    return {
        name: TemplateSpec(status=entry["status"], template=entry.get("template"))
        for name, entry in data["templates"].items()
    }


def load_predicates() -> dict[str, PredicateSpec]:
    data = yaml.safe_load(PREDICATES_FILE.read_text(encoding="utf-8"))
    return {
        entry["name"]: PredicateSpec(
            name=entry["name"],
            args=tuple(entry["args"]),
            source_of_truth=entry["source_of_truth"],
        )
        for entry in data["predicates"]
    }


def load_vocabulary() -> ClosedVocabulary:
    data = yaml.safe_load(VOCABULARY_FILE.read_text(encoding="utf-8"))
    flattened: set[str] = set()
    for key, value in data.items():
        if key == "config_paths":
            continue
        if isinstance(value, dict):
            flattened.update(value.keys())
    config_paths = frozenset(data.get("config_paths") or [])
    return ClosedVocabulary(flattened=frozenset(flattened), config_paths=config_paths)
