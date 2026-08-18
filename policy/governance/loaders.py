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
    return {}
