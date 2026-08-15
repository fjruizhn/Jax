#!/usr/bin/env python3
"""
Genera generated/CORPUS.md desde rules/*.yaml — solo plantilla, sin lógica
interpretativa. Mismo principio que el renderer de REFORMAS-v3.md §3.1.6:
slots para valores del YAML, sin ramas condicionales sobre el contenido de
las reglas, sin texto generado que no provenga directamente de un campo.

generated/CORPUS.md NUNCA se edita a mano — ver policy/README.md.

Uso: policy/tools/generate_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

POLICY_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = POLICY_DIR / "rules"
GENERATED_DIR = POLICY_DIR / "generated"
VERSION_FILE = POLICY_DIR / "VERSION"

sys.path.insert(0, str(POLICY_DIR / "tools"))
from corpus_hash import compute_hash, rule_id_from_filename  # noqa: E402

STATUS_ORDER = ["NORMATIVA", "NORMATIVA_PENDIENTE", "CULTURAL", "HISTORICA"]

RULE_TEMPLATE = """### {id} — {statement_first_line}

- **Enunciado:** {statement}
- **Origen:** {origin}
- **Estado:** {status}
- **Mecanismo de cumplimiento:** {mechanism}
- **Test:** {test}
- **Versión:** {version} · **Creada:** {created} · **Enmendada por:** {amended_by}
{notes_block}
"""


def load_rules() -> list[dict]:
    files = sorted(RULES_DIR.glob("*.yaml"), key=rule_id_from_filename)
    rules = []
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        data["_file"] = f.name
        rules.append(data)
    return rules


def render_rule(r: dict) -> str:
    statement = str(r.get("statement", "")).strip()
    first_line = statement.splitlines()[0] if statement else "(sin enunciado)"
    enforcement = r.get("enforcement") or {}
    notes = r.get("notes")
    notes_block = f"- **Notas:** {notes}\n" if notes else ""
    return RULE_TEMPLATE.format(
        id=r.get("id", "?"),
        statement_first_line=first_line,
        statement=statement,
        origin=r.get("origin", "?"),
        status=r.get("status", "?"),
        mechanism=enforcement.get("mechanism") or "null",
        test=enforcement.get("test") or "null",
        version=r.get("version", "?"),
        created=r.get("created", "?"),
        amended_by=r.get("amended_by") or "null",
        notes_block=notes_block,
    )


def main() -> int:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    digest, files = compute_hash()

    version_line = "?"
    if VERSION_FILE.exists():
        for line in VERSION_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("version:"):
                version_line = line.split(":", 1)[1].strip()

    counts = {s: 0 for s in STATUS_ORDER}
    for r in rules:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1

    out = []
    out.append("<!-- GENERADO AUTOMÁTICAMENTE por policy/tools/generate_corpus.py")
    out.append("     NO EDITAR A MANO — ver policy/README.md -->")
    out.append("")
    out.append("# CORPUS — JAX/Axioma, reglas normativas")
    out.append("")
    out.append(f"**Versión del corpus:** {version_line}")
    out.append(f"**SHA256:** `{digest}`")
    out.append(f"**Reglas:** {len(rules)}")
    out.append("")
    out.append("| Estado | Cantidad |")
    out.append("|---|---|")
    for s in STATUS_ORDER:
        out.append(f"| {s} | {counts.get(s, 0)} |")
    out.append("")
    out.append("---")
    out.append("")

    for status in STATUS_ORDER:
        group = [r for r in rules if r.get("status") == status]
        if not group:
            continue
        out.append(f"## {status}")
        out.append("")
        for r in group:
            out.append(render_rule(r))

    GENERATED_DIR_FILE = GENERATED_DIR / "CORPUS.md"
    GENERATED_DIR_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Generado: {GENERATED_DIR_FILE} ({len(rules)} reglas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
