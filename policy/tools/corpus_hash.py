#!/usr/bin/env python3
"""
Calcula el sha256 del corpus de reglas: concatenación ordenada por id de
todos los .yaml de rules/, byte a byte (sin parsear YAML, sin normalizar
espacios) — el hash certifica el contenido exacto de los archivos fuente.

Uso:
    policy/tools/corpus_hash.py            imprime el hash
    policy/tools/corpus_hash.py --write    lo escribe en VERSION (línea 2)
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = POLICY_DIR / "rules"
VERSION_FILE = POLICY_DIR / "VERSION"


def rule_id_from_filename(path: Path) -> str:
    """El id es el prefijo antes del primer '-' en {ID}-{slug}.yaml."""
    return path.stem.split("-", 1)[0]


def compute_hash() -> tuple[str, list[Path]]:
    files = sorted(RULES_DIR.glob("*.yaml"), key=rule_id_from_filename)
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest(), files


def main() -> int:
    if not RULES_DIR.exists():
        print(f"ERROR: no existe {RULES_DIR}", file=sys.stderr)
        return 1

    digest, files = compute_hash()
    print(f"sha256: {digest}")
    print(f"archivos incluidos ({len(files)}):")
    for f in files:
        print(f"  {rule_id_from_filename(f)}  {f.name}")

    if "--write" in sys.argv:
        if not VERSION_FILE.exists():
            print(f"ERROR: no existe {VERSION_FILE}", file=sys.stderr)
            return 1
        content = VERSION_FILE.read_text(encoding="utf-8")
        new_content = re.sub(
            r"^sha256:.*$",
            f"sha256: {digest}",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if new_content == content and "sha256:" not in content:
            print(f"ERROR: {VERSION_FILE} no tiene una línea 'sha256: ...' para reemplazar", file=sys.stderr)
            return 1
        VERSION_FILE.write_text(new_content, encoding="utf-8")
        print(f"\nVERSION actualizado: {VERSION_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
