#!/usr/bin/env python3
"""
Calcula el sha256 de policy/templates/render_templates.yaml, byte a byte
— mismo algoritmo que policy/tools/corpus_hash.py (que hashea el corpus
de rules/), aplicado a un solo archivo en vez de a un directorio.

Uso:
    policy/tools/template_hash.py            imprime el hash
    policy/tools/template_hash.py --write    lo escribe en VERSION, línea
                                              'templates_sha256:' (nueva
                                              si no existe, reemplazada si
                                              ya existe)
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_FILE = POLICY_DIR / "templates" / "render_templates.yaml"
VERSION_FILE = POLICY_DIR / "VERSION"


def compute_hash() -> str:
    return hashlib.sha256(TEMPLATES_FILE.read_bytes()).hexdigest()


def main() -> int:
    if not TEMPLATES_FILE.exists():
        print(f"ERROR: no existe {TEMPLATES_FILE}", file=sys.stderr)
        return 1

    digest = compute_hash()
    print(f"templates_sha256: {digest}")

    if "--write" in sys.argv:
        if not VERSION_FILE.exists():
            print(f"ERROR: no existe {VERSION_FILE}", file=sys.stderr)
            return 1
        content = VERSION_FILE.read_text(encoding="utf-8")
        if re.search(r"^templates_sha256:.*$", content, flags=re.MULTILINE):
            new_content = re.sub(
                r"^templates_sha256:.*$",
                f"templates_sha256: {digest}",
                content,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            sep = "" if content.endswith("\n") else "\n"
            new_content = f"{content}{sep}templates_sha256: {digest}\n"
        VERSION_FILE.write_text(new_content, encoding="utf-8")
        print(f"\nVERSION actualizado: {VERSION_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
