#!/usr/bin/env python3
"""
Valida el esquema de rules/*.yaml — ver policy/README.md y
/opt/jax/docs/FASE-0.5-CORPUS.md para la definición de cada campo y estado.

Falla (exit 1) si:
  - falta un campo obligatorio
  - status no es uno de los cuatro válidos
  - status == NORMATIVA con enforcement.test == null
  - hay ids duplicados entre archivos
  - el id del archivo no coincide con el id declarado dentro del YAML

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

POLICY_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = POLICY_DIR / "rules"

VALID_STATUSES = {"NORMATIVA", "NORMATIVA_PENDIENTE", "CULTURAL", "HISTORICA"}

REQUIRED_TOP_FIELDS = {
    "id", "statement", "origin", "status", "enforcement",
    "version", "created", "amended_by",
}
REQUIRED_ENFORCEMENT_FIELDS = {"mechanism", "test"}


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return [f"{path.name}: YAML inválido — {e}"]

    if not isinstance(data, dict):
        return [f"{path.name}: el documento no es un mapeo YAML"]

    missing = REQUIRED_TOP_FIELDS - data.keys()
    if missing:
        errors.append(f"{path.name}: faltan campos obligatorios: {sorted(missing)}")

    rule_id = data.get("id")
    if rule_id:
        expected_prefix = path.stem.split("-", 1)[0]
        if rule_id != expected_prefix:
            errors.append(
                f"{path.name}: id declarado '{rule_id}' no coincide con el "
                f"prefijo del nombre de archivo '{expected_prefix}'"
            )

    status = data.get("status")
    if status is not None and status not in VALID_STATUSES:
        errors.append(f"{path.name}: status '{status}' no es uno de {sorted(VALID_STATUSES)}")

    enforcement = data.get("enforcement")
    if not isinstance(enforcement, dict):
        errors.append(f"{path.name}: 'enforcement' debe ser un mapeo con 'mechanism' y 'test'")
    else:
        missing_enf = REQUIRED_ENFORCEMENT_FIELDS - enforcement.keys()
        if missing_enf:
            errors.append(f"{path.name}: faltan campos en 'enforcement': {sorted(missing_enf)}")
        if status == "NORMATIVA" and enforcement.get("test") is None:
            errors.append(
                f"{path.name}: status=NORMATIVA requiere enforcement.test != null "
                f"(no puede ser NORMATIVA sin un test que la haga cumplir hoy)"
            )

    statement = data.get("statement")
    if statement is not None and not str(statement).strip():
        errors.append(f"{path.name}: 'statement' está vacío")

    return errors


def main() -> int:
    if not RULES_DIR.exists():
        print(f"ERROR: no existe {RULES_DIR}", file=sys.stderr)
        return 1

    files = sorted(RULES_DIR.glob("*.yaml"))
    if not files:
        print(f"ERROR: no hay archivos .yaml en {RULES_DIR}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    seen_ids: dict[str, str] = {}

    for f in files:
        all_errors.extend(validate_file(f))
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        rid = data.get("id")
        if rid:
            if rid in seen_ids:
                all_errors.append(
                    f"{f.name}: id duplicado '{rid}' (ya usado en {seen_ids[rid]})"
                )
            else:
                seen_ids[rid] = f.name

    print(f"Archivos revisados: {len(files)}")
    print(f"Ids únicos: {len(seen_ids)}")

    if all_errors:
        print(f"\nFALLÓ — {len(all_errors)} error(es):")
        for e in all_errors:
            print(f"  ✗ {e}")
        return 1

    print("\n✓ Corpus válido — todos los archivos cumplen el esquema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
