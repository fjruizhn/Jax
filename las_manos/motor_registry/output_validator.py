"""
LAS MANOS — Motor Registry: validador de outputs de motores.

Valida la respuesta del motor contra el schema declarado en la capability.
Falla abierto por schema: si el schema es desconocido, emite warning y sigue.
Nunca bloquea un job por un schema inválido o faltante.

Schemas soportados:
  code_swarm.v1       — plan, steps, patches, tests, risk_notes, human_review_needed
  code_patch.v1       — diff, files_modified, description
  architecture_review.v1 — summary, risks, recommendations
  bug_hunt.v1         — bugs_found, severity, reproduction_steps, suggested_fix

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SCHEMAS: dict[str, list[str]] = {
    "code_swarm.v1": [
        "plan", "steps", "patches", "tests", "risk_notes", "human_review_needed",
    ],
    "code_patch.v1": [
        "diff", "files_modified", "description",
    ],
    "architecture_review.v1": [
        "summary", "risks", "recommendations",
    ],
    "bug_hunt.v1": [
        "bugs_found", "severity", "reproduction_steps", "suggested_fix",
    ],
}


def validate(content: str, schema_name: str) -> dict[str, Any]:
    """
    Valida `content` contra `schema_name`.

    Devuelve dict con:
        validated:      bool
        parsed:         dict | None    (si el content era JSON válido)
        missing_fields: list[str]      (campos requeridos ausentes)
        raw:            str            (content original, siempre presente)
        warning:        str | None
    """
    result: dict[str, Any] = {
        "validated": False,
        "parsed": None,
        "missing_fields": [],
        "raw": content,
        "warning": None,
    }

    # Intentar parsear como JSON
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        result["warning"] = "El motor devolvió texto libre, no JSON"
        return result

    if not isinstance(parsed, dict):
        result["warning"] = f"JSON válido pero no es un objeto (tipo: {type(parsed).__name__})"
        return result

    result["parsed"] = parsed

    # Si no hay schema_name, JSON válido alcanza
    if not schema_name:
        result["validated"] = True
        result["warning"] = "No se especificó schema — validación estructural omitida"
        return result

    # Schema desconocido: warning pero no falla
    required = SCHEMAS.get(schema_name)
    if required is None:
        result["validated"] = True
        result["warning"] = f"Schema '{schema_name}' desconocido — validación de campos omitida"
        logger.warning("Schema desconocido: '%s'", schema_name)
        return result

    # Verificar campos requeridos
    missing = [f for f in required if f not in parsed]
    if missing:
        result["missing_fields"] = missing
        result["warning"] = f"Campos faltantes para '{schema_name}': {missing}"
        return result

    result["validated"] = True
    return result
