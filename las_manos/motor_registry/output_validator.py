"""
LAS MANOS — Motor Registry: validador de outputs de motores.

Valida la respuesta del motor contra el schema declarado en la capability.
Fail-open ACOTADO (P10): un schema declarado pero sin validación de campos
implementada (ver _KNOWN_UNIMPLEMENTED_SCHEMAS) emite warning y sigue. Un
schema que ni siquiera está declarado ahí -- typo, capability mal
configurada -- falla cerrado: el caller lo trata como salida inválida.

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

# Schemas declarados HOY en capability.output_schema (DB de produccion,
# verificado 2026-08-25) pero sin validacion de campos implementada --
# fail-open EXPLICITO y ACOTADO a esta lista (deuda registrada, no un
# descuido): critique, design, generate, pipeline_analysis, reason,
# reconcile, validate_consistency. Implementarlos requiere saber que
# campos devuelve cada capability realmente -- fuera de este fix (ver
# DEUDA.md, item nuevo "schemas de capability sin validacion de campos").
#
# Cualquier OTRO nombre que llegue acá y no esté en SCHEMAS tampoco en
# esta lista es un caso distinto: typo, capability mal configurada, o un
# nombre nuevo que alguien se olvidó de declarar acá -- ESE sí falla
# cerrado (P10).
_KNOWN_UNIMPLEMENTED_SCHEMAS: frozenset[str] = frozenset({
    "critique.v1", "design.v1", "generate.v1", "analysis.v1",
    "reason.v1", "reconcile.v1", "validation.v1",
})


def validate(content: str, schema_name: str, has_tool_calls: bool = False) -> dict[str, Any]:
    """
    Valida `content` contra `schema_name`.

    Devuelve dict con:
        validated:      bool
        parsed:         dict | None    (si el content era JSON válido)
        missing_fields: list[str]      (campos requeridos ausentes)
        raw:            str            (content original, siempre presente)
        warning:        str | None
        skipped:        bool           (True si la validación de forma no
                                         aplica -- ver has_tool_calls)

    has_tool_calls (GAP2 Fase1, 2026-08-19): cuando el motor devuelve
    tool_calls, la salida estructurada viaja por ESE canal, no por
    `content` -- `content` puede venir vacío legítimamente (verificado
    en vivo: Ollama+qwen3.6 devuelve content="" con tool_calls poblado).
    Validar `content` como si fuera texto libre en ese caso generaría
    "El motor devolvió texto libre, no JSON" -- falso: no fue un intento
    fallido de responder en JSON, fue una respuesta correcta por el otro
    canal. Los dos canales son mutuamente excluyentes; la exclusión se
    resuelve ACÁ (una sola fuente de verdad), no en cada caller.
    """
    if has_tool_calls:
        return {
            "validated": False,
            "parsed": None,
            "missing_fields": [],
            "raw": content,
            "warning": None,
            "skipped": True,
        }

    result: dict[str, Any] = {
        "validated": False,
        "parsed": None,
        "missing_fields": [],
        "raw": content,
        "warning": None,
        "skipped": False,
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

    required = SCHEMAS.get(schema_name)
    if required is None:
        if schema_name in _KNOWN_UNIMPLEMENTED_SCHEMAS:
            # Fail-open EXPLICITO: declarado en produccion, sin schema de
            # campos implementado todavia. Ver _KNOWN_UNIMPLEMENTED_SCHEMAS.
            result["validated"] = True
            result["warning"] = (
                f"Schema '{schema_name}' declarado pero sin schema de campos "
                "implementado — validación omitida (deuda conocida)"
            )
            logger.warning("Schema declarado-pendiente sin validar: '%s'", schema_name)
            return result
        # Schema realmente desconocido: ni implementado ni declarado como
        # pendiente -- typo o capability mal configurada. Fail-closed (P10):
        # el caller (worker.py) reintenta una vez y despues marca FAILED,
        # en vez de completar el job creyendo que algo se validó.
        result["warning"] = (
            f"Schema '{schema_name}' no reconocido (ni implementado ni "
            "declarado como pendiente) — posible typo o capability mal "
            "configurada"
        )
        logger.error("Schema NO reconocido, fallando cerrado: '%s'", schema_name)
        return result

    # Verificar campos requeridos
    missing = [f for f in required if f not in parsed]
    if missing:
        result["missing_fields"] = missing
        result["warning"] = f"Campos faltantes para '{schema_name}': {missing}"
        return result

    result["validated"] = True
    return result
