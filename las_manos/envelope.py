"""
LAS MANOS — Intent Envelope.

Cada llamada de una faceta a LAS MANOS viaja dentro de este sobre. Sin sobre
completo y válido, LAS MANOS falla cerrado (HTTP 422). Baja el Contrato Final
de Activación que la Mesa (Thot + Jekyll + Fernando) firmó el 15-jun-2026 a
una especificación EJECUTABLE: 18 campos + 9 condiciones de fallo cerrado.

"El contrato existe cuando el sistema puede negarse a obedecerlo mal."

Dos capas de validación:
  1. Estructural (Pydantic, IntentEnvelope): campo faltante o mal tipado → 422.
  2. Semántica (validate): las condiciones CRUZADAS del contrato.

Este módulo es PURO: sin I/O, sin red, testeable en aislamiento.
"En la duda, el sistema se niega."

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

# Operaciones mutantes (idéntico a policy.py y a la spec, línea 71).
MUTATING_CAPABILITIES = {"ssh_exec", "write_file", "rsync", "kill_process"}


class MemoryRef(BaseModel):
    """Una memoria que informa la intención. La memoria informa, no autoriza."""
    id: str
    type: str
    has_provenance: bool


class IntentEnvelope(BaseModel):
    # ── Los 18 campos del contrato (Mesa, 15-jun-2026) ──────────────────
    trace_id: str                                                          # 1
    facet_id: Literal["thot", "hipatia", "jekyll", "hyde", "jax_local"]    # 2
    actor_type: Literal["faceta", "voz", "persona"]                        # 3
    origin_of_authority: str                                               # 4
    verification_label: Literal[
        "web_verified", "internal_knowledge", "local_context", "unverified"
    ]                                                                      # 5
    intent_summary: str                                                    # 6
    requested_capability: Literal[
        "ssh_exec_readonly", "ssh_exec", "read_file", "write_file",
        "list_dir", "rsync", "http_get", "kill_process",
        "audit_log_read", "validate_json", "validate_yaml",
    ]                                                                      # 7
    target_environment: Literal["local", "staging", "prod", "bridge"]      # 8
    risk_level: Literal["none", "low", "medium", "high"]                   # 9
    memory_refs_used: list[MemoryRef]                                      # 10
    freshness_required: bool                                               # 11
    dry_run_required: bool                                                 # 12
    policy_required: bool                                                  # 13
    human_gate_required: bool                                              # 14
    approval_token: str | None                                            # 15 (condicional, pero la CLAVE es obligatoria)
    rollback_plan: str | None                                            # 16 (condicional, pero la CLAVE es obligatoria)
    kill_switch_scope: Literal["global", "per_operation", "none"]          # 17
    fail_closed_behavior: str                                              # 18

    # ── Carga operacional (NO parte de los 18 campos del contrato) ──────
    # El sobre la envuelve para que policy.check y los workers sigan igual.
    # target_environment es el AMBIENTE declarado; target_host es el IP real.
    target_host: str
    params: dict = Field(default_factory=dict)

    # Los 18 campos exigen su CLAVE presente (incluso approval_token /
    # rollback_plan, que admiten valor null pero no ausencia). Pydantic
    # rechaza cualquier clave faltante con 422 — esa es la condición 1.
    model_config = {"extra": "forbid"}


@dataclass
class EnvelopeResult:
    ok: bool
    reason: str
    code: str = "ENVELOPE_REJECTED"


def validate(env: IntentEnvelope) -> EnvelopeResult:
    """Capa 2 — las condiciones CRUZADAS del contrato (las que Pydantic no ve).

    Cubre las condiciones 2, 4, 5, 8, 9 + la obligatoriedad condicional del
    approval_token (campo 15) + no-vacío de los campos de texto del contrato.
    Quedan fuera (por diseño, ya las maneja el sistema):
      - cond. 1 (campo faltante) y 3 (enum inválido) → capa Pydantic (422).
      - cond. 6 (token reutilizado)                  → human gate existente.
      - cond. 7 (policy no puede decidir)            → policy.check.
    Devuelve al PRIMER fallo: en la duda, el sistema se niega.
    """
    # Condición 2 — origin_of_authority no vacío.
    if not env.origin_of_authority.strip():
        return EnvelopeResult(False, "origin_of_authority vacío (condición 2)")

    # Campos 6 y 18 — texto legible obligatorio, no vacío.
    if not env.intent_summary.strip():
        return EnvelopeResult(False, "intent_summary vacío (campo 6)")
    if not env.fail_closed_behavior.strip():
        return EnvelopeResult(False, "fail_closed_behavior vacío (campo 18)")

    # Condición 4 — ninguna memoria sin procedencia.
    for ref in env.memory_refs_used:
        if not ref.has_provenance:
            return EnvelopeResult(
                False,
                f"memoria '{ref.id}' sin procedencia (has_provenance=false) "
                f"(condición 4). La memoria informa, no autoriza.",
            )

    mutating = env.requested_capability in MUTATING_CAPABILITIES

    # Condición 5 — prod + mutante + sin human gate declarado = mentira peligrosa.
    if env.target_environment == "prod" and mutating and not env.human_gate_required:
        return EnvelopeResult(
            False,
            "target_environment=prod + operación mutante + "
            "human_gate_required=false (condición 5)",
        )

    # Campo 15 — si declara human gate, el token debe venir (presencia, no validez).
    if env.human_gate_required and not (env.approval_token and env.approval_token.strip()):
        return EnvelopeResult(
            False,
            "human_gate_required=true sin approval_token presente (campo 15)",
        )

    # Condición 8 — operación mutante no puede tener kill_switch_scope=none.
    if mutating and env.kill_switch_scope == "none":
        return EnvelopeResult(
            False,
            "operación mutante con kill_switch_scope=none (condición 8)",
        )

    # Condición 9 — operación mutante en prod exige rollback_plan.
    if mutating and env.target_environment == "prod" and not (
        env.rollback_plan and env.rollback_plan.strip()
    ):
        return EnvelopeResult(
            False,
            "operación mutante en prod sin rollback_plan (condición 9)",
        )

    return EnvelopeResult(True, "Envelope válido")
