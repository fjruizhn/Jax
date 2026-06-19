"""
Jacobs — Política de orquestación.

Candados duros: no diferibles, no configurables en v0.1.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jacobs.models import VALID_INVOKERS

KILL_SWITCH_PATH = Path("/etc/jax/PAUSE")

MAX_STEPS_PER_PIPELINE  = 20
MAX_PARALLEL_PIPELINES  = 3
MAX_SUBPIPELINE_DEPTH   = 1


@dataclass
class PolicyResult:
    ok:     bool
    reason: str


def check_kill_switch() -> bool:
    """True si el kill switch está activo."""
    return KILL_SWITCH_PATH.exists()


def validate_create(
    invoked_by: str,
    mode: str,
    max_steps: int,
    active_count: int,
    subpipeline_depth: int = 0,
    subpipeline_token: str | None = None,
) -> PolicyResult:
    """Valida si se puede crear un pipeline nuevo."""

    if invoked_by not in VALID_INVOKERS:
        return PolicyResult(
            ok=False,
            reason=f"invoked_by '{invoked_by}' no autorizado. Aceptados: {sorted(VALID_INVOKERS)}",
        )

    if invoked_by == "ada" and not subpipeline_token:
        return PolicyResult(
            ok=False,
            reason="ada requiere subpipeline_token para invocar Jacobs",
        )

    if max_steps > MAX_STEPS_PER_PIPELINE:
        return PolicyResult(
            ok=False,
            reason=f"max_steps={max_steps} excede límite duro ({MAX_STEPS_PER_PIPELINE})",
        )

    if active_count >= MAX_PARALLEL_PIPELINES:
        return PolicyResult(
            ok=False,
            reason=(
                f"Ya hay {active_count} pipelines activos. "
                f"Límite duro: {MAX_PARALLEL_PIPELINES}"
            ),
        )

    if subpipeline_depth > MAX_SUBPIPELINE_DEPTH:
        return PolicyResult(
            ok=False,
            reason=f"Profundidad de sub-pipeline ({subpipeline_depth}) excede límite ({MAX_SUBPIPELINE_DEPTH})",
        )

    if check_kill_switch():
        return PolicyResult(ok=False, reason="Kill switch activo — Jacobs detenido")

    return PolicyResult(ok=True, reason="OK")


def validate_resume(invoked_by: str) -> PolicyResult:
    """Solo Fernando puede reanudar un pipeline interrumpido."""
    if invoked_by != "Fernando":
        return PolicyResult(
            ok=False,
            reason="Solo Fernando puede reanudar un pipeline interrumpido",
        )
    return PolicyResult(ok=True, reason="OK")


def hyde_requires_human_gate() -> bool:
    """Hyde siempre requiere human_gate. Candado duro."""
    return True
