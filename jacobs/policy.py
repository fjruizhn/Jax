"""
Jacobs — Política de orquestación.

Candados duros: no diferibles, no configurables en v0.1.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import dataclass

from interruptor import interruptor_activo
from jacobs.models import INVOKER_ADA, INVOKER_PLATAFORMA, MAX_STEPS_PER_PIPELINE, VALID_INVOKERS

# MAX_PARALLEL_PIPELINES se espeja en jax-platform backend/ajustes.py (familia
# `tope_pipelines` de scripts/check_mirror_sync.py, frente C 2026-09-16): es el
# máximo del ajuste max_pipelines de Admin. Cambiarlo acá exige cambiar la copia
# en el mismo paso. MAX_STEPS_PER_PIPELINE vive en jacobs/models.py (E-13).
MAX_PARALLEL_PIPELINES  = 3
# MAX_SUBPIPELINE_DEPTH se borró (frente F, 2026-09-16): era un literal que
# ningún llamador alimentaba. La profundidad vive en la fila del token y el
# límite en JAX_MAX_SUBPIPELINE_DEPTH (jacobs/subpipelines.py).


@dataclass
class PolicyResult:
    ok:     bool
    reason: str


def check_kill_switch() -> bool:
    """True si el kill switch está puesto (archivo de JAX_KILL_SWITCH_PATH).

    Sin la variable lanza InterruptorSinConfigurar: sin saber dónde está el
    freno no se ejecuta nada. Un error al mirarlo que no sea "no existe"
    cuenta como PUESTO (interruptor.py)."""
    return interruptor_activo()


def validate_create(
    invoked_by: str,
    mode: str,
    max_steps: int,
    subpipeline_token: str | None = None,
    parent_pipeline_id: str | None = None,
) -> PolicyResult:
    """Valida si se puede crear un pipeline nuevo.

    Para ada controla la FORMA (token y padre presentes; nadie más puede
    traerlos). La VALIDEZ del token no se decide acá: es un consumo atómico en
    la base (subpipelines.consumir_token_subpipeline), que routes.create_pipeline
    corre justo después de esta función.

    EL CUPO TAMPOCO SE DECIDE ACÁ (2026-09-17). Hasta hoy esta función recibía
    `active_count` y comparaba contra MAX_PARALLEL_PIPELINES: un conteo leído
    en una consulta y decidido en otra, que sólo era un límite de verdad
    porque routes.py lo envolvía en un candado global del proceso. Ese candado
    serializaba toda la creación (~43 delegaciones/s, y la Mesa esperando
    detrás de Ada). Ahora el cupo lo hace cumplir la base en UNA sentencia:
    `jacobs/cupo.py::reservar_cupo`. La constante sigue viviendo acá porque es
    la que `scripts/check_mirror_sync.py` espeja contra jax-platform."""

    if invoked_by not in VALID_INVOKERS:
        return PolicyResult(
            ok=False,
            reason=f"invoked_by '{invoked_by}' no autorizado. Aceptados: {sorted(VALID_INVOKERS)}",
        )

    if invoked_by == INVOKER_ADA and (not subpipeline_token or not parent_pipeline_id):
        return PolicyResult(
            ok=False,
            reason="ada requiere subpipeline_token y parent_pipeline_id para invocar Jacobs",
        )

    if invoked_by != INVOKER_ADA and (subpipeline_token or parent_pipeline_id):
        return PolicyResult(
            ok=False,
            reason=(
                f"invoked_by '{invoked_by}' no puede presentar subpipeline_token "
                "ni parent_pipeline_id"
            ),
        )

    if max_steps > MAX_STEPS_PER_PIPELINE:
        return PolicyResult(
            ok=False,
            reason=f"max_steps={max_steps} excede límite duro ({MAX_STEPS_PER_PIPELINE})",
        )

    if check_kill_switch():
        return PolicyResult(ok=False, reason="Kill switch activo — Jacobs detenido")

    return PolicyResult(ok=True, reason="OK")


def validate_resume(invoked_by: str) -> PolicyResult:
    """Solo la plataforma (jax-platform, en nombre de un usuario autenticado y
    dueño del pipeline -- eso lo verifica jax-platform antes de reenviar)
    puede reanudar un pipeline interrumpido o aprobar un paso."""
    if invoked_by != INVOKER_PLATAFORMA:
        return PolicyResult(
            ok=False,
            reason=(
                f"invoked_by '{invoked_by}' no puede reanudar ni aprobar: "
                f"solo '{INVOKER_PLATAFORMA}'"
            ),
        )
    return PolicyResult(ok=True, reason="OK")
