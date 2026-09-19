"""
Jacobs — Política de orquestación.

Candados duros: no diferibles, no configurables en v0.1.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import dataclass

from interruptor import interruptor_activo
from jacobs.models import (
    INVOKER_ADA,
    INVOKER_PLATAFORMA,
    MAX_STEPS_PER_PIPELINE,
    VALID_INVOKERS,
    PipelineStatus,
)

# MAX_PARALLEL_PIPELINES se espeja en jax-platform backend/ajustes.py (familia
# `tope_pipelines` de scripts/check_mirror_sync.py, frente C 2026-09-16): es el
# máximo del ajuste max_pipelines de Admin. Cambiarlo acá exige cambiar la copia
# en el mismo paso. MAX_STEPS_PER_PIPELINE vive en jacobs/models.py (E-13).
MAX_PARALLEL_PIPELINES  = 3


# ---------------------------------------------------------------------------
#  QUÉ ESTADOS OCUPAN CUPO — ESTO ES UN CONTRATO, NO UNA LISTA DE CONVENIENCIA
# ---------------------------------------------------------------------------
# El cupo se hace cumplir metiendo una condición DENTRO de cada escritura que lo
# consume (`jacobs/cupo.py` para el INSERT de crear; `jacobs/store.py` para los
# UPDATE de continuar, resume y approve-step). Las tres sentencias cuentan con
# estos estados y NADA MÁS, y por eso la lista vive acá, en un módulo que no
# importa a ninguno de los dos: una segunda copia se desincroniza sola.
#
# Agregar un estado a `PipelineStatus` sin decidir de qué lado cae **cuenta mal
# el cupo en producción y no lo avisa nadie**. Riesgo concreto y con fecha: el
# frente G agrega `queued`, `awaiting_approval` y `waiting_children`.
# Por eso la partición es EXHAUSTIVA y hay controles que se ponen rojos solos
# (`tests/test_creacion_sin_candado_global.py`, `jacobs/_cupo_io_test.py`).
ESTADOS_QUE_OCUPAN_CUPO = (PipelineStatus.pending, PipelineStatus.running)

#: El otro lado, declarado y no implícito. `interrupted` NO ocupa cupo: es la
#: semántica heredada de `store.pipeline_count_active()` y esta rama no la
#: cambia (un interrumpido espera un /resume humano; si ocupara cupo, tres
#: interrupciones sin atender frenarían la Mesa entera).
ESTADOS_SIN_CUPO = (
    PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.aborted,
    PipelineStatus.interrupted, PipelineStatus.expired,
    # Ronda de arreglo 2 (2026-09-18-arbitro-devuelve): `disputed` es
    # terminal -- el árbitro agotó el tope de devoluciones y el pipeline
    # dejó de correr, igual que `completed`/`failed`. No ocupa cupo.
    PipelineStatus.disputed,
)

#: La lista para un `IN (...)` de SQL. Literal y no parámetros: son valores del
#: enum, no datos del usuario, y así la sentencia queda legible en un EXPLAIN.
SQL_ESTADOS_VIVOS = ",".join(f"'{e.value}'" for e in ESTADOS_QUE_OCUPAN_CUPO)

#: El JOIN que le mete la condición del cupo a un UPDATE. MariaDB no deja
#: subconsultar en el `SET`/`WHERE` de un UPDATE en todas las versiones, pero un
#: JOIN con tabla derivada sí, y —medido el 2026-09-17 contra MariaDB 12.3 con
#: 10, 25 y 50 corrutinas— sostiene el cupo exacto sin un solo error.
SQL_JOIN_CUPO = (
    f"JOIN (SELECT COUNT(*) AS c FROM jacobs_pipelines "
    f"WHERE status IN ({SQL_ESTADOS_VIVOS})) cupo_x"
)


class ContencionAlReservar(Exception):
    """No se pudo DECIDIR el cupo: las escrituras del cupo se trabaron entre sí
    (deadlock de InnoDB) más veces de las que el presupuesto de espera permite.

    NO es un error del sistema y NO es un pedido inválido: es contención, y lo
    que corresponde es volver a intentar. Por eso el llamador responde **503
    con `contencion_al_reservar`** y un `Retry-After`, no un 500 (que manda a
    buscar un defecto que no existe) ni el 422 del cupo (que diría que el
    pedido está mal, y no lo está).

    Fail-closed igual: no se creó ni se reanudó nada.
    """

    def __init__(self, intentos: int, espera_total: float) -> None:
        self.intentos, self.espera_total = intentos, espera_total
        super().__init__(
            f"no se pudo decidir el cupo tras {intentos} intentos "
            f"({espera_total:.2f} s de espera): contención en la base"
        )


class CupoAgotado(Exception):
    """El cupo global está lleno. Vive acá, y no en cupo.py o store.py, para que
    los dos puedan levantarla sin importarse entre sí."""

    def __init__(self, activos: int, limite: int = MAX_PARALLEL_PIPELINES) -> None:
        self.activos, self.limite = activos, limite
        super().__init__(f"Ya hay {activos} pipelines activos. Límite duro: {limite}")

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
