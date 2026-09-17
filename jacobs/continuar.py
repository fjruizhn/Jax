"""Jacobs — continuar un pipeline abortado o vencido (spec 2026-09-17 §5).

UNA función de servicio para el endpoint (POST /jacobs/pipeline/{id}/continue
y /continue/preflight) y para el CLI (tools/jacobs_relaunch.py): ninguno tiene
lógica propia.

Reglas (§5.2): sólo `plataforma` (el dueño lo verifica jax-platform); estados
aborted (cualquier causa, D2) o expired; kill switch 423; dentro del candado y
contra MAX_PARALLEL_PIPELINES; se reusan los pasos con ref LEGIBLE y se
rehacen todos los demás; reasignar solo pasos a correr, con clean-room y
gobernanza sobre el plan completo; pre-vuelo sobre los pendientes; escrituras
en una transacción que incrementa la época.

Lo de antes de la transacción NO escribe filas de estado. El único registro
previo es el evento PREVUELO_RECHAZADO (auditoría, igual que al crear).

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal

from jacobs import store
from jacobs.candado import candado_de_creacion
from jacobs.executor import RefIlegible, _load_ref
from jacobs.models import MOTOR_FACETS, VALID_FACETS, Pipeline, PipelineStatus, Step, StepStatus
from jacobs.plan import PlanRejected, _check_cleanroom, _validate_plan_capabilities
from jacobs.policy import MAX_PARALLEL_PIPELINES, check_kill_switch, validate_resume
from jacobs.prevuelo import prevuelo
from jacobs.prevuelo_reglas import formatear_usd

logger = logging.getLogger("jacobs.continuar")

ESTADOS_CONTINUABLES = frozenset({PipelineStatus.aborted, PipelineStatus.expired})


class ContinuarRechazado(Exception):
    def __init__(self, status_code: int, code: str, detalle):
        super().__init__(f"{status_code} {code}: {detalle}")
        self.status_code = status_code
        self.code = code
        self.detalle = detalle

    def cuerpo(self) -> dict:
        if isinstance(self.detalle, dict):
            return {"code": self.code, **self.detalle}
        return {"code": self.code, "detalle": self.detalle}


@dataclass
class Analisis:
    pipeline: Pipeline
    plan: list[Step]
    contexto: dict
    pasos_a_correr: list[int]
    pasos_reusados: list[int]
    reasignados: dict[str, dict] = field(default_factory=dict)


def _ref_legible(ref: str) -> bool:
    """SÍNCRONA (lee artifacts de disco): se llama por asyncio.to_thread."""
    try:
        _load_ref(ref)
    except RefIlegible as exc:
        logger.warning("continuar: ref ilegible, el paso se rehace (%s)", exc)
        return False
    return True


def _texto_limite(activos: int) -> str:
    return f"Ya hay {activos} pipelines activos. Límite duro: {MAX_PARALLEL_PIPELINES}"


async def analizar(pipeline_id: str, invoked_by: str, reasignar: dict[str, str] | None) -> Analisis:
    politica = validate_resume(invoked_by)
    if not politica.ok:
        raise ContinuarRechazado(403, "invocador_no_autorizado", politica.reason)
    pipeline = await store.pipeline_get(pipeline_id)
    if pipeline is None:
        raise ContinuarRechazado(404, "no_existe", f"Pipeline '{pipeline_id}' no encontrado")
    if pipeline.status not in ESTADOS_CONTINUABLES:
        mensaje = ("tiene /resume" if pipeline.status == PipelineStatus.interrupted
                   else "solo se continúan pipelines aborted o expired")
        raise ContinuarRechazado(409, "estado_no_continuable",
                                 {"status": pipeline.status.value, "mensaje": mensaje})
    if check_kill_switch():
        raise ContinuarRechazado(423, "kill_switch", "Kill switch activo — no se puede continuar")

    # Los pasos VIGENTES (jacobs_steps), no la foto de creación en `plan`.
    plan = await store.steps_by_pipeline(pipeline_id)
    if [s.step_index for s in plan] != list(range(len(plan))):
        raise ContinuarRechazado(409, "plan_inconsistente",
                                 "los pasos guardados no son 0..N-1 sin huecos")

    contexto = dict(pipeline.context)
    reusados: list[int] = []
    a_correr: list[int] = []
    for paso in plan:
        clave = f"step_{paso.step_index}_ref"
        ref = contexto.get(clave)
        if ref and await asyncio.to_thread(_ref_legible, ref):
            reusados.append(paso.step_index)
            continue
        contexto.pop(clave, None)
        # La aprobación humana de hyde era para la corrida anterior (desvío 10).
        contexto.pop(f"hyde_approved_{paso.step_id}", None)
        paso.status = StepStatus.pending
        paso.error = None
        paso.started_at = None
        paso.finished_at = None
        paso.output_ref = None
        a_correr.append(paso.step_index)

    reasignados: dict[str, dict] = {}
    invalidas: list[dict] = []
    for clave, faceta in (reasignar or {}).items():
        try:
            indice = int(clave)
        except (TypeError, ValueError):
            invalidas.append({"paso": clave, "motivo": "el índice de paso no es un entero"})
            continue
        if indice not in a_correr:
            invalidas.append({"paso": indice, "motivo": "solo se reasignan pasos a correr, no los reusados"})
            continue
        if faceta not in VALID_FACETS:
            invalidas.append({"paso": indice, "motivo": f"faceta desconocida: '{faceta}'"})
            continue
        paso = plan[indice]
        reasignados[str(indice)] = {"de": paso.facet, "a": faceta}
        paso.facet = faceta
        paso.motor = faceta if faceta in MOTOR_FACETS else None
    if invalidas:
        raise ContinuarRechazado(422, "reasignacion_invalida", invalidas)

    codigo = "reasignacion_invalida" if reasignados else "plan_rechazado"
    violaciones = _check_cleanroom(plan)
    if violaciones:
        raise ContinuarRechazado(422, codigo, [v.to_dict() for v in violaciones])
    try:
        await _validate_plan_capabilities(plan)
    except PlanRejected as exc:
        raise ContinuarRechazado(422, codigo, [v.to_dict() for v in exc.violations]) from exc

    return Analisis(pipeline, plan, contexto, a_correr, reusados, reasignados)


def _pasos(a: Analisis) -> dict:
    return {"pasos_a_correr": a.pasos_a_correr, "pasos_reusados": a.pasos_reusados}


async def _prevuelo_de(a: Analisis, user_id: str | None, tenant_id: str | None):
    return await prevuelo(
        a.plan, a.contexto, pendientes=set(a.pasos_a_correr),
        user_id=user_id or a.pipeline.user_id, tenant_id=tenant_id or a.pipeline.tenant_id,
    )


async def previsualizar(pipeline_id: str, invoked_by: str, reasignar: dict[str, str] | None = None,
                        user_id: str | None = None, tenant_id: str | None = None) -> dict:
    """Reglas 1-8 sin escribir (§5.1). 403 y 404 se relanzan; el resto se
    devuelve como `continuable: False` con su motivo."""
    try:
        a = await analizar(pipeline_id, invoked_by, reasignar)
    except ContinuarRechazado as r:
        if r.status_code in (403, 404):
            raise
        return {"continuable": False, "motivo": r.cuerpo(),
                "pasos_a_correr": [], "pasos_reusados": [], "veredicto": None}
    veredicto = await _prevuelo_de(a, user_id, tenant_id)
    activos = await store.pipeline_count_active()
    motivo = None
    if activos >= MAX_PARALLEL_PIPELINES:
        motivo = {"code": "limite_de_activos", "detalle": _texto_limite(activos)}
    elif not veredicto.ok:
        motivo = {"code": "prevuelo_rechazado", "detalle": "el pre-vuelo encontró violaciones: ver `veredicto`"}
    return {"continuable": motivo is None, "motivo": motivo, **_pasos(a), "veredicto": veredicto.to_dict()}


async def continuar(pipeline_id: str, invoked_by: str, reasignar: dict[str, str] | None = None,
                    user_id: str | None = None, tenant_id: str | None = None,
                    costo_max_aceptado_usd: Decimal | None = None) -> tuple[dict, Pipeline]:
    async with candado_de_creacion:
        a = await analizar(pipeline_id, invoked_by, reasignar)
        activos = await store.pipeline_count_active()
        if activos >= MAX_PARALLEL_PIPELINES:
            raise ContinuarRechazado(429, "limite_de_activos", _texto_limite(activos))
        veredicto = await _prevuelo_de(a, user_id, tenant_id)
        if not veredicto.ok:
            await store.event_append(pipeline_id, "PREVUELO_RECHAZADO",
                                     {"code": "prevuelo_rechazado", **veredicto.to_dict()})
            raise ContinuarRechazado(422, "prevuelo_rechazado", veredicto.to_dict())
        if costo_max_aceptado_usd is not None and veredicto.costo_max_usd > costo_max_aceptado_usd:
            raise ContinuarRechazado(409, "costo_supera_lo_aceptado", {
                "costo_max_aceptado_usd": formatear_usd(costo_max_aceptado_usd), **veredicto.to_dict(),
            })

        indice = min(a.pasos_a_correr) if a.pasos_a_correr else len(a.plan)
        nueva = await store.continuar_transaccion(
            pipeline_id, a.pipeline.run_epoch, a.pipeline.status,
            [a.plan[i] for i in a.pasos_a_correr], a.plan, a.contexto, indice,
        )
        if nueva is None:
            raise ContinuarRechazado(409, "estado_no_continuable", {
                "status": None,
                "mensaje": "otro pedido cambió el pipeline mientras se preparaba este: volvé a consultarlo",
            })
        continuado = a.pipeline.model_copy(update={
            "plan": a.plan, "context": a.contexto, "status": PipelineStatus.running,
            "run_epoch": nueva, "current_step_index": indice,
        })
        await store.event_append(pipeline_id, "PIPELINE_CONTINUED", {
            "by": invoked_by, "from_status": a.pipeline.status.value, "run_epoch": nueva,
            **_pasos(a), "reasignados": a.reasignados, "costo_max_usd": formatear_usd(veredicto.costo_max_usd),
        })

    respuesta = {
        "pipeline_id": pipeline_id, "status": "running", "run_epoch": nueva, **_pasos(a),
        "costo_max_usd": formatear_usd(veredicto.costo_max_usd),
        "pasos_costo": [c.to_dict() for c in veredicto.pasos_costo],
    }
    return respuesta, continuado
