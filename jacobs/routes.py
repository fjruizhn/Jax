"""
Jacobs — Endpoints FastAPI.

En honor al Prof. Raúl Jacobs — maestro, mentor, director.
En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from redaccion import recortar_redactado

from jacobs import store
from jacobs.artifacts import read_artifact
from jacobs.executor import run_pipeline
from jacobs.models import (
    VALID_INVOKERS,
    Pipeline,
    PipelineCreateRequest,
    PipelineStatus,
    Step,
    StepSpec,
    StepStatus,
)
from jacobs.plan import PlanBuilder, PlanRejected
from jacobs.prevuelo import prevuelo
from jacobs.prevuelo_reglas import Veredicto
from jacobs.policy import (
    check_kill_switch,
    validate_create,
    validate_resume,
)

router = APIRouter(prefix="/jacobs", tags=["jacobs"])

# Añadido 2026-09-16: `_resolve_ref` tragaba el fallo de lectura sin dejar
# rastro. Un error que no se registra en ningún lado no existe para nadie.
logger = logging.getLogger(__name__)

_plan_builder = PlanBuilder()

# T2 (2026-08-19): active_count = await store.pipeline_count_active() y el
# INSERT posterior corrían en conexiones DB separadas, sin lock ni
# transacción -- dos POST /pipeline concurrentes podían leer el mismo
# active_count y ambos pasar validate_create(), superando
# MAX_PARALLEL_PIPELINES. Jacobs corre en un solo proceso uvicorn (sin
# --workers, confirmado por systemctl/ps) así que un asyncio.Lock() de
# proceso es válido y cubre el caso real. Se prefiere sobre una transacción
# con SELECT...FOR UPDATE porque _plan_builder.build() (adentro de la
# sección crítica) tarda 20-40s llamando a un LLM externo -- mantener esa
# transacción/fila lockeada todo ese tiempo arriesgaría agotar el pool de
# conexiones bajo carga real; un lock en memoria no reserva conexión DB.
_pipeline_create_lock = asyncio.Lock()


async def _build_plan_or_reject(
    pipeline_id: str, objective: str, max_steps: int, steps_spec: list[dict] | None,
) -> list[Step]:
    """T2 (2026-08-21, diagnóstico pipeline 19ad2c42-cdf): único punto de
    entrada a _plan_builder.build() -- si el plan es inejecutable,
    PlanRejected sube ANTES de que cualquiera de los dos endpoints llame a
    store.pipeline_create()/pipeline_add_steps(), así que jacobs_pipelines/
    jacobs_steps nunca llegan a tener fila para este plan. El evento en
    jacobs_events no depende de que exista una fila en jacobs_pipelines --
    la tabla no tiene FK a pipeline_id (confirmado: DESCRIBE jacobs_events),
    así que registrar el rechazo bajo este pipeline_id (nunca persistido
    como pipeline real) es seguro."""
    try:
        return await _plan_builder.build(
            pipeline_id=pipeline_id, objective=objective,
            max_steps=max_steps, steps_spec=steps_spec,
        )
    except PlanRejected as exc:
        await store.event_append(
            pipeline_id, "PLAN_REJECTED",
            {"violations": [v.to_dict() for v in exc.violations]},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _prevuelo_o_503(
    steps: list[Step],
    contexto: dict,
    *,
    pendientes: set[int] | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> Veredicto:
    """Spec 2026-09-17 §8: sin pre-vuelo no se crea ni se continúa. Un error
    (DB caída, catálogo ilegible) es 503 prevuelo_no_disponible, nunca un 500
    genérico ni un pase libre."""
    try:
        return await prevuelo(steps, contexto, pendientes=pendientes,
                              user_id=user_id, tenant_id=tenant_id)
    except Exception as exc:
        motivo = recortar_redactado(f"{type(exc).__name__}: {exc}", 300)
        logger.error("pre-vuelo no disponible: %s", motivo, exc_info=True)
        raise HTTPException(
            status_code=503, detail={"code": "prevuelo_no_disponible", "motivo": motivo},
        ) from exc


def _resumen_de_costo(veredicto: Veredicto) -> dict:
    return {
        "costo_max_usd": str(veredicto.costo_max_usd),
        "pasos_costo": [c.to_dict() for c in veredicto.pasos_costo],
    }


# ----------------------------------------------------------------
#  POST /jacobs/preflight  — pre-vuelo sin escribir
# ----------------------------------------------------------------

class PreflightRequest(BaseModel):
    invoked_by: str
    # Obligatorio y no vacío: sin pasos, build() planifica con un LLM pago y un
    # pre-vuelo no puede gastar (desvío 6 del plan 2026-09-17).
    steps:      list[StepSpec] = Field(min_length=1)
    objective:  str = ""
    user_id:    str | None = None
    tenant_id:  str | None = None


@router.post("/preflight")
async def preflight(req: PreflightRequest) -> dict:
    """Pre-vuelo de un plan explícito (spec 2026-09-17 §4.7). Responde 200 con
    el veredicto AUNQUE rechace. No crea filas, no escribe eventos, no toma el
    candado. La sonda sí registra su evento de salud y su uso: se paga.

    El camino por objetivo (planificación por LLM) NO pasa por acá: lo gastado
    en planificar ya está gastado; crear corre el pre-vuelo sobre el plan que
    devolvió el LLM."""
    if req.invoked_by not in VALID_INVOKERS:
        raise HTTPException(status_code=403, detail=f"invoked_by '{req.invoked_by}' no autorizado")
    if len(req.steps) > 20:
        raise HTTPException(status_code=422, detail=f"{len(req.steps)} pasos excede el límite duro (20)")
    steps_spec = [s.model_dump() for s in req.steps]
    try:
        steps = await _plan_builder.build(
            pipeline_id=str(uuid.uuid4()), objective=req.objective,
            max_steps=len(steps_spec), steps_spec=steps_spec,
        )
    except PlanRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    veredicto = await _prevuelo_o_503(
        steps, {"objective": req.objective}, user_id=req.user_id, tenant_id=req.tenant_id,
    )
    return veredicto.to_dict()


# ----------------------------------------------------------------
#  POST /jacobs/plan  — genera plan sin ejecutar
# ----------------------------------------------------------------

class PlanRequest(BaseModel):
    name:       str
    objective:  str
    invoked_by: str
    mode:       str
    max_steps:  int = 20
    steps:      list[StepSpec] | None = None


@router.post("/plan")
async def plan_only(req: PlanRequest) -> dict:
    """Genera un plan de steps sin ejecutar nada (dry_run de planificación)."""

    if req.max_steps > 20:
        raise HTTPException(
            status_code=422,
            detail=f"max_steps={req.max_steps} excede límite duro (20)",
        )
    if req.invoked_by not in VALID_INVOKERS:
        raise HTTPException(
            status_code=403,
            detail=f"invoked_by '{req.invoked_by}' no autorizado",
        )
    if check_kill_switch():
        raise HTTPException(
            status_code=423,
            detail="Kill switch activo — Jacobs detenido",
        )

    pipeline_id = str(uuid.uuid4())
    steps_spec = [s.model_dump() for s in req.steps] if req.steps else None
    steps = await _build_plan_or_reject(pipeline_id, req.objective, req.max_steps, steps_spec)

    return {
        "pipeline_id": pipeline_id,
        "name": req.name,
        "mode": req.mode,
        "invoked_by": req.invoked_by,
        "objective": req.objective,
        "plan": [s.model_dump() for s in steps],
        "step_count": len(steps),
        "executed": False,
    }


# ----------------------------------------------------------------
#  POST /jacobs/pipeline  — crea y ejecuta
# ----------------------------------------------------------------

@router.post("/pipeline")
async def create_pipeline(req: PipelineCreateRequest, background: BackgroundTasks) -> dict:
    """Crea un pipeline y lo ejecuta en background."""

    # Lock de proceso: active_count (lectura) y pipeline_create (escritura)
    # deben ser atómicos entre sí para que MAX_PARALLEL_PIPELINES sea un
    # límite real, no una lectura optimista. Incluye build() adentro a
    # propósito -- ver justificación en _pipeline_create_lock arriba.
    async with _pipeline_create_lock:
        active_count = await store.pipeline_count_active()
        policy = validate_create(
            invoked_by=req.invoked_by,
            mode=req.mode,
            max_steps=req.max_steps,
            active_count=active_count,
            subpipeline_token=req.subpipeline_token,
        )
        if not policy.ok:
            status_code = 423 if "kill switch" in policy.reason.lower() else 422
            raise HTTPException(status_code=status_code, detail=policy.reason)

        pipeline_id = str(uuid.uuid4())
        steps_spec = [s.model_dump() for s in req.steps] if req.steps else None
        steps = await _build_plan_or_reject(pipeline_id, req.objective, req.max_steps, steps_spec)

        # Asignar pipeline_id a cada step
        for step in steps:
            step.pipeline_id = pipeline_id

        # Pre-vuelo (spec 2026-09-17 §4.7): DESPUÉS de build() y ANTES de
        # crear filas, dentro del candado. dry_run también ("¿cuánto costaría?").
        veredicto = await _prevuelo_o_503(
            steps, {"objective": req.objective}, user_id=req.user_id, tenant_id=req.tenant_id,
        )
        if not veredicto.ok:
            cuerpo = {"code": "prevuelo_rechazado", **veredicto.to_dict()}
            await store.event_append(pipeline_id, "PREVUELO_RECHAZADO", cuerpo)
            raise HTTPException(status_code=422, detail=cuerpo)
        if (req.costo_max_aceptado_usd is not None
                and veredicto.costo_max_usd > req.costo_max_aceptado_usd):
            raise HTTPException(status_code=409, detail={
                "code": "costo_supera_lo_aceptado",
                "costo_max_aceptado_usd": str(req.costo_max_aceptado_usd),
                **veredicto.to_dict(),
            })
        costo = _resumen_de_costo(veredicto)

        now = time.time()
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=req.name,
            invoked_by=req.invoked_by,
            user_id=req.user_id,
            tenant_id=req.tenant_id,
            mode=req.mode,
            plan=steps,
            max_steps=req.max_steps,
            context={"objective": req.objective},
            created_at=now,
            updated_at=now,
        )

        await store.pipeline_create(pipeline)
        for step in steps:
            await store.step_upsert(step)
        await store.event_append(pipeline_id, "PIPELINE_CREATED", {
            "name": req.name, "mode": req.mode, "steps": len(steps), **costo,
        })

    # dry_run: no ejecuta en background, solo completa inmediatamente
    if req.mode == "dry_run":
        await store.pipeline_update_status(pipeline_id, PipelineStatus.completed)
        await store.event_append(pipeline_id, "DRY_RUN_COMPLETE")
        return {
            "pipeline_id": pipeline_id,
            "status": "completed",
            "mode": "dry_run",
            "plan": [s.model_dump() for s in steps],
            "note": "dry_run — ningún step fue ejecutado",
            **costo,
        }

    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "running",
        "mode": req.mode,
        "step_count": len(steps),
        "message": "Pipeline iniciado. Consultar GET /jacobs/pipeline/{id}",
        **costo,
    }


# ----------------------------------------------------------------
#  GET /jacobs/pipeline/{id}
# ----------------------------------------------------------------

@router.get("/pipeline/{pipeline_id}")
async def get_pipeline(pipeline_id: str) -> dict:
    pipeline = await store.pipeline_get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' no encontrado")

    steps = await store.steps_by_pipeline(pipeline_id)
    return {
        "pipeline": pipeline.model_dump(exclude={"plan"}),
        "steps": [s.model_dump() for s in steps],
    }


# ----------------------------------------------------------------
#  POST /jacobs/pipeline/{id}/cancel
# ----------------------------------------------------------------

@router.post("/pipeline/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str) -> dict:
    pipeline = await store.pipeline_get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' no encontrado")
    if pipeline.status in (PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.aborted):
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline ya finalizado con status '{pipeline.status.value}'",
        )
    await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)
    await store.event_append(pipeline_id, "PIPELINE_CANCELLED", {"by": "API request"})
    return {"pipeline_id": pipeline_id, "status": "aborted"}


# ----------------------------------------------------------------
#  POST /jacobs/pipeline/{id}/resume
# ----------------------------------------------------------------

class ResumeRequest(BaseModel):
    invoked_by: str


@router.post("/pipeline/{pipeline_id}/resume")
async def resume_pipeline(
    pipeline_id: str, req: ResumeRequest, background: BackgroundTasks
) -> dict:
    """Reanuda un pipeline interrumpido. Solo el rol 'plataforma' puede hacerlo."""
    policy = validate_resume(req.invoked_by)
    if not policy.ok:
        raise HTTPException(status_code=403, detail=policy.reason)

    pipeline = await store.pipeline_get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' no encontrado")
    if pipeline.status != PipelineStatus.interrupted:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline no está en status 'interrupted' (actual: '{pipeline.status.value}')",
        )
    if check_kill_switch():
        raise HTTPException(status_code=423, detail="Kill switch activo — no se puede reanudar")

    # Época (spec 2026-09-17 §5.3): se toma ANTES de tocar pasos. Si otro
    # resume ganó, este no escribe ni lanza nada: dos ejecutores del mismo
    # pipeline es exactamente lo que la época existe para impedir.
    nueva_epoca = await store.pipeline_tomar_epoca(
        pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,),
    )
    if nueva_epoca is None:
        raise HTTPException(
            status_code=409,
            detail="Otro pedido ya reanudó o cambió este pipeline mientras se procesaba este",
        )
    await store.event_append(
        pipeline_id, "PIPELINE_RESUMED", {"by": req.invoked_by, "run_epoch": nueva_epoca},
    )

    # Desbloquear TODOS los steps en estado blocked (una ola supervised pudo
    # dejar varios). El executor recalcula las olas desde los refs en context,
    # así que basta con poner los blocked en pending para que entren a su ola.
    steps = await store.steps_by_pipeline(pipeline_id)
    for s in steps:
        if s.status == StepStatus.blocked:
            s.status = StepStatus.pending
            await store.step_upsert(s)

    pipeline.plan = steps
    pipeline.run_epoch = nueva_epoca
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "resuming",
        "from_index": pipeline.current_step_index,
        "run_epoch": nueva_epoca,
    }


# ----------------------------------------------------------------
#  POST /jacobs/pipeline/{id}/approve-step
# ----------------------------------------------------------------

class ApproveStepRequest(BaseModel):
    invoked_by: str


@router.post("/pipeline/{pipeline_id}/approve-step")
async def approve_step(
    pipeline_id: str, req: ApproveStepRequest, background: BackgroundTasks
) -> dict:
    """
    Aprueba el step bloqueado en hyde y lo ejecuta.
    Válido si: pipeline.mode == "supervised" O step.facet == "hyde".
    Requiere que el step esté en status blocked_human_gate.
    Solo el rol 'plataforma' puede aprobar.
    """
    policy = validate_resume(req.invoked_by)
    if not policy.ok:
        raise HTTPException(status_code=403, detail=policy.reason)

    pipeline = await store.pipeline_get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' no encontrado")
    if pipeline.status != PipelineStatus.interrupted:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline no está interrumpido (actual: '{pipeline.status.value}')",
        )
    if check_kill_switch():
        raise HTTPException(status_code=423, detail="Kill switch activo — no se puede aprobar")

    steps = await store.steps_by_pipeline(pipeline_id)

    # Buscar TODOS los steps en human gate (una ola puede tener varios hyde).
    gated = [s for s in steps if s.status == StepStatus.blocked_human_gate]
    if not gated:
        raise HTTPException(
            status_code=409, detail="No hay steps pendientes de aprobación (human gate)"
        )

    # En modo no-supervised, approve-step solo aplica a hyde. En supervised,
    # aplica a cualquier step de la ola pausada.
    if pipeline.mode != "supervised":
        non_hyde = [s for s in gated if s.facet != "hyde"]
        if non_hyde:
            raise HTTPException(
                status_code=422,
                detail="approve-step solo válido para hyde o pipelines en modo supervised",
            )

    for current_step in gated:
        if current_step.facet == "hyde":
            pipeline.context[f"hyde_approved_{current_step.step_id}"] = True

    # Época (desvío 9 del plan 2026-09-17): approve-step lanza run_pipeline
    # igual que resume. Las marcas hyde_approved_* viajan en el MISMO UPDATE
    # que toma la época (antes iban en un pipeline_update_status aparte).
    nueva_epoca = await store.pipeline_tomar_epoca(
        pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,), pipeline.context,
    )
    if nueva_epoca is None:
        raise HTTPException(
            status_code=409,
            detail="Otro pedido ya aprobó o cambió este pipeline mientras se procesaba este",
        )

    approved_indices = []
    for current_step in gated:
        await store.event_append(
            pipeline_id, "STEP_APPROVED",
            {"step_index": current_step.step_index, "facet": current_step.facet,
             "by": req.invoked_by, "run_epoch": nueva_epoca},
            current_step.step_id,
        )
        current_step.status = StepStatus.pending
        current_step.error  = None
        await store.step_upsert(current_step)
        approved_indices.append(current_step.step_index)

    pipeline.plan = steps
    pipeline.run_epoch = nueva_epoca
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id":     pipeline_id,
        "status":          "resuming",
        "approved_steps":  approved_indices,
        "run_epoch":       nueva_epoca,
    }


# ----------------------------------------------------------------
#  GET /jacobs/pipeline/{id}/results
# ----------------------------------------------------------------

def _resolve_ref(ref: str) -> tuple[str, list]:
    """Extrae texto y fuentes de un output_ref. Retorna (result, sources, error).

    `error` es None cuando todo fue bien —— incluido el caso legítimo de «no hay
    ref». Cuando NO es None, el resultado vacío es un fallo de lectura y NO
    debe presentarse como el contenido del paso.
    """
    if not ref:
        return "", [], None
    try:
        if ref.startswith("inline:"):
            data = json.loads(ref[7:])
        elif ref.startswith("artifact://"):
            data = read_artifact(ref)
        else:
            return ref, [], None
        result = str(data.get("result") or data.get("text") or json.dumps(data))
        sources = data.get("sources") or []
        return result, sources, None
    except Exception as exc:  # noqa: BLE001  # fail-closed: se devuelve el motivo, nunca un vacío que parezca un resultado
        # ARREGLADO 2026-09-16: antes `return "", []` sin una sola línea de log.
        # El endpoint construía el step con status='completed' y result='', y el
        # usuario veía un paso completado con resultado vacío —— indistinguible
        # de uno que legítimamente no produjo texto. Además se perdían las
        # `sources` (el grounding del Principio VIII) sin ninguna señal.
        logger.error("No se pudo resolver output_ref %r: %s", ref, exc, exc_info=True)
        return "", [], f"el resultado de este paso no se pudo leer: {exc}"


@router.get("/pipeline/{pipeline_id}/results")
async def get_pipeline_results(pipeline_id: str) -> dict:
    """Devuelve el pipeline con el contenido completo del resultado de cada step."""
    pipeline = await store.pipeline_get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' no encontrado")

    steps = await store.steps_by_pipeline(pipeline_id)

    steps_data = []
    for step in steps:
        result_text, sources, ref_error = _resolve_ref(step.output_ref or "")
        duration = None
        if step.started_at and step.finished_at:
            duration = round(step.finished_at - step.started_at, 2)
        steps_data.append({
            "step_index":        step.step_index,
            "facet":             step.facet,
            "capability":        step.capability,
            "name":              step.input.get("name", f"{step.facet} — {step.capability}"),
            "status":            step.status.value,
            "result":            result_text,
            "sources":           sources,
            "duration_seconds":  duration,
            "error":             step.error or ref_error,
            "result_unavailable": ref_error is not None,
        })

    total_duration = None
    if pipeline.created_at and pipeline.updated_at:
        total_duration = round(pipeline.updated_at - pipeline.created_at, 2)

    return {
        "pipeline_id":           pipeline_id,
        "name":                  pipeline.name,
        "status":                pipeline.status.value,
        "steps":                 steps_data,
        "total_duration_seconds": total_duration,
    }


# ----------------------------------------------------------------
#  GET /jacobs/pipeline/{id}/events
# ----------------------------------------------------------------

@router.get("/pipeline/{pipeline_id}/events")
async def get_events(pipeline_id: str) -> dict:
    events = await store.events_by_pipeline(pipeline_id)
    return {"pipeline_id": pipeline_id, "events": events}
