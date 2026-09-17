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
from pydantic import BaseModel

from jacobs import store
from jacobs.artifacts import read_artifact
from jacobs.executor import run_pipeline
from jacobs.models import (
    INVOKER_ADA,
    MAX_STEPS_PER_PIPELINE,
    VALID_INVOKERS,
    Pipeline,
    PipelineCreateRequest,
    PipelineStatus,
    Step,
    StepSpec,
    StepStatus,
)
from jacobs.plan import PlanBuilder, PlanRejected
from jacobs.policy import (
    check_kill_switch,
    validate_create,
    validate_resume,
)
from jacobs.subpipelines import ConsumoRechazado, Motivo, consumir_token_subpipeline

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


# ----------------------------------------------------------------
#  POST /jacobs/plan  — genera plan sin ejecutar
# ----------------------------------------------------------------

class PlanRequest(BaseModel):
    name:       str
    objective:  str
    invoked_by: str
    mode:       str
    max_steps:  int = MAX_STEPS_PER_PIPELINE
    steps:      list[StepSpec] | None = None


@router.post("/plan")
async def plan_only(req: PlanRequest) -> dict:
    """Genera un plan de steps sin ejecutar nada (dry_run de planificación)."""

    if req.max_steps > MAX_STEPS_PER_PIPELINE:
        raise HTTPException(
            status_code=422,
            detail=f"max_steps={req.max_steps} excede límite duro ({MAX_STEPS_PER_PIPELINE})",
        )
    if req.invoked_by not in VALID_INVOKERS:
        raise HTTPException(
            status_code=403,
            detail=f"invoked_by '{req.invoked_by}' no autorizado",
        )
    # Frente F: Ada no planifica por acá. Un sub-pipeline entra solo por
    # POST /pipeline con un subpipeline_token; /plan no consume tokens y
    # planificar llama a un LLM.
    if req.invoked_by == INVOKER_ADA:
        raise HTTPException(
            status_code=403,
            detail="ada no planifica por /plan: un sub-pipeline se crea por /pipeline con subpipeline_token",
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
            parent_pipeline_id=req.parent_pipeline_id,
        )
        if not policy.ok:
            status_code = 423 if "kill switch" in policy.reason.lower() else 422
            raise HTTPException(status_code=status_code, detail=policy.reason)

        pipeline_id = str(uuid.uuid4())

        # Frente F (2026-09-16): un hijo de Ada consume su token ACÁ, después de
        # validate_create (un 423 del kill switch no lo quema) y ANTES de
        # planificar (no se sostiene una transacción los 20-40 s del LLM). Padre
        # y profundidad salen de la fila del token, nunca del cuerpo.
        parent_pipeline_id: str | None = None
        parent_step: str | None = None
        depth = 0
        if req.invoked_by == INVOKER_ADA:
            consumo = await consumir_token_subpipeline(
                req.subpipeline_token, req.parent_pipeline_id, pipeline_id,
            )
            if isinstance(consumo, ConsumoRechazado):
                raise HTTPException(
                    status_code=403,
                    detail=f"subpipeline_token rechazado: {consumo.motivo.value}",
                )
            parent_pipeline_id = consumo.parent_pipeline_id
            parent_step = consumo.parent_step
            depth = consumo.depth

        steps_spec = [s.model_dump() for s in req.steps] if req.steps else None
        try:
            steps = await _build_plan_or_reject(pipeline_id, req.objective, req.max_steps, steps_spec)
        except HTTPException:
            if parent_pipeline_id is not None:
                await store.event_append(pipeline_id, "SUBPIPELINE_RECHAZADO", {
                    "fase": "plan",
                    "motivo": Motivo.PLAN_RECHAZADO.value,
                    "parent_pipeline_id": parent_pipeline_id,
                })
            raise

        # Asignar pipeline_id a cada step
        for step in steps:
            step.pipeline_id = pipeline_id

        now = time.time()
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=req.name,
            invoked_by=req.invoked_by,
            user_id=req.user_id,
            tenant_id=req.tenant_id,
            parent_pipeline_id=parent_pipeline_id,
            depth=depth,
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
            "name": req.name, "mode": req.mode, "steps": len(steps),
            "parent_pipeline_id": parent_pipeline_id, "depth": depth,
        })
        if parent_pipeline_id is not None:
            await store.event_append(
                parent_pipeline_id, "SUBPIPELINE_CREADO",
                {"hijo_pipeline_id": pipeline_id, "depth": depth},
                parent_step,
            )

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
        }

    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "running",
        "mode": req.mode,
        "step_count": len(steps),
        "message": "Pipeline iniciado. Consultar GET /jacobs/pipeline/{id}",
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

    await store.event_append(pipeline_id, "PIPELINE_RESUMED", {"by": req.invoked_by})

    # Desbloquear TODOS los steps en estado blocked (una ola supervised pudo
    # dejar varios). El executor recalcula las olas desde los refs en context,
    # así que basta con poner los blocked en pending para que entren a su ola.
    steps = await store.steps_by_pipeline(pipeline_id)
    for s in steps:
        if s.status == StepStatus.blocked:
            s.status = StepStatus.pending
            await store.step_upsert(s)

    pipeline.plan = steps
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "resuming",
        "from_index": pipeline.current_step_index,
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

    approved_indices = []
    for current_step in gated:
        await store.event_append(
            pipeline_id, "STEP_APPROVED",
            {"step_index": current_step.step_index, "facet": current_step.facet,
             "by": req.invoked_by},
            current_step.step_id,
        )
        current_step.status = StepStatus.pending
        current_step.error  = None
        await store.step_upsert(current_step)
        if current_step.facet == "hyde":
            pipeline.context[f"hyde_approved_{current_step.step_id}"] = True
        approved_indices.append(current_step.step_index)

    # Persistir las marcas hyde_approved_* en context antes de reanudar.
    await store.pipeline_update_status(
        pipeline_id, pipeline.status, pipeline.current_step_index, pipeline.context
    )

    pipeline.plan = steps
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id":     pipeline_id,
        "status":          "resuming",
        "approved_steps":  approved_indices,
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
