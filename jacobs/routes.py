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
from decimal import Decimal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, model_validator

from redaccion import recortar_redactado

from jacobs import store
from jacobs import continuar as servicio_continuar
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
    validar_costo_max_aceptado,
)
from jacobs.plan import PlanBuilder, PlanRejected
from jacobs.prevuelo import prevuelo
from jacobs.prevuelo_reglas import Veredicto, formatear_usd
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
# m6 (re-revisión final, 2026-09-17): hoy la sección crítica es más grande que
# lo que dice el párrafo de arriba -- cubre build() + el pre-vuelo con sus
# sondas + la transacción, y continuar.py toma el MISMO objeto. El inventario
# completo y por qué se deja así está en jacobs/candado.py.
from jacobs.candado import candado_de_creacion as _pipeline_create_lock  # noqa: E402  (2026-09-17: compartido con continuar.py)


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


def _motivo_redactado(exc: Exception) -> str:
    """Redacción + recorte compartidos por todo sitio que convierte un error
    inesperado en 503 prevuelo_no_disponible (create, preflight, continue) --
    un solo lugar, para no desincronizar el largo o el filtro de secretos
    entre ellos (contexto Task 11)."""
    return recortar_redactado(f"{type(exc).__name__}: {exc}", 300)


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
        motivo = _motivo_redactado(exc)
        logger.error("pre-vuelo no disponible: %s", motivo, exc_info=True)
        raise HTTPException(
            status_code=503, detail={"code": "prevuelo_no_disponible", "motivo": motivo},
        ) from exc


def _resumen_de_costo(veredicto: Veredicto) -> dict:
    # Ruling R18: mismo grano de 6 decimales que Veredicto.to_dict() -- este
    # resumen alimenta PIPELINE_CREATED y las respuestas 200/dry_run, no pasa
    # por to_dict() directo.
    return {
        "costo_max_usd": formatear_usd(veredicto.costo_max_usd),
        "pasos_costo": [c.to_dict() for c in veredicto.pasos_costo],
    }


async def _prevuelo_de_reanudacion(
    pipeline: Pipeline, pasos: list[Step], *, aprobados_ahora: frozenset[str] = frozenset(),
) -> tuple[dict, dict, list[Step]]:
    """Ola final F2 (Ruling R32, criterio de Fernando "nada gasta sin
    pre-vuelo"): resume y approve-step lanzan run_pipeline igual que continue,
    así que corren el pre-vuelo ANTES de tomar la época, sobre los pasos SIN
    ref legible (la misma regla 5 de continue, servicio_continuar.
    separar_por_ref) -- en approve-step eso es la ola COMPLETA que se va a
    lanzar, no sólo el paso aprobado; hyde no cobra (regla existente). El
    contexto que recibe el pre-vuelo no lleva refs de pasos a rehacer.

    - ok=False -> 422 {"code": "prevuelo_rechazado", **veredicto} y evento
      PREVUELO_RECHAZADO con el mismo cuerpo; ni época ni pasos se tocan.
    - no puede correr -> 503 prevuelo_no_disponible, motivo redactado.
    - SIN costo_max_aceptado_usd: el consentimiento se dio al crear/continuar.

    Devuelve (resumen de costo para la respuesta 200, contexto sin las refs
    de los pasos a rehacer, pasos cuya ref era ILEGIBLE). Pasada final R34:
    resume y approve-step persisten ESE contexto al tomar la época y vuelven
    a poner esos pasos en pending, como continue -- si no, run_pipeline los
    daba por hechos (ref presente) y sus dependientes fallaban al leerla."""
    try:
        _, a_correr, contexto = await servicio_continuar.separar_por_ref(pasos, pipeline.context)
    except Exception as exc:  # fail-closed: sin saber qué pasos se corren no hay pre-vuelo, y sin pre-vuelo no se lanza (spec §8)
        raise _no_disponible(exc) from exc
    veredicto = await _prevuelo_o_503(
        pasos, contexto, pendientes=set(a_correr),
        user_id=pipeline.user_id, tenant_id=pipeline.tenant_id,
    )
    if not veredicto.ok:
        cuerpo = {"code": "prevuelo_rechazado", **veredicto.to_dict()}
        await store.event_append(pipeline.pipeline_id, "PREVUELO_RECHAZADO", cuerpo)
        raise HTTPException(status_code=422, detail=cuerpo)
    a_rehacer = set(a_correr)
    ilegibles = [p for p in pasos
                 if p.step_index in a_rehacer and pipeline.context.get(f"step_{p.step_index}_ref")]
    # Ruling R37 (desvío 10, como continue): un paso que se rehace pierde su
    # hyde_approved_*: la aprobación humana de una corrida no autoriza la
    # siguiente. Salvo la que approve-step está dando en ESTE pedido.
    for p in ilegibles:
        if p.step_id not in aprobados_ahora:
            contexto.pop(f"hyde_approved_{p.step_id}", None)
    return _resumen_de_costo(veredicto), contexto, ilegibles


def _rehacer(paso: Step) -> None:
    """Un paso cuya ref no se lee vuelve a correr (mismo reset que continue)."""
    paso.status = StepStatus.pending
    paso.error = None
    paso.started_at = None
    paso.finished_at = None
    paso.output_ref = None


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
    devolvió el LLM.

    Ruling R17 (fix round 1, Task 9): a diferencia de los rechazos PREEXISTENTES
    de texto de POST /jacobs/pipeline (que la Mesa ya consume como
    jacobs_rechazo y no se tocan en esta rama), todo rechazo de este endpoint
    NUEVO va como dict {code, detalle}."""
    if req.invoked_by not in VALID_INVOKERS:
        raise HTTPException(status_code=403, detail={
            "code": "invocador_no_autorizado",
            "detalle": f"invoked_by '{req.invoked_by}' no autorizado",
        })
    # I1 / Ruling R54 (2026-09-17): el pre-vuelo GASTA -- la sonda paga una
    # llamada por cada clave sin evento `ok` fresco y la registra en
    # axioma_usage (spec §4.5) -- así que el kill switch lo frena como a todo
    # camino que gasta. Era el único que no lo miraba: el desvío 6 del plan lo
    # justificaba con "no ejecuta nada", cierto antes de que entrara la sonda.
    # Va DESPUÉS de validar invoked_by (un invocador no autorizado no se entera
    # de si Jacobs está frenado) y ANTES de build().
    if check_kill_switch():
        raise HTTPException(status_code=423, detail={
            "code": "kill_switch",
            "detalle": "Kill switch activo — Jacobs detenido",
        })
    if len(req.steps) > 20:
        raise HTTPException(status_code=422, detail={
            "code": "plan_rechazado",
            "detalle": f"{len(req.steps)} pasos excede el límite duro (20)",
        })
    steps_spec = [s.model_dump() for s in req.steps]
    try:
        steps = await _plan_builder.build(
            pipeline_id=str(uuid.uuid4()), objective=req.objective,
            max_steps=len(steps_spec), steps_spec=steps_spec,
        )
    except PlanRejected as exc:
        raise HTTPException(status_code=422, detail={
            "code": "plan_rechazado",
            "detalle": str(exc),
        }) from exc
    except Exception as exc:  # fail-closed: build() con pasos explícitos sólo lee la gobernanza de la base; si no puede, no hay pre-vuelo (spec §8: 503, nunca 500 genérico ni veredicto)
        # Ruling R38 (2026-09-17): el OperationalError 2013 de
        # get_motor_governance() bajo carga salía de acá sin atrapar y el
        # cliente recibía un 500.
        motivo = _motivo_redactado(exc)
        logger.error("pre-vuelo no disponible (gobernanza del plan): %s", motivo, exc_info=True)
        raise HTTPException(
            status_code=503, detail={"code": "prevuelo_no_disponible", "motivo": motivo},
        ) from exc
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
    """Crea un pipeline y lo ejecuta en background.

    Escritura (R38 fix round 1, 2b): pipeline, pasos y PIPELINE_CREATED (y en
    dry_run el completed y DRY_RUN_COMPLETE) van en UNA transacción bajo el
    candado de activos. Si algo falla ANTES del COMMIT, nada queda escrito y
    responde 503 {"code": "prevuelo_no_disponible", "motivo"}.

    COMMIT cortado (Ruling R41): si la conexión se corta o vence DURANTE el
    COMMIT, el resultado es incierto -- el servidor pudo haberlo confirmado.
    Responde el mismo 503 con un `detalle` que lo dice: el pipeline puede
    existir, y hay que revisar la lista antes de reintentar (un reintento
    ciego podría crearlo dos veces).

    Spec 2026-09-17 §4.7: en el camino por objetivo (req.steps=None, el plan
    lo arma _build_plan_or_reject() llamando a un LLM pago), el pre-vuelo
    corre DESPUÉS, sobre el plan que el LLM ya devolvió -- lo gastado en
    planificar ya está gastado, igual que en /jacobs/preflight."""

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
            # Ruling R19 (fix round 1, Task 9): pipeline_id es el MISMO que se
            # usa para el evento -- campo extra, la whitelist de la Mesa lo
            # ignora si no lo consume.
            cuerpo = {"code": "prevuelo_rechazado", "pipeline_id": pipeline_id, **veredicto.to_dict()}
            await store.event_append(pipeline_id, "PREVUELO_RECHAZADO", cuerpo)
            raise HTTPException(status_code=422, detail=cuerpo)
        if (req.costo_max_aceptado_usd is not None
                and veredicto.costo_max_usd > req.costo_max_aceptado_usd):
            raise HTTPException(status_code=409, detail={
                "code": "costo_supera_lo_aceptado",
                # Ruling R18: mismo grano de 6 decimales que el resto de los
                # montos de Jacobs -- el valor puede llegar sin cuantizar.
                "costo_max_aceptado_usd": formatear_usd(req.costo_max_aceptado_usd),
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

        # F3 (ola final, Ruling R31): el conteo de arriba sólo evita planificar
        # y sondear en vano; el cupo se decide recontando DENTRO del candado
        # con nombre de MariaDB y escribiendo en el mismo bloque -- el
        # asyncio.Lock no cruza al CLI de continuar, que corre en otro proceso.
        #
        # R38, fix round 1 (2b): pipeline, pasos, PIPELINE_CREATED y, en
        # dry_run, el completed y DRY_RUN_COMPLETE van en UNA transacción
        # sobre la conexión del candado. Antes iban por conexiones separadas
        # (del pool, esperando turno con el GET_LOCK tomado): un vencimiento a
        # mitad dejaba un pipeline sin sus pasos y un 500. Ahora es todo o
        # nada, acotado por JAX_DB_CONNECT_TIMEOUT_SECONDS; cualquier falla es
        # 503 prevuelo_no_disponible sin nada escrito.
        estado_tx = store.EstadoDeTransaccion()
        try:
            async with store.candado_de_activos() as conexion_del_candado:
                active_count = await store.pipeline_count_active(conexion=conexion_del_candado)
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

                async with asyncio.timeout(store.db_connect_timeout_seconds()):
                    async with store.transaccion(conexion_del_candado, estado_tx) as tx:
                        await store.pipeline_create(pipeline, conexion=tx)
                        for step in steps:
                            await store.step_upsert(step, conexion=tx)
                        await store.event_append(pipeline_id, "PIPELINE_CREATED", {
                            "name": req.name, "mode": req.mode, "steps": len(steps), **costo,
                        }, conexion=tx)
                        if req.mode == "dry_run":
                            await store.pipeline_update_status(
                                pipeline_id, PipelineStatus.completed, conexion=tx)
                            await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", conexion=tx)
        except HTTPException:
            raise
        except Exception as exc:  # fail-closed: candado, recuento o transacción de creación que no se pudo completar -> nada quedó escrito (la transacción se descarta) y no se crea (spec §8)
            motivo = _motivo_redactado(exc)
            detalle = {"code": "prevuelo_no_disponible", "motivo": motivo}
            if estado_tx.incierta:
                # R41: el COMMIT salió y la respuesta no llegó.
                logger.error("crear: COMMIT del pipeline %s con resultado INCIERTO: %s", pipeline_id, motivo)
                detalle["detalle"] = (
                    f"Resultado incierto: la conexión se cortó mientras se confirmaba. El pipeline "
                    f"{pipeline_id} puede existir: revisá la lista de pipelines antes de reintentar."
                )
            else:
                logger.error("crear: no se pudo escribir el pipeline: %s", motivo)
            raise HTTPException(status_code=503, detail=detalle) from exc

    # dry_run: no ejecuta en background; ya quedó completed en la transacción
    if req.mode == "dry_run":
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
    # F6 (ola final): compare-and-set con la época y el status LEÍDOS. Sin
    # condición, un /continue o /resume que tomó la época entre la lectura y
    # esta escritura quedaba pisado (el pipeline recién relanzado pasaba a
    # aborted). Si cambió, 409 de texto como el resto de los 409 de cancel.
    if not await store.pipeline_update_status_si_epoca(
        pipeline_id, pipeline.run_epoch, PipelineStatus.aborted, desde=(pipeline.status,),
    ):
        logger.info("cancel de %s no escrito: cambió después de leerlo (época %s, status %s)",
                    pipeline_id, pipeline.run_epoch, pipeline.status.value)
        raise HTTPException(
            status_code=409,
            detail="El pipeline cambió mientras se cancelaba (otro pedido lo reanudó, continuó o "
                   "terminó): volvé a consultarlo",
        )
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

    steps = await store.steps_by_pipeline(pipeline_id)
    # F2 (Ruling R32): pre-vuelo antes de tomar la época -- un rechazo no deja
    # rastro de estado (sólo el evento PREVUELO_RECHAZADO).
    costo, contexto, ilegibles = await _prevuelo_de_reanudacion(pipeline, steps)

    # Época (spec 2026-09-17 §5.3): se toma ANTES de tocar pasos. Si otro
    # resume ganó, este no escribe ni lanza nada: dos ejecutores del mismo
    # pipeline es exactamente lo que la época existe para impedir.
    # Pasada final R34: si había refs ilegibles, el contexto sin ellas viaja
    # en el MISMO UPDATE que toma la época.
    if ilegibles:
        nueva_epoca = await store.pipeline_tomar_epoca(
            pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,), contexto,
        )
    else:
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
    for s in steps:
        if s.status == StepStatus.blocked:
            s.status = StepStatus.pending
            await store.step_upsert(s)
    for s in ilegibles:
        _rehacer(s)
        await store.step_upsert(s)

    pipeline.plan = steps
    pipeline.context = contexto
    pipeline.run_epoch = nueva_epoca
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "resuming",
        "from_index": pipeline.current_step_index,
        "run_epoch": nueva_epoca,
        **costo,
    }


# ----------------------------------------------------------------
#  POST /jacobs/pipeline/{id}/continue  (spec 2026-09-17 §5)
# ----------------------------------------------------------------

class ContinueRequest(BaseModel):
    invoked_by:             str
    reasignar:              dict[str, str] | None = None
    user_id:                str | None = None
    tenant_id:              str | None = None
    costo_max_aceptado_usd: Decimal | None = None

    @model_validator(mode="after")
    def _validar_costo(self) -> "ContinueRequest":
        # Fix round 1 (revisión de Task 11): mismo guardia que
        # PipelineCreateRequest -- reusa jacobs.models.validar_costo_max_aceptado
        # en vez de repetir el umbral y el mensaje acá.
        validar_costo_max_aceptado(self.costo_max_aceptado_usd)
        return self


class ContinuePreflightRequest(BaseModel):
    invoked_by: str
    reasignar:  dict[str, str] | None = None
    user_id:    str | None = None
    tenant_id:  str | None = None


def _no_disponible(exc: Exception) -> HTTPException:
    motivo = _motivo_redactado(exc)
    logger.error("continuar no disponible: %s", motivo, exc_info=True)
    return HTTPException(status_code=503, detail={"code": "prevuelo_no_disponible", "motivo": motivo})


@router.post("/pipeline/{pipeline_id}/continue")
async def continue_pipeline(
    pipeline_id: str, req: ContinueRequest, background: BackgroundTasks
) -> dict:
    """Continúa un pipeline aborted o expired (spec §5.1): reusa los pasos con
    ref legible, rehace el resto (con reasignación opcional) y corre el
    pre-vuelo sobre los pendientes. 403/404/409/422/423/429 con
    {"code", ...}; 503 si el análisis o el pre-vuelo no pueden correr."""
    try:
        respuesta, pipeline = await servicio_continuar.continuar(
            pipeline_id, req.invoked_by, reasignar=req.reasignar, user_id=req.user_id,
            tenant_id=req.tenant_id, costo_max_aceptado_usd=req.costo_max_aceptado_usd,
        )
    except servicio_continuar.ContinuarRechazado as rechazo:
        raise HTTPException(status_code=rechazo.status_code, detail=rechazo.cuerpo()) from rechazo
    except Exception as exc:  # fail-closed: sin pre-vuelo/análisis no se continúa (spec §8, desvío 20)
        raise _no_disponible(exc) from exc
    background.add_task(run_pipeline, pipeline)
    return respuesta


@router.post("/pipeline/{pipeline_id}/continue/preflight")
async def continue_preflight(pipeline_id: str, req: ContinuePreflightRequest) -> dict:
    """Reglas 1-8 de continuar SIN escribir (spec §5.1): lo que la Mesa muestra
    antes de confirmar."""
    try:
        return await servicio_continuar.previsualizar(
            pipeline_id, req.invoked_by, reasignar=req.reasignar,
            user_id=req.user_id, tenant_id=req.tenant_id,
        )
    except servicio_continuar.ContinuarRechazado as rechazo:
        raise HTTPException(status_code=rechazo.status_code, detail=rechazo.cuerpo()) from rechazo
    except Exception as exc:  # fail-closed: sin pre-vuelo no hay vista previa confiable (spec §8, desvío 20)
        raise _no_disponible(exc) from exc


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

    # F2 (Ruling R32): pre-vuelo de la ola completa que se va a lanzar (todos
    # los pasos sin ref legible), antes de tomar la época y de persistir las
    # marcas de hyde.
    costo, contexto, ilegibles = await _prevuelo_de_reanudacion(
        pipeline, steps, aprobados_ahora=frozenset(s.step_id for s in gated if s.facet == "hyde"),
    )
    # Pasada final R34: el contexto sin las refs ilegibles (con las marcas
    # hyde_approved_* intactas) es el que viaja con la época.
    pipeline.context = contexto

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
    for s in ilegibles:
        _rehacer(s)
        await store.step_upsert(s)

    pipeline.plan = steps
    pipeline.run_epoch = nueva_epoca
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id":     pipeline_id,
        "status":          "resuming",
        "approved_steps":  approved_indices,
        "run_epoch":       nueva_epoca,
        **costo,
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
