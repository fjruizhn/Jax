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

from jacobs import cupo, store
from jacobs import continuar as servicio_continuar
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
    validar_costo_max_aceptado,
)
from jacobs.plan import PlanBuilder, PlanRejected
from jacobs.prevuelo import prevuelo
from jacobs.prevuelo_reglas import Veredicto, formatear_usd
from jacobs.policy import (
    MAX_PARALLEL_PIPELINES,
    ContencionAlReservar,
    CupoAgotado,
    check_kill_switch,
    validate_create,
    validate_resume,
)
from jacobs.subpipelines import (
    ConsumoRechazado, Motivo, consumir_token_subpipeline, hash_token, token_ref,
)

router = APIRouter(prefix="/jacobs", tags=["jacobs"])

# Añadido 2026-09-16: `_resolve_ref` tragaba el fallo de lectura sin dejar
# rastro. Un error que no se registra en ningún lado no existe para nadie.
logger = logging.getLogger(__name__)

_plan_builder = PlanBuilder()

# HISTORIA DE LOS CANDADOS QUE YA NO ESTÁN (2026-09-17, decisión de Fernando y
# del coordinador). Acá vivieron, uno después del otro:
#   - `_pipeline_create_lock = asyncio.Lock()` (T2, 2026-08-19): candado GLOBAL
#     del proceso alrededor del recuento, el consumo del token de Ada, la
#     planificación (20-40 s de LLM) y el INSERT, porque leer el conteo en una
#     consulta y decidir en otra es una lectura optimista;
#   - después `jacobs/candado.py::candado_de_creacion` (el mismo objeto, mudado
#     para compartirlo con continuar.py) más `store.candado_de_activos()`, un
#     GET_LOCK del servidor MariaDB, porque el candado de proceso no cruzaba al
#     CLI (jax#209, Ruling R31).
# Los dos hacían cumplir el límite y los dos serializaban: la medición del
# frente G dio ~43 delegaciones/s, y la creación de la MESA esperaba detrás de
# las delegaciones de Ada.
#
# QUEDA UN SOLO MECANISMO, y es una REGLA, no un objeto: **la condición del cupo
# viaja dentro de la escritura que lo consume** (`jacobs/cupo.py`). Un candado
# hay que acordarse de pedirlo; una condición adentro del INSERT o del UPDATE se
# aplica sola aunque quien escriba un camino nuevo no sepa que hay un límite. El
# barrido `tests/test_creacion_sin_candado_global.py` impide que vuelvan.


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

async def _auditar_token_quemado(
    hijo_pipeline_id: str,
    parent_pipeline_id: str,
    fase: str,
    motivo: Motivo,
    excepcion: str | None = None,
    ref_token: str | None = None,
) -> None:
    """Deja SUBPIPELINE_RECHAZADO para un token que ya se consumió y cuyo hijo
    no llegó a crearse. Best-effort: si el evento no se puede escribir se loguea
    (sin token, sin mensaje de la excepción) y el llamador relanza igual su
    error original -- un fallo de auditoría no puede tapar la causa (M-2)."""
    payload = {"fase": fase, "motivo": motivo.value, "parent_pipeline_id": parent_pipeline_id}
    if excepcion is not None:
        payload["excepcion"] = excepcion
    if ref_token is not None:
        payload["token_ref"] = ref_token
    try:
        await store.event_append(hijo_pipeline_id, "SUBPIPELINE_RECHAZADO", payload)
    except Exception as exc_evento:  # fail-soft: el error original lo relanza create_pipeline; tragar este evita que un fallo de auditoría reemplace la causa real
        logger.error(
            "SUBPIPELINE_RECHAZADO sin registrar: hijo=%s padre=%s fase=%s motivo=%s error_evento=%s",
            hijo_pipeline_id, parent_pipeline_id, fase, motivo.value, type(exc_evento).__name__,
        )


async def _soltar_reserva(pipeline_id: str) -> None:
    """Devuelve el cupo de una reserva que no llegó a ser pipeline.

    Best-effort A PROPÓSITO y NO es fail-open: si el DELETE falla, el cupo queda
    OCUPADO —el lado restrictivo— y el reaper cosecha la fila `pending` a los
    300 s. Lo que no puede pasar es que un fallo al liberar reemplace el error
    original que trajo al llamador hasta acá.
    """
    try:
        await cupo.soltar_reserva(pipeline_id)
    except Exception as exc:  # fail-soft: soltar es best-effort; el error original manda y el reaper cosecha la fila pending
        logger.error(
            "cupo: reserva %s no liberada (%s); queda ocupada hasta el reaper",
            pipeline_id, type(exc).__name__,
        )


def _es_deadlock(exc: BaseException) -> bool:
    """Un 1213 de InnoDB, venga envuelto o no. Se mira el código, no el texto:
    el mensaje del servidor cambia con la versión y el idioma."""
    args = getattr(exc, "args", ())
    return bool(args) and args[0] == 1213


def _contencion_503(exc: ContencionAlReservar) -> HTTPException:
    """La contención NO es un 500 (2026-09-17, revisión del autor del mecanismo
    retirado). Un 500 dice "me rompí" y manda a alguien a buscar un defecto que
    no existe; lo que pasó es que dos escrituras del cupo se trabaron y no se
    destrabaron dentro del presupuesto de espera. Tampoco es el 422 del cupo:
    422 significa "tu pedido no es válido", y este pedido está perfecto.

    503 + `Retry-After: 1` = "volvé a intentar", que es exactamente lo que hay
    que hacer. El segundo sale del presupuesto de espera (~0,96 s): reintentar
    antes es pedirle a la base que se trabe de nuevo.
    """
    return HTTPException(
        status_code=503,
        detail={
            "code": "contencion_al_reservar",
            "intentos": exc.intentos,
            "espera_s": round(exc.espera_total, 3),
        },
        headers={"Retry-After": "1"},
    )


async def _cupo_o_429() -> None:
    """Compuerta BARATA de cupo, ANTES de cualquier cosa que salga a la red.

    POR QUÉ ESTÁ ACÁ Y NO SÓLO EN LA ESCRITURA (2026-09-17, revisión del autor
    del mecanismo retirado). `resume` y `approve-step` corren el pre-vuelo, y el
    pre-vuelo **sondea facetas: es una llamada PAGA**. Si el cupo se mirara sólo
    en el UPDATE —que va después—, un pedido rechazado por falta de lugar ya
    habría gastado dinero. No es estética: es plata que se va sin que nadie la
    vea.

    NO decide: la decisión sigue siendo la condición que viaja dentro del
    UPDATE (`pipeline_tomar_epoca(cupo_maximo=...)`). Esto es una compuerta que
    ahorra el gasto en el caso claro. Si la lectura queda vieja y el cupo se
    llena entre medio, el UPDATE rechaza igual: fail-closed sin depender de
    esta lectura.

    `tests/test_cupo_en_todos_los_caminos.py` fija el ORDEN: falla si alguien
    pone el pre-vuelo antes.
    """
    activos = await cupo.activos()
    if activos >= MAX_PARALLEL_PIPELINES:
        raise HTTPException(status_code=429, detail={
            "code": "limite_de_activos",
            "detalle": str(CupoAgotado(activos, MAX_PARALLEL_PIPELINES)),
        })


@router.post("/pipeline")
async def create_pipeline(req: PipelineCreateRequest, background: BackgroundTasks) -> dict:
    """Crea un pipeline y lo ejecuta en background.

    Escritura (R38 fix round 1, 2b): los pasos, PIPELINE_CREATED y —en
    dry_run— el completed y DRY_RUN_COMPLETE van en UNA transacción, junto con
    el UPDATE que completa la RESERVA de cupo tomada al principio. Si algo falla
    ANTES del COMMIT, nada queda escrito, la reserva se suelta y responde 503
    {"code": "prevuelo_no_disponible", "motivo"}.

    COMMIT cortado (Ruling R41): si la conexión se corta o vence DURANTE el
    COMMIT, el resultado es incierto -- el servidor pudo haberlo confirmado.
    Responde el mismo 503 con un `detalle` que lo dice: el pipeline puede
    existir, y hay que revisar la lista antes de reintentar (un reintento
    ciego podría crearlo dos veces).

    Spec 2026-09-17 §4.7: en el camino por objetivo (req.steps=None, el plan
    lo arma _build_plan_or_reject() llamando a un LLM pago), el pre-vuelo
    corre DESPUÉS, sobre el plan que el LLM ya devolvió -- lo gastado en
    planificar ya está gastado, igual que en /jacobs/preflight."""


    # EL CUPO LO HACE CUMPLIR LA BASE (2026-09-17, decisión de Fernando y del
    # coordinador). `validate_create` ya no recibe `active_count`: las
    # comprobaciones baratas (rol, forma, max_steps, kill switch) van primero y
    # el cupo lo decide LA RESERVA, una sola sentencia que cuenta e inserta a la
    # vez. No hay candado de proceso ni GET_LOCK: la condición viaja dentro de
    # la escritura que consume el cupo.
    policy = validate_create(
        invoked_by=req.invoked_by,
        mode=req.mode,
        max_steps=req.max_steps,
        subpipeline_token=req.subpipeline_token,
        parent_pipeline_id=req.parent_pipeline_id,
    )
    if not policy.ok:
        status_code = 423 if "kill switch" in policy.reason.lower() else 422
        raise HTTPException(status_code=status_code, detail=policy.reason)

    pipeline_id = str(uuid.uuid4())
    now = time.time()

    # La reserva se toma ANTES del token de Ada y ANTES de planificar y sondear.
    # Ese orden conserva los invariantes que sostenían los candados, no sólo el
    # del conteo:
    #   - ningún token de sub-pipeline se quema sin que haya cupo;
    #   - no se planifica ni se sondea en vano con el cupo lleno -- eso hacía el
    #     recuento suelto de antes, pero ahora la respuesta es AUTORITATIVA en
    #     vez de una lectura optimista que había que reconfirmar después;
    #   - las planificaciones en vuelo siguen acotadas: como máximo tantas como
    #     cupo hay. Antes el candado las acotaba a UNA, y de ahí salía el cuello.
    # La fila nace SIN plan y, para un hijo de Ada, SIN identidad: user_id y
    # tenant_id del cuerpo no se escriben nunca (I-3); los trae la fila del token.
    es_de_ada = req.invoked_by == INVOKER_ADA
    pipeline = Pipeline(
        pipeline_id=pipeline_id,
        name=req.name,
        invoked_by=req.invoked_by,
        user_id=None if es_de_ada else req.user_id,
        tenant_id=None if es_de_ada else req.tenant_id,
        mode=req.mode,
        plan=[],
        max_steps=req.max_steps,
        context={"objective": req.objective},
        created_at=now,
        updated_at=now,
    )
    try:
        hay_lugar = await cupo.reservar_cupo(pipeline)
    except ContencionAlReservar as exc:
        raise _contencion_503(exc) from exc
    if not hay_lugar:
        # 0 filas afectadas = cupo agotado. Mismo 422 y mismo texto que daba
        # validate_create. El conteo es sólo para el MENSAJE: decidió la base.
        raise HTTPException(
            status_code=422,
            detail=str(cupo.CupoAgotado(await cupo.activos(), MAX_PARALLEL_PIPELINES)),
        )

    # Desde acá la reserva ocupa un lugar del cupo: todo camino de salida que no
    # deje un pipeline de verdad tiene que devolverlo.
    soltar_la_reserva = True
    try:
        # Frente F (2026-09-16): un hijo de Ada consume su token ACÁ, después de
        # validate_create (un 423 del kill switch no lo quema) y ANTES de
        # planificar (no se sostiene una transacción los 20-40 s del LLM). Padre,
        # profundidad e identidad salen de la fila del token, nunca del cuerpo.
        parent_pipeline_id: str | None = None
        parent_step: str | None = None
        depth = 0
        user_id, tenant_id = req.user_id, req.tenant_id
        if req.invoked_by == INVOKER_ADA:
            # Residual de I-2 (R13): el UPDATE del consumo se confirma solo; si
            # DESPUÉS falla la relectura (error de base, invariante roto) o llega
            # una cancelación, el token puede estar quemado sin hijo. BaseException
            # a propósito: CancelledError no es Exception. Se audita best-effort
            # (hash parcial, nunca el token) y se relanza el error ORIGINAL: la
            # creación no sigue. El padre del payload es el declarado; si el token
            # se quemó, coincide con el de la fila (el UPDATE lo exige).
            try:
                consumo = await consumir_token_subpipeline(
                    req.subpipeline_token, req.parent_pipeline_id, pipeline_id,
                    user_id=req.user_id, tenant_id=req.tenant_id,
                )
            except BaseException as exc:  # fail-closed: audita y relanza siempre
                await _auditar_token_quemado(
                    pipeline_id, req.parent_pipeline_id, "consumo", Motivo.CREACION_FALLIDA,
                    excepcion=type(exc).__name__,
                    ref_token=token_ref(hash_token(req.subpipeline_token)),
                )
                raise
            if isinstance(consumo, ConsumoRechazado):
                raise HTTPException(
                    status_code=403,
                    detail=f"subpipeline_token rechazado: {consumo.motivo.value}",
                )
            parent_pipeline_id = consumo.parent_pipeline_id
            parent_step = consumo.parent_step
            depth = consumo.depth
            # Identidad del hijo = la del padre (fila), nunca la del cuerpo
            # (revisión final, I-3): el executor le carga el uso a ella.
            user_id, tenant_id = consumo.user_id, consumo.tenant_id

        # Revisión final (I-2, frente F): desde acá el token ya está quemado.
        # CUALQUIER error antes de que el hijo exista (plan, pre-vuelo,
        # transacción) deja SUBPIPELINE_RECHAZADO y se relanza el
        # error ORIGINAL. Este primer bloque cubre plan y pre-vuelo (fase
        # `plan`); el de la escritura, más abajo, cubre la fase `creacion`.
        try:
            steps_spec = [s.model_dump() for s in req.steps] if req.steps else None
            steps = await _build_plan_or_reject(pipeline_id, req.objective, req.max_steps, steps_spec)

            # Asignar pipeline_id a cada step
            for step in steps:
                step.pipeline_id = pipeline_id

            # Pre-vuelo (spec 2026-09-17 §4.7): DESPUÉS de build() y ANTES de
            # crear filas. dry_run también ("¿cuánto costaría?").
            veredicto = await _prevuelo_o_503(
                steps, {"objective": req.objective}, user_id=user_id, tenant_id=tenant_id,
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
                # Identidad del hijo = la del padre (frente F, I-3), no la del cuerpo.
                user_id=user_id,
                tenant_id=tenant_id,
                parent_pipeline_id=parent_pipeline_id,
                depth=depth,
                mode=req.mode,
                plan=steps,
                max_steps=req.max_steps,
                context={"objective": req.objective},
                # El árbitro devuelve (spec 2026-09-18 §3.4): PERSISTIDO
                # desde la creación -- antes se validaba contra el pre-vuelo
                # (arriba) y se descartaba; una devolución automática, horas
                # después, necesita seguir viéndolo.
                costo_max_aceptado_usd=req.costo_max_aceptado_usd,
                created_at=now,
                updated_at=now,
            )
        except HTTPException:
            if parent_pipeline_id is not None:
                await _auditar_token_quemado(
                    pipeline_id, parent_pipeline_id, "plan", Motivo.PLAN_RECHAZADO)
            raise
        except BaseException as exc:  # CancelledError incluida: audita y relanza
            if parent_pipeline_id is not None:
                await _auditar_token_quemado(
                    pipeline_id, parent_pipeline_id, "creacion", Motivo.CREACION_FALLIDA,
                    excepcion=type(exc).__name__,
                )
            raise


        # R38, fix round 1 (2b): pasos, PIPELINE_CREATED y, en dry_run, el
        # completed y DRY_RUN_COMPLETE van en UNA transacción junto con
        # `completar_reserva`, el UPDATE que le pone el plan a la fila
        # reservada. Todo o nada: si algo falla antes del COMMIT no queda nada
        # escrito y la reserva se suelta en el `except` de afuera.
        estado_tx = store.EstadoDeTransaccion()
        try:
            async with store.conexion_del_pool() as conn:
                async with asyncio.timeout(store.db_connect_timeout_seconds()):
                    async with store.transaccion(conn, estado_tx) as tx:
                        await cupo.completar_reserva(pipeline, conexion=tx)
                        for step in steps:
                            await store.step_upsert(step, conexion=tx)
                        await store.event_append(pipeline_id, "PIPELINE_CREATED", {
                            "name": req.name, "mode": req.mode, "steps": len(steps),
                            "parent_pipeline_id": parent_pipeline_id, "depth": depth,
                            **costo,
                        }, conexion=tx)
                        if req.mode == "dry_run":
                            await store.pipeline_update_status(
                                pipeline_id, PipelineStatus.completed, conexion=tx)
                            await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", conexion=tx)
        except HTTPException:
            # Frente F (I-2): el token ya está quemado y el hijo NO existe.
            if parent_pipeline_id is not None:
                await _auditar_token_quemado(
                    pipeline_id, parent_pipeline_id, "creacion", Motivo.CREACION_FALLIDA)
            raise
        except BaseException as exc:  # CancelledError incluida (frente F, I-2): audita antes de decidir qué se responde
            if parent_pipeline_id is not None:
                await _auditar_token_quemado(
                    pipeline_id, parent_pipeline_id, "creacion", Motivo.CREACION_FALLIDA,
                    excepcion=type(exc).__name__,
                )
            if estado_tx.incierta:
                # R41 con reserva: el COMMIT salió y la respuesta no llegó, así
                # que la fila PUEDE ser un pipeline completo. Soltarla borraría
                # un pipeline real. Se deja ocupando cupo; si quedó `pending` la
                # cosecha el reaper a los 300 s. Fail-closed por el lado que no
                # destruye datos.
                soltar_la_reserva = False
            if not isinstance(exc, Exception):
                raise  # cancelación: no se convierte en un 503
            if _es_deadlock(exc) and not estado_tx.incierta:
                # Contención DENTRO de la transacción (el UPDATE que completa la
                # reserva y los pasos pelean por los mismos candados de rango).
                # No se reintenta acá -- reintentar significaría rehacer la
                # transacción entera --, pero tampoco se etiqueta mal: es
                # contención, y decirlo `prevuelo_no_disponible` mandaría a
                # revisar el pre-vuelo, que no tuvo nada que ver. Medido: 2 de
                # ~250.000 pedidos en la carga del 2026-09-17.
                raise _contencion_503(ContencionAlReservar(1, 0.0)) from exc
            # fail-closed: transacción de creación que no se pudo completar ->
            # nada quedó escrito y no se crea (spec §8).
            motivo = _motivo_redactado(exc)
            detalle = {"code": "prevuelo_no_disponible", "motivo": motivo}
            if estado_tx.incierta:
                logger.error("crear: COMMIT del pipeline %s con resultado INCIERTO: %s", pipeline_id, motivo)
                detalle["detalle"] = (
                    f"Resultado incierto: la conexión se cortó mientras se confirmaba. El pipeline "
                    f"{pipeline_id} puede existir: revisá la lista de pipelines antes de reintentar."
                )
            else:
                logger.error("crear: no se pudo escribir el pipeline: %s", motivo)
            raise HTTPException(status_code=503, detail=detalle) from exc
        soltar_la_reserva = False
    except BaseException:
        if soltar_la_reserva:
            await _soltar_reserva(pipeline_id)
        raise

    # Con la fila completa y los pasos escritos el hijo EXISTE; un fallo de
    # este evento no es "token quemado sin hijo" (frente F) ni deja reserva.
    if parent_pipeline_id is not None:
        await store.event_append(
            parent_pipeline_id, "SUBPIPELINE_CREADO",
            {"hijo_pipeline_id": pipeline_id, "depth": depth},
            parent_step,
        )

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
    if pipeline.status in (
        PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.aborted,
        # Ronda de arreglo 2 (2026-09-18-arbitro-devuelve): `disputed` es
        # terminal (ESTADOS_SIN_CUPO) -- no se cancela lo que ya terminó de
        # correr, aunque nadie haya resuelto la objeción todavía.
        PipelineStatus.disputed,
    ):
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
    # El cupo ANTES del pre-vuelo: sondear cuesta plata (ver _cupo_o_429).
    await _cupo_o_429()
    costo, contexto, ilegibles = await _prevuelo_de_reanudacion(pipeline, steps)

    # Época (spec 2026-09-17 §5.3): se toma ANTES de tocar pasos. Si otro
    # resume ganó, este no escribe ni lanza nada: dos ejecutores del mismo
    # pipeline es exactamente lo que la época existe para impedir.
    # Pasada final R34: si había refs ilegibles, el contexto sin ellas viaja
    # en el MISMO UPDATE que toma la época.
    # `cupo_maximo` (2026-09-17): reanudar OCUPA CUPO -- el pipeline pasa de
    # `interrupted`, que no cuenta, a correr --, y hasta hoy este camino no
    # miraba MAX_PARALLEL_PIPELINES: ni el candado de proceso ni el GET_LOCK lo
    # tomaban. Con tres interrumpidos y tres `resume` se pasaba el límite. La
    # condición viaja ahora dentro del mismo UPDATE que toma la época.
    try:
        if ilegibles:
            nueva_epoca = await store.pipeline_tomar_epoca(
                pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,), contexto,
                cupo_maximo=MAX_PARALLEL_PIPELINES,
            )
        else:
            nueva_epoca = await store.pipeline_tomar_epoca(
                pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,),
                cupo_maximo=MAX_PARALLEL_PIPELINES,
            )
    except CupoAgotado as exc:
        raise HTTPException(status_code=429, detail={
            "code": "limite_de_activos", "detalle": str(exc),
        }) from exc
    except ContencionAlReservar as exc:
        raise _contencion_503(exc) from exc
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
    # El cupo ANTES del pre-vuelo: sondear cuesta plata (ver _cupo_o_429).
    await _cupo_o_429()
    costo, contexto, ilegibles = await _prevuelo_de_reanudacion(
        pipeline, steps, aprobados_ahora=frozenset(s.step_id for s in gated if s.facet == "hyde"),
    )
    # Pasada final R34: el contexto sin las refs ilegibles (con las marcas
    # hyde_approved_* intactas) es el que viaja con la época.
    pipeline.context = contexto

    # Época (desvío 9 del plan 2026-09-17): approve-step lanza run_pipeline
    # igual que resume. Las marcas hyde_approved_* viajan en el MISMO UPDATE
    # que toma la época (antes iban en un pipeline_update_status aparte).
    # `cupo_maximo`: aprobar un paso relanza el pipeline, así que ocupa cupo
    # igual que resume, y hasta hoy tampoco lo miraba (ver resume_pipeline).
    try:
        nueva_epoca = await store.pipeline_tomar_epoca(
            pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,), pipeline.context,
            cupo_maximo=MAX_PARALLEL_PIPELINES,
        )
    except CupoAgotado as exc:
        raise HTTPException(status_code=429, detail={
            "code": "limite_de_activos", "detalle": str(exc),
        }) from exc
    except ContencionAlReservar as exc:
        raise _contencion_503(exc) from exc
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
            # Task 9 (2026-09-18, historial-y-arreglos-de-pipeline): el
            # historial de la Mesa necesita el prompt EXACTO, el modelo REAL
            # y de qué pasos dependía -- los tres ya sobreviven el round-trip
            # por jacobs_steps (jacobs/_modelo_real_test.py,
            # jacobs/_pipeline_results_historial_test.py) y antes se tiraban
            # acá, en la lista blanca del dict de salida.
            "prompt":            step.input.get("prompt"),
            "modelo_real":       step.modelo_real,
            "depends_on":        step.depends_on,
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
