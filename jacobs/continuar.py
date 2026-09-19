"""Jacobs — continuar un pipeline abortado o vencido (spec 2026-09-17 §5).

UNA función de servicio para el endpoint (POST /jacobs/pipeline/{id}/continue
y /continue/preflight) y para el CLI (tools/jacobs_relaunch.py): ninguno tiene
lógica propia.

Reglas (§5.2): sólo `plataforma` (el dueño lo verifica jax-platform); estados
aborted (cualquier causa, D2) o expired; kill switch 423; contra
MAX_PARALLEL_PIPELINES, con la condición del cupo DENTRO del UPDATE que revive
el pipeline (2026-09-17: ya no hay candado, ni de proceso ni con nombre); se reusan los pasos con ref LEGIBLE y se
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

from redaccion import recortar_redactado

from jacobs import store
from jacobs.executor import RefIlegible, _load_ref
from jacobs.models import MOTOR_FACETS, Pipeline, PipelineStatus, Step, StepStatus
from jacobs.plan import PlanRejected, _check_cleanroom, _validate_plan_capabilities
from jacobs.policy import (
    MAX_PARALLEL_PIPELINES,
    ContencionAlReservar,
    CupoAgotado,
    check_kill_switch,
    validate_resume,
)
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


async def separar_por_ref(plan: list[Step], contexto: dict) -> tuple[list[int], list[int], dict]:
    """Regla 5 de continuar (§5.2), compartida con resume y approve-step (ola
    final F2, Ruling R32): un paso se REUSA si tiene `step_{i}_ref` en el
    contexto y la ref se lee; todos los demás se corren. Devuelve
    (reusados, a_correr, contexto SIN las refs de los pasos a correr) -- el
    pre-vuelo necesita ese contexto: con la ref de un paso a rehacer mediría
    a sus dependientes con una salida que no va a existir. No toca `plan` ni
    `contexto` (copia). Las refs se leen en un hilo (disco)."""
    limpio = dict(contexto)
    reusados: list[int] = []
    a_correr: list[int] = []
    for paso in plan:
        clave = f"step_{paso.step_index}_ref"
        ref = limpio.get(clave)
        if ref and await asyncio.to_thread(_ref_legible, ref):
            reusados.append(paso.step_index)
            continue
        limpio.pop(clave, None)
        a_correr.append(paso.step_index)
    return reusados, a_correr, limpio


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

    reusados, a_correr, contexto = await separar_por_ref(plan, pipeline.context)
    for paso in (plan[i] for i in a_correr):
        # La aprobación humana de hyde era para la corrida anterior (desvío 10).
        contexto.pop(f"hyde_approved_{paso.step_id}", None)
        paso.status = StepStatus.pending
        paso.error = None
        paso.started_at = None
        paso.finished_at = None
        paso.output_ref = None

    # Merge 2026-09-17 (E-03): las facetas válidas salen de la tabla `facet`
    # (status='active'), no de una lista fija -- `VALID_FACETS` dejó de existir
    # en el frente E. Se lee PEREZOSAMENTE y una sola vez: un continue sin
    # `--reasignar` (el caso normal) no agrega una lectura de gobernanza.
    facetas_activas: frozenset | None = None

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
        if facetas_activas is None:
            facetas_activas = (await store.get_motor_governance())["facets"]
        if faceta not in facetas_activas:
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
    # SIN CANDADO (2026-09-17). Este recuento suelto sólo evita sondear en vano
    # con el cupo lleno: NO decide. El que decide está dentro de la sentencia
    # que escribe, más abajo.
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
    # Regla 10 (Ruling R22): el evento va DENTRO de la misma transacción
    # que escribe los pasos y el pipeline, no después del commit. La
    # época nueva se conoce de antemano (epoca_leida+1, bajo el
    # SELECT...FOR UPDATE de la transacción): si la transacción pierde la
    # carrera devuelve None y este payload nunca se inserta.
    evento_payload = {
        "by": invoked_by, "from_status": a.pipeline.status.value, "run_epoch": a.pipeline.run_epoch + 1,
        **_pasos(a), "reasignados": a.reasignados, "costo_max_usd": formatear_usd(veredicto.costo_max_usd),
    }
    # EL CUPO LO DECIDE LA ESCRITURA. Ya no hay GET_LOCK ni candado de proceso:
    # `store.continuar_transaccion` lleva la condición del cupo DENTRO de su
    # UPDATE, así que cuenta y escribe en la misma sentencia y se interbloquea
    # con el INSERT de crear aunque el otro corra en OTRO PROCESO (el CLI) --
    # que era exactamente lo que el GET_LOCK vino a arreglar. Ahora se arregla
    # solo: un camino nuevo hereda el límite sin acordarse de pedir nada.
    #
    # m1: con plazo, como la transacción de crear. El SELECT ... FOR UPDATE
    # puede esperar un lock de fila hasta innodb_lock_wait_timeout (50 s por
    # defecto); al vencer, la conexión se cierra en su propio finally (nada
    # queda a medias) y el endpoint responde 503. Sin candado con nombre, esa
    # espera ya no frena a los demás procesos.
    estado_tx = store.EstadoDeTransaccion()
    try:
        async with asyncio.timeout(store.db_connect_timeout_seconds()):
            nueva = await store.continuar_transaccion(
                pipeline_id, a.pipeline.run_epoch, a.pipeline.status,
                [a.plan[i] for i in a.pasos_a_correr], a.plan, a.contexto, indice,
                evento_payload=evento_payload, estado=estado_tx,
                # El árbitro devuelve (spec 2026-09-18 §3.4): un /continue
                # humano con un tope nuevo lo PERSISTE, no solo lo valida
                # para este pedido -- si no, una devolución automática
                # minutos después de este continue seguiría viendo el tope
                # de la creación original (o ninguno).
                costo_max_aceptado_usd=costo_max_aceptado_usd,
            )
    except ContencionAlReservar as exc:
        # Contención, no falla: 503 para que el llamador reintente. El CLI sale
        # con error y el endpoint traduce el código (ver routes._contencion_503).
        raise ContinuarRechazado(503, "contencion_al_reservar", {
            "intentos": exc.intentos, "espera_s": round(exc.espera_total, 3),
        }) from exc
    except CupoAgotado as exc:
        # El UPDATE no tocó la fila, y la época y el status ya estaban
        # verificados bajo el candado de FILA unas líneas más arriba: el único
        # motivo que queda es que no hay lugar. 429, el mismo rechazo que da el
        # recuento de arriba.
        raise ContinuarRechazado(
            429, "limite_de_activos", _texto_limite(exc.activos)) from exc
    except BaseException as exc:
        # Mismo mecanismo que crear (R41): si el corte o el plazo caen
        # DURANTE el COMMIT, el servidor pudo haberlo confirmado -- la
        # época quedaría incrementada y el pipeline en `running` sin
        # que nadie encolara run_pipeline (lo rescata el reaper). El
        # 503 lo dice; un reintento a ciegas lo continuaría dos veces.
        # Si el corte fue ANTES del COMMIT no hay nada escrito y el
        # error sube tal cual (el endpoint responde 503 sin `detalle`).
        if not estado_tx.incierta:
            raise
        raise ContinuarRechazado(503, "prevuelo_no_disponible", {
            "motivo": recortar_redactado(f"{type(exc).__name__}: {exc}", 300),
            "detalle": (
                f"Resultado incierto: la conexión se cortó mientras se confirmaba. "
                f"El continue del pipeline {pipeline_id} puede haber empezado: "
                f"revisá su estado antes de reintentar."
            ),
        }) from exc
    if nueva is None:
        raise ContinuarRechazado(409, "estado_no_continuable", {
            "status": None,
            "mensaje": "otro pedido cambió el pipeline mientras se preparaba este: volvé a consultarlo",
        })
    continuado = a.pipeline.model_copy(update={
        "plan": a.plan, "context": a.contexto, "status": PipelineStatus.running,
        "run_epoch": nueva, "current_step_index": indice,
        "costo_max_aceptado_usd": (
            costo_max_aceptado_usd if costo_max_aceptado_usd is not None
            else a.pipeline.costo_max_aceptado_usd
        ),
    })

    respuesta = {
        "pipeline_id": pipeline_id, "status": "running", "run_epoch": nueva, **_pasos(a),
        "costo_max_usd": formatear_usd(veredicto.costo_max_usd),
        "pasos_costo": [c.to_dict() for c in veredicto.pasos_costo],
    }
    return respuesta, continuado
