"""
LAS MANOS — Motor Registry: endpoints HTTP.

POST /motor/dispatch           — crea un job (pasa por policy, falla cerrado)
GET  /motor/job/{job_id}       — consulta estado de un job
POST /motor/job/{job_id}/cancel — solicita cancelación

El router se registra en server.py al conectar el Motor Registry.
Este módulo no modifica server.py — eso es responsabilidad del Commit 2.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from motor_registry.catalog import MotorCatalog
from motor_registry.facet_policy import check_facet_admission
from motor_registry.job_store import JobStore
from motor_registry.models import (
    FacetAuthorizeRequest,
    FacetAuthorizeResponse,
    JobStatus,
    MotorDispatchRequest,
    MotorDispatchResponse,
    GovernedDispatchRequest,
    MotorJobView,
)
from motor_registry.policy import MotorPolicy
import facet_resolver  # su sello (mtime de un archivo) invalida también el catálogo
from motor_registry import job_tasks
from motor_registry import worker as motor_worker
from interruptor import interruptor_activo, ruta_del_interruptor

BASE_DIR = Path(__file__).resolve().parent.parent

_STORE = JobStore(str(BASE_DIR / "logs" / "motor_jobs.jsonl"))
_B7_EVIDENCE_RECORDER = None

def configure_b7_evidence_recorder(recorder) -> None:
    """Startup composition only; the legacy HTTP request cannot choose it.

    Tambien valida al arrancar la configuracion del registro de denegaciones
    (timeout y concurrencia): un valor invalido es EntornoInvalido y el servicio
    no levanta, igual que el resto de las variables de /etc/jax/.env."""
    global _B7_EVIDENCE_RECORDER
    _B7_EVIDENCE_RECORDER = recorder
    configurar_registro_de_denegaciones()


# --- Registro de la evidencia de denegacion de /motor/dispatch -----------------
# Corre en el camino de CUALQUIER pedido denegado por el middleware, asi que esta
# acotado: fuera del hilo del bucle, con limite de tiempo y con tope de escrituras
# en vuelo. Lo que no cabe se OMITE (y se cuenta), nunca se acumula ni bloquea.
VARIABLE_TIMEOUT_DENEGACION = "JAX_B7_DENEGACION_TIMEOUT_S"
VARIABLE_CONCURRENCIA_DENEGACION = "JAX_B7_DENEGACION_CONCURRENCIA"
TIMEOUT_DENEGACION_POR_DEFECTO = 3.0
CONCURRENCIA_DENEGACION_POR_DEFECTO = 4
TIMEOUT_DENEGACION_MAXIMO = 60.0
CONCURRENCIA_DENEGACION_MAXIMA = 64
#: Cada cuanto, como maximo, se resume cuantas evidencias se omitieron.
INTERVALO_AVISO_OMITIDAS_S = 60.0

#: Resultado de registrar_denegacion_de_dispatch.
EVIDENCIA_REGISTRADA = "registrada"
EVIDENCIA_FALLIDA = "fallida"                  # error o timeout al escribirla
EVIDENCIA_OMITIDA = "omitida_por_saturacion"   # sin cupo inmediato
EVIDENCIA_SIN_REGISTRADOR = "sin_registrador"  # B7 no esta compuesto

_DENEGACION: dict | None = None


def _numero_del_entorno(entorno, nombre: str, defecto, tipo, minimo, maximo):
    from config_entorno import EntornoInvalido
    crudo = (entorno.get(nombre) or "").strip()
    if not crudo:
        return defecto
    try:
        valor = tipo(crudo)
    except ValueError as exc:
        raise EntornoInvalido(f"{nombre}={crudo!r} no es un numero valido ({tipo.__name__}).") from exc
    if not (minimo < valor <= maximo if tipo is float else minimo <= valor <= maximo):
        raise EntornoInvalido(f"{nombre}={crudo!r} fuera de rango ({minimo} a {maximo}).")
    return valor


def configurar_registro_de_denegaciones(entorno=None) -> dict:
    """Lee y VALIDA la configuracion (fail-closed) y reinicia el estado: semaforo,
    contador de omitidas y aviso pendiente. Sin variables, usa los defectos."""
    import os
    global _DENEGACION
    entorno = os.environ if entorno is None else entorno
    timeout = _numero_del_entorno(entorno, VARIABLE_TIMEOUT_DENEGACION, TIMEOUT_DENEGACION_POR_DEFECTO,
                                  float, 0.0, TIMEOUT_DENEGACION_MAXIMO)
    cupo = _numero_del_entorno(entorno, VARIABLE_CONCURRENCIA_DENEGACION, CONCURRENCIA_DENEGACION_POR_DEFECTO,
                               int, 1, CONCURRENCIA_DENEGACION_MAXIMA)
    if timeout != timeout:  # NaN
        from config_entorno import EntornoInvalido
        raise EntornoInvalido(f"{VARIABLE_TIMEOUT_DENEGACION} no puede ser NaN.")
    _DENEGACION = {"timeout": timeout, "cupo": cupo, "semaforo": asyncio.Semaphore(cupo),
                   "omitidas": 0, "omitidas_total": 0, "aviso_pendiente": False}
    return _DENEGACION


def _estado_denegacion() -> dict:
    return _DENEGACION if _DENEGACION is not None else configurar_registro_de_denegaciones()


def _avisar_omitidas(estado: dict) -> None:
    n, estado["omitidas"], estado["aviso_pendiente"] = estado["omitidas"], 0, False
    logger.warning(
        "B7: %d evidencia(s) de /motor/dispatch denegado omitida(s) por saturacion en los ultimos %.0f s "
        "(tope de %d escrituras en vuelo); las denegaciones se devolvieron igual",
        n, INTERVALO_AVISO_OMITIDAS_S, estado["cupo"],
    )


def _contar_omitida(estado: dict) -> None:
    estado["omitidas"] += 1
    estado["omitidas_total"] += 1
    if not estado["aviso_pendiente"]:
        estado["aviso_pendiente"] = True
        asyncio.get_running_loop().call_later(INTERVALO_AVISO_OMITIDAS_S, _avisar_omitidas, estado)
# _CATALOG/_POLICY arrancan None -- se pueblan en el startup hook de
# server.py (init_motor_catalog, abajo). [motors.*]/[capabilities.*] de
# config.toml ya no se leen (R4 -- catalogo en DB). Ningun otro modulo
# importa estos dos nombres directamente (verificado: grep -rn "_CATALOG"
# solo los usa este archivo), asi que reasignarlos acá es seguro.
_CATALOG: MotorCatalog | None = None
_POLICY: MotorPolicy | None = None
# Set only by the service composition root.  The route fails closed until the
# authoritative execution store is present; request-body fields never replace
# stored artifacts.
_GOVERNED_EXECUTION_STORE = None
_GOVERNED_DISPATCH_CLAIMER = None
_B7_WORKER_RESULT_INGESTOR = None

logger = logging.getLogger(__name__)


# Instante de RELOJ DE PARED (time.time()) en que se cargó el catálogo --
# el mismo reloj que estampa facet_resolver._tocar_sello(). Nunca monotonic:
# compararlo con un mtime da un veredicto constante (ver _entrada_sellada).
_CATALOG_LOADED_AT_WALL: float | None = None
# Una sola recarga a la vez: N dispatches que ven el sello nuevo a la par no
# disparan N consultas, y ninguno ve _CATALOG/_POLICY a medias.
_CATALOG_LOCK = asyncio.Lock()


async def _load_catalog(conexion=None) -> None:
    """Carga el catálogo y reemplaza _CATALOG y _POLICY JUNTOS, solo si la
    consulta terminó bien. El instante se toma ANTES de consultar: si el
    sello se estampa mientras la consulta está en vuelo, la próxima
    comparación lo ve más nuevo y recarga (mismo criterio que resolve_facet).

    `conexion` (ola final F8 de Jacobs, 2026-09-17): el pre-vuelo recarga por
    la conexión de su pool; sin ella, from_db() abre y cierra la suya."""
    global _CATALOG, _POLICY, _CATALOG_LOADED_AT_WALL
    started_at_wall = time.time()
    catalog = await (MotorCatalog.from_db() if conexion is None else MotorCatalog.from_db(conexion=conexion))
    policy = MotorPolicy(catalog)
    _CATALOG, _POLICY, _CATALOG_LOADED_AT_WALL = catalog, policy, started_at_wall


async def init_motor_catalog() -> None:
    """Llamado desde el startup hook de server.py. Si la DB no responde al
    arrancar, _CATALOG queda None; el primer dispatch vuelve a intentar
    (_ensure_catalog_fresh) y, si tampoco puede, rechaza explícito."""
    await _load_catalog()


def _catalog_is_stale() -> bool:
    if _CATALOG is None:
        return True
    mtime = facet_resolver._seal_mtime()
    # None = sin señal, nunca "invalidar" (contrato de facet_resolver). `>=`
    # y no `>`: un empate de resolución del filesystem se lee como "cambió".
    return mtime is not None and mtime >= _CATALOG_LOADED_AT_WALL


async def _ensure_catalog_fresh(conexion=None) -> None:
    """Recarga el catálogo si el sello quedó más nuevo que su carga.

    También lo usa el pre-vuelo de Jacobs (jacobs/prevuelo_catalogo.py::
    resolver_motores, ola final F8): evalúa el MISMO catálogo que el despacho
    va a usar, con esta misma invalidación, en vez de un from_db() por
    pedido. `conexion` es la que el pre-vuelo tomó del pool del store de
    Jacobs (jacobs/store.py::conexion_del_pool), sólo para la recarga.

    Existe por la regresión del 2026-09-12 (13:03-13:29): el catálogo se
    cargaba UNA vez al arrancar, la migración `generate` 5 -> 15 corrió al
    arrancar jax-platform, y este proceso siguió rechazando 900 s contra su
    copia de 300 s hasta que alguien lo reinició. Los escritores de
    `motor`/`capability`/`capability_motor` (migraciones y admin de
    jax-platform) estampan el sello de facet_resolver tras commitear; los
    rebinds de facets también, y el catálogo depende de `model` vía
    model_ref, así que les sirve igual.

    Costo medido en hall9000 (2026-09-12): el chequeo es un os.stat, p50
    0,70 us; una recarga es from_db(), p50 0,95 ms (p95 1,41 ms).

    RECARGA FALLIDA -> FALLA CERRADO (503) y se reintenta en el próximo
    dispatch. Mantener el catálogo viejo sería un gate fail-open: puede
    seguir autorizando un motor o capability que se acaba de revocar. No
    cuesta disponibilidad real: el dispatch ya depende de la DB (credenciales)."""
    if not _catalog_is_stale():
        return
    async with _CATALOG_LOCK:
        if not _catalog_is_stale():  # otro dispatch ya recargó mientras esperábamos
            return
        try:
            await _load_catalog(conexion)
        except Exception as exc:  # noqa: BLE001 -- se re-lanza como 503, no se traga
            logger.error("Motor Registry: el catálogo cambió y no se pudo recargar: %r", exc)
            raise HTTPException(
                status_code=503,
                detail="Motor Registry: el catálogo cambió y no se pudo recargar desde la DB",
            ) from exc

router = APIRouter(prefix="/motor", tags=["motor_registry"])


def configure_governed_execution_store(store, dispatch_claimer=None, worker_result_ingestor=None) -> None:
    """Composition-root hook for fixed Block 6/B7 dependencies only."""
    global _GOVERNED_EXECUTION_STORE, _GOVERNED_DISPATCH_CLAIMER, _B7_WORKER_RESULT_INGESTOR
    _GOVERNED_EXECUTION_STORE = store
    _GOVERNED_DISPATCH_CLAIMER = dispatch_claimer
    _B7_WORKER_RESULT_INGESTOR = worker_result_ingestor


def _log_worker_exception(task: asyncio.Task, *, job_id: str) -> None:
    """Done-callback: cierra el punto ciego del create_task fire-and-forget.
    Cualquier excepción que escape de worker.run (incl. fuera de su try interno)
    queda en el log con traceback completo, en vez de morir en silencio."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Excepción no capturada en worker del job %s:\n%s",
            job_id,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )


def _ingest_governed_worker_completion(task: asyncio.Task, *, execution_id: str, job_id: str) -> None:
    """Fixed post-worker bridge; a legacy job never reaches this callback."""
    _log_worker_exception(task, job_id=job_id)
    if task.cancelled() or task.exception() is not None or _B7_WORKER_RESULT_INGESTOR is None:
        return
    view = _STORE.get(job_id)
    if view is None or view.status is not JobStatus.COMPLETED or not view.result_path:
        return
    try:
        _B7_WORKER_RESULT_INGESTOR.ingest_governed_completion(
            view.result_path, execution_id=execution_id, job_id=job_id)
    except Exception:  # fail-soft: post-completion evidence cannot resurrect or alter a completed job.
        logger.exception("No se pudo ingerir resultado B7 del job gobernado %s", job_id)


@router.post("/governed-dispatch", response_model=MotorDispatchResponse, status_code=202)
async def governed_dispatch(req: GovernedDispatchRequest) -> MotorDispatchResponse:
    """Launch only a durably claimed, authoritative Block 6 execution."""
    if _GOVERNED_EXECUTION_STORE is None:
        raise HTTPException(status_code=503, detail="governed execution store no inicializado")
    await _ensure_catalog_fresh()
    if _CATALOG is None:
        raise HTTPException(status_code=503, detail="Motor Registry: catálogo no inicializado todavía")
    # Resolve before any state/job work; a missing kill-switch configuration
    # fails closed without touching authoritative execution state.
    route = ruta_del_interruptor()
    try:
        record = _GOVERNED_EXECUTION_STORE.load_execution(req.execution_id)
        authorization = _GOVERNED_EXECUTION_STORE.load_authorization(record.authorization_id)
        request = authorization.execution_request
        if (record.decision_id != authorization.decision_id or
                record.execution_authorization_hash != authorization.execution_authorization_hash or
                record.execution_request_hash != request.execution_request_hash):
            raise ValueError("binding almacenado inválido")
        cap = _CATALOG.get_capability(request.capability)
        motor = _CATALOG.get_motor(request.motor)
        if (cap is None or motor is None or not motor.enabled or
                request.authenticated_caller_id not in cap.allowed_callers or
                request.motor not in cap.allowed_motors or not cap.sandbox_only or not motor.sandbox_only):
            raise ValueError("catálogo actual no permite execution")
        if interruptor_activo(route):
            raise ValueError("kill switch activo")
        if _GOVERNED_DISPATCH_CLAIMER is None:
            from policy.execution_control.service import dispatch_execution
            claimer = dispatch_execution
        else:
            claimer = _GOVERNED_DISPATCH_CLAIMER
        # Reserve an opaque job identity, but do not create a job yet.  The
        # B6/B7 shared transaction must commit its exact mapping before any
        # task exists; evidence failure therefore still leaves no job/worker.
        job_id = str(uuid.uuid4())
        # This is the atomic state claim immediately before job/task creation.
        claimer(_GOVERNED_EXECUTION_STORE, record, authorization,
                           now_utc=datetime.now(timezone.utc), kill_switch_active=False,
                           job_id=job_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"GOVERNED_DISPATCH_REJECTED: {exc}") from exc

    _STORE.create(caller=request.authenticated_caller_id, capability=request.capability,
        motor=request.motor, trace_id=req.trace_id, prompt=request.prompt,
        recursion_depth=0, pipeline_id=None, job_id=job_id, tenant_id=request.tenant_id,
        user_id=request.user_id, project_id=None)
    task = asyncio.create_task(motor_worker.run(job_id=job_id, motor=request.motor,
        capability=request.capability, prompt=request.prompt, context=request.projection()["context"],
        store=_STORE, catalog=_CATALOG, kill_switch_path=str(route), user_id=request.user_id,
        tenant_id=request.tenant_id, caller=request.authenticated_caller_id,
        timeout_seconds=request.timeout_seconds, pipeline_id=None))
    task.add_done_callback(lambda t: _ingest_governed_worker_completion(
        t, execution_id=record.execution_id, job_id=job_id))
    job_tasks.register(job_id, task)
    return MotorDispatchResponse(job_id=job_id, status=JobStatus.PENDING, motor=request.motor,
        capability=request.capability, trace_id=req.trace_id)


async def registrar_denegacion_de_dispatch(correlacion: str) -> str:
    """Registra la evidencia B7 de que el despacho legacy fue DENEGADO y devuelve
    EVIDENCIA_REGISTRADA / _FALLIDA / _OMITIDA / _SIN_REGISTRADOR. Nunca levanta y
    nunca bloquea al llamador mas que el timeout configurado.

    - Fuera del hilo del bucle (asyncio.to_thread): la escritura es sincrona.
    - Con limite de tiempo (JAX_B7_DENEGACION_TIMEOUT_S, 3 s): si vence cuenta
      como fallida (log ERROR con `correlacion`). El hilo no se puede cortar; el
      cupo se libera cuando el hilo termina de verdad, no al vencer el plazo.
    - Con tope de escrituras en vuelo (JAX_B7_DENEGACION_CONCURRENCIA, 4): sin
      cupo inmediato NO se escribe la evidencia individual, se cuenta como omitida
      y un WARNING agregado, como maximo por minuto, dice cuantas.
    - Un fallo deja log de ERROR con `correlacion` y la causa; no se loguea nada
      del pedido.

    Lo llama el middleware (`auth_servicio.proteger`), que es donde la
    denegacion de POST /motor/dispatch ocurre de verdad -- ninguna identidad
    tiene permiso sobre esa ruta --, y dispatch() como defensa en profundidad."""
    registrador = _B7_EVIDENCE_RECORDER
    if registrador is None:
        return EVIDENCIA_SIN_REGISTRADOR
    estado = _estado_denegacion()
    semaforo = estado["semaforo"]
    if semaforo.locked():
        _contar_omitida(estado)
        return EVIDENCIA_OMITIDA
    await semaforo.acquire()  # con cupo, no cede el control: la decision es atomica
    try:
        escritura = asyncio.ensure_future(asyncio.to_thread(registrador.record_governed_dispatch_denied))
    except BaseException:
        semaforo.release()
        raise

    vencida = []

    def _terminada(tarea: asyncio.Future) -> None:
        semaforo.release()  # el cupo vuelve cuando el hilo termino, venza o no el plazo
        if not tarea.cancelled() and tarea.exception() is not None and vencida:
            logger.error("B7: la evidencia de /motor/dispatch denegado fallo despues de vencer el plazo correlacion=%s",
                         correlacion, exc_info=tarea.exception())

    escritura.add_done_callback(_terminada)
    try:
        observacion = await asyncio.wait_for(asyncio.shield(escritura), timeout=estado["timeout"])
    except asyncio.TimeoutError:
        vencida.append(True)
        logger.error(
            "B7: la evidencia de /motor/dispatch denegado no termino en %.1f s (la denegacion se devuelve igual) correlacion=%s",
            estado["timeout"], correlacion,
        )
        return EVIDENCIA_FALLIDA
    except Exception:  # fail-soft: legacy dispatch remains rejected if evidence persistence is unavailable.
        logger.exception(
            "B7: no se pudo registrar la evidencia de /motor/dispatch denegado (la denegacion se devuelve igual) correlacion=%s",
            correlacion,
        )
        return EVIDENCIA_FALLIDA
    # El registrador devuelve la observacion ya persistida (EnforcementObservation):
    # su observation_id es el vinculo exacto con la fila B7. Va en el log junto a
    # la correlacion para unirlos sin depender de la hora. (El subject de esa fila
    # es "denial:<uuid propio>", no la correlacion: cambiarlo exigiria policy/**.)
    logger.info(
        "B7: evidencia de /motor/dispatch denegado registrada observation_id=%s correlacion=%s",
        getattr(observacion, "observation_id", None), correlacion,
    )
    return EVIDENCIA_REGISTRADA


@router.post("/dispatch", response_model=MotorDispatchResponse, status_code=202)
async def dispatch(req: MotorDispatchRequest) -> MotorDispatchResponse:
    # Block 6: every catalog capability is governed in V1.  This legacy
    # transport endpoint must never consume a gate, create a job, or start a
    # worker; its request body is not an execution authority artifact.
    #
    # DEFENSA EN PROFUNDIDAD: en produccion este cuerpo no se alcanza por HTTP.
    # `proteger(app)` (auth_servicio) niega POST /motor/dispatch con 403 a toda
    # identidad, y es el middleware quien registra la evidencia. Esto solo corre
    # si alguien quita el permiso del middleware o llama a la funcion directo.
    # Cuerpo del 410: {"code", "correlacion", "evidencia_registrada"}, con
    # `correlacion` = hex si la evidencia NO quedo registrada y null si quedo.
    # (El 403 del middleware es distinto en eso: su `correlacion` es SIEMPRE hex.)
    correlacion = uuid.uuid4().hex
    estado = await registrar_denegacion_de_dispatch(correlacion)
    registrada = estado == EVIDENCIA_REGISTRADA
    raise HTTPException(status_code=410, detail={
        "code": "GOVERNED_EXECUTION_REQUIRED",
        "correlacion": None if registrada else correlacion,
        "evidencia_registrada": registrada,
    })


@router.post("/authorize-facet", response_model=FacetAuthorizeResponse)
async def authorize_facet(req: FacetAuthorizeRequest) -> FacetAuthorizeResponse:
    """Sincrono, sin job ni polling -- solo corre check_facet_admission()
    y devuelve el veredicto. Usado por jax-platform (Mesa web) antes de
    despachar a un facet HTTP-directo -- ver docs/superpowers/specs/
    2026-08-27-http-facets-motor-policy-governance-design.md."""
    allowed, reason = await check_facet_admission(req.caller, req.facet)
    return FacetAuthorizeResponse(allowed=allowed, reason=reason)


@router.get("/job/{job_id}", response_model=MotorJobView)
async def get_job(job_id: str) -> MotorJobView:
    view = _STORE.get(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado")
    return view


@router.post("/job/{job_id}/cancel", response_model=MotorJobView)
async def cancel_job(job_id: str) -> MotorJobView:
    view = _STORE.get(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado")
    if view.status in (
        JobStatus.COMPLETED, JobStatus.FAILED,
        JobStatus.CANCELLED, JobStatus.REJECTED,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' ya está en estado terminal: {view.status.value}",
        )
    _STORE.update(job_id, status=JobStatus.CANCELLED.value, finished_at=time.time())
    # La etiqueta sola no paraba nada: worker.run nunca la leía y el job
    # seguía llamando al modelo. Cortar la tarea es lo que corta el gasto.
    job_tasks.cancel(job_id)
    return _STORE.get(job_id)
