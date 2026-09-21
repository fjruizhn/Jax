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
    MotorJobView,
)
from motor_registry.policy import MotorPolicy
import facet_resolver  # su sello (mtime de un archivo) invalida también el catálogo
from motor_registry import job_tasks
from motor_registry import worker as motor_worker
from interruptor import ruta_del_interruptor
import human_gate

BASE_DIR = Path(__file__).resolve().parent.parent

_STORE = JobStore(str(BASE_DIR / "logs" / "motor_jobs.jsonl"))
# _CATALOG/_POLICY arrancan None -- se pueblan en el startup hook de
# server.py (init_motor_catalog, abajo). [motors.*]/[capabilities.*] de
# config.toml ya no se leen (R4 -- catalogo en DB). Ningun otro modulo
# importa estos dos nombres directamente (verificado: grep -rn "_CATALOG"
# solo los usa este archivo), asi que reasignarlos acá es seguro.
_CATALOG: MotorCatalog | None = None
_POLICY: MotorPolicy | None = None

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


def _rechazado(req: MotorDispatchRequest, motor: str | None, razon: str) -> MotorDispatchResponse:
    """Un pedido rechazado deja su job REJECTED con la razón (testigo)."""
    job_id = _STORE.create(
        caller=req.caller,
        capability=req.capability,
        motor=motor or "none",
        trace_id=req.trace_id,
        prompt=req.prompt,
        recursion_depth=req.recursion_depth,
        pipeline_id=req.pipeline_id,
    )
    _STORE.update(
        job_id,
        status=JobStatus.REJECTED.value,
        finished_at=time.time(),
        error=razon,
    )
    return MotorDispatchResponse(
        job_id=job_id,
        status=JobStatus.REJECTED,
        motor="none",
        capability=req.capability,
        trace_id=req.trace_id,
        rejected_reason=razon,
    )


@router.post("/dispatch", response_model=MotorDispatchResponse, status_code=202)
async def dispatch(req: MotorDispatchRequest) -> MotorDispatchResponse:
    await _ensure_catalog_fresh()
    if _POLICY is None or _CATALOG is None:
        raise HTTPException(status_code=503, detail="Motor Registry: catálogo no inicializado todavía")

    result = _POLICY.check(
        caller=req.caller,
        capability=req.capability,
        motor=req.motor,
        context_keys=list(req.context.keys()),
        recursion_depth=req.recursion_depth,
        human_gate_token=req.human_gate_token,
        timeout_seconds=req.timeout_seconds,
    )

    if not result.allowed:
        return _rechazado(req, result.resolved_motor, result.reason)

    # Human gate (2026-09-17): la política sólo mira que el token ESTÉ; acá se
    # consume contra la base (un string inventado ya no aprueba). Va DESPUÉS
    # de la política: un pedido que la política rechaza no quema el token.
    cap = _CATALOG.get_capability(req.capability)
    if cap is not None and cap.requires_human_gate:
        veredicto = await human_gate.consumir_token_gate(
            req.human_gate_token, uso=f"motor:{req.trace_id}"[:128],
        )
        if not veredicto.aceptado:
            return _rechazado(req, result.resolved_motor, f"Human gate: {veredicto.motivo.value}")

    # La ruta del freno se resuelve en CADA dispatch y ANTES de crear el job:
    # sin la variable del freno el pedido falla cerrado sin dejar un job
    # `pending` huérfano (revisión final del frente B, 2026-09-17).
    ruta_del_freno = str(ruta_del_interruptor())
    job_id = _STORE.create(
        caller=req.caller,
        capability=req.capability,
        motor=result.resolved_motor,
        trace_id=req.trace_id,
        prompt=req.prompt,
        recursion_depth=req.recursion_depth,
        pipeline_id=req.pipeline_id,
    )

    task = asyncio.create_task(
        motor_worker.run(
            job_id=job_id,
            motor=result.resolved_motor,
            capability=req.capability,
            prompt=req.prompt,
            context=req.context,
            store=_STORE,
            catalog=_CATALOG,
            kill_switch_path=ruta_del_freno,
            user_id=req.user_id,
            tenant_id=req.tenant_id,
            caller=req.caller,
            timeout_seconds=req.timeout_seconds,
            pipeline_id=req.pipeline_id,
        )
    )
    task.add_done_callback(lambda t: _log_worker_exception(t, job_id=job_id))
    job_tasks.register(job_id, task)

    return MotorDispatchResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        motor=result.resolved_motor,
        capability=req.capability,
        trace_id=req.trace_id,
    )


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
