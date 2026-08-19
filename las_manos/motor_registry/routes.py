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

import tomllib
from fastapi import APIRouter, HTTPException

from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore
from motor_registry.models import (
    JobStatus,
    MotorDispatchRequest,
    MotorDispatchResponse,
    MotorJobView,
)
from motor_registry.policy import MotorPolicy
from motor_registry import worker as motor_worker

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.toml"

with open(CONFIG_PATH, "rb") as _f:
    _CONFIG = tomllib.load(_f)

_STORE = JobStore(str(BASE_DIR / "logs" / "motor_jobs.jsonl"))
# _CATALOG/_POLICY arrancan None -- se pueblan en el startup hook de
# server.py (init_motor_catalog, abajo). [motors.*]/[capabilities.*] de
# config.toml ya no se leen (R4 -- catalogo en DB). Ningun otro modulo
# importa estos dos nombres directamente (verificado: grep -rn "_CATALOG"
# solo los usa este archivo), asi que reasignarlos acá es seguro.
_CATALOG: MotorCatalog | None = None
_POLICY: MotorPolicy | None = None
_KILL_SWITCH_PATH: str = _CONFIG.get("server", {}).get("kill_switch_path", "/etc/jax/PAUSE")

logger = logging.getLogger(__name__)


async def init_motor_catalog() -> None:
    """Llamado desde el startup hook de server.py. Falla cerrado: si la DB
    no responde al arrancar, _CATALOG queda None y cada dispatch rechaza
    explícito (ver check en dispatch abajo) en vez de arrancar con un
    catálogo vacío en silencio.

    LIMITACIÓN CONOCIDA (no es bug, fuera de alcance de este fix wave): este
    catálogo se carga UNA sola vez, acá, al arrancar el proceso. Un motor
    nuevo dado de alta vía el form de admin (Task 9) queda en la DB pero NO
    lo ve este proceso hasta reiniciar jax-las-manos.service — no hay
    endpoint de reload todavía."""
    global _CATALOG, _POLICY
    _CATALOG = await MotorCatalog.from_db()
    _POLICY = MotorPolicy(_CATALOG)

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


@router.post("/dispatch", response_model=MotorDispatchResponse, status_code=202)
async def dispatch(req: MotorDispatchRequest) -> MotorDispatchResponse:
    if _POLICY is None or _CATALOG is None:
        raise HTTPException(status_code=503, detail="Motor Registry: catálogo no inicializado todavía")

    result = _POLICY.check(
        caller=req.caller,
        capability=req.capability,
        motor=req.motor,
        context_keys=list(req.context.keys()),
        recursion_depth=req.recursion_depth,
        human_gate_token=req.human_gate_token,
    )

    if not result.allowed:
        job_id = _STORE.create(
            caller=req.caller,
            capability=req.capability,
            motor=result.resolved_motor or "none",
            trace_id=req.trace_id,
            prompt=req.prompt,
            recursion_depth=req.recursion_depth,
        )
        _STORE.update(
            job_id,
            status=JobStatus.REJECTED.value,
            finished_at=time.time(),
            error=result.reason,
        )
        return MotorDispatchResponse(
            job_id=job_id,
            status=JobStatus.REJECTED,
            motor="none",
            capability=req.capability,
            trace_id=req.trace_id,
            rejected_reason=result.reason,
        )

    job_id = _STORE.create(
        caller=req.caller,
        capability=req.capability,
        motor=result.resolved_motor,
        trace_id=req.trace_id,
        prompt=req.prompt,
        recursion_depth=req.recursion_depth,
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
            kill_switch_path=_KILL_SWITCH_PATH,
            user_id=req.user_id,
            tenant_id=req.tenant_id,
            caller=req.caller,
        )
    )
    task.add_done_callback(lambda t: _log_worker_exception(t, job_id=job_id))

    return MotorDispatchResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        motor=result.resolved_motor,
        capability=req.capability,
        trace_id=req.trace_id,
    )


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
    return _STORE.get(job_id)
