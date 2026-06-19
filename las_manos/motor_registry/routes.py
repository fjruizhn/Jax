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
import time
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
_CATALOG = MotorCatalog(_CONFIG)
_POLICY = MotorPolicy(_CATALOG)
_KILL_SWITCH_PATH: str = _CONFIG.get("server", {}).get("kill_switch_path", "/etc/jax/PAUSE")

router = APIRouter(prefix="/motor", tags=["motor_registry"])


@router.post("/dispatch", response_model=MotorDispatchResponse, status_code=202)
async def dispatch(req: MotorDispatchRequest) -> MotorDispatchResponse:
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

    asyncio.create_task(
        motor_worker.run(
            job_id=job_id,
            motor=result.resolved_motor,
            capability=req.capability,
            prompt=req.prompt,
            context=req.context,
            store=_STORE,
            catalog=_CATALOG,
            kill_switch_path=_KILL_SWITCH_PATH,
        )
    )

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
