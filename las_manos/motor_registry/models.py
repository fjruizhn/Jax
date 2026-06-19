"""
LAS MANOS — Motor Registry: modelos de datos.

Define los tipos que circulan por el Motor Registry: JobStatus,
MotorDispatchRequest, MotorDispatchResponse, MotorJobView.

Estos modelos son la frontera entre quien pide y quien ejecuta.
Sin modelo válido, el motor no arranca.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"


class MotorDispatchRequest(BaseModel):
    caller: str                           # faceta que solicita (hyde, ada, kimi, ...)
    capability: str                       # p.ej. "code_swarm", "refactor"
    motor: str | None = None              # motor específico; None = elegir según catalog
    prompt: str                           # instrucción al motor
    context: dict[str, Any] = Field(default_factory=dict)
    recursion_depth: int = 0
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    human_gate_token: str | None = None   # requerido si capability.requires_human_gate
    sandbox: bool = True                   # True = no escribe fuera del sandbox

    model_config = {"extra": "forbid"}


class MotorDispatchResponse(BaseModel):
    job_id: str
    status: JobStatus
    motor: str
    capability: str
    trace_id: str
    rejected_reason: str | None = None    # si status == REJECTED, el motivo


class MotorJobView(BaseModel):
    job_id: str
    status: JobStatus
    motor: str
    capability: str
    caller: str
    trace_id: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result_summary: str | None = None     # hint; el resultado real va al log
