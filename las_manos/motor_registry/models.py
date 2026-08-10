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
    # TRUST BOUNDARY (2026-08-10): user_id/tenant_id llegan del body del
    # request SIN verificacion independiente en este endpoint -- la confianza
    # descansa por completo en que el caller (Jacobs) ya los valido antes de
    # despachar (jax-platform Task 4) y en que jax-las-manos.service solo
    # bindea a 127.0.0.1. Este repo ya tuvo IDOR real por asumir el limite de
    # confianza incorrecto (ver CONTEXT.md 2026-08-08) -- si este puerto se
    # expone alguna vez publicamente, esto se vuelve un IDOR inmediato.
    user_id:    str | None = None
    tenant_id:  str | None = None

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
