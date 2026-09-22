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
    CANCELLING = "cancelling"  # 2026-09-21 (N-3, endpoint de Procesamiento
    # de Archivos): un pedido de cancelación no puede matar un hilo del SO
    # en vuelo (Python no lo permite) -- este estado es honesto sobre eso:
    # "se pidió, se dejó de programar trabajo nuevo, lo que ya corría sigue
    # hasta que termina solo". Sólo CANCELLED (arriba) es terminal.
    REJECTED  = "rejected"
    TOOLS_REQUESTED = "tools_requested"  # GAP2 Fase1 (2026-08-19): el modelo
    # pidió tool_calls y Fase 1 no ejecuta nada -- NO es completed (el
    # trabajo no terminó) ni failed (no hubo error), es un estado propio
    # que dice la verdad: hubo un pedido de herramientas sin atender.


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
    # GAP2 Fase3 (2026-08-19): presupuesto de tiempo del bucle de
    # tool-calling en worker.py -- REUSA step.timeout_seconds (jacobs),
    # nunca un timeout nuevo por turno. None = sin presupuesto extra (solo
    # el timeout por-llamada de motor.default_timeout_seconds), compat con
    # cualquier caller que no lo mande.
    timeout_seconds: int | None = None
    # Task 7b (2026-09-18, historial-y-arreglos-de-pipeline): el pipeline de
    # Jacobs que despacha este job (Pipeline.pipeline_id, jacobs/models.py),
    # para que record_motor_usage() lo escriba en axioma_usage.pipeline_id y
    # el historial pueda sumar el costo real por pipeline. None = quien
    # despacha no viene de un pipeline (un caller directo contra este
    # endpoint, sin Jacobs de por medio) -- válido, no un hueco.
    pipeline_id: str | None = None
    # Block 6: direct dispatch is never a governed authority boundary.  The
    # governed adapter supplies these references after durable validation.
    execution_id: str | None = None
    decision_id: str | None = None
    execution_request_hash: str | None = None
    authorization_id: str | None = None
    execution_authorization_hash: str | None = None

    model_config = {"extra": "forbid"}


class MotorDispatchResponse(BaseModel):
    job_id: str
    status: JobStatus
    motor: str
    capability: str
    trace_id: str
    rejected_reason: str | None = None    # si status == REJECTED, el motivo


class GovernedDispatchRequest(BaseModel):
    """Only an authoritative execution identity crosses this endpoint."""
    execution_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_config = {"extra": "forbid"}


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
    result_summary: str | None = None     # primeros 200 caracteres
    # Salida completa (2026-09-12): antes no se guardaba en ningún lado y los
    # pasos siguientes de Jacobs recibían solo result_summary.
    result_path: str | None = None
    # Ronda de arreglo 1 de Task 1 (2026-09-18, historial-y-arreglos-de-
    # pipeline): el model_id REAL que despachó este job (motor_entry.model
    # en worker.py, el mismo que va en el payload a la API -- worker.py:171
    # y worker.py:725). 'motor' de arriba es el NOMBRE del motor
    # (kimi/jax_local), no el modelo; sin este campo, Jacobs no tenía forma
    # de saber qué modelo corrió un paso de motor sin inventarlo. None
    # hasta que worker.py resuelve motor_entry con éxito -- un job que
    # falla ANTES de eso (motor inexistente, transporte no soportado)
    # nunca llegó a saber qué modelo iba a usar.
    model: str | None = None
    # Task 7b (2026-09-18, historial-y-arreglos-de-pipeline): el mismo
    # pipeline_id que llegó en MotorDispatchRequest, expuesto acá por
    # completitud (así el job se puede leer también por pipeline desde
    # afuera) -- sin este campo, JobStore.get() lo filtraría en silencio
    # igual que filtraba `model` antes de la ronda de arreglo 1 de Task 1.
    pipeline_id: str | None = None


class FacetAuthorizeRequest(BaseModel):
    caller: str
    facet: str


class FacetAuthorizeResponse(BaseModel):
    allowed: bool
    reason: str
