"""
Jacobs — Modelos de datos.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# T2 (2026-08-21, diagnóstico pipeline 19ad2c42-cdf): single source de qué
# facets despachan por HTTP directo vs. Motor Registry de LAS MANOS. Vivía
# duplicado como _HTTP_FACETS/_MOTOR_FACETS locales en executor.py -- movido
# acá porque plan.py también lo necesita (validación pre-persist, T2) y
# plan.py no puede importar executor.py (executor.py ya importa de plan.py,
# sería circular). executor.py ahora importa estos dos nombres desde acá en
# vez de definirlos.
HTTP_FACETS = frozenset({"hipatia", "jekyll", "thot", "ada"})
MOTOR_FACETS = frozenset({"kimi", "jax_local"})


class PipelineStatus(str, Enum):
    pending     = "pending"
    running     = "running"
    completed   = "completed"
    failed      = "failed"
    aborted     = "aborted"
    interrupted = "interrupted"
    expired     = "expired"  # T4 (2026-08-19): cosechado por jacobs/reaper.py,
                              # distinto de aborted (decisión humana/API explícita)


class StepStatus(str, Enum):
    pending             = "pending"
    running             = "running"
    completed           = "completed"
    failed              = "failed"
    skipped             = "skipped"
    blocked             = "blocked"
    blocked_human_gate  = "blocked_human_gate"


# `invoked_by` es un ROL de quien pide, no el nombre de una persona (tanda A,
# 2026-09-14, decisión de Fernando). "plataforma" = pedido de jax-platform en
# nombre de un usuario autenticado; QUIÉN es viaja en user_id/tenant_id. Las
# filas viejas de jacobs_pipelines con "Fernando" quedan como están: historia.
INVOKER_PLATAFORMA = "plataforma"
# Ada solo crea sub-pipelines: presenta un subpipeline_token emitido por Jacobs
# para un padre y un paso de Ada en ejecución (jacobs/subpipelines.py).
INVOKER_ADA = "ada"
VALID_INVOKERS = frozenset({INVOKER_PLATAFORMA, "jax_local", INVOKER_ADA})

VALID_MODES = frozenset({"dry_run", "supervised", "autonomous"})

# Tope duro de steps por pipeline (E-13, 2026-09-16). Vive acá y no en
# policy.py porque policy importa models: al revés sería un import circular.
# policy.py, routes.py, plan.py y el validador de abajo lo importan de acá.
MAX_STEPS_PER_PIPELINE = 20


class Step(BaseModel):
    step_id:          str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id:      str = ""
    step_index:       int = 0
    facet:            str
    motor:            str | None = None  # R4: motor explícito, separado de facet.
                                          # None = MotorPolicy._resolve_motor() elige por competencia.
    capability:       str
    input:            dict[str, Any] = Field(default_factory=dict)
    output_ref:       str | None = None
    status:           StepStatus = StepStatus.pending
    timeout_seconds:  int = 300
    retries_allowed:  int = 0
    skip_on_fail:     bool = False
    depends_on:       list[int] = Field(default_factory=list)  # step_index de dependencias
    trace_id:         str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at:       float | None = None
    finished_at:      float | None = None
    error:            str | None = None


class Pipeline(BaseModel):
    pipeline_id:        str = Field(default_factory=lambda: str(uuid.uuid4()))
    name:               str
    orchestrator:       str = "Jacobs"
    invoked_by:         str
    user_id:            str | None = None
    tenant_id:          str | None = None
    # Ronda 5 (2026-08-20, T1): reemplaza el owner file en filesystem
    # (~/jax/pipelines/{id}_owner.json, escrito por jax-platform). None =
    # jax-platform todavia no confirmo haber recibido pipeline_id y
    # completado su propio bookkeeping (resource_manager, engine_state) --
    # mismo significado que "owner file ausente" antes, pero sin cruzar de
    # repo ni depender de que ambos servicios corran en el mismo host.
    owner_ack_at:       float | None = None
    # Frente F (2026-09-16): un hijo de Ada guarda de quién es hijo y a qué
    # profundidad. Los dos salen de la fila del token, nunca del pedido.
    parent_pipeline_id: str | None = None
    depth:              int = 0
    mode:               str
    status:             PipelineStatus = PipelineStatus.pending
    plan:               list[Step] = Field(default_factory=list)
    plan_version:       int = 1
    current_step_index: int = 0
    # Época de corrida (spec 2026-09-17 §5.3): la toma cada ejecutor al
    # arrancar; resume, approve-step y continue la INCREMENTAN. Toda escritura
    # del ejecutor es condicional a su época y a status='running': una corrida
    # superada (cancelada, vencida, continuada por otro) no escribe nada.
    run_epoch:          int = 0
    max_steps:          int = MAX_STEPS_PER_PIPELINE
    context:            dict[str, Any] = Field(default_factory=dict)
    created_at:         float = 0.0
    updated_at:         float = 0.0
    # dedication: interno, no expuesto en API


def validar_costo_max_aceptado(costo: Decimal | None) -> None:
    """Spec 2026-09-17 §6.1: costo_max_aceptado_usd no puede ser negativo.
    Compartido por PipelineCreateRequest (acá abajo) y ContinueRequest
    (jacobs/routes.py, Task 11 fix round 1) para no duplicar el umbral ni el
    mensaje en dos sitios -- una revisión encontró que ContinueRequest había
    quedado sin este guardia y un valor negativo llegaba hasta el servicio,
    donde salía como 409 costo_supera_lo_aceptado en vez de 422."""
    if costo is not None and costo < 0:
        raise ValueError("costo_max_aceptado_usd no puede ser negativo")


class PipelineCreateRequest(BaseModel):
    name:               str
    objective:          str
    invoked_by:         str
    user_id:            str | None = None
    tenant_id:          str | None = None
    mode:               str
    max_steps:          int = MAX_STEPS_PER_PIPELINE
    steps:              list[StepSpec] | None = None
    # Frente F: solo para invoked_by="ada". La profundidad NO es un campo: un
    # `depth` o `subpipeline_depth` en el JSON se ignora (nadie lo lee).
    subpipeline_token:  str | None = Field(default=None, min_length=1, max_length=128)
    parent_pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)
    # Spec 2026-09-17 §6.1: el costo que el humano confirmó en la Mesa. Si el
    # pre-vuelo interno da MÁS, Jacobs responde 409 costo_supera_lo_aceptado
    # sin crear: la condición la hace cumplir quien gasta.
    costo_max_aceptado_usd: Decimal | None = None

    # Validadores POR CAMPO, no de modelo (revisión final del frente F,
    # 2026-09-16, hallazgo I-1): un error de un `model_validator` lleva en
    # `input` el cuerpo ENTERO, y FastAPI lo devuelve en el 422 -- con el
    # `subpipeline_token` en claro. Uno por campo solo eco-ea ese campo. La
    # forma por rol (token/padre según invoked_by) NO vive acá: la decide
    # policy.validate_create, que responde 422 con `policy.reason`, sin el
    # token. Un campo requerido faltante sigue trayendo el cuerpo en `input`
    # desde Pydantic: en LAS MANOS lo tapa el handler global de
    # RequestValidationError (las_manos/server.py), que devuelve solo nombres
    # de campos.
    @field_validator("invoked_by")
    @classmethod
    def _invoked_by_valido(cls, valor: str) -> str:
        if valor not in VALID_INVOKERS:
            raise ValueError(f"invoked_by '{valor}' inválido. Aceptados: {sorted(VALID_INVOKERS)}")
        return valor

    @field_validator("mode")
    @classmethod
    def _mode_valido(cls, valor: str) -> str:
        if valor not in VALID_MODES:
            raise ValueError(f"mode '{valor}' inválido. Aceptados: {sorted(VALID_MODES)}")
        return valor

    @field_validator("max_steps")
    @classmethod
    def _max_steps_valido(cls, valor: int) -> int:
        if valor < 1 or valor > MAX_STEPS_PER_PIPELINE:
            raise ValueError(
                f"max_steps debe estar entre 1 y {MAX_STEPS_PER_PIPELINE} (límite duro v0.1)"
            )
        return valor

    # Spec 2026-09-17 §6.1. Por campo y no en un model_validator, por la misma
    # razón del bloque de arriba (hallazgo I-1 del frente F): un error de
    # model_validator devuelve el cuerpo entero en el 422, con el
    # subpipeline_token en claro.
    @field_validator("costo_max_aceptado_usd")
    @classmethod
    def _costo_max_aceptado_valido(cls, valor: Decimal | None) -> Decimal | None:
        validar_costo_max_aceptado(valor)
        return valor


class StepSpec(BaseModel):
    facet:           str
    capability:      str
    prompt:          str = ""
    input:           dict[str, Any] = Field(default_factory=dict)
    # None = sin override explicito del caller. Ronda 4 (2026-08-20, T2.a):
    # ANTES era int=300 -- un default de Pydantic, no una decision del
    # caller. routes.py convierte StepSpec a dict via model_dump(), que
    # SIEMPRE incluye los defaults; PlanBuilder._from_spec() interpretaba
    # esa presencia como "el caller pidio 300s", pisando el default por-
    # capability (hoy `capability.max_execution_minutes` en la DB; hasta
    # 2026-09-01 era un dict en plan.py) incluso cuando el caller
    # nunca toco el campo. Confirmado en produccion real: jacobs_steps
    # muestra 'reconcile' con timeout_seconds=300 en 1 de 3 corridas reales
    # pese a estar en el dict con valor 900. None deja que _from_spec()
    # distinga "ausente" de "override real" y aplique el default correcto.
    timeout_seconds: int | None = None
    skip_on_fail:    bool = False
    depends_on:      list[int] = Field(default_factory=list)


# Evitar forward-reference con StepSpec antes de Step
PipelineCreateRequest.model_rebuild()
