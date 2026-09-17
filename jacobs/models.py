"""
Jacobs — Modelos de datos.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    max_steps:          int = MAX_STEPS_PER_PIPELINE
    context:            dict[str, Any] = Field(default_factory=dict)
    created_at:         float = 0.0
    updated_at:         float = 0.0
    # dedication: interno, no expuesto en API


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

    @model_validator(mode="after")
    def validate_fields(self) -> "PipelineCreateRequest":
        if self.invoked_by not in VALID_INVOKERS:
            raise ValueError(
                f"invoked_by '{self.invoked_by}' inválido. Aceptados: {sorted(VALID_INVOKERS)}"
            )
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode '{self.mode}' inválido. Aceptados: {sorted(VALID_MODES)}"
            )
        if self.max_steps < 1 or self.max_steps > MAX_STEPS_PER_PIPELINE:
            raise ValueError(
                f"max_steps debe estar entre 1 y {MAX_STEPS_PER_PIPELINE} (límite duro v0.1)"
            )
        trae_contrato = self.subpipeline_token is not None or self.parent_pipeline_id is not None
        if self.invoked_by != INVOKER_ADA and trae_contrato:
            raise ValueError(
                "subpipeline_token y parent_pipeline_id solo los acepta invoked_by='ada'"
            )
        if self.invoked_by == INVOKER_ADA and (
            self.subpipeline_token is None or self.parent_pipeline_id is None
        ):
            raise ValueError("invoked_by='ada' exige subpipeline_token y parent_pipeline_id")
        return self


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
