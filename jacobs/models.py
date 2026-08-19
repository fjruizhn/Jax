"""
Jacobs — Modelos de datos.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PipelineStatus(str, Enum):
    pending     = "pending"
    running     = "running"
    completed   = "completed"
    failed      = "failed"
    aborted     = "aborted"
    interrupted = "interrupted"


class StepStatus(str, Enum):
    pending             = "pending"
    running             = "running"
    completed           = "completed"
    failed              = "failed"
    skipped             = "skipped"
    blocked             = "blocked"
    blocked_human_gate  = "blocked_human_gate"


VALID_FACETS = frozenset({
    "hipatia", "jekyll", "thot", "ada", "kimi", "hyde", "jax_local",
})

VALID_INVOKERS = frozenset({"Fernando", "jax_local", "ada"})

VALID_MODES = frozenset({"dry_run", "supervised", "autonomous"})


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
    mode:               str
    status:             PipelineStatus = PipelineStatus.pending
    plan:               list[Step] = Field(default_factory=list)
    plan_version:       int = 1
    current_step_index: int = 0
    max_steps:          int = 20
    context:            dict[str, Any] = Field(default_factory=dict)
    created_at:         float = 0.0
    updated_at:         float = 0.0
    # dedication: interno, no expuesto en API


class PipelineCreateRequest(BaseModel):
    name:             str
    objective:        str
    invoked_by:       str
    user_id:          str | None = None
    tenant_id:        str | None = None
    mode:             str
    max_steps:        int = 20
    steps:            list[StepSpec] | None = None
    subpipeline_token: str | None = None

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
        if self.max_steps < 1 or self.max_steps > 20:
            raise ValueError("max_steps debe estar entre 1 y 20 (límite duro v0.1)")
        return self


class StepSpec(BaseModel):
    facet:           str
    capability:      str
    prompt:          str = ""
    input:           dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 300
    skip_on_fail:    bool = False
    depends_on:      list[int] = Field(default_factory=list)


class StepResult(BaseModel):
    step_id:    str
    status:     StepStatus
    output_ref: str | None = None
    error:      str | None = None
    duration_s: float | None = None


# Evitar forward-reference con StepSpec antes de Step
PipelineCreateRequest.model_rebuild()
