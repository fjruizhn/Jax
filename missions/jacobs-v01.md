faceta: hyde

# Jacobs v0.1 — El Director

__module_name__ = "Jacobs"
__role__ = "Director ejecutivo de pipelines multi-faceta"
__dedication__ = "Prof. Raúl Jacobs — maestro, mentor, director."
__version__ = "0.1.0"

## Contexto
JAX tiene 7 facetas operativas y LAS MANOS (FastAPI 127.0.0.1:7777) como sistema nervioso inhibitorio.
Jacobs es el sistema nervioso ejecutivo — módulo de orquestación puro, sin LLM propio.
Vive en ~/jax/jacobs/ como módulo independiente que se integra a LAS MANOS.

## Estructura a crear

~/jax/jacobs/
  __init__.py
  models.py       — PipelineStatus, StepStatus, Pipeline, Step, PipelineCreateRequest, StepResult
  plan.py         — PlanBuilder: genera plan desde objetivo usando JAX Local
  executor.py     — StepExecutor: ejecuta un step invocando la faceta correcta
  store.py        — PipelineStore: MariaDB para pipelines y steps
  policy.py       — JacobsPolicy: valida invocador, límites, candados
  routes.py       — endpoints FastAPI
  artifacts.py    — guarda outputs >1MB en ~/jax/jacobs/artifacts/

## Modelos (models.py)

PipelineStatus enum: pending, running, completed, failed, aborted, interrupted
StepStatus enum: pending, running, completed, failed, skipped, blocked

Pipeline fields:
  pipeline_id: str (uuid)
  name: str
  orchestrator: str = "Jacobs"
  invoked_by: str  # Fernando|jax_local|ada
  mode: str  # dry_run|supervised|autonomous
  status: PipelineStatus
  plan: list[Step]
  plan_version: int = 1
  current_step_index: int = 0
  max_steps: int = 20
  context: dict  # outputs de steps anteriores (refs, no contenido)
  created_at: float
  updated_at: float
  dedication: str = "Prof. Raúl Jacobs — maestro, mentor, director."

Step fields:
  step_id: str (uuid)
  pipeline_id: str
  step_index: int
  facet: str  # hipatia|jekyll|thot|ada|kimi|hyde|jax_local
  capability: str
  input: dict
  output_ref: str | None  # artifact://jacobs/{pipeline_id}/{step_id}/output.json
  status: StepStatus
  timeout_seconds: int = 300
  retries_allowed: int = 0
  skip_on_fail: bool = False
  trace_id: str (uuid)
  started_at: float | None
  finished_at: float | None
  error: str | None

## Endpoints (routes.py)

POST /jacobs/plan              — genera plan sin ejecutar (dry_run)
POST /jacobs/pipeline          — crea y ejecuta pipeline
GET  /jacobs/pipeline/{id}     — consulta estado
POST /jacobs/pipeline/{id}/cancel
POST /jacobs/pipeline/{id}/resume   — solo para status=interrupted, requiere invoked_by=Fernando
GET  /jacobs/pipeline/{id}/events   — lista de audit events del pipeline

## Policy (policy.py)

Candados NO diferibles:
- invoked_by solo acepta: Fernando, jax_local, ada (con subpipeline_token)
- max_steps_per_pipeline = 20 (duro, no configurable en v0.1)
- max_parallel_pipelines = 3 (duro)
- max_subpipeline_depth = 1
- no auto_recovery post-reinicio
- kill switch revisado antes de CADA step
- outputs >1MB → artifact_ref, no en context JSON
- Hyde siempre requiere human_gate = true
- modo autonomous NO disponible en v0.1 (rechazar con error claro)

## Executor (executor.py)

Para cada step:
1. Revisar kill switch — si activo: step → failed, pipeline → aborted
2. Marcar step → running, audit event
3. Calcular deadline (timeout_seconds)
4. Invocar faceta según step.facet:
   - hipatia/jekyll/thot/jax_local: HttpMuscle o OllamaMuscle de JAX
   - kimi/ada: POST /motor/dispatch en LAS MANOS
   - hyde: POST /execute en LAS MANOS con human_gate=True
5. Si output > 1MB → guardar en artifacts/, poner ref en step.output_ref
6. Si output <= 1MB → poner en step.output_ref como inline JSON
7. Marcar step → completed, audit event
8. Si falla: step → failed, audit event
   - Si skip_on_fail=True: continuar
   - Si skip_on_fail=False: pipeline → aborted

## Store (store.py)

Usar MariaDB (mismas credenciales que jax_memory: JAX_DB_HOST, JAX_DB_USER, JAX_DB_PASSWORD).
Base de datos: jax_memory (misma base, tablas nuevas).

Tablas:
CREATE TABLE IF NOT EXISTS jacobs_pipelines (
  pipeline_id VARCHAR(36) PRIMARY KEY,
  name TEXT NOT NULL,
  invoked_by VARCHAR(50) NOT NULL,
  mode VARCHAR(20) NOT NULL,
  status VARCHAR(20) NOT NULL,
  plan JSON,
  current_step_index INT DEFAULT 0,
  max_steps INT DEFAULT 20,
  context_refs JSON,
  created_at DOUBLE NOT NULL,
  updated_at DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS jacobs_steps (
  step_id VARCHAR(36) PRIMARY KEY,
  pipeline_id VARCHAR(36) NOT NULL,
  step_index INT NOT NULL,
  facet VARCHAR(30) NOT NULL,
  capability VARCHAR(50) NOT NULL,
  input_ref TEXT,
  output_ref TEXT,
  status VARCHAR(20) NOT NULL,
  timeout_seconds INT DEFAULT 300,
  retries_allowed INT DEFAULT 0,
  skip_on_fail BOOLEAN DEFAULT FALSE,
  trace_id VARCHAR(36),
  started_at DOUBLE,
  finished_at DOUBLE,
  error TEXT
);

## Integración con LAS MANOS

Agregar en ~/jax/las_manos/server.py:
from jacobs.routes import router as jacobs_router
app.include_router(jacobs_router)

IMPORTANTE: jacobs/ debe estar en el PYTHONPATH de LAS MANOS.
Copiar o symlink ~/jax/jacobs/ → ~/jax/las_manos/jacobs/
O agregar ~/jax al sys.path en server.py.

## Plan de prueba obligatorio

1. POST /jacobs/plan con objetivo simple:
   {"name": "test", "objective": "investiga qué es HAMMURABI", "invoked_by": "Fernando", "mode": "dry_run"}
   → debe devolver plan con steps sin ejecutar

2. Verificar que modo autonomous rechaza con error claro

3. Verificar que max_steps=21 rechaza

4. Verificar kill switch: touch /etc/jax/PAUSE → pipeline debe abortar en step 1

5. POST /jacobs/pipeline en modo supervised con 2 steps simples (jax_local → jax_local)
   → primer step ejecuta, segundo step queda blocked esperando aprobación

## Verificaciones obligatorias
- py_compile en todos los archivos con ~/jax/las_manos/.venv/bin/python
- curl /health → alive
- curl /jacobs/plan → plan válido sin ejecución
- Todas las pruebas del plan de prueba

## NO tocar
- audit.py, policy.py, envelope.py de LAS MANOS
- motor_registry/ (ya funciona)
- config.toml de las_manos (solo agregar si es necesario)

Escribir resultado en ~/jax/missions/jacobs-v01_result.md

En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.
