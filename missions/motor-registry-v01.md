faceta: hyde

# Motor Registry v0.1 — Commit 1: Skeleton con frenos

## Contexto
LAS MANOS es FastAPI en ~/jax/las_manos/ (puerto 7777). Ya existe:
- server.py, audit.py, policy.py, planner.py, envelope.py
- workers/: ssh_worker.py, file_worker.py, rsync_worker.py
- config.toml con permisos por faceta

## Tarea
Construir el Motor Registry v0.1 — SOLO el skeleton con frenos. NO llamar a Kimi todavía.

### Archivos a crear:
1. ~/jax/las_manos/motor_registry/__init__.py (vacío)
2. ~/jax/las_manos/motor_registry/models.py — JobStatus enum + MotorDispatchRequest + MotorDispatchResponse + MotorJobView (Pydantic)
3. ~/jax/las_manos/motor_registry/job_store.py — JSONL append-only, indexed by job_id
4. ~/jax/las_manos/motor_registry/catalog.py — lee [motors.*] y [capabilities.*] de config.toml
5. ~/jax/las_manos/motor_registry/policy.py — valida caller autorizado, capability existe, sandbox, recursion_depth, forbidden_context_keys
6. ~/jax/las_manos/motor_registry/routes.py — tres endpoints: POST /motor/dispatch, GET /motor/job/{job_id}, POST /motor/job/{job_id}/cancel
7. ~/jax/las_manos/motor_registry/worker.py — STUB: acepta job, marca running, espera, marca completed. SIN llamar Kimi todavía.

### Agregar a config.toml de LAS MANOS:

[motors.kimi]
enabled = true
provider = "kimi"
api_key_env = "KIMI_API_KEY"
api_url = "https://api.moonshot.ai/v1/chat/completions"
model = "kimi-k2.7-code"
max_context_tokens = 256000
sandbox_only = true
default_timeout_seconds = 600
supports_reasoning = true
reasoning_default_visibility = "audit_only"

[motors.ada]
enabled = false
provider = "zhipu"
api_key_env = "ZHIPU_API_KEY"
api_url = "https://api.z.ai/api/paas/v4/chat/completions"
model = "glm-5.2"
max_context_tokens = 1000000
sandbox_only = true
default_timeout_seconds = 600
supports_reasoning = false

[capabilities.code_swarm]
allowed_motors = ["kimi"]
allowed_callers = ["hyde", "ada", "kimi"]
risk_level = "high"
sandbox_only = true
requires_human_gate = true
max_execution_minutes = 30
max_recursion_depth = 1
output_schema = "code_swarm.v1"
fallback_motor = "ada"
fallback_mode = "manual_only"
forbidden_paths = [".env", "secrets/", "private_keys/", "credentials/"]

[capabilities.refactor]
allowed_motors = ["kimi"]
allowed_callers = ["hyde", "ada"]
risk_level = "medium"
sandbox_only = true
requires_human_gate = false
max_execution_minutes = 10
max_recursion_depth = 0
output_schema = "code_patch.v1"

[capabilities.architecture_review]
allowed_motors = ["ada"]
allowed_callers = ["hyde"]
risk_level = "medium"
sandbox_only = true
requires_human_gate = false
max_execution_minutes = 5
output_schema = "architecture_review.v1"

[capabilities.bug_hunt]
allowed_motors = ["kimi"]
allowed_callers = ["hyde", "ada"]
risk_level = "high"
sandbox_only = true
requires_human_gate = true
max_execution_minutes = 15
output_schema = "bug_hunt.v1"

### Reglas absolutas:
- NO ejecutar nada en producción
- NO modificar archivos existentes de LAS MANOS (solo agregar)
- Backup de config.toml antes de modificarlo
- Cada archivo nuevo debe pasar python -m py_compile antes de continuar
- El worker.py es STUB — no llama ninguna API externa
- El sistema debe poder negarse: probar que POST /motor/dispatch rechaza caller no autorizado

### Verificación final:
1. py_compile en todos los archivos nuevos
2. grep -r "import" motor_registry/ — sin imports rotos
3. python3 -c "from motor_registry.models import JobStatus, MotorDispatchRequest" debe funcionar
4. Escribir resultado en ~/jax/missions/motor-registry-v01_result.md

En memoria de Jairo Urbina.
