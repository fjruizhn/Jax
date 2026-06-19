faceta: hyde

# Motor Registry v0.2 — Commit 2: Kimi real + sandbox + output validator

## Contexto
Motor Registry v0.1 está operativo en LAS MANOS (127.0.0.1:7777).
Archivos en ~/jax/las_manos/motor_registry/
Todos los archivos existentes pasan py_compile con .venv de las_manos.
El worker.py es STUB — no llama ninguna API externa todavía.

## Tareas

### 1. Agregar campo sandbox a MotorDispatchRequest (models.py)
Agregar:
    sandbox: bool = True

### 2. Conectar Kimi real en worker.py
- Leer KIMI_API_KEY de os.environ
- Leer api_url y model del catalog (motors.kimi)
- Llamar https://api.moonshot.ai/v1/chat/completions con httpx async
- Separar content y reasoning_content (Kimi siempre devuelve ambos)
- reasoning_content → loguear en job_store como metadata, NO devolver al caller
- content → result_summary del job
- Timeout: usar default_timeout_seconds del catalog (600s)
- Si falla la API → job status = failed, error = mensaje humanizado
- Kill switch: revisar /etc/jax/PAUSE antes de llamar y durante (cada 5s con asyncio)

### 3. Crear output_validator.py
Validador flexible para estos schemas:
- code_swarm.v1: requiere plan, steps, patches, tests, risk_notes, human_review_needed
- code_patch.v1: requiere diff, files_modified, description
- architecture_review.v1: requiere summary, risks, recommendations
- bug_hunt.v1: requiere bugs_found, severity, reproduction_steps, suggested_fix

Si el motor devuelve JSON válido con los campos → validated = true
Si devuelve texto libre → intentar parsear con json.loads, si falla → validated = false, raw guardado en audit
Nunca fallar cerrado por schema — schema inválido = warning, no error fatal

### 4. Integrar output_validator en worker.py
Después de recibir content de Kimi:
- Intentar validar contra output_schema de la capability
- Guardar resultado validado o raw en job_store
- result_summary = primeras 200 chars del content si no es JSON

### 5. Verificaciones obligatorias
- py_compile en todos los archivos modificados con .venv/bin/python
- Probar dispatch con caller="hyde", capability="refactor", prompt="refactoriza el archivo de prueba"
- Verificar que job llega a status=completed con result_summary real de Kimi
- Verificar que reasoning_content NO aparece en GET /motor/job/{id}
- Verificar kill switch: crear /etc/jax/PAUSE antes del dispatch → job debe fallar con killed_by_switch

### 6. NO tocar
- server.py (ya conectado)
- config.toml (ya tiene motors.kimi)
- audit.py, policy.py, envelope.py de LAS MANOS principal

Escribir resultado en ~/jax/missions/motor-registry-v02_result.md

En memoria de Jairo Urbina.
