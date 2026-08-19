# R4 — Motor desacoplado de faceta: catálogo en DB

**Fecha:** 2026-08-18
**Estado:** Diseño validado en brainstorming, pendiente de plan de implementación.
**Contexto:** `docs/REFORMAS-v3.1.md`, R4 ("Motor, rol y faceta") y R3 (capacidades por contrato).

---

## 1. Motivación

El objetivo no es conectar Kimi al chat conversacional. Es que un motor se
elija **por la tarea**, no por el nombre de la faceta que lo presenta —
y que agregar un motor nuevo al catálogo sea una fila, no un commit.

**Hallazgo que origina este diseño:** el mecanismo de selección por
competencia ya existe y está sin usar. `motor_registry/policy.py::MotorPolicy._resolve_motor()`
resuelve `motor=None` iterando `capability.allowed_motors` y devuelve el
primero habilitado — pero el único caller real (`jacobs/executor.py::_invoke_motor`)
siempre manda `motor: step.facet` explícito, nunca `None`. La capacidad
estaba construida; el nombre de faceta la anulaba. Esto es R4 en una frase.

**Criterios de aceptación:**

1. Qwen (`qwen3-coder:30b`) ejecuta una tarea de código real vía Pipeline,
   con las capabilities que esa tarea requiere — no conversando desde la
   faceta `jax_local`.
2. Kimi ejecuta una tarea de razonamiento profundo vía Pipeline, por su
   transporte `motor_registry` real (ya funciona hoy; el gap es que el
   motor no se elige, se hereda de la faceta).
3. Se puede elegir qué motor atiende una tarea, o dejar que el sistema
   elija por competencia declarada, no por nombre de faceta.
4. **Criterio decisivo:** dar de alta un motor que hoy no existe en el
   catálogo — sin tocar código, vía formulario de Admin/Modelos — y que
   compita por su capability. Si esto no pasa, R4 no está hecho, sin
   importar cuánto funcionen los casos 1-3.

## 2. Fuera de alcance (explícito, con razón)

- **Comando** (pestaña de la Mesa): no pasa por Jacobs — dispara el CLI
  viejo (`jax/core/main.py`) con `facet: 'hyde'` hardcodeado en
  `jax-platform/backend/api/command.py:37`. Deuda con nombre: es un
  cuarto camino sin gobernanza, primo de `_HTTP_FACETS`. No se toca acá.
- **Unificación A7** (`ops.*` de `las_manos/policy.py` vs `capabilities.*`
  de `motor_registry`): coexisten. Cuál sobrevive se decide con datos de
  qué roles se usan, y esos datos los produce R4 corriendo — resolverlo
  antes sería orden invertido.
- **`jax/core/credential_resolver.py`**: segunda copia del resolver,
  separada de la que usa `las_manos/`. Encontrada de paso, no investigada.
  Anotada como deuda, probablemente el mismo síntoma que Comando.
- **Abstracción general del registro**: no se construye por adelantado.
  Emerge de los 4 criterios de aceptación (R5 — carga antes que
  infraestructura), no al revés.

## 3. Lo que ya existe y se reutiliza (auditado contra schema real, no supuesto)

| Necesito | Ya existe en | Nota |
|---|---|---|
| Auth por proveedor (`api_key`/`none`/`subprocess`) | `provider.auth_type` | Fila `ollama → 'none', is_local=1` ya sembrada — no hace falta crearla |
| Guard para no exigir credencial en transportes sin auth | `jax-platform/backend/facet_resolver.py:81-82` | `if transport not in ("ollama","subprocess"): credential = await resolve_credential_instrumented(...)` — replicar tal cual en `worker.py`, no rediseñar |
| Catálogo de modelos (context window, pricing, deprecación, digest) | tabla `model`, FK `provider_id` | `model.id=1` ya es `ollama/qwen3-coder:30b` — cero dato nuevo que sembrar para Qwen |
| Enum de transporte | `facet.transport` | Mismo set de valores (`http_openai_compat`, `http_gemini`, `motor_registry`, `ollama`, `subprocess`) se reusa en `motor.transport` |
| Patrón de dispatch por transporte | `chat.py::_invoke_facet` | `worker.py` hoy tiene una sola función hardcodeada (`_call_kimi`) en vez de este patrón — se generaliza, no se inventa uno nuevo |
| Patrón binding→modelo con FK | `facet_binding.model_ref → model.id` | Mismo patrón para `motor.model_ref` |

**Lo que se elimina como consecuencia** (no es limpieza aparte, es lo que
el movimiento a DB hace innecesario):
- `_MOTOR_PROVIDER_MAP` hardcodeado en `worker.py` (`{"kimi":"moonshot","ada":"zhipu"}`)
  → `motor.model_ref → model.provider_id`, FK real.
- `[motors.X].provider/api_key_env/api_url/model/max_context_tokens` en
  `config.toml` → `provider.base_url` + `credential` + `model.*` vía
  `model_ref`.

## 4. Esquema nuevo

Migración única en `jax-platform/backend/db/migrations.py` (ya es dueño
de `provider`/`model`/`facet`/`facet_binding` en esta misma DB
compartida — dos migradores sobre la misma base es cómo hoy existen dos
puntos de conexión y dos resolvers de credenciales, error que no se
repite acá). `~/jax/las_manos/motor_registry/catalog.py` lee estas
tablas en runtime — leer no es ser dueño del schema, mismo principio que
`facet_resolver.py` ya aplica del lado de la Mesa.

```sql
CREATE TABLE motor (
  `key` VARCHAR(50) PRIMARY KEY,
  model_ref INT NOT NULL,
  transport ENUM('http_openai_compat','http_gemini','motor_registry','ollama','subprocess') NOT NULL,
  max_tokens INT NULL,
  default_timeout_seconds INT NOT NULL DEFAULT 600,
  supports_reasoning BOOLEAN NOT NULL DEFAULT FALSE,
  reasoning_default_visibility ENUM('audit_only','visible') DEFAULT 'audit_only',
  sandbox_only BOOLEAN NOT NULL DEFAULT TRUE,
  status ENUM('active','disabled') NOT NULL DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (model_ref) REFERENCES model(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE capability (
  `key` VARCHAR(50) PRIMARY KEY,
  risk_level ENUM('low','medium','high') NOT NULL,
  sandbox_only BOOLEAN NOT NULL DEFAULT TRUE,
  requires_human_gate BOOLEAN NOT NULL DEFAULT FALSE,
  max_execution_minutes INT NOT NULL,
  max_recursion_depth INT NOT NULL DEFAULT 0,
  output_schema VARCHAR(100) NULL,
  fallback_motor VARCHAR(50) NULL,
  fallback_mode ENUM('manual_only','auto') NULL,
  allowed_callers JSON NOT NULL,
  forbidden_paths JSON NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (fallback_motor) REFERENCES motor(`key`),
  CONSTRAINT chk_allowed_callers CHECK (JSON_VALID(allowed_callers))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE capability_motor (
  capability_key VARCHAR(50) NOT NULL,
  motor_key VARCHAR(50) NOT NULL,
  priority INT NOT NULL DEFAULT 0,
  PRIMARY KEY (capability_key, motor_key),
  FOREIGN KEY (capability_key) REFERENCES capability(`key`),
  FOREIGN KEY (motor_key) REFERENCES motor(`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

`priority` reemplaza el orden implícito de la lista `allowed_motors` de
TOML — `_resolve_motor()` hoy toma "el primero habilitado"; con DB, ese
orden es explícito y editable sin reordenar un archivo. Convención:
**menor `priority` gana primero** (0 = primer intento), mismo sentido
que "el primero de la lista" en TOML — `_resolve_motor()` pasa a iterar
`ORDER BY priority ASC` en vez del orden de aparición en el archivo.

**Seed de la migración** (los datos reales de `config.toml`, portados,
no reinventados): `motor` para `kimi`/`ada` (`model_ref` resuelto contra
las filas `model` ya existentes de moonshot/zhipu — a verificar en el
plan si ya están sembradas o hace falta un `INSERT` puntual), `capability`
para las 6+ definidas hoy (`code_swarm`, `implementation`,
`architecture_review`, `bug_hunt`, `pipeline_analysis`, y las que falten
por relevar), y `capability_motor` desde cada `allowed_motors`.

## 5. Componentes a cambiar

1. **Migración** (`jax-platform/backend/db/migrations.py`): las 3 tablas
   + seed desde `config.toml` actual.
2. **`~/jax/las_manos/motor_registry/catalog.py`**: lee las 3 tablas en
   vez de `[motors.*]`/`[capabilities.*]` de `config.toml`. Mismo pool
   de conexión que `credential_resolver.py` ya usa contra esta DB.
3. **`~/jax/las_manos/motor_registry/worker.py`**: `_call_kimi` se
   generaliza a dispatch por `motor.transport` (formas ya escritas en
   `chat.py`, adaptadas al loop de kill-switch existente). Antes de
   resolver credencial: guard `if motor.transport not in ("ollama",
   "subprocess")`, igual que `facet_resolver.py:81-82`.
4. **Alta de Qwen**: fila `motor` con `key='jax_local'`, `model_ref=1`
   (ya existe: `ollama/qwen3-coder:30b`), `transport='ollama'`. Sin dato
   nuevo de modelo que sembrar.
5. **`~/jax/jacobs/executor.py`**: `step.motor` (nuevo campo opcional,
   separado de `step.facet`). Si viene, se pasa tal cual a
   `_invoke_motor`. Si no, se manda `motor=None` — activa
   `_resolve_motor()`. `jax_local` se suma al conjunto gobernado (hoy
   `_MOTOR_FACETS={"kimi"}`) para pasar por el job-queue en vez de
   `_invoke_ollama` directo.
6. **`jax-platform/frontend/.../PipelineModal.jsx`** + endpoint
   `pipelines.py`: reemplazar el checkbox-de-faceta-con-capability-fija
   por elegir capability (tipo de tarea) y, opcional, motor explícito —
   vacío = auto, resuelto por competencia vía `capability_motor`.
7. **Admin/Modelos — form de alta de motor** (ÚLTIMA tarea del plan,
   deliberado): CRUD sobre `motor`/`capability`/`capability_motor`,
   reusando el patrón de admin ya construido en Bloque D para
   provider/model. Va al final porque cablea algo ya verificado por
   INSERT directo, no diseña a ciegas — el criterio de aceptación #4 se
   prueba primero por INSERT/migración de seed, y el form es la
   consecuencia, no la apuesta.

## 6. Testing

- Migración: test de idempotencia (mismo patrón que las tablas
  existentes — `_table_exists`/`_column_exists` guards).
- `catalog.py` contra DB: reemplazar los tests actuales que asumen
  `config.toml` por fixtures de DB (mismo patrón que
  `test_facet_bindings.py` ya prueba del lado de la Mesa).
- `worker.py`: extender `_worker_max_tokens_test.py` (ya existe, mockea
  `httpx.AsyncClient.post` y credencial) para cubrir el dispatch por
  transporte y el guard de credencial-opcional con `transport='ollama'`.
- E2E real (no mockeado): un job de Qwen por `code_swarm` o
  `implementation` completando con `finish_reason` sano (mismo runbook
  de verificación que `docs/runbooks/verificar-truncamiento-kimi.md` ya
  documenta), y un job de Kimi vía Pipeline con motor elegido
  explícitamente vs. `None` (auto).
- Criterio #4: motor nuevo dado de alta por INSERT directo (antes del
  form) compite y se despacha — sin este test pasando, no se construye
  el form. **Debe ser un motor real que responda, no un stub que solo
  prueba el parseo del catálogo.** No hace falta credencial nueva:
  `provider`/`credential` ya tienen filas activas para `openai`,
  `anthropic` y `gemini` (las usa Thot/Hyde/Hipatia del lado de la
  Mesa) — ninguno de los tres existe hoy como `motor` en
  `motor_registry` (solo `kimi`/`ada`). Registrar cualquiera de esos
  tres como `motor` nuevo + `capability_motor` prueba el camino
  completo con infraestructura ya viva, sin setup adicional.
- **Orden estricto:** la tarea del form de Admin (§5.7) no puede
  empezar hasta que el criterio #4 pase por INSERT directo. Si el plan
  la secuencia antes, se está diseñando UI contra un backend sin
  verificar.

## 7. Riesgos / decisiones abiertas para el plan

- `capability_motor` para Qwen: ¿qué capabilities gana desde el día uno
  (`implementation`/`refactor` junto a Kimi) vs. cuáles quedan
  exclusivas de Kimi por ahora? Es una decisión de dato (seed), no de
  código — se puede ajustar sin tocar nada una vez que el mecanismo
  funciona.
- `jax_local` sumado a `_MOTOR_FACETS` cambia su camino de ejecución
  (job-queue gobernado vs. `_invoke_ollama` directo) — verificar que
  `GPU_SEMAPHORE=1` (una sola inferencia Ollama a la vez, regla crítica
  de `~/jax/CLAUDE.md`) se sigue respetando dentro del job-queue.
