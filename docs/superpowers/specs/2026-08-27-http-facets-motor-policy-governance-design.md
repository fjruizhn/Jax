# `_HTTP_FACETS` sin gobernanza del Motor Registry — diseño

En memoria de Jairo Urbina.

## Problema

`hipatia`/`jekyll`/`thot`/`ada` despachan HTTP directo desde dos caminos —
`jacobs/executor.py` (pipelines) y `jax-platform/backend/api/chat.py::_invoke_facet`
(Mesa web) — sin pasar nunca por `MotorPolicy.check()`. De los 8 checks que
sí protegen a `kimi`/`jax_local`, solo el NIVEL A (existencia de la
capability en la DB, Bloque 3 2026-08-21) aplica a estos 4 facets. Los otros
7 — `allowed_callers`, `requires_human_gate`, `recursion_depth`, claves
prohibidas, resolución de motor, `sandbox_only`, techo de timeout — nunca
corrieron para ninguno de los dos caminos.

Alcance de esta ronda: **ambos caminos** (Jacobs y Mesa web). Cerrar solo
uno deja el mismo gap estructural en el otro — precedente reciente:
`motor.model_ref` volvió 5 días después de "cerrado" con dos de tres
caminos protegidos y el tercero sin guard.

## Investigación previa (evidencia, no inferencia)

- Los checks 2-5 y 8 de `MotorPolicy.check()` (`las_manos/motor_registry/policy.py:57-144`)
  **nunca corren para `kimi`/`jax_local` dentro de Jacobs tampoco** — corren
  server-side, dentro de `las_manos`, solo cuando `_invoke_motor` llama a
  `POST /motor/dispatch`. Jacobs solo pre-valida NIVEL A/B (existencia +
  `allowed_motors`/`allowed_callers` parcial) antes de ese HTTP.
- `Jacobs corre DENTRO del proceso de `las_manos`**: `jax-las-manos.service`
  arranca `uvicorn server:app` con `WorkingDirectory=/home/fruiz/jax/las_manos`,
  y `las_manos/jacobs` es un symlink real a `/home/fruiz/jax/jacobs` — mismo
  intérprete, mismo proceso. Jacobs puede importar `motor_registry.policy`/
  `motor_registry.catalog` directo, sin red.
- `capability.sandbox_only` (5/5 filas relevantes en `1`) no tiene NINGÚN
  lector en el código (`grep -rn "cap.sandbox_only\|capability.sandbox_only\|entry\[.sandbox_only.\]\|entry.get(.sandbox_only"` → 0 resultados). El único
  `sandbox_only` que se enforce es `motor.sandbox_only` (`policy.py:116-123`,
  vía `MotorCatalog.get_motor()`), una columna de la tabla `motor`, donde
  hipatia/jekyll/thot/ada no tienen fila — no son motores en este esquema.
- `capability.max_execution_minutes` no gobierna nada hoy, ni para
  `kimi`/`jax_local` del lado de Jacobs: `step.timeout_seconds` sale de
  `_CAPABILITY_TIMEOUT_SECONDS` (dict hardcodeado, `jacobs/plan.py:106-110`),
  y un `timeout_seconds` explícito en el spec lo pisa sin validar contra el
  techo (`jacobs/plan.py:390/400`, `_validate_plan_capabilities` no lo
  chequea — su propio docstring dice que solo aplica a `_MOTOR_FACETS`, y
  ni así valida timeout). El check 8 real solo corre server-side en
  `las_manos`, solo para `kimi`/`jax_local`.
- `MotorDispatchRequest` (`las_manos/motor_registry/models.py:34-43`) usa
  `context={}`, `recursion_depth=0`, `human_gate_token=None` como default —
  y el payload que Jacobs manda hoy para `kimi`/`jax_local` no los incluye,
  así que esos son los valores reales que ya recibe `MotorPolicy.check()`
  para el camino existente.
- `jax-platform` ya llama a `las_manos:7777` por HTTP para otras cosas
  (`JACOBS_URL`, `/health` en `admin/dashboard.py`) — agregar un endpoint
  más no es un patrón de integración nuevo.

## Decisiones ya tomadas (no reabrir sin evidencia nueva)

1. **`capability.sandbox_only` → vestigial esta ronda.** No se le inventa
   semántica (el candidato obvio, egress de red, es un ítem que ya se
   difirió a propósito). Se anota en `DEUDA.md` y en un comentario de schema
   como "columna sin lector, verificado por grep, no confiar en su valor" —
   no se borra todavía.
2. **Techo de timeout (`max_execution_minutes`) → NO entra esta ronda.**
   Motivo estructural: activarlo ahora significaría validar contra
   `capability.max_execution_minutes` mientras `plan.py` sigue despachando
   con el dict hardcodeado — dos valores que pueden divergir, reconciliados
   solo por un script manual (`scripts/check_timeout_consistency.py`). Sería
   un check que afirma imponer un techo que el ejecutor real no usa. El
   ceiling y la deduplicación del dict se resuelven juntos, en una ronda
   aparte, con `plan.py` leyendo `capability.max_execution_minutes` en vivo
   para el default Y para validar el override.
3. **Approach elegido: C** — partir `MotorPolicy.check()` en dos funciones
   reusables en vez de reimplementar los checks en cada call site (approach
   B, que hubiera creado una tercera implementación de la misma regla) o
   migrar los 4 facets al Motor Registry completo (approach A, que cambia el
   modelo de despacho síncrono de la Mesa web sin necesidad).

## Arquitectura

### 1. `las_manos/motor_registry/policy.py` — split, sin cambiar comportamiento

Nueva función `check_capability_admission(caller, capability, context_keys,
recursion_depth, human_gate_token)` — firma SIN `motor` ni `timeout_seconds`,
porque ninguno de los checks que cubre los necesita. Cubre los checks 1-5
(capability existe, caller autorizado, human gate, recursion depth, claves
prohibidas). El check 8 (techo de timeout) NO entra en esta función por la
decisión del punto 2 — se queda exclusivamente en `check()`, sin cambios,
para preservar el comportamiento existente de `kimi`/`jax_local`.

`check()` (la función pública actual, firma sin tocar) pasa a ser un
wrapper: llama `check_capability_admission()` primero (checks 1-5), después
6-7 (resolver motor + `motor.sandbox_only`, sin tocar) y por último 8
(techo, sin tocar) — mismo orden exacto de evaluación que hoy, "al primer
fallo". **Cero cambio de comportamiento para `kimi`/`jax_local`** — mismo
`MotorPolicyResult`, mismos mensajes de rechazo. Requisito 1 (abajo) exige
que la suite existente pase sin modificarla.

### 2. Jacobs (`jacobs/executor.py::validate_capability`) — import directo

Para facets en `_HTTP_FACETS`, después del NIVEL A existente, llamar
`check_capability_admission(caller="jacobs", capability=step.capability,
context_keys=list(step.input.keys()), recursion_depth=0,
human_gate_token=None)` — import directo de `motor_registry.policy`, sin
red (mismo proceso). Si `allowed=False`, se rechaza igual que hoy rechaza
NIVEL A (mensaje `str`, no reenrutable — mismo criterio que el resto de
`validate_capability`).

`context_keys=list(step.input.keys())` mirror del `context_keys=list(req.context.keys())`
que ya usa `/motor/dispatch` para `kimi`/`jax_local` — Step.input es el
dict equivalente de contexto para un step de Jacobs.

### 3. Nuevo endpoint `POST /motor/authorize` en `las_manos`

Síncrono, sin job ni polling — corre `check_capability_admission()` y
devuelve `{"allowed": bool, "reason": str}`. Mismo patrón de request/response
que ya conoce este servicio, pero sin crear ningún job.

### 4. Mesa web (`jax-platform/backend/api/chat.py::_invoke_facet`)

Antes de despachar a `hipatia`/`jekyll`/`thot`/`ada`: `POST` a
`http://127.0.0.1:7777/motor/authorize` con
`{"caller": "jax_platform_chat", "capability": <la que resuelva el facet>,
"context_keys": [], "recursion_depth": 0, "human_gate_token": null}`.

`context_keys=[]`: el payload de un turno de chat es texto libre
(`req.message`), no un dict con claves nombradas — no hay equivalente real
de "contexto con posibles secretos" que mapear ahí, a diferencia de
`step.input` en Jacobs. Documentado, no fabricado.

**Fail-closed (Requisito 3):** timeout, error HTTP, o respuesta
inesperada de `las_manos` → Mesa web deniega, no despacha. Nunca "no pude
verificar, sigo igual" — P10 lo prohíbe explícitamente.

**Caller `jax_platform_chat`:** identifica el origen real de la llamada
(el backend de jax-platform, módulo `chat.py`), no el nombre de producto
("Mesa web" es el nombre que ve el usuario, no la pieza de infraestructura
que llama). Consistente con `"jacobs"` como caller (nombre del módulo que
despacha, no "pipelines" ni "Jacobs Director").

### 5. Migración de datos — ANTES de activar el enforcement

`UPDATE capability SET allowed_callers = JSON_ARRAY_APPEND(allowed_callers, '$', 'jax_platform_chat')`
sobre las 5 filas relevantes (`research`, `analysis`, `design`, `reconcile`,
`validate_consistency`) — mismo acceso que Mesa web ya tiene hoy a los 4
facets, sin restringir ni ampliar nada (Requisito 4). El enforcement se
activa DESPUÉS de esta migración, nunca antes — si se invierte el orden,
Mesa web se rompe el día del deploy.

## Testing (Requisitos 1 y 2, no negociables)

1. **Cero cambio de comportamiento kimi/jax_local:** correr la suite
   existente de `policy.py`/Motor Registry tal cual, sin modificar ningún
   test. Si algo falla y la corrección es "actualizar el test", eso es señal
   de que el refactor cambió comportamiento — PARAR y reportar, no ajustar
   el test para que pase.
2. **Test negativo real en los 3 caminos** — un caller NO autorizado
   rechazado de verdad, no solo "el código compila":
   - `kimi`/`jax_local` vía `check()` completo (ya debería existir cobertura
     parecida; confirmar que sigue pasando post-split).
   - Jacobs-HTTP: un caller ficticio sin `"jacobs"` en `allowed_callers`
     de una capability HTTP → `validate_capability()` rechaza el step.
   - Mesa-web-HTTP: mismo caso contra `check_capability_admission()`
     llamada por `/motor/authorize`, Y un test de integración liviano que
     confirme que `_invoke_facet` deniega cuando `/motor/authorize`
     responde `allowed=False` O cuando la llamada falla/hace timeout
     (fail-closed, Requisito 3 — ambos casos, no solo el explícito).
3. **Verificación en vivo post-cambio (Requisito 4):** con el enforcement
   activo y la migración de `allowed_callers` aplicada, confirmar en el chat
   real de la Mesa web que los 4 facets (hipatia/jekyll/thot/ada) siguen
   respondiendo — no alcanza con tests unitarios para este punto, tiene que
   ser una corrida real.

## Qué queda explícitamente sin gobernar después de este PR

- **Techo de timeout:** `timeout_seconds` sigue gobernado solo por
  `_CAPABILITY_TIMEOUT_SECONDS` (default) sin validación de techo para
  ningún facet — ni HTTP ni motor, desde ningún camino. Este PR no lo toca.
  Queda anotado en `DEUDA.md` como ítem aparte, explícito, para no sugerir
  que el techo se está imponiendo cuando no es cierto.
- **`capability.sandbox_only`:** columna sin lector, marcada vestigial en
  schema/`DEUDA.md`, no borrada, sin semántica nueva inventada.
- **`human_gate_token` para HTTP-directo:** ninguna de las 5 capabilities
  requiere human gate hoy (`requires_human_gate=0` en las 5 filas), así que
  el check pasa trivialmente. Si algún día se pone `requires_human_gate=1`
  en alguna de ellas, el dispatch HTTP-directo (Jacobs o Mesa web) empezará
  a rechazar TODO — correcto por fail-closed (no hay mecanismo de pasar un
  token real por ninguno de los dos caminos hoy), pero vale dejarlo anotado
  para que no sorprenda el día que pase.

## `DEUDA.md` — actualización al cerrar

El bullet `_HTTP_FACETS sin gobernanza del Motor Registry` se marca CERRADO
con referencia a esta ronda, listando explícitamente los 5 checks que ahora
sí aplican (1-5, no 8) y el porqué del 6-7 (N/A, no hay motor que resolver)
y el 8 (diferido, con la razón estructural). Se agregan dos entradas nuevas
si no existían ya con este detalle: `capability.sandbox_only` vestigial, y
el techo de timeout como deuda separada y explícita (probablemente ya
cubierta por el ítem existente de `_CAPABILITY_TIMEOUT_SECONDS`/dedup, a
verificar contra el texto real al momento de cerrar).
