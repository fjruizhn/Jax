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
- **Jacobs corre DENTRO del proceso de `las_manos`**: `jax-las-manos.service`
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

### 3. CORREGIDO — Mesa web NO tiene concepto de `capability`, y `check_capability_admission()` no aplica ahí

**Esta sección del spec original afirmaba algo falso, sin verificarlo.**
Decía "`capability`: la que resuelva el facet" — no existe tal resolución.
Verificado leyendo `chat.py` completo: un turno de chat es texto libre
(`req.message`) enrutado a un facet por keyword-matching (`_WEB_TIEBREAK`,
`_HIPATIA_KW`/`_JEKYLL_KW`/`_THOT_KW`/`_ADA_KW`, líneas 302-307). No hay
NINGÚN mapeo facet→capability en `jax-platform` (`grep -n
"\"hipatia\"\|\"jekyll\"\|\"thot\"\|\"ada\"" backend/*.py backend/api/*.py`
→ solo keyword-routing y strings de UI, cero relación con `capability`).
`research`/`analysis`/`design`/`reconcile`/`validate_consistency` son
capabilities de **step de pipeline** (un objetivo concreto con output
esperado) — no de "tener una conversación". Inventar cuál correspondería a
cada facet habría sido autorizar contra una capability ficticia — la
instancia número diez del patrón "código que afirma algo que no hace",
fabricada dentro del PR que viene a cerrar gobernanza. Descartado.

**La pregunta correcta para Mesa web no es "¿está autorizada la capability
X?" — es "¿puede `jax_platform_chat` hablar con el facet `ada`?".** Es un
check de nivel FACET, no de nivel CAPABILITY, y **no reusa
`check_capability_admission()`** — es una función nueva y más chica,
`check_facet_admission(caller, facet)`, que no toca la tabla `capability`
en absoluto.

**Dónde vive el dato (Requisito 2 — fuente única, no un dict hardcodeado):**
columna nueva `facet.allowed_callers`, mismo estilo que `capability.allowed_callers`
pero NULLABLE (mirror exacto del patrón ya existente de
`capability.forbidden_paths`: `LONGTEXT ... NULL CHECK (allowed_callers IS
NULL OR json_valid(allowed_callers))`). La tabla `facet` ya existe
(`jax-platform/backend/db/migrations.py:222-238`, `CREATE_FACET`, una fila
por facet — confirmado en vivo: 7 filas hoy, `ada`/`hipatia`/`hyde`/
`jax_local`/`jekyll`/`kimi`/`thot`) — es la fuente única y correcta para un
dato por-facet, mismo lugar donde ya viven `max_latency_ms`/
`max_cost_per_1k_usd` (otro ítem abierto de `DEUDA.md`, sin relación con
este). NULL en vez de `[]` a propósito: para los 3 facets fuera de alcance
(`hyde`, `jax_local`, `kimi`) no se fabrica un valor — se declara "sin
configurar" explícitamente, y el check fail-closed trata NULL como
denegación, no como "lista vacía = nadie" ambiguo con "no configurado
todavía".

`check_facet_admission()` vive en `las_manos/motor_registry/facet_policy.py`
(archivo nuevo, responsabilidad propia — no entra en `policy.py`, que es
"puro, sin I/O", y esta función SÍ hace una query): consulta
`facet.allowed_callers`, `NULL` o caller ausente → `(False, razón)`.

### 4. Nuevo endpoint `POST /motor/authorize-facet` en `las_manos`

Síncrono, sin job ni polling — corre `check_facet_admission()` y devuelve
`{"allowed": bool, "reason": str}`. Nombre explícito (`-facet`, no
`/motor/authorize` genérico) para no confundirlo con una futura variante a
nivel capability que hoy nadie necesita (Jacobs usa el import directo,
sección 2 — no hace falta exponer `check_capability_admission()` por HTTP
para nada en esta ronda).

### 5. Mesa web (`jax-platform/backend/api/chat.py::_invoke_facet`)

Antes de despachar, **solo si `facet in {"hipatia","jekyll","thot","ada"}`**
(kimi/jax_local/hyde no se tocan — mismo alcance de siempre): `POST` a
`http://127.0.0.1:7777/motor/authorize-facet` con
`{"caller": "jax_platform_chat", "facet": facet}`.

**Fail-closed (Requisito 3):** timeout, error HTTP, o respuesta
inesperada de `las_manos` → Mesa web deniega, no despacha. Nunca "no pude
verificar, sigo igual" — P10 lo prohíbe explícitamente.

**Caller `jax_platform_chat`:** identifica el origen real de la llamada
(el backend de jax-platform, módulo `chat.py`), no el nombre de producto
("Mesa web" es el nombre que ve el usuario, no la pieza de infraestructura
que llama). Consistente con `"jacobs"` como caller (nombre del módulo que
despacha, no "pipelines" ni "Jacobs Director").

### 6. Migración de datos — ANTES de activar el enforcement

Una sola migración, en `jax-platform` (`facet.allowed_callers` es la única
columna nueva que este diseño necesita — `jax_platform_chat` NUNCA se
agrega a `capability.allowed_callers`, porque nada en este diseño llama
`check_capability_admission()` con ese caller; Mesa web no toca `capability`
en absoluto desde la corrección de la sección 3). Mismo patrón idempotente
que `_fix_file_write_gate_and_auditor` (guarda `WHERE`, no pisa un valor
manual futuro):

```sql
UPDATE facet SET allowed_callers = JSON_ARRAY('jacobs', 'jax_platform_chat')
WHERE `key` IN ('hipatia', 'jekyll', 'thot', 'ada') AND allowed_callers IS NULL;
```

Mismo acceso que existe hoy (nadie estaba bloqueado, ahora queda explícito),
sin restringir ni ampliar nada (Requisito 4/5). El enforcement se activa
DESPUÉS de esta migración, nunca antes — si se invierte el orden, Mesa web
se rompe el día del deploy.

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
     de una capability HTTP → `validate_capability()` rechaza el step
     (vía `check_capability_admission()`).
   - Mesa-web-HTTP: **dos tests separados**, porque son dos capas
     distintas:
     - `check_facet_admission()` en aislamiento: un facet con
       `allowed_callers` que no incluye `jax_platform_chat` (o `NULL`) →
       `(False, razón)`.
     - Integración en `chat.py`: `_invoke_facet` deniega cuando
       `/motor/authorize-facet` responde `allowed=False` **Y** cuando la
       llamada falla/hace timeout — este segundo caso simula `las_manos`
       caído de verdad (conexión rechazada, no un mock que devuelve un
       error prolijo), fail-closed (Requisito 3), es el test que prueba
       que el gate gatea cuando más importa.
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
- **`human_gate_token` para el camino de Jacobs:** ninguna de las 5
  capabilities requiere human gate hoy (`requires_human_gate=0` en las 5
  filas), así que el check pasa trivialmente. Si algún día se pone
  `requires_human_gate=1` en alguna de ellas, Jacobs-HTTP empezará a
  rechazar TODO — correcto por fail-closed (no hay mecanismo de pasar un
  token real hoy), pero vale dejarlo anotado para que no sorprenda.
- **Mesa web: `requires_human_gate`, `recursion_depth`, claves prohibidas,
  y techo de timeout — los cuatro son N/A por diseño, no "se saltearon".**
  Después de la corrección de la sección 3, Mesa web valida por FACET
  (`check_facet_admission`), no por CAPABILITY — y esos cuatro checks viven
  exclusivamente en `check_capability_admission()`/`check()`, atados a la
  tabla `capability`, que Mesa web ya no consulta para este camino. No es
  que este PR decida no aplicarlos — es que la pregunta que responden
  ("¿esta capability de pipeline permite este presupuesto/profundidad/
  claves?") no tiene sentido para un turno de chat libre. Si algún día Mesa
  web necesita gobernanza más fina que "¿puede este caller hablar con este
  facet?", es una capability real para chat (approach (b) que se descartó
  esta ronda) — diseño nuevo, no una extensión de este PR.

## `DEUDA.md` — actualización al cerrar

El bullet `_HTTP_FACETS sin gobernanza del Motor Registry` se marca CERRADO
con referencia a esta ronda, distinguiendo los dos caminos por separado (son
gobernanza de forma distinta, no lo mismo aplicado dos veces):
- **Jacobs:** checks 1-5 de `MotorPolicy` ahora aplican vía
  `check_capability_admission()`. 6-7 N/A (no hay motor que resolver para
  un facet HTTP). 8 diferido, con la razón estructural (dict hardcodeado en
  `plan.py` sin dedup con la DB).
- **Mesa web:** NO usa `MotorPolicy` ni la tabla `capability` — usa
  `check_facet_admission()`, un check nuevo y más chico sobre
  `facet.allowed_callers`. Los checks 2-8 de `MotorPolicy` son N/A acá por
  diseño (atados a `capability`, que Mesa web no consulta), no "quedaron
  pendientes".

Se agregan dos entradas nuevas si no existían ya con este detalle:
`capability.sandbox_only` vestigial, y el techo de timeout como deuda
separada y explícita (probablemente ya cubierta por el ítem existente de
`_CAPABILITY_TIMEOUT_SECONDS`/dedup, a verificar contra el texto real al
momento de cerrar).
