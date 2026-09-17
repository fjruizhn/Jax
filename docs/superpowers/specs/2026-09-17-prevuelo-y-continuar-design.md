# Pre-vuelo antes de gastar y continuar pipelines abortados — diseño

**Fecha:** 2026-09-17 · **Autor:** Mr. Hyde · **Repos:** `jax` (Jacobs, LAS MANOS) y `jax-platform` (Mesa)
**Decisiones:** Fernando, 2026-09-16/17, por preguntas cerradas; GO autónomo hasta producto final.

## 0. Por qué

El pipeline `ef9b2d6e-6697-4dc3-9b9d-cd15d88acde2` («leyes de energía de Honduras… informe HTML»)
se abortó el 2026-09-16 a las 19:40 en el paso 4. Kimi (kimi-k3) se cortó por `motor.max_tokens=8000`:
7.997 tokens se fueron en razonamiento y **cero** en la respuesta. Los pasos 0-3 (Hipatia, Ada, Jekyll, Ada)
habían terminado y tienen su resultado en `context_refs` (`step_0_ref`..`step_3_ref`). Costo registrado
del pipeline: $0,35, de los cuales $0,14 fueron de Kimi, que no produjo nada.

Tres defectos, medidos:
1. **El tope de Kimi es nuestro, no del proveedor.** `model.max_output_tokens` de kimi-k3 = 131.072
   (`source=provider_api`, 2026-09-12). Los 8.000 fueron «un valor de partida, ajustar si se repite un corte»
   (CONTEXT.md, 2026-08-10). Se repitió.
2. **Nada valida antes de gastar.** `PlanBuilder.build()` valida capability, clean-room y techo de timeout,
   pero no resuelve credencial, salud, límite de salida ni costo. El error apareció a los 16 minutos,
   después de cuatro pasos pagos.
3. **Un pipeline abortado no se puede continuar.** `POST /jacobs/pipeline/{id}/resume` sólo acepta
   `interrupted`. El ejecutor ya salta los pasos con `step_N_ref` (`executor.py:1109-1114`), pero no hay
   camino que lo use para un `aborted`. El CLI `tools/jacobs_relaunch.py` lo intenta y tiene tres agujeros:
   usa la foto vieja del plan, no borra las refs de los pasos a rehacer y se salta el candado y el límite.

## 1. Decisiones de Fernando (mandan sobre este documento)

| # | Pregunta | Decisión |
|---|---|---|
| D1 | Tope de salida de Kimi | **El máximo del modelo** (`motor.max_tokens=0` → catálogo) |
| D2 | Qué abortados se continúan | **Todos**: por fallo, por cancelación manual y por kill switch |
| D3 | El paso fallido al continuar | **Se puede cambiar de faceta** (y pasa por la validación del plan) |
| D4 | Qué revisa el pre-vuelo | **Los cuatro**: tope suficiente, credencial viva, faceta sana, costo máximo con confirmación |
| D5 | Faceta sin datos de salud | **Sondear en el momento**; si no responde, rechazar |
| D6 | Dónde vive | **Jacobs** (enfoque A): crear y continuar pasan obligatoriamente por el pre-vuelo; la Mesa muestra y confirma |

Consecuencia declarada de D1: **Ada recibe el mismo tratamiento** (`motor.max_tokens=0`). En los
pipelines Ada ya recibe el tope del catálogo por el camino HTTP directo (`executor.py:381`); los 8.000 sólo
le aplican por Motor Registry. Dos criterios para lo mismo es la clase de divergencia que se viene cerrando.

## 2. Alcance

**Entra:**
- A. Tope de Kimi y Ada al máximo del catálogo, por el endpoint gobernado.
- B. Pre-vuelo en Jacobs (endpoint propio + obligatorio en crear y continuar).
- C. Continuar un pipeline `aborted` o `expired`, con cambio de faceta opcional por paso pendiente.
- D. **Un solo ejecutor por pipeline** (época de corrida): hoy cancelar no detiene nada y el reaper
  tampoco. Sin esto, continuar un cancelado que sigue corriendo lo ejecuta dos veces. Es un bypass de la
  propiedad que esta ronda construye, así que es alcance, no ampliación.
- E. Mesa: pre-vuelo dentro del modal de pipeline, confirmación de costo en ventana propia, botón
  «Continuar» con cambio de faceta, y propagación de los 4xx de Jacobs en resume, cancel, get, results y
  continue (hoy responden 200 con el cuerpo de error, así que el aviso de rechazo nunca aparece).
- F. Datos del catálogo que el pre-vuelo necesita: `max_output_tokens` y precios de los modelos
  vinculados hoy a una faceta, medidos contra el proveedor.
- G. Rescate del pipeline `ef9b2d6e` con el mecanismo nuevo, después del deploy.

**No entra (y por qué):**
- Reintento automático: gasta sin decisión humana (enfoque C, descartado).
- El retiro del fallback a `.env` de `resolve_credential_instrumented` y la copia vieja de credencial en
  `resolve_facet`: son del frente E y de B1.4. El pre-vuelo lee la verdad de la DB (§4.3) y lo declara.
- Pre-vuelo del LLM planificador en `_from_objective`: la Mesa siempre manda pasos explícitos
  (`buildSteps`). El camino por objetivo gasta en planificar antes del pre-vuelo; queda declarado en §4.7.

## 3. Arquitectura

```
Mesa (PipelineModal / ContinuarModal)
   │ POST /api/pipelines/preflight            ─┐
   │ POST /api/pipelines        (+costo_confirmado_usd)
   │ POST /api/pipelines/{id}/continue (+reasignar, +costo_confirmado_usd)
   ▼                                           │ umbral de confirmación: ajuste `pipeline_confirmar_usd`
jax-platform api/pipelines.py  ────────────────┘
   │ POST /jacobs/preflight
   │ POST /jacobs/pipeline            (corre el pre-vuelo bloqueante dentro del candado)
   │ POST /jacobs/pipeline/{id}/continue (idem)
   ▼
Jacobs: jacobs/prevuelo.py  →  catálogo (model, motor_resolved, facet_binding, capability, credential)
                            →  salud (facet_health_event) → sonda si hace falta (jacobs/sonda.py)
        jacobs/executor.py  →  run_pipeline con época de corrida (store.run_epoch)
```

**Quién hace cumplir qué:** los cuatro chequeos técnicos los hace cumplir **Jacobs**, siempre, para
cualquier llamador (Mesa, REPL, emisor de Ada, scripts). La **confirmación de costo** es consentimiento
humano y vive donde está el humano: jax-platform. Jacobs devuelve el costo máximo en cada respuesta de
pre-vuelo y en la de crear/continuar, para que ningún llamador lo ignore sin verlo.

## 4. Pre-vuelo (`jacobs/prevuelo.py`)

### 4.1 Entrada y salida

```python
async def prevuelo(pasos: list[Step], contexto: dict) -> Veredicto   # contexto = refs ya existentes (continuar)
@dataclass(frozen=True)
class Veredicto:
    ok: bool                       # no hay violaciones bloqueantes
    violaciones: list[Violacion]   # (paso, faceta, regla, detalle) — regla ∈ REGLAS
    costo_max_usd: Decimal         # suma de los pasos acotables
    pasos_costo: list[CostoPaso]   # (paso, faceta, modelo, llamadas_max, tokens_in_max, tokens_out_max, usd_max | None, motivo)
    sondeadas: list[str]           # facetas sondeadas en este pre-vuelo
```

`REGLAS = {"tope_insuficiente", "sin_contrato_de_salida", "credencial_ausente", "faceta_caida", "faceta_inexistente"}`.

Sólo se evalúan los pasos **pendientes** (en crear, todos; en continuar, los que no tienen ref válida).
`assemble` es mecánico y no cuesta. `hyde` no cobra por token (suscripción): costo 0 declarado con
motivo `suscripcion`, sin chequeo de tope, credencial ni salud (no se sondea, igual que el canario de la Mesa).

### 4.2 Resolución por paso

Para cada faceta distinta: la fila de `facet` activa + `facet_binding(role='primary')` + `model` +
`provider` (la misma consulta que `_query_facet`, sin resolver la credencial). Una faceta sin fila →
`faceta_inexistente`. Para `MOTOR_FACETS` además `motor_resolved` (transporte, `max_tokens`,
`has_tool_access`). Una consulta por pre-vuelo, no por paso.

### 4.3 Credencial viva

Proveedor con transporte que cobra (`http_openai_compat`, `http_gemini`): debe existir
`credential(provider_id, state='active')`. **Se lee la DB, no el resolver**: `resolve_credential_instrumented`
cae a `.env` aunque la credencial esté revocada, y `resolve_facet` puede servir una credencial vieja 300 s.
El pre-vuelo no hereda esos dos comportamientos (§2, no entra su retiro). `ollama` y `subprocess` no llevan.

### 4.4 Tope de salida suficiente

- Nueva columna **`capability.min_output_tokens INT NOT NULL DEFAULT 0`** (DDL en jax-platform
  `db/migrations.py`, como el resto de `capability`).
- Tope efectivo del paso = el mismo cálculo del despacho real:
  - Motor Registry: `min(motor.max_tokens, model.max_output_tokens)` con 0 = sin tope propio (`worker._limite_del_motor`).
  - HTTP `http_openai_compat`/`ollama`: `model.max_output_tokens` (`contrato_dispatch.limite_de_salida`).
  - `http_gemini`: el proveedor aplica su `outputTokenLimit` aunque no se mande; el tope efectivo es
    `model.max_output_tokens` de la fila.
- `model.max_output_tokens` NULL en un paso que cobra → **`sin_contrato_de_salida`** (bloquea): sin tope no
  hay costo máximo. El detalle dice qué fila completar. Es el mismo fallo cerrado de `contrato_dispatch`.
- Tope efectivo < `capability.min_output_tokens` → **`tope_insuficiente`**, con los dos números y cuál subir.
- **Semilla de `min_output_tokens`, medida, no inventada:** máximo de `completion_tokens` de corridas
  **completadas** por capability (`motor_jobs.jsonl` `_usage` + `axioma_usage.tokens_out` unido a
  `jacobs_steps` por `job_id`/ventana de tiempo), redondeado hacia arriba a múltiplo de 1024. Capability sin
  corridas medibles → 0, declarado en el commit. La tabla de valores y el comando van en el plan.

### 4.5 Faceta sana (D5)

- Se lee de `facet_health_event` el último evento **de nivel proveedor** de la faceta dentro de
  `HEALTH_WINDOW_SECONDS` (2 h, `jacobs/facet_health.py:16`). Nivel proveedor = outcome ∈ {`ok`,
  `provider_error`}. Los outcomes `gate_*`, `unbound`, `unsupported_transport` son del gate de la Mesa
  (kimi siempre escribe `unsupported_transport` porque la Mesa no lo despacha): **no dicen nada del
  proveedor** y no cuentan. `probe_error` es falla de la sonda, no del proveedor: tampoco cuenta.
- Último evento de nivel proveedor = `ok` → sana.
- Cualquier otra cosa (sin evento, `provider_error`, fuera de ventana) → **sondear ahora** (`jacobs/sonda.py`).
  Responde → sana; no responde → **`faceta_caida`** (bloquea) con el error redactado.
- Decisión de diseño: un `provider_error` viejo no bloquea por sí solo, se vuelve a medir. Decide un
  dato fresco, no uno de hace una hora.
- **La sonda:** una llamada mínima por el transporte real de la faceta (`resolve_facet` + mismo cliente que
  el executor), mensaje fijo de sonda, límite de salida mínimo del contrato del modelo (el nombre del
  parámetro sale de `model.max_tokens_param`). «Responde» = HTTP 2xx del proveedor, **aunque el contenido
  venga cortado por el límite**: mide disponibilidad, no calidad. Timeout propio
  `JAX_PREVUELO_SONDA_TIMEOUT_S` (config, validado). El resultado se registra en `facet_health_event`
  con `source='preflight'` (nuevo valor del ENUM, DDL en jax-platform) y outcome `ok`/`provider_error`, así el
  próximo pre-vuelo dentro de 2 h no vuelve a sondear. Su uso se registra en `axioma_usage` con
  `request_type='preflight_probe'` (se paga y se ve). Varias facetas se sondean en paralelo.
- Crear y continuar después de un `/preflight` de la Mesa no vuelven a sondear en la práctica: la sonda del
  primer pre-vuelo dejó un evento `ok` fresco. No hay flag para saltearse la sonda.

### 4.6 Costo máximo estimado

Por paso que cobra:
- `tokens_out_max` = tope efectivo (§4.4).
- `tokens_in_max` = ⌈(chars del prompt del paso + Σ min(chars de cada dependencia, `MAX_DEP_CONTEXT_CHARS`)
  + chars de la persona) / `JAX_PREVUELO_CHARS_POR_TOKEN`⌉. Las dependencias que ya tienen ref (continuar)
  se miden de verdad; las que no existen todavía cuentan `MAX_DEP_CONTEXT_CHARS` (el peor caso del armado,
  `executor.py:50`). `JAX_PREVUELO_CHARS_POR_TOKEN` es config con default conservador **2** (cota inferior de
  chars por token; menos chars por token = más tokens = más costo): sobreestima, nunca subestima.
- `llamadas_max` = 1, ×2 si la faceta es `http_gemini` con grounding `required_web` (reintento de grounding),
  ×2 si la capability tiene `output_schema` implementado en `output_validator.SCHEMAS` (reintento de schema),
  ×`MOTOR_TOOL_LOOP_MAX_ITERATIONS` si el motor tiene `has_tool_access`.
- `usd_max` = llamadas_max × (tokens_in_max × price_input + tokens_out_max × price_output) / 1e6.
- Precio NULL en un paso que cobra → `usd_max=None` con motivo `sin_precio`, y **cuenta como no acotado**:
  la Mesa pide confirmación siempre que haya un paso no acotado (§6.2). No bloquea: el precio es dato de
  referencia (models.dev), no un contrato del despacho.
- `jax_local` (ollama local): costo 0, motivo `local`.

### 4.7 Dónde corre

- `POST /jacobs/preflight` `{invoked_by, steps}` → 200 con el `Veredicto` (aunque `ok=False`). No crea filas,
  no toma el candado. Valida `invoked_by` como `/plan`.
- `POST /jacobs/pipeline`: dentro del candado, **después** de `build()` y **antes** de `pipeline_create`. `ok=False`
  → 422 `{code:"prevuelo_rechazado", violaciones, costo_max_usd, pasos_costo}` + evento `PREVUELO_RECHAZADO`
  con el mismo contenido. `ok=True` → la respuesta de creación suma `costo_max_usd` y `pasos_costo`, y el
  evento `PIPELINE_CREATED` también.
- `POST /jacobs/pipeline/{id}/continue`: igual, sobre los pasos pendientes (§5).
- `_from_objective` (planificación por LLM): el pre-vuelo corre sobre el plan que devolvió el LLM; lo gastado
  en planificar ya está gastado y se declara en la doc del endpoint.
- `dry_run`: corre el pre-vuelo igual (es el uso natural de «¿cuánto costaría?»).

## 5. Continuar (`POST /jacobs/pipeline/{id}/continue`)

### 5.1 Contrato

```
body: {invoked_by, reasignar?: {"<step_index>": "<faceta>"}, user_id?, tenant_id?}
200: {pipeline_id, status:"running", run_epoch, pasos_a_correr:[i...], pasos_reusados:[i...], costo_max_usd, pasos_costo}
403 invoked_by · 404 no existe · 409 estado no continuable | costo_supera_lo_aceptado · 422 prevuelo_rechazado | reasignacion_invalida · 423 kill switch · 429 limite de activos
```

`POST /jacobs/pipeline/{id}/continue/preflight` con el mismo body (sin `costo_max_aceptado_usd`) aplica las
reglas 1-8 **sin escribir** y devuelve `{continuable, motivo?, pasos_a_correr, pasos_reusados, veredicto}`. Es lo
que muestra la Mesa antes de confirmar. `continue` acepta además `costo_max_aceptado_usd` (§6.1).

### 5.2 Reglas

1. `validate_resume(invoked_by)` (sólo `plataforma`). El dueño lo verifica jax-platform (como resume).
2. Estado continuable: `aborted` (cualquier causa, D2) o `expired`. `completed`, `running`, `pending`,
   `interrupted` → 409 con el estado actual (`interrupted` tiene `/resume`).
3. Kill switch activo → 423 (continuar ejecuta).
4. **Dentro de `_pipeline_create_lock`** y contra `MAX_PARALLEL_PIPELINES` (hoy `/resume` se lo salta:
   continuar no).
5. **Pasos reusados:** los que tienen `step_{i}_ref` en context **y** su ref se lee (`_load_ref`). Una ref
   ilegible (artifact borrado) → el paso se rehace, se quita la ref, y queda en el pre-vuelo con su costo.
6. **Pasos a correr:** todos los demás. Se ponen `pending`, `error=None`, `started_at=None`, `finished_at=None`.
7. **Reasignar** (D3): sólo pasos a correr. Faceta ∈ `VALID_FACETS`. Se cambia `facet`; `motor` se recalcula
   (faceta de motor → `motor=faceta`; HTTP → `None`). La capability no cambia. El plan resultante completo pasa
   por `_check_cleanroom` y `_validate_plan_capabilities` → si falla, 422 `reasignacion_invalida` con las
   violaciones, sin tocar nada. `store.step_upsert` hoy no actualiza `facet` ni `input` en el ON DUPLICATE: se
   agrega `facet`. La columna `plan` JSON de `jacobs_pipelines` se reescribe con los pasos vigentes (hoy queda la
   foto de creación, y el relanzador lee esa foto: se corrige en el mismo movimiento).
8. Pre-vuelo sobre los pasos a correr (§4) → 422 si rechaza.
9. **Época nueva** (§5.3), status `running`, evento `PIPELINE_CONTINUED {by, from_status, run_epoch,
   pasos_a_correr, pasos_reusados, reasignados:{i:{de,a}}, costo_max_usd}`, `run_pipeline` en background.
10. Todo lo anterior a la escritura (1-8) no modifica la DB. Las escrituras de 6-9 van en una transacción.

### 5.3 Un solo ejecutor por pipeline: época de corrida

- Columna **`jacobs_pipelines.run_epoch INT NOT NULL DEFAULT 0`** (DDL en `jacobs/store.py::init_tables`, patrón
  idempotente existente; el CI de jax-platform crea el esquema de Jacobs con ese mismo `init_tables`).
- `run_pipeline(pipeline)` toma `epoca = pipeline.run_epoch` al arrancar.
- **Toda escritura de estado del pipeline desde el ejecutor** (`pipeline_update_status` en run_pipeline y
  `_fail_step`) pasa a condicional: `UPDATE … WHERE pipeline_id=%s AND run_epoch=%s AND status='running'`
  (para la transición inicial a running, `status IN (...)` según corresponda). 0 filas → el ejecutor **perdió la
  época o el pipeline ya no está corriendo** (cancelado, expirado, continuado por otro): registra
  `RUN_SUPERSEDED {epoca, status_actual}` una vez y **termina sin escribir nada más**.
- Antes de cada ola, el ejecutor relee `status` y `run_epoch` (una consulta por PK); si no es su época o no está
  `running`, termina igual.
- `cancel`, el kill switch y el reaper **no** tocan la época: cambian el status, y eso ya detiene al ejecutor en la
  próxima escritura o la próxima ola. `continue` y `resume` **incrementan** la época (resume también, para que
  un `resume` doble no lance dos ejecutores).
- Límite honesto: un paso que ya está despachado al proveedor **no** se interrumpe; su resultado se descarta
  si llega tarde (la escritura de `step_N_ref` en context también es condicional a la época: el context se
  persiste con la misma condición). Es el mismo alcance que el kill switch entre olas. Se documenta.
- `_fail_step` deja de escribir `aborted` por su cuenta cuando `run_pipeline` lo va a escribir igual: el doble
  `PIPELINE_ABORTED` con payloads distintos (sorpresa medida) se reduce a uno, el de la ola, con
  `{at_wave, failed_steps, errores:{i:error}}`.

### 5.4 Relanzador CLI

`tools/jacobs_relaunch.py` pasa a llamar la misma función de servicio que el endpoint (`jacobs/continuar.py`),
sin lógica propia: se cierran sus tres agujeros por construcción.

## 6. Mesa (jax-platform)

### 6.1 Backend `api/pipelines.py`

- `POST /api/pipelines/preflight` `{steps}` → reenvía a `/jacobs/preflight` con la identidad inyectada y agrega
  `umbral_usd` (ajuste) y `requiere_confirmacion = costo_max_usd > umbral or hay paso no acotado`.
- `POST /api/pipelines` acepta `costo_confirmado_usd: Decimal | None`. Secuencia: cupo → **pre-vuelo** (Jacobs) →
  si `ok=False` → 422 `prevuelo_rechazado` (con violaciones) → si `requiere_confirmacion` y
  (`costo_confirmado_usd` es None o < `costo_max_usd`) → **409 `confirmacion_de_costo`** con el veredicto → crear.
- El costo puede cambiar entre el pre-vuelo y la creación (otro precio, otra sonda). Jacobs arranca en
  background al crear, así que la Mesa no puede corregir después. Por eso `POST /jacobs/pipeline` y `/continue`
  aceptan `costo_max_aceptado_usd` opcional: si viene y el pre-vuelo interno da más, Jacobs responde 409
  `costo_supera_lo_aceptado` **sin crear**. La Mesa lo manda siempre que haya confirmación. La condición la hace
  cumplir quien gasta, no quien muestra.
- `POST /api/pipelines/{id}/continue` `{reasignar?, costo_confirmado_usd?}`: dueño → mismo flujo (pre-vuelo con
  los pasos pendientes: Jacobs expone `POST /jacobs/pipeline/{id}/continue/preflight` que devuelve pasos a correr,
  reusados y veredicto sin escribir) → 422/409 → continue.
- **Propagación de errores:** resume, cancel, get, results y continue dejan de devolver 200 con el cuerpo de error:
  status de Jacobs ≥ 400 → mismo status con `{code:"jacobs_rechazo", status, motivo}` (el helper de create se
  extrae y se reusa). Cuerpo no-JSON con status ≥ 400 → `jacobs_rechazo` con el texto recortado, no
  `jacobs_no_responde`.
- Ajuste nuevo **`pipeline_confirmar_usd`** en `ajustes.DEFINICIONES` (Decimal ≥ 0, default sembrado **0,50**,
  editable en el panel de ajustes existente). 0 = confirmar siempre.
- `GET /api/pipelines/{id}/continuable` no hace falta: el preflight de continue lo contesta.

### 6.2 Frontend

- **`PipelineModal`**: «Planificar y ejecutar» ahora (1) llama `/pipelines/preflight`; (2) violaciones → se muestran
  **dentro del modal** (lista por paso, `text-peligro`), el modal no se cierra; (3) `requiere_confirmacion` →
  ventana `ConfirmarCostoDialogo` (sobre `Dialogo`, no destructiva: sin suma) con el costo máximo, el desglose
  por paso y los pasos no acotados; (4) confirma → `POST /pipelines` con `costo_confirmado_usd`. Errores
  409/422 de la creación vuelven al modal. `handleSubmit` con try/finally.
- **`RightPanel`**: un pipeline `aborted`/`expired` muestra el motivo (último `STEP_FAILED`/cancelación/kill
  switch/reaper, texto i18n por causa) y el botón **«Continuar»**, que abre `ContinuarPipelineModal`: pasos
  reusados (marcados como listos), pasos a correr con selector de faceta (mismas opciones y
  bloqueo de clean-room que `PipelineModal`), costo máximo del preflight de continue, y el mismo flujo de
  confirmación. La lista de abortados sale de `GET /api/pipelines` (existe, filtrado por dueño).
- i18n es/en para todo texto nuevo (paridad por test); colores por tokens (`peligro`, `aviso`, `info`); sin
  diálogos del navegador (el escaneo existente lo vigila). Montos con el locale activo (formato de moneda).
- Eventos de WebSocket: `pipeline_continued` refresca el panel igual que `pipeline_step_changed`.

## 7. Datos (sección F y A)

- **A.** `PATCH /api/admin/motors/kimi` y `/ada` con `max_tokens=0`, como superadmin, con respaldo de la fila y
  lectura antes/después. El PATCH hoy no audita ni valida el rango (acepta negativos). Se agrega la validación
  `max_tokens >= 0` (un negativo llega al proveedor como tope inválido: es defecto del camino que se usa). La
  auditoría del PATCH de motores es otra propiedad (quién cambió qué) y va a `DEUDA.md` con fecha; la evidencia
  de este cambio queda en el respaldo de la fila y en la Biblioteca.
- **F.** Para cada modelo vinculado hoy a una faceta que cobra (`facet_binding role='primary'`), completar
  `model.max_output_tokens` NULL con el valor **medido contra la API del proveedor** (Gemini:
  `GET models/{id}` → `outputTokenLimit`; OpenAI-compatibles: el campo del listado si existe, si no, la
  documentación del proveedor citada en la migración). Va como migración de datos versionada en
  jax-platform (patrón `_fix_*` idempotente con marcador), no como UPDATE suelto. Un modelo cuyo valor no se
  pueda medir queda NULL y el pre-vuelo lo rechaza con `sin_contrato_de_salida`: se declara en el commit.

## 8. Errores y fallos

- DB caída durante el pre-vuelo → excepción → 503 `prevuelo_no_disponible` en Jacobs (no 500 genérico, no
  fail-open: sin pre-vuelo no se crea).
- Sonda con timeout → `faceta_caida` con `detalle="timeout de sonda (Ns)"`.
- Registro de la sonda en `facet_health_event` falla → el veredicto se mantiene (el dato es la respuesta del
  proveedor, no la fila); se loguea `WARNING` y se cuenta. Marcado `# fail-soft:` con razón (P10).
- Carrera entre dos `continue` del mismo pipeline: el candado serializa; el segundo ve `running` → 409.
- Carrera `cancel` durante `continue`: gana el último que escribe status; si gana cancel, el ejecutor nuevo
  termina en su primera escritura (§5.3).

## 9. Pruebas

- **Pre-vuelo (puros):** cada regla con un caso que pasa y uno que bloquea; salud con eventos `gate_*` que no
  cuentan; sonda ok/timeout/HTTP 5xx; costo con cada multiplicador; precio NULL → no acotado; `hyde` y `jax_local`.
- **Pre-vuelo (DB, `jax_memory_test`):** consulta real de catálogo/credencial/salud contra el esquema migrado
  (job `jacobs-gobernanza-db`).
- **Continuar:** cada estado continuable y no continuable; refs ilegibles; reasignación válida e inválida
  (clean-room); candado y límite; transacción (fallo a mitad → nada cambió).
- **Época:** ejecutor viejo que pierde la época no escribe (test que **falla contra el código de hoy**: cancelar y
  ver que la ola siguiente reescribe `running`); doble continue; resume doble.
- **Mesa backend:** 409 de confirmación, 422 con violaciones, propagación de 4xx en los 5 endpoints (cada uno
  falla contra el código de hoy), ajuste nuevo.
- **Mesa frontend:** modal no se cierra con violaciones; confirmación en ventana propia; continuar con reasignación;
  i18n paridad; escaneo de diálogos del navegador.
- **Canarios de CI:** cada suite nueva entra a su job y se ve roja una vez a propósito antes de mergear.
- **Carga (LAS CUATRO, #4):** `/jacobs/preflight` con catálogo y salud frescos (sin sondas) a c=25 sobre
  `jax_memory_test` en instancia aislada: p95 y rps registrados; y `continue` concurrente sobre el mismo
  pipeline (sólo uno gana). Índices: `EXPLAIN` de las consultas nuevas (`facet_health_event (facet, ts)` existe;
  `credential (provider_id, state)` se verifica).

## 10. Orden y convivencia con otros frentes

Los frentes E y F (jax) y B (jax-platform) de la otra instancia tocan `jacobs/routes.py`, `executor.py`,
`store.py`, `policy.py` y el resume de la Mesa. Esta rama sale de `origin/master` y **se rebasea sobre master
antes de mergear**; los conflictos se resuelven preservando lo de ellos. No se tocan sus worktrees ni ramas.
El deploy se hace con 0 pipelines en vuelo.

## 11. Rescate (sección G)

Después del deploy, con Kimi en el máximo del catálogo: `continue` del pipeline `ef9b2d6e` por la API de Jacobs
con `invoked_by=plataforma`, sin reasignar. Evidencia: pasos 0-3 reusados (sin nuevas filas de uso para ellos),
paso 4 `completed` con `finish_reason=stop`, paso 5 (Thot) `completed`, pipeline `completed`, costo real contra el
costo máximo estimado. El resultado queda para Fernando en la Mesa.
