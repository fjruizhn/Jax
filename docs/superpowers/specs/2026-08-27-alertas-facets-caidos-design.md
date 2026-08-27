# Alertas de facets caídos — diseño

Fecha: 2026-08-27
Ítem: primer "Bloquea trabajo" de `DEUDA.md` — *"No existe ninguna señal
que avise cuando un facet deja de responder"*.
Repos afectados: `jax-platform` (escritor + sonda) y `jax` (lector +
alerta).

---

## 1. Diagnóstico — qué existe hoy

### 1.1 Los seis estados de una invocación, y qué los distingue

`_invoke_facet` (`jax-platform/backend/api/chat.py:781-875`) tiene seis
caminos de salida. Hacia afuera son casi indistinguibles:

| # | Estado | Qué devuelve | Rastro persistente hoy |
|---|---|---|---|
| 1 | Responde bien | `(texto, UsageInfo(...))` | fila en `axioma_usage` (`request_type='chat'`) |
| 2 | Error del proveedor | **lanza** → `chat()` → HTTP 502 | ninguno en DB; solo `journalctl` |
| 3 | Gate deniega correctamente | `("⚠️ … acceso no autorizado.", None)` | ninguno; `logger.warning("authorize-facet denied … reason=…")` |
| 4 | Gate deniega por error | **el MISMO string**, `None` | ninguno; `logger.warning("authorize-facet unreachable … error=…")` |
| 5 | Sin binding activo | `("⚠️ … sin binding activo configurado.", None)` | ninguno |
| 6 | Transporte no soportado | `("⚠️ … transporte 'X' no soportado…", None)` | ninguno |

**Conclusión central: ninguna señal estructurada distingue 2 a 6.** Los
estados 3 y 4 sí están separados, pero **únicamente en el texto libre de
dos `logger.warning`** (`chat.py:819-836`; el comentario del código
explica por qué se separaron ahí). Por encima de esa línea de log son
idénticos: mismo string al usuario, mismo HTTP 200, misma ausencia de fila
en `axioma_usage`.

Un `⚠️` con `usage=None` lo trata `chat()` como `is_canned=True` — el
mismo trato que la respuesta enlatada de `hyde`. **Un fail-closed
disparado por error se ve, en cada capa por encima de `_invoke_facet`,
igual que un turno exitoso.**

`journalctl` retiene 663 MB rotativos, sin ventana garantizada.

### 1.2 Las cinco señales candidatas, todas descartadas

Las cuatro de `DEUDA.md` (`/health`, `facet.status`, `model.status`,
errores por turno) siguen descartadas por las razones ya registradas ahí.
Se suma una quinta, no listada antes:

- **`model_catalog.record_resolved_version()`** (`model_catalog.py:312`)
  detecta drift de versión y crea un `model_binding_proposal` — precedente
  de la decisión D1.1, *"la alerta ES la proposal pendiente, sin tabla de
  log aparte"*. **No sirve para esto:** corre en `on_response`, o sea
  únicamente cuando la llamada al proveedor **tuvo éxito**. Evidencia: el
  drift de `thot` se registró recién el 2026-08-27 10:05
  (`model_catalog drift facet=thot from=gpt-5.5-2026-04-23
  to=gpt-5.6-terra proposal_id=7`), cuando `thot` volvió a responder —
  tres días después de romperse.

Regla que se repite en las cinco: **todo lo que existe se alimenta de
éxitos.**

### 1.3 El hallazgo que reencuadra el ítem: no hay tráfico que observar

`DEUDA.md` propone *"una señal derivada del tráfico real"*. El tráfico
real, medido:

```
messages, por día y rol (14 días, consultado 2026-08-27)
2026-08-14  user 3  | jax_local 3
2026-08-17  user 1  | jax_local 1
2026-08-18  user 5  | jax_local 2, jekyll 2, hipatia 1
2026-08-19  user 3  | jax_local 3
2026-08-27  user 16 | jekyll 3, hipatia 3, thot 1, ada 3
```

**Del 2026-08-20 al 2026-08-26 inclusive: cero turnos de chat.** Ni uno.

`facet_binding` fija la ventana del incidente: `thot` quedó rebindeado a
`gpt-5.6-terra` con `approved_at = 2026-08-24 11:08:01`; se descubrió el
08-27. **Durante los tres días que `thot` estuvo roto, nadie lo llamó.**

Un detector pasivo derivado del tráfico —contar errores por ventana,
mirar caída de tasa de éxito— **no habría detectado nada**. El modo de
falla real no es *"un facet que funcionaba empieza a fallar bajo carga"*;
es **"un facet queda roto y nadie lo ejercita durante días"**. La
observación pasiva es estructuralmente incapaz de cerrar ese gap con este
perfil de uso.

Corolario: la ausencia de filas en `axioma_usage` es ambigua por diseño
(solo se escribe en éxito), así que *"cero éxitos"* significa
simultáneamente *"roto"* y *"nadie lo usó"*. Cualquier umbral sobre
éxitos daría falsa alarma todas las noches.

**Por eso la v1 es una sonda activa, no un contador pasivo.** Es la
diferencia con lo que `DEUDA.md` anticipaba, y es la decisión de diseño
más importante del documento.

### 1.4 El momento de riesgo es un evento discreto y registrado

Los tres rebindings del 2026-08-24 (`thot` 11:08:01, `ada` y `jax_local`
14:18) están en `facet_binding.approved_at`. El cambio de estado que
rompió `thot` **quedó grabado con timestamp**. Una sonda disparada por ese
evento baja la detección de tres días a minutos.

**`facet_binding` tiene DOS escritores, no uno:**

1. `POST /api/admin/models/proposals/{id}/approve` — `api/admin/models.py:148`
2. `PUT /api/admin/facet-bindings/{facet_key}` — `api/admin/facet_bindings.py:87`

El docstring del primero registra que esta misma duplicación ya causó un
incidente: *"volvio a fallar 5 dias despues porque PUT
/api/admin/facet-bindings/{key} escribia facet_binding sin pasar por
aca, y nada sincronizaba motor para ese camino"*. Colgar la sonda solo de
`approve` reproduce ese bug exactamente. **Va en los dos, vía un helper
compartido.**

(Nota: el `CLAUDE.md` global afirma que `approve` es el único escritor de
`facet_binding`. Ya no es exacto — corregirlo es una acción aparte de esta
ronda.)

### 1.5 Infraestructura reusable, verificada corriendo

- **`jacobs/reaper.py`** — loop de 300 s, **corriendo en producción**
  (`journalctl -u jax-las-manos`, reconciliación horaria, última corrida
  verificada 2026-08-27 10:38). Tiene `send_telegram_alert()` con entrega
  verificada (T1, 2026-08-19) y el patrón *esperado vs real por facet* ya
  implementado.
- **`jax-platform`** — precedente de loop en background:
  `asyncio.create_task(start_owner_file_cleanup())` en el `lifespan`
  (`main.py:87`), con su `while True / try / except fail-soft / sleep`.
- **CI real en ambos repos**: `jax-platform` corre la suite del backend
  sin DB con piso de cobertura (`JAX_CI_MIN_PASSED=70`); `jax` corre un
  job por archivo de test.

El scheduler, el canal de alerta y el patrón de agregación **ya existen y
están probados**. Lo único que falta es la fuente de verdad.

---

## 2. Diseño

Cuatro piezas. **La salud se calcula en un solo lugar.**

```
jax-platform                                  jax (las_manos)
────────────────────────────────────          ────────────────────────
_invoke_facet ──┐                             reaper (cada 300 s)
                ├──> facet_health_event ────>   facet_health.evaluate()
canary loop ────┤        (jax_memory)              │
(1 h + rebind)  │                                  ├─> facet_health_alert
                ┘                                  └─> send_telegram_alert()
```

### 2.1 `facet_health_event` — la única fuente de verdad

```sql
CREATE TABLE facet_health_event (
  id      BIGINT AUTO_INCREMENT PRIMARY KEY,
  facet   VARCHAR(50) NOT NULL,
  outcome ENUM('ok','provider_error','gate_denied','gate_unreachable',
               'unbound','unsupported_transport','probe_error') NOT NULL,
  source  ENUM('chat','canary_periodic','canary_rebind') NOT NULL,
  detail  VARCHAR(255) NULL,
  ts      DOUBLE NOT NULL,
  KEY idx_facet_ts (facet, ts),
  KEY idx_ts (ts)
);
```

`ts` es epoch `DOUBLE`, **no `TIMESTAMP`** — misma decisión y misma razón
que `jacobs_events.ts`: inmune a la timezone de sesión. La limpieza de
`axioma_usage` del 2026-08-21 perdió 90 de 106 filas por comparar un
`TIMESTAMP` contra un string de fecha; ese error no se repite acá.

`detail` es diagnóstico interno, nunca se muestra en UI — no entra en i18n.

Migración vía `backend/db/migrations.py`, que es el mecanismo que ya
resolvió el drift de esquema de la ronda anterior.

### 2.2 Escritor — instrumentar el chokepoint

El cuerpo actual de `_invoke_facet` pasa a `_invoke_facet_dispatch(...)`,
que devuelve `(texto, usage, outcome)` — **el `outcome` es un literal
tipado en cada punto de retorno, nunca se deduce del texto**. Deducirlo
del string sería frágil y además esos strings son texto de usuario en
español (política i18n).

`_invoke_facet` queda como envoltorio y conserva su firma pública
`(facet, config, user_id, message, semantic_context) -> tuple[str,
UsageInfo | None]`, más un parámetro nuevo **keyword-only**
`source: str = "chat"`.

**Criterio de cierre de esta partición, no negociable:** los dos tests
que ya llaman a `_invoke_facet` directo
(`test_chat_facet_validation.py:333`,
`test_chat_resolved_version_capture.py:126`) deben seguir pasando **sin
tocarlos**. Si alguno hay que modificarlo, la firma pública cambió y eso
no es lo acordado: **se para y se reporta**, no se ajusta el test. Mismo
criterio que se usó con `check()` en la ronda anterior. Un test editado
para acomodar un refactor deja de ser evidencia de que el refactor no
rompió nada.

```
try:
    texto, usage, outcome = await _invoke_facet_dispatch(...)
except Exception as e:
    _record_health(facet, "provider_error", source, f"{type(e).__name__}: {e}")
    raise                      # <-- SIEMPRE re-lanza
_record_health(facet, outcome, source, ...)
return texto, usage
```

- El `except` **re-lanza siempre**. No puede volverse fail-open; el
  scanner P10 del CI lo verifica mecánicamente.
- Instrumentar **dentro** de `_invoke_facet`, no en un envoltorio que el
  llamador tenga que acordarse de usar: así el chat real y la sonda quedan
  cubiertos por construcción, sin una segunda ruta que pueda divergir.
- `_record_health` es fail-soft ante error de DB (un fallo escribiendo
  salud no puede tumbar un turno de chat) pero **nunca silencioso**:
  loguea. Y la ausencia resultante se convierte en `unknown`, no en `ok`
  — ver §2.5. Ese `except` fail-soft lleva el comentario marcador que
  exige el scanner P10 (`# fail-soft: …` con el motivo), igual que los
  de `owner_cleanup.py` y `state.py`; sin el marcador el CI lo rechaza,
  que es exactamente lo que debe pasar con un `except` sin justificar.

### 2.3 Sonda — garantiza que haya tráfico

`jax-platform/backend/jax_engine/facet_canary.py`, mismo patrón que
`owner_cleanup.py`, arrancada con `asyncio.create_task()` en el
`lifespan`.

- **Intervalo:** `CANARY_INTERVAL_SECONDS = 3600`.
- **Qué sondea:** `config["personalities"].keys() - {"hyde"}` — el mismo
  conjunto contra el que `chat()` valida `req.facet` (`chat.py:902`), o
  sea exactamente lo que un usuario puede elegir. `hyde` se excluye porque
  `chat()` lo corta antes del dispatch con una respuesta enlatada.
  **No se filtra por transporte a propósito**: así `kimi` se sondea y
  reporta `unsupported_transport` en vez de quedar invisible (§4).
- **Cómo:** llama a `_invoke_facet(...)`, **la misma función que usa el
  chat real**, no una reimplementación. `_invoke_facet` no escribe
  `axioma_usage` ni `messages` (eso ocurre en `chat()`), así que la sonda
  no contamina datos de usuario ni memoria semántica.
- **La sonda NO puede saltearse el gate.** Pasa por
  `POST /motor/authorize-facet` con el mismo caller `jax_platform_chat`
  que el chat real, porque los estados 3 y 4 —los que hoy son
  indistinguibles— **solo ocurren dentro del gate**. Una sonda que
  resolviera el facet por su cuenta y llamara al proveedor directo
  reportaría `ok` con el gate denegando a todos los usuarios reales:
  sería el detector mintiendo en verde, la falla que este ítem existe
  para eliminar.
- **Usuario sintético:** `CANARY_USER_ID`, un id que no puede colisionar
  con uno real. `_invoke_facet` solo **lee** `_conversations`, nunca
  escribe, así que la sonda no ocupa lugar en el LRU.
- **Trampa conocida, con test obligatorio:** `_is_model_identity_question(message)`
  cortocircuitea **antes** del dispatch y devuelve una respuesta enlatada.
  Si el mensaje de sonda pareciera una pregunta de identidad, la sonda
  reportaría `ok` **sin haber tocado al proveedor**. Un test verifica que
  `CANARY_MESSAGE` no dispara ese cortocircuito.
- **Costo:** ~1k tokens por sonda, 4 facets con proveedor pago, cada hora
  ≈ 96 llamadas/día. Aceptado explícitamente por Fernando.
- Si la sonda falla **antes** de llegar a `_invoke_facet` (no pudo leer
  config, etc.), escribe `probe_error`. Un fallo del detector es un
  evento, no un silencio.

### 2.4 Sonda por rebinding

Requisito: **debe dispararse DESPUÉS de que el binding quedó aprobado**, y
su resultado debe ser visible en esa misma ventana.

Helper compartido llamado desde **los dos** escritores de `facet_binding`
(§1.4), en cada caso **después del `conn.commit()`**, encolado con
`BackgroundTasks`. FastAPI corre las background tasks después de emitir la
respuesta, o sea estrictamente después del commit: el orden "primero
aprobado, después sondeado" está garantizado por construcción, no por
convención.

Escribe con `source='canary_rebind'`.

**Visibilidad en la misma ventana:** el reaper evalúa la salud **en cada
barrido de 300 s** (no cada hora). Un rebinding que rompe un facet se
alerta en ≤5 minutos, sin necesidad de un segundo alertador en
`jax-platform` — la lectura es barata (una consulta), lo caro es la
llamada al LLM y esa sigue siendo horaria.

*Alternativa rechazada:* `await` inline de la sonda dentro del endpoint de
approve. Sería más visible (el resultado viajaría en la respuesta) pero
cuelga la request del admin de una llamada a un proveedor externo: si el
proveedor se cuelga, el approve se cuelga.

### 2.5 Lector y alerta — y por qué ausencia de datos NO es salud

`jax/jacobs/facet_health.py`, invocado desde `start_reaper_loop()` en cada
barrido. **Es el único lugar donde se calcula la salud.**

Estado por facet sobre una ventana de `2 × CANARY_INTERVAL` (2 h):

| Estado | Condición |
|---|---|
| `ok` | el evento más reciente del facet en la ventana es `ok` |
| `down` | el evento más reciente es cualquier otro `outcome` |
| `unknown` | **cero eventos del facet en la ventana** |

**`unknown` alerta igual que `down`.** Este es el requisito explícito de
Fernando y la lección del `OK -- 0/0 dispatches (0.0% gap)` del propio
reaper —verde sobre cero eventos— y del scanner P10 de `jax`, que estuvo
verde meses reportando `1 passed` sobre cero archivos escaneados.

Reglas que lo hacen cumplible, no solo declarado:

1. El chequeo es `if total_eventos == 0: unknown`. **Nunca**
   `if fallos == 0: ok`. Esa segunda forma es el bug, escrita como
   código.
2. `unknown` cubre de un solo tiro: sonda muerta, escritor roto,
   `jax-platform` caído, y un facet agregado al picker que nadie sondea.
3. Si **ningún** facet tiene eventos en la ventana, se alerta una sola vez
   por el sistema entero (*"la sonda no está corriendo"*), no cuatro veces
   por facet — el diagnóstico es distinto y el mensaje debe decirlo.
   Esa alerta agregada **pasa por el mismo ledger y la misma supresión**
   que las de facet, bajo la clave centinela `__system__` en
   `facet_health_alert`: se avisa en la transición y después a lo sumo
   cada `ALERT_REPEAT_SECONDS`. Sin esto, una sonda que muere un viernes
   produciría un mensaje por barrido — 288 por día — que es ruido, y el
   ruido es una forma de no avisar. `__system__` no puede colisionar con
   un facet real: los nombres de facet vienen de `facet.key`, ninguno
   empieza con guión bajo, y un test lo verifica.
4. `probe_error` es un `outcome` como cualquier otro: un detector que
   falla produce un evento, no un silencio.

**Anti-spam.** Se alerta en **transiciones** (`ok→down`, `ok→unknown`,
`down→ok` como recuperación), no en cada barrido. El estado previo vive en
`facet_health_alert (facet, state, first_seen_ts, notified_ts)` — persiste
entre reinicios de `las_manos`, a diferencia de `engine_state`, que es RAM
por proceso y se borra sola (motivo por el que no sirve como señal, §1.2
de `DEUDA.md`). Mientras un facet sigue caído, se repite el aviso a lo
sumo cada `ALERT_REPEAT_SECONDS = 6h`.

`facet_health_alert` **no es una segunda fuente de verdad de salud**: es
el registro de *qué ya se avisó*. La salud se sigue calculando
exclusivamente desde `facet_health_event`. La distinción es la misma que
entre un valor y su acuse de recibo.

**Retención.** El mismo barrido borra las filas de `facet_health_event`
con más de `HEALTH_EVENT_RETENTION_DAYS = 30`. Sin esto la tabla crece
para siempre (~96 filas/día solo de sonda) — es el error que motivó
`owner_cleanup.py`, cuyo docstring dice textual *"Had no cleanup, so the
directory grew forever"*. 30 días es holgadamente mayor que la ventana de
2 h que usa el lector: la poda no puede fabricar un `unknown`.

**Canal:** `send_telegram_alert()` del reaper, sin duplicar el sender. El
mensaje incluye facet, estado, `outcome`, `detail` y desde cuándo.

---

## 3. Cómo se prueba que detecta

Un detector que nunca detectó nada es una hipótesis. Se rompe a propósito
y se verifica que avisa — igual que el CI que se puso rojo al romper el
guard de `api/command.py`.

**En vivo, cada uno reversible:**

1. `facet.allowed_callers = '[]'` en un facet → debe registrar
   `gate_denied` y alertar. Revertir.
2. `LAS_MANOS_URL` a un puerto muerto → debe registrar
   `gate_unreachable`, **distinto** del anterior. **Ésta es la prueba de
   que los estados 3 y 4 quedaron separados**, que es el objetivo central
   del ítem.
3. Rebindear un facet a un modelo inexistente → `provider_error`, alertado
   por el camino de rebinding en ≤5 min. Revertir.
4. Parar la sonda con eventos frescos en la tabla, esperar la ventana →
   debe pasar a `unknown` y alertar. **Es la prueba del punto A**:
   ausencia de datos no es salud.

**En CI:**

- `jax-platform`, job `backend-tests-no-db` (tiene piso de cobertura, así
  que no puede quedar verde sin correr): escritor con DB parcheada, mapeo
  de los seis `outcome`, que el `except` **re-lanza**, y que
  `CANARY_MESSAGE` no dispara `_is_model_identity_question`.

- **Test propio y explícito: la sonda no puede reportar `ok` con el gate
  denegando.** Gate parcheado para devolver `allowed:false` → la sonda
  debe registrar `gate_denied`, y **la aserción es sobre el `outcome`
  exacto, no sobre "no fue ok"**. Segundo caso en el mismo test: gate que
  lanza (las_manos inalcanzable) → `gate_unreachable`, distinto del
  anterior.

  No queda cubierto de refilón por el escenario 1 de §3 (romper en vivo):
  ese se corre una vez y no protege de nada después. Éste corre en cada
  push. Es el caso que, si un refactor futuro lo rompe, devuelve el
  sistema **exactamente** al problema que esta ronda vino a cerrar — una
  sonda verde sobre un gate que deniega a todos los usuarios reales — y
  lo devuelve en silencio, que es la parte peor.
- `jax`, job nuevo `facet-health-unit` sobre `jacobs/_facet_health_test.py`
  (un job por archivo, como los cuatro existentes): la máquina de estados
  pura, con foco en `unknown` — cero eventos, y eventos viejos fuera de
  ventana.

---

## 4. Qué NO cubre esta v1 — declarado explícito

Una alerta que se crea más completa de lo que es sería una instancia del
patrón dentro del ítem que existe para detectar el patrón.

- **Jacobs no queda cubierto.** `_dispatch_step` no se instrumenta. Un
  facet puede estar sano para Mesa web y fallar en un pipeline por su
  propio camino de admisión (`check_capability_admission`).
- **`kimi` y `hyde` no son sondeables de verdad.** Transportes
  `motor_registry` y `subprocess`, que `_invoke_facet` no despacha. `kimi`
  **sí se sondea** y va a reportar `unsupported_transport` — es
  precisamente lo que se quiere ver (§5). `hyde` queda fuera del conjunto.
- **La sonda no prueba lo que prueba un usuario real:** prompt corto, sin
  historial, sin contexto semántico, sin tool use. Un facet que falla solo
  con contexto largo pasa verde.
- **No detecta degradación de calidad**, solo disponibilidad. Un facet que
  responde basura cuenta `ok`.
- **La salud se calcula sobre datos que pueden estar incompletos.** Si el
  escritor pierde filas en silencio, la salud miente. Mitigado
  parcialmente por `unknown` (pérdida total se ve), no por pérdida
  parcial. Conectado con §6.
- **Un facet nuevo en el picker sin sondear** queda cubierto por `unknown`
  únicamente si aparece en `config["personalities"]`. Si se agrega al
  frontend sin agregarlo ahí, es invisible — y sigue siendo la misma
  duplicación `FACET_ORDER` (frontend) vs `personalities` (backend) que ya
  existe hoy.

---

## 5. `kimi` — anotado, no decidido en esta ronda

`kimi` está en el picker de la Mesa web (`BottomBar.jsx:12`,
`LeftPanel.jsx:6`) y en `config.toml` como personality, pero su
`facet.transport` es `motor_registry`, que `_invoke_facet` no despacha:
cae al `return` final,
`"⚠️ kimi no está disponible: transporte 'motor_registry' no soportado en
la Mesa web."` **Es inalcanzable, y lo es desde que existe el picker.**

**Deducido de código + fila de DB, NO probado en vivo** — verificarlo
requeriría un JWT y un turno real de chat. Queda escrito así a propósito.

Es una decisión de producto (¿rutear `kimi` por Motor Registry desde Mesa
web, o sacarlo del picker?) y no se toma en esta ronda.

**El dato que importa: es exactamente la clase de falla que este ítem
existe para detectar, invisible hoy. Cuando la v1 esté corriendo, `kimi`
debe aparecer como caído. Si no aparece, la v1 tiene un hueco** — y ese es
el criterio de aceptación, no una observación de color.

---

## 6. Aparte: 6 turnos sin respuesta guardada el 2026-08-27

El 2026-08-27 hay 16 filas `user` en `messages` y 10 de facet: **6 turnos
sin fila de respuesta**.

Dos explicaciones compatibles con la evidencia, **ninguna probada**:

1. Los 502 de `thot` — el camino de excepción de `chat()` no guarda fila
   del asistente.
2. Pérdidas de escritura de `save_message()`, que es fire-and-forget sin
   confirmación ni id de vuelta (deuda ya registrada,
   `save-message-fire-and-forget-sin-garantia`).

**Importa para este diseño**, no es color: si el escritor pierde filas en
silencio, la salud se calcula sobre datos incompletos — el mismo riesgo
listado en §4. Se anota como ítem propio, separado del de alertas, sin
resolverse acá.

---

En memoria de Jairo Urbina.
