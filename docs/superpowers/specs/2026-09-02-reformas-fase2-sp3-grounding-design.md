# REFORMAS-v3 Fase 2 — Sub-proyecto 3: grounding por snapshot inyectado

**Fecha:** 2026-09-02
**Estado:** diseño aprobado por secciones, pendiente de revisión del documento
**Repos:** `jax` (gobernanza pura) + `jax-platform` (integración Mesa web)
**Predecesor:** Sub-proyecto 2 (`2026-08-18-reformas-fase2-sp2-integracion-real-design.md`)
y el arreglo del prompt de contrato (`jax-platform` #43, mergeado 2026-09-02,
`master` 36b7728).

---

## 1. El problema, medido

`validator.py:249` corta antes de resolver:

```python
if claim.authority == "INFERIDO":
    return Verdict(status="AUTHORITY_INVALID", ...)
```

y `shadow_validation.py:190` fija `authority="INFERIDO"` en duro para todo
claim. Los resolvers reales existen y funcionan, pero son **inalcanzables**:
ningún claim de la Mesa web puede llegar a ser resuelto contra fuente de
verdad.

Medido contra `jax_memory` (:3308) el 2026-09-02, después de que #43
arreglara la emisión:

| tabla | valor |
|---|---|
| `shadow_messages` | 30 filas, 2026-08-18 → 2026-09-01 |
| filas 1–22 (antes de #43) | `has_claim = 0` en las 22 |
| filas 23–27 (pregunta orgánica, 5 facetas) | `has_claim = 0` |
| filas 28–29 (pedido explícito, `jax_local`, `jekyll`) | `has_claim = 1` |
| fila 30 (`thot`, pedido explícito) | `has_claim = 0` — se negó |
| `shadow_claim_verdicts` | 4 filas, **todas** `AUTHORITY_INVALID` |

O sea: el canal ya emite, y todo lo que emite muere en autoridad. Ese es el
objeto de este sub-proyecto.

**Lo que este spec NO asume:** que arreglar la autoridad haga que las facetas
afirmen espontáneamente. Ver §8.

---

## 2. Decisión de enfoque: pre-grounding inyectado

De tres caminos considerados —tool-calling real en la Mesa web; pre-grounding
inyectado; reencuadrar la autoridad como propiedad del validador— se elige el
**pre-grounding inyectado**.

**Razón.** Cumple §3.1.4 al pie sin enmendar el corpus, y no cierra la puerta
a los otros dos. El tool-calling es el diseño correcto pero prematuro: todavía
no sabemos qué predicados quieren usar las facetas, y ese dato lo va a dar el
shadow con el prompt ya arreglado. Construir la cola de resolvers antes de esa
medición es diseñarla a ciegas.

### 2.1 Por qué el pre-grounding no fuerza la lectura de §3.1.4

§3.1.4 define `OBSERVADO` como *"se leyó estado del sistema sin mutarlo"* y
exige *"hash de resultado de herramienta"*. El texto es **neutral respecto de
quién leyó** — no dice que la lectura la haga el modelo. §3.1.2 muestra
`"provenance_ref": "tool_result:sha256:..."` sin definir el campo.

P08 (*"la autodeclaración de calidad por parte del modelo no es
contractual"*) se **refuerza**, no se estira: el modelo no declara autoridad
ni procedencia. Solo señala qué línea del snapshot está afirmando. Autoridad y
`provenance_ref` los escribe el servidor tras verificar.

### 2.2 La condición sin la cual el mecanismo sería un lavado

Citar no puede alcanzar. Si el servidor inyecta
`CAPABILITY_AVAILABLE(name=write_file, mode=mutating)` en `/capabilities/3` y
el modelo emite `CAPABILITY_AVAILABLE(name=write_file, mode=read_only)`
citando `/capabilities/3`, el puntero es real y el claim es falso.

**Conceder `OBSERVADO` por "el puntero resuelve" lavaría priors del modelo
como observación** — exactamente lo que P08 prohíbe, y dejaría §3.1.4 vacío.

Por lo tanto: `OBSERVADO` se concede **si y solo si** el puntero resuelve a una
entrada del snapshot **y los args del claim coinciden exactamente con esa
entrada**. Una citación que no coincide no es `INFERIDO`: tiene estado propio
(§4).

---

## 3. Qué entra al snapshot

**Solo hechos de predicados que tienen resolver real.** El límite lo decide el
código, no una lista curada: el snapshot se genera desde las mismas fuentes
que consulta el validador.

**Invariante:** *todo hecho inyectado tiene quién lo re-resuelva.* El snapshot
no puede contener autoridad sin fuente de verdad, y crece solo si crece el
número de resolvers — acoplado a trabajo real, no a criterio.

**Primer corte: `CAPABILITY_AVAILABLE` solo** (11 entradas, las `[ops.*]` de
`las_manos/config.toml`).

Dos razones. La barata: `FACET_EXISTS` necesita que se le construya el
resolver y arrastra la decisión de §3.1. La fuerte: con el snapshot cubriendo
un solo predicado, los claims de los otros siete llegan **sin grounding en el
mismo tráfico, mismas facetas, mismos días** — un grupo de control dentro de
la misma ronda, que separa "el mecanismo no funciona" de "esta faceta no
afirma" sin necesitar una segunda. Es el mismo método del probe explícito que
salvó el diagnóstico de SP2.

Por eso **el sufijo de contrato sigue ofreciendo los 8 predicados**. Recortarlo
a los que tienen grounding destruiría el control y dejaría de medir qué querrían
afirmar las facetas — el dato que decide el alcance de SP4.

### 3.1 `FACET_EXISTS`: el predicado no está mal definido, la fuente que se
### buscó primero era la equivocada

Al construir un snapshot de prueba con facetas, `engine` salió `desconocido`.
Diagnóstico: `las_manos/config.toml [facets]` **no es un registro de
facetas**. Sus entradas son `allowed_ops` / `allowed_envs` / `can_write_prod` —
es la tabla de autorización identity-based (la tensión P04 del corpus). No
tiene `engine` porque no es su trabajo tenerlo.

La fuente correcta es `facet_binding.provider_id` en `jax_memory`, que da el
valor exacto: `ollama`, `anthropic`, `openai`, `gemini`, `deepseek`, `zhipu`,
`moonshot`.

**Precisión que el corpus hoy no tiene:** `predicates.yaml` declara
`source_of_truth: "Configuración de facetas"`, que es vago. Cuando se
construya el resolver de `FACET_EXISTS` (fuera del alcance de este spec) debe
fijarse **cuál binding**: `role='primary'`, no los `fallback_*`.

### 3.2 Divergencia de listas de facetas: medida, y NO es la clase de
### `PROVIDER_ENV_KEYS`

Se sospechó una tercera instancia del patrón "mapa replicado con copia
incompleta". Medido:

```
facet_binding (role=primary):      7   ada hipatia hyde jax_local jekyll kimi thot
config.toml [personalities]:       7   ← idénticas
las_manos/config.toml [facets]:    5   ← las 7 menos jax_local y kimi
solo en personalities: jax_local, kimi
solo en las_manos:     (ninguna)
```

No hay copia divergente: `las_manos [facets]` es **subconjunto propio y de otra
cosa** — las facetas que LAS MANOS gatea con `allowed_ops`. `jax_local` y
`kimi` no están porque no invocan ops de LAS MANOS. Dos conjuntos con
propósitos distintos que se solapan, no dos copias de uno.

**La instancia sospechada no existe. No se registra como hallazgo.**

---

## 4. Semántica de veredictos

### 4.1 Orden de chequeos

El orden es normativo: define qué falla se le imputa a quién.

```
0. snapshot del turno en ERROR      → authority=INFERIDO → GROUNDING_UNAVAILABLE
1. predicado desconocido            → UNKNOWN_PREDICATE
2. claves de args mal               → ARGS_MISMATCH
3. SIN resolver para el predicado   → RESOLVER_NOT_IMPLEMENTED
4. con resolver, sin puntero        → authority=INFERIDO → AUTHORITY_INVALID
5. puntero que no resuelve, o args que no coinciden → PROVENANCE_MISMATCH
6. acreditado                       → authority=OBSERVADO → corre el resolver
                                      → VALID / FACT_MISMATCH / SOURCE_CONFLICT
```

**El paso 3 va antes del 5 a propósito:** `JOB_STATUS` con puntero inventado
da ausencia-de-resolver, no citación-falsa. No se puede acusar de falsear una
cita a quien nunca tuvo dónde citarla.

**El paso 0 va antes que todo por lo mismo, en el otro sentido:** si el
snapshot del turno falló al construirse, la falla es del sistema. Un claim de
ese turno no es `PROVENANCE_MISMATCH` (no hay contra qué comparar) ni
`RESOLVER_NOT_IMPLEMENTED` (el resolver existe).

Estos cuatro estados separan cuatro cosas que hoy se confunden bajo una sola
etiqueta:

| condición | de quién es la falla |
|---|---|
| `RESOLVER_NOT_IMPLEMENTED` | deuda del sistema: no hay resolver |
| `GROUNDING_UNAVAILABLE` | falla del sistema: el snapshot no se construyó |
| `AUTHORITY_INVALID` | conducta del modelo: se ofreció grounding y no citó |
| `PROVENANCE_MISMATCH` | conducta del modelo: citó algo que no dice eso |

### 4.2 Estados nuevos

**`PROVENANCE_MISMATCH`** — aprobado.

**`GROUNDING_UNAVAILABLE`** — propuesto, **y contradice la decisión de "un solo
estado nuevo"**. Se propone igual porque la alternativa es peor: reusar
`AUTHORITY_INVALID` y distinguir por join a
`shadow_messages.grounding_snapshot_sha256 = 'ERROR'` obliga a que **cada
análisis futuro se acuerde de hacer el join**. Eso convierte "mirar no es
medir" en una propiedad del esquema, y contaría una falla del sistema como
conducta del modelo en cualquier consulta que omita el join.

**Punto abierto para la revisión: si se rechaza, el spec vuelve a
`AUTHORITY_INVALID` + join, con la consecuencia escrita.**

**No se agrega `UNGROUNDED`.** Con la definición acordada —ausencia de resolver
para el predicado— es la misma condición que `RESOLVER_NOT_IMPLEMENTED`, que ya
existe en el `Literal` de `Verdict` y hoy es inalcanzable solo porque
`authority` corta antes. Bajo el orden nuevo pasa a ser alcanzable y su nombre
ya lo describe. Agregar un sinónimo sería la deriva de nombres que este
proyecto viene pagando.

Longitudes contra `status VARCHAR(30)`: `PROVENANCE_MISMATCH` 19,
`GROUNDING_UNAVAILABLE` 21, el más largo existente `RESOLVER_NOT_IMPLEMENTED`
24. **Entran sin ampliar la columna.**

### 4.3 Autoridad y veredicto son dimensiones distintas

No son alternativas: responden preguntas distintas.

- **Autoridad** — ¿el claim nació de leer estado real o de priors del modelo?
  Es una pregunta sobre el **origen**, y el origen es histórico: se acredita
  **contra el snapshot**, siempre.
- **Veredicto** — ¿es cierto **ahora**? Lo contesta el resolver contra estado
  vivo.

Corren las dos, en ese orden. Un claim `OBSERVADO` + `FACT_MISMATCH` es
información legítima y nueva: el modelo citó bien y el mundo cambió entre la
inyección y la validación. Colapsar las dos capas perdería la distinción entre
"el modelo citó mal" y "el mundo cambió", que son fallas de naturaleza opuesta.

**Requisito de esquema:** hoy `shadow_claim_verdicts` **no tiene columna de
`authority`** — se fija en `shadow_validation.py:190` y se descarta. Sin la
columna, `OBSERVADO + FACT_MISMATCH` no es que se pise: la mitad no se guarda.
Ver §6.

---

## 5. Mecanismo

### 5.1 Inyección

`chat.py` construye el snapshot desde el `ValidationContext` que el validador
ya usa (`ctx.ops`, `ctx.mutating_capabilities`), lo serializa canónicamente,
saca su sha256, y anexa al system prompt **solo la lista con sus
`evidence_pointer`**. El hash **no viaja al prompt**.

**El hash no se le pide al modelo** porque hay exactamente un snapshot por
turno y el servidor sabe cuál inyectó. Es más barato (§7) y más fuerte en P08:
el modelo no puede siquiera intentar falsificar la procedencia — solo puede
señalar una línea.

Forma del bloque:

```
HECHOS VERIFICADOS — leídos del sistema por el servidor. Para afirmar uno,
poné su evidence_pointer en el claim.
  capabilities:
    /capabilities/0: name=audit_log_read, mode=read_only
    /capabilities/1: name=http_get, mode=read_only
    ...
```

### 5.2 Serialización canónica — y por qué el orden es normativo

```python
json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

y la lista de capabilities **ordenada por `name`**, no por orden de aparición
en el TOML.

**Razón:** si el orden dependiera del archivo, agregar una op en el medio
correría todos los `evidence_pointer` de las siguientes y rompería las citas
**sin que cambie ningún hecho**. La clave de orden es parte del contrato, no
un detalle de implementación.

### 5.3 Normalización de args — una sola función para los dos lados

La comparación del paso 5 usa **exactamente la misma normalización que produjo
el snapshot**: `str()` sobre cada valor, `strip()`, comparación exacta,
expuesta como una única función que llaman inyección y acreditación.

**Razón:** un `PROVENANCE_MISMATCH` disparado por diferencia de formato es un
detector que se entrena a ignorar. Si el estado puede aparecer por una causa
que no es la que nombra, deja de significar lo que dice.

### 5.4 Fallo de construcción del snapshot

`build_snapshot()` **falla ruidoso**. No devuelve vacío, no devuelve `None`.

P10 es la única regla `NORMATIVA` del corpus: *"ningún validador o gate puede
fallar abierto ante error o ausencia de señal, incluyendo vía excepción sin
capturar"*.

`chat.py` captura ese fallo, **no inyecta bloque**, responde el turno igual
(el grounding es medición, no puede tumbar un chat — mismo criterio que el
encolado de shadow validation) y marca el turno con
`grounding_snapshot_sha256 = 'ERROR'` y el motivo en `grounding_snapshot`.

**Tres estados distinguibles, no dos:**

| `grounding_snapshot_sha256` | significado |
|---|---|
| `NULL` | turno sin grounding **por diseño** |
| `'ERROR'` | el snapshot **falló al construirse** |
| 64 hex | snapshot real, reconstruible |

Un fallo del sistema indistinguible de una ausencia deliberada es exactamente
lo que este proyecto viene pagando.

### 5.5 Persistencia y paso al validador

`run_shadow_validation` recibe el snapshot **como quinto argumento**: el exacto
que se inyectó en **ese** turno. Sin tabla nueva, sin lookup por hash, sin
política de retención propia.

**Invariante:** un claim solo se acredita contra el snapshot de su propio
turno. Un hash de turno anterior no coincide, y eso es correcto — el snapshot
es determinista sobre el estado, así que un hash distinto significa que el
estado cambió.

El validador **persiste el snapshot recibido junto con su sha256**. Es rastro
de auditoría y material para SP4: poder reconstruir qué vio el modelo y
retractar un veredicto **por comparación, no por apariencia**.

---

## 6. Alcance

### 6.1 Adentro

| repo | archivo | qué |
|---|---|---|
| `jax` | `policy/governance/grounding.py` *(nuevo)* | `build_snapshot(ctx)`, `render(snapshot)`, `snapshot_sha256(snapshot)`, `normalize_args(args)`, `accredit(raw_claim, snapshot)` — **los cinco puros** |
| `jax` | `policy/governance/validator.py` | `PROVENANCE_MISMATCH` y `GROUNDING_UNAVAILABLE` en el `Literal` de `Verdict`; reordenar chequeos según §4.1 |
| `jax-platform` | `backend/api/chat.py` | construir/renderizar/anexar el snapshot; **cambiar la forma del claim en el sufijo de contrato**; pasar el snapshot al background task |
| `jax-platform` | `backend/shadow_validation.py` | acreditar antes de validar; persistir snapshot, sha256, authority y evidence_pointer |
| `jax-platform` | migración | ver §6.2 |

`grounding.py` no hace I/O: `ValidationContext` ya trae `ops` y
`mutating_capabilities`. Toda la lógica de gobernanza queda pura y testeable
sin DB — misma separación que fijó SP1 (`claims.py` puro, `loaders`/`validator`
con I/O).

### 6.2 Migración explícita

`shadow_messages`:
```sql
ALTER TABLE shadow_messages
  ADD COLUMN grounding_snapshot        longtext NULL,
  ADD COLUMN grounding_snapshot_sha256 char(64) NULL;
```

`shadow_claim_verdicts`:
```sql
ALTER TABLE shadow_claim_verdicts
  ADD COLUMN authority        varchar(12) NULL,
  ADD COLUMN evidence_pointer varchar(100) NULL;
```

Ninguna columna de texto libre existente se reusa. `shadow_messages` tiene
`degradation_reason (text)`, que tiene dueño semántico; reusarla sería elegir
un campo por su tipo y no por su significado.

### 6.3 El bloqueo que hay que levantar en el contrato

El sufijo de contrato generado desde el YAML **ya está mergeado** (#43). Pero
ese mismo sufijo dice hoy, en `chat.py:576`:

> `Cada claim es {"predicate": "...", "args": {...}} — SOLO esos dos campos,
> nada más.`

y cierra con *"No incluyas ningún otro campo"*. **El contrato actual le prohíbe
al modelo emitir `evidence_pointer`.** El mecanismo entero es inalcanzable
hasta cambiar la forma del claim. Es dependencia del mismo PR, no un ajuste de
redacción posterior.

### 6.4 Afuera, explícito

Resolver de `FACET_EXISTS` (segunda vuelta, con §3.1 ya decidido); los otros 6
predicados; tool-calling; frontend; y cualquier cambio al conjunto de
predicados ofrecidos en el sufijo.

---

## 7. Costo en el prompt — medido, no estimado

Tokenizador real de `qwen3.6:35b-a3b-q4_K_M` (modelo activo de `jax_local`
según `facet_binding`), vía `prompt_eval_count` de `/api/generate` con
`num_predict: 0`.

**Desglose por variante del bloque** (11 capabilities):

| variante | tokens | chars |
|---|---|---|
| A) `tool_result:sha256:<64>` + instrucción larga | 299 | 871 |
| B) id corto (12 hex) + instrucción larga | 246 | 805 |
| C) id corto + instrucción corta | 231 | 756 |
| **D) sin hash en el prompt, solo `evidence_pointer`** | **212** | **715** |

El sha256 completo costaba **53 tokens (18% del bloque)** para que el modelo lo
copiara de vuelta. La variante D lo elimina del prompt: −29% contra A, y más
fuerte en P08 (§5.1).

**Costo de D contra el prompt real de cada faceta** (`system_prompt` + sufijo
de contrato de 417 tok):

| faceta | prompt hoy | +grounding | total | grounding % |
|---|---|---|---|---|
| jax_local | 1192 | 212 | 1404 | 15.1% |
| hyde | 699 | 212 | 911 | **23.3%** |
| jekyll | 898 | 212 | 1110 | 19.1% |
| hipatia | 954 | 212 | 1166 | 18.2% |
| thot | 1093 | 212 | 1305 | 16.2% |
| ada | 913 | 212 | 1125 | 18.8% |
| kimi | 1036 | 212 | 1248 | 17.0% |

**No es barato:** entre un sexto y un cuarto del prompt de sistema, peor en
`hyde`, que tiene la persona más corta. Se declara así, sin adornar. El costo
crece linealmente con el número de hechos inyectados, que es la razón
estructural del invariante de §3: el snapshot no puede crecer por criterio.

---

## 8. Lo que este sub-proyecto NO va a poder demostrar

**Que el grounding haga que las facetas afirmen espontáneamente.**

Medido el 2026-09-01 con el prompt ya arreglado: ante una pregunta orgánica
sobre el estado del sistema, **las 5 facetas devolvieron `[]`**. Solo ante un
pedido explícito emitieron claims, y `thot` mantuvo la negativa incluso
entonces. Sus personas se niegan —correctamente— a afirmar sin evidencia.

El snapshot les da **con qué acreditar, no motivo para afirmar**. Son dos
cosas distintas y este sub-proyecto solo entrega la primera.

**Por lo tanto, y queda escrito de antemano para que no se reinterprete
después: si tras SP3 el tráfico orgánico sigue en `[]`, eso es dato, no
fallo.** Es la medición que dice que lo que falta es otra cosa —presumiblemente
que las facetas tengan una razón de tarea para afirmar, no solo permiso— y esa
otra cosa es alcance de SP4, no deuda de éste.

El precedente que obliga a escribir esta sección: el docstring de
`shadow_validation.py` afirmaba *"Resultado esperado: 100% AUTHORITY_INVALID
esta ronda"* y se leyó durante 14 días como si fuera lo medido. Una predicción
sin marcar se convierte en un registro falso.

---

## 9. Pruebas

TDD. La parte pura primero, porque ahí está la lógica de gobernanza.

### 9.1 Unitarias en `jax` (sin I/O, sin DB)

Los cuatro estados **provocados a propósito**, más las propiedades del
snapshot:

1. `JOB_STATUS` **con puntero inventado** → `RESOLVER_NOT_IMPLEMENTED`, **no**
   `PROVENANCE_MISMATCH`. Fija el orden 3-antes-de-5.
2. `CAPABILITY_AVAILABLE` **con puntero inventado** → `PROVENANCE_MISMATCH`,
   **no** `RESOLVER_NOT_IMPLEMENTED`.
3. `CAPABILITY_AVAILABLE` sin puntero → `AUTHORITY_INVALID`, `authority=INFERIDO`.
4. **Puntero válido, args que no coinciden** → `PROVENANCE_MISMATCH`.
   *Es el test que sostiene el diseño entero:* sin él, citar lava priors como
   observación (§2.2).
5. Puntero válido con args exactos → `authority=OBSERVADO`, `provenance_ref`
   escrito por el servidor, el resolver corre.
6. Snapshot en `ERROR` + claim **con puntero** → `GROUNDING_UNAVAILABLE`,
   `authority=INFERIDO`, **no** `PROVENANCE_MISMATCH`.
7. Propiedades del snapshot: hash determinista sobre el mismo estado, distinto
   ante cambio; `render()` **no** contiene el hash; **agregar una op no mueve
   el `evidence_pointer` de las demás** (test del orden canónico de §5.2);
   `build_snapshot` que falla **lanza** (P10), no devuelve vacío.
8. **`FACT_MISMATCH` ejercitado rompiendo la rama a mano:** `build_snapshot`
   desde `ctx` A; el claim cita bien una entrada de ese snapshot y se acredita
   contra **él** (→ `OBSERVADO`); después `validate` corre el resolver contra
   `ctx` B, que tiene esa op de menos → `FACT_MISMATCH`. Las dos capas de §4.3
   ejercitadas en el mismo test, con datos distintos a propósito.
   *Necesario porque en producción hoy no puede dispararse (§9.4). Una rama sin
   test es una hipótesis.*

### 9.2 Integración en `jax-platform`

Contra el service container `mariadb:11.8` del job `backend-tests-con-db`:

- Cada estado de §4.1 produce su fila en `shadow_claim_verdicts` con
  `authority` y `evidence_pointer` correctos.
- `shadow_messages` guarda snapshot y sha256.
- Turno sin grounding → `NULL` en ambas columnas, nada se rompe.
- Turno con snapshot fallido → `'ERROR'`, distinguible de `NULL` y de un hash.

### 9.3 El camino de producción, no solo el sintético

`chat.py` arma el prompt y el snapshot, y **el que viaja al background task es
el mismo que se renderizó**.

Es la juntura donde un cambio de firma se escapa con CI entero en verde —
lección 22 (`#98`→`#99`): `build()` tenía dos caminos de entrada y todos los
tests entraban por el que no usa producción.

### 9.4 Propiedad diferida: el detector de drift nace inerte

`_validation_context()` es `@lru_cache(maxsize=1)` — config estática por
proceso, por política explícita (Lección operativa #6 de
`jax-platform/CLAUDE.md`: un cambio real exige reiniciar). Y
`_resolve_capability_available` lee `ctx.ops`, **el mismo objeto cacheado** del
que sale el snapshot.

Inyección y resolución ven los mismos datos dentro de un proceso: **para
`CAPABILITY_AVAILABLE`, `FACT_MISMATCH` no puede dispararse por drift.**

La capa doble de §4.3 sigue siendo el diseño correcto, pero recién tiene efecto
cuando llegue un predicado con fuente genuinamente viva (`JOB_STATUS`,
`ENGINE_STATUS`, `MEMORY_ENTRY_EXISTS`). **Se declara como propiedad diferida,
no como capacidad entregada**, y por eso la prueba 8 la ejercita rompiendo la
rama a mano.

Corolario: en el primer corte un claim bien citado dará `VALID` casi siempre.
**Lo que hay que medir son los otros estados.**

### 9.5 Verificación en vivo

El método que salvó el diagnóstico de SP2:

- **Sonda explícita** (pedir el claim) **y pregunta orgánica**, por separado.
  Sin la sonda explícita, un resultado en `[]` se reporta como "el mecanismo no
  funciona", que sería falso.
- Medición contra `jax_memory` directo, no contra el cuerpo del PR.
- **Contaminación declarada:** las filas de sonda no son muestra de
  comportamiento espontáneo y se marcan como tales.
- Pisos de CI medidos antes/después en el job real, no estimados.

---

## 10. Puntos abiertos para la revisión

1. **`GROUNDING_UNAVAILABLE`** (§4.2) es un segundo estado nuevo y contradice
   la decisión de "un solo estado nuevo". La alternativa —`AUTHORITY_INVALID` +
   join a `grounding_snapshot_sha256='ERROR'`— está escrita con su
   consecuencia. Decisión pendiente.
2. **`hyde` paga 23.3% de su prompt** en grounding y es la faceta con la
   persona más corta. No se propone excluirla (rompería la uniformidad del
   canal), pero el número queda anotado por si el costo pesa más que el dato.
