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

### 3.3 `build_snapshot` lee SOLO la fuente que el resolver resuelve de verdad

Confirmado contra el código (2026-09-02), y es condición de que este spec sea
diferible respecto de la deuda del catálogo:

`_resolve_capability_available` tiene dos ramas: `in_ops` (lee `ctx.ops`, que
`load_validation_context()` toma de `[ops.*]` en `las_manos/config.toml`) e
`in_catalog` (lee `ctx.catalog`, un `MotorCatalog(config)` construido **desde el
mismo TOML**). Desde el Bloque 3 las capabilities viven en la DB
(`MotorCatalog.from_db()`), `[capabilities.*]` del TOML tiene 0 entradas, y
**la rama `in_catalog` es código muerto en producción**. Lo tapaba un test que
afirmaba lo contrario, rojo en `master` sin que nadie lo viera porque
`tests/test_governance_*.py` no corría en CI. Ver DEUDA.md (ítem del
2026-09-02) y el tripwire
`test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production`.

**`build_snapshot(ctx)` lee `ctx.ops` y `ctx.mutating_capabilities`, y nada
más.** No toca `ctx.catalog`. Por lo tanto el snapshot cubre exactamente lo que
la rama viva del resolver puede confirmar, y el invariante de §3 ("todo hecho
inyectado tiene quién lo re-resuelva") se cumple hoy. **Si `build_snapshot`
tocara la rama muerta, esta deuda no sería diferible**: se estaría inyectando
como observado algo que ningún resolver puede verificar.

Consecuencia declarada: una capability que exista solo en la DB no aparece en
el snapshot **ni** puede dar `VALID` hoy. Cuando el validador pase a leer la DB,
el tripwire se pone rojo y obliga a revisar esta sección y `SECTION_PREDICATE`.

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

Dos, los dos aprobados. Ninguno amplía la columna: `PROVENANCE_MISMATCH` 19,
`GROUNDING_UNAVAILABLE` 21, contra `status VARCHAR(30)` y el más largo
existente `RESOLVER_NOT_IMPLEMENTED` 24.

**`PROVENANCE_MISMATCH`** — el claim trae puntero y el puntero no resuelve a
una entrada del snapshot, **o** resuelve y los args no coinciden con ella.
Conducta del modelo.

**`GROUNDING_UNAVAILABLE`** — el snapshot del turno falló al construirse
(`grounding_snapshot_sha256 = 'ERROR'`). Condición cerrada:

```
sha256 = 'ERROR'  →  paso 0  →  authority = INFERIDO, status = GROUNDING_UNAVAILABLE
```

Se aplica a **todo** claim del turno, traiga puntero o no, tenga resolver o no.
Falla del sistema. Se le da estado propio porque la alternativa —reusar
`AUTHORITY_INVALID` y desambiguar por join a `shadow_messages`— imputaría al
modelo un fallo del servidor y obligaría a desambiguar a mano en cada
análisis: la misma razón por la que "sin resolver" no se mezcla con "no citó".

**No se agrega `UNGROUNDED`.** Con la definición acordada —ausencia de resolver
para el predicado— es la misma condición que `RESOLVER_NOT_IMPLEMENTED`, que ya
existe en el `Literal` de `Verdict` y hoy es inalcanzable solo porque
`authority` corta antes. Bajo el orden nuevo pasa a ser alcanzable y su nombre
ya lo describe. Agregar un sinónimo sería la deriva de nombres que este
proyecto viene pagando.

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

**Razón:** si el orden dependiera del archivo, **reordenar** el TOML sin
cambiar ningún hecho correría los `evidence_pointer` y rompería las citas. Con
orden por `name`, la promesa es exacta: **mismo contenido ⇒ mismo hash y mismos
punteros, en cualquier orden de archivo.** Agregar o quitar una op sí puede
mover punteros — y debe: el snapshot cambió, y un claim del turno anterior no
debe acreditarse contra el nuevo (§5.5). La clave de orden es parte del
contrato, no un detalle de implementación.

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

Cada fila es un test con su entrada completa y su salida esperada. Snapshot
base **S** = construido desde `ctx` A, que tiene `write_file` en
`/capabilities/10` con `mode=mutating` (orden por `name`, §5.2).

| # | snapshot | predicado | `evidence_pointer` | args | → `authority` | → `status` | fija |
|---|---|---|---|---|---|---|---|
| 1 | S | `JOB_STATUS` | `/capabilities/10` (inventado: el predicado no tiene entradas) | `{job_id, status}` válidos | `INFERIDO` | `RESOLVER_NOT_IMPLEMENTED` | paso 3 antes del 5: sin resolver no hay citación falsa posible |
| 2 | S | `CAPABILITY_AVAILABLE` | `/capabilities/99` (índice inexistente) | `{name=write_file, mode=mutating}` | `INFERIDO` | `PROVENANCE_MISMATCH` | puntero que no resuelve |
| 3 | S | `CAPABILITY_AVAILABLE` | *(ausente)* | `{name=write_file, mode=mutating}` | `INFERIDO` | `AUTHORITY_INVALID` | se ofreció grounding y no citó |
| 4 | S | `CAPABILITY_AVAILABLE` | `/capabilities/10` | `{name=write_file, mode=read_only}` | `INFERIDO` | `PROVENANCE_MISMATCH` | **la citación falsa (§2.2): puntero real, args que no coinciden** |
| 5 | S | `CAPABILITY_AVAILABLE` | `/capabilities/10` | `{name=write_file, mode=mutating}` | `OBSERVADO` | `VALID` | camino feliz; `provenance_ref = tool_result:sha256:<sha de S>` escrito por el servidor |
| 6 | `ERROR` | `CAPABILITY_AVAILABLE` | `/capabilities/10` | `{name=write_file, mode=mutating}` | `INFERIDO` | `GROUNDING_UNAVAILABLE` | paso 0 antes que todo; **no** `PROVENANCE_MISMATCH` aunque el puntero "parezca" válido |
| 7a | S vs S' | — | — | — | — | `sha256(S) == sha256(S')` si `ctx` es igual; `!=` si difiere en una op | hash determinista |
| 7b | S | — | — | — | — | `sha256(S) not in render(S)` | el hash no viaja al prompt |
| 7c | S vs S+`aaa_op` | — | — | — | — | `/capabilities/N` de `write_file` **no cambia** al agregar una op que ordena antes... | — ver nota |
| 7d | `ctx` roto | — | — | — | — | `build_snapshot` **lanza**; nunca devuelve `[]` ni `None` | P10 |
| 8 | S (de `ctx` A) | `CAPABILITY_AVAILABLE` | `/capabilities/10` | `{name=write_file, mode=mutating}` | `OBSERVADO` | `FACT_MISMATCH` | acreditación contra S; `validate` con `ctx` B = A sin `write_file`. Las dos capas de §4.3 con datos distintos a propósito |

**Nota sobre 7c, corregida al escribir la tabla.** Ordenar por `name` hace
que los punteros sean estables ante **cambios del archivo** (reordenar el TOML
no mueve nada), pero **no** ante inserciones que ordenen antes: agregar
`aaa_op` sí corre el índice de todas las demás. Eso es correcto y deseable —
el snapshot cambió, su hash cambió, y un claim del turno anterior no debe
acreditarse contra el nuevo (§5.5). Lo que 7c prueba entonces es lo que sí se
promete: **reordenar las entradas del TOML sin cambiar su contenido no cambia
ni el hash ni ningún puntero.** La promesa de §5.2 queda reescrita en esos
términos.

Además, para cada uno de 1–6 y 8, la fila persistida cumple:

- **`authority` es SIEMPRE el valor derivado por el servidor.** Nunca lo que
  mandó el modelo. Si el modelo incluye un campo `authority` en el claim, no
  entra en esa columna: queda en el raw del claim, en `detail`. (Corrige una
  redacción anterior de esta nota que contradecía §5.)
- **`evidence_pointer` se persiste tal como se recibió** (`NULL` si ausente),
  sin normalizar — para poder ver después qué mandó el modelo.

### 9.1b Punteros malformados — cada caso por separado

Un puntero que mata la background task es fail-open (P10): el turno queda sin
validar y nadie lo ve. Cada uno de estos es un test propio, sobre snapshot S y
`CAPABILITY_AVAILABLE` con args válidos:

| puntero | → `status` | además |
|---|---|---|
| `""` | `PROVENANCE_MISMATCH` | la task completa |
| `"capabilities/10"` (sin `/` inicial) | `PROVENANCE_MISMATCH` | la task completa |
| `"/capabilities/abc"` | `PROVENANCE_MISMATCH` | la task completa |
| `"/capabilities/-1"` | `PROVENANCE_MISMATCH` | la task completa; **no** indexa desde el final |
| 300 caracteres | `PROVENANCE_MISMATCH` | `evidence_pointer` persistido **truncado a 100** (el `varchar`), el original completo en `detail`, el `INSERT` no falla |

En todos: sin excepción, la background task completa y la fila existe.

### 9.2 Integración en `jax-platform`

Contra el service container `mariadb:11.8` del job `backend-tests-con-db`:

- Cada estado de §4.1 produce su fila en `shadow_claim_verdicts` con
  `authority` y `evidence_pointer` correctos.
- `shadow_messages` guarda snapshot y sha256.
- Turno con snapshot fallido → `'ERROR'`, distinguible de `NULL` y de un hash.
- **Toda fila que escribe `run_shadow_validation` tiene `grounding_snapshot_sha256`
  distinto de `NULL`** — hash o `'ERROR'`, nunca vacío. Ver la nota siguiente.

**`NULL` después de SP3 es solo filas legadas. Medido, no supuesto:**

- `run_shadow_validation` tiene **un único sitio de encolado** (`chat.py:1106`).
- `contract` es `None` cuando `is_canned` (`chat.py:1072`), que son las
  salidas tempranas de `_invoke_facet_dispatch` (sin binding, gate denegado,
  respuesta de identidad, transporte no soportado). Con `contract=None`,
  `run_shadow_validation` **retorna antes del primer `INSERT`**
  (`shadow_validation.py:160`): esos turnos no producen fila `NULL` —
  **no producen fila**.
- Si `_invoke_facet` lanza, la línea 1106 no se alcanza: tampoco hay fila.

Por lo tanto, todo turno que llega a insertar pasó por el camino que construye
el snapshot, y lleva hash o `'ERROR'`. **La única forma de que una fila nueva
quede `NULL` sería que alguien llamara a `run_shadow_validation` sin
snapshot.** Se cierra por firma: **el quinto argumento es obligatorio, sin
default.** Un caller nuevo que lo omita falla al llamar, no produce una fila
ambigua. Los claims de una fila `NULL` no se acreditan porque no existen: son
filas anteriores a esta migración y no tienen veredictos que revisar.

### 9.3 El camino de producción, no solo el sintético

**Aserción concreta.** Con el proveedor parcheado para capturar el payload
saliente y `add_safe_task` parcheado para capturar sus argumentos, un
`POST /api/chat` real produce **un solo objeto snapshot** que se verifica en
sus dos consumidores:

1. `render(snapshot_capturado) in system_prompt_que_salió` — lo que vio el
   modelo es la representación de ese objeto, y no otra.
2. `snapshot_sha256(snapshot_capturado) == grounding_snapshot_sha256`
   persistido para ese turno — lo que se guardó es el hash de ese mismo
   objeto.

**Por qué no es "sha256 del bloque renderizado == columna":** el hash
persistido es del **JSON canónico** (§5.2), no del texto renderizado, y el
texto renderizado **deliberadamente no contiene el hash** (§5.1). Hashear el
texto haría que un cambio de redacción de la instrucción cambiara el hash sin
que cambie ningún hecho. La aserción de arriba verifica la misma propiedad —
un objeto, dos consumidores, ninguno divergente— sin atar el hash al texto.

Es la juntura donde un cambio de firma se escapa con CI entero en verde —
lección 22 (`#98`→`#99`): `build()` tenía dos caminos de entrada y todos los
tests entraban por el que no usa producción.

**Regresión del contrato (§6.3):** un test que falla si el sufijo vuelve a
decir *"SOLO esos dos campos"* o *"No incluyas ningún otro campo"* sin admitir
`evidence_pointer`. Restaurar la línea 576 pone rojo.

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

## 10. Decisiones de la revisión

1. **`GROUNDING_UNAVAILABLE`: aprobado** como segundo estado nuevo (§4.2). La
   alternativa `AUTHORITY_INVALID` + join se descartó por imputar al modelo un
   fallo del servidor y desambiguarse a mano.
2. **`hyde` paga 23.3% de su prompt: anotado, no se excluye.** Excluir una
   faceta del canal por costo sería fail-open selectivo — la regla se
   cumpliría en todas menos en la que más cuesta.
