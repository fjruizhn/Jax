# REFORMAS-v3 Fase 2, Sub-proyecto 1 — Schema, validador y renderer de claims — Design

**Fecha:** 2026-08-17
**Spec fuente:** `/opt/jax/docs/REFORMAS-v3.md` §3.1 (R1), sha256
`4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660`.
**Progreso previo:** `jax-platform perf` no aplica aquí; ver memoria
`reformas-v3-progreso` para el estado completo de REFORMAS-v3 al
2026-08-15 (Fases 0, 0.5, 1 completas; Fase 2 en brainstorming).

## Alcance

Fase 2 de REFORMAS-v3 (§5) se dividió en dos sub-proyectos, cada uno con
spec propio:

- **Sub-proyecto 1 (este documento):** schema de claim, validador
  determinista contra fuentes reales, barrido de vocabulario cerrado,
  renderer por plantillas con hash versionado. Todo probado con claims
  **sintéticos** — construidos a mano en tests, no producidos por ninguna
  faceta. **No toca `jax-platform/backend/api/chat.py` ni ningún otro
  punto de integración con tráfico real.**
- **Sub-proyecto 2 (fuera de alcance, spec futuro):** separación de
  canales `claim`/`analysis`/`judgment` en el contrato de salida de las
  facetas, cambio de prompt de cada faceta para emitir JSON estructurado,
  manejo de salida malformada (relevante en particular para Kimi, que
  trunca a 488 bytes — defecto de integración conocido, sin resolver),
  integración real en modo *shadow validation* (§3.1.8) sobre la Mesa web.

Este documento cubre **solo el sub-proyecto 1**.

## Fuera de alcance, explícitamente

- Cambiar el contrato de salida de ninguna faceta.
- Conectar el validador a tráfico real (Mesa web, REPL).
- Shadow validation, canario o enforcement (§3.1.8) — eso presupone un
  consumidor real, que es sub-proyecto 2.
- Resolvers reales para `ENGINE_STATUS`, `FACET_EXISTS`, `CONFIG_VALUE`,
  `AUDIT_EVENT_EXISTS`, `JOB_STATUS`, `MEMORY_ENTRY_EXISTS` — quedan
  `RESOLVER_NOT_IMPLEMENTED` con motivo explícito (ver §4). Ninguno tiene
  consumidor identificado hoy (R5).

## Por qué solo dos resolvers reales

Aplicación de R5 ("ninguna capacidad se construye sin carga concreta que
la consuma") al interior del sub-proyecto: de los 8 predicados de
`policy/vocabulary/predicates.yaml`, solo `CAPABILITY_AVAILABLE` y
`FILE_EXISTS` tienen consumidor identificado — son los dos que, junto con
el barrido de vocabulario, bloquean las 4 afirmaciones falsas del
Apéndice B (criterio de aceptación escrito de R1):

| # Apéndice B | Bloqueado por |
|---|---|
| 1 ("JAX no tiene acceso a los logs") | `CAPABILITY_AVAILABLE` (sin evidencia) o barrido si va como `judgment` |
| 2 ("no puedo invocar a Hyde...") | `CAPABILITY_AVAILABLE`/`ENGINE_STATUS` sin procedencia — cae bajo `CAPABILITY_AVAILABLE`, que sí se implementa |
| 3 (`trae a hyde` como comando inventado) | Barrido de vocabulario — `trae a hyde` no está en `closed_vocabulary.yaml` |
| 4 (`/var/log/jax*` sin verificar) | `FILE_EXISTS` |

**`ENGINE_STATUS` verificado y descartado esta ronda:** la única fuente
parecida a "salud de motor" es la tabla `model` de MariaDB, poblada por
`jax-platform/backend/model_catalog.py` vía sync periódico contra las
APIs de los providers + Ollama. Dos razones independientes para no
usarla, cualquiera de las dos basta: (1) vive en `jax-platform`, otro
repo/servicio — acoplar el validador de `jax/policy/` a esa DB es el
acoplamiento prematuro que R5 prohíbe sin carga concreta; (2) semántica
distinta — "¿el provider sigue ofreciendo este modelo?" no es "¿este
motor responde ahora?". Implementarlo habría dado un predicado que
responde una pregunta distinta a la que su nombre sugiere.

Los otros 5 predicados no tienen fuente identificada ni consumidor —
quedan como deuda **declarada**, no oculta.

## 1. Estructura de módulos

Nuevo paquete `policy/governance/`, paralelo a `vocabulary/`, `templates/`,
`tools/`, `rules/` — organiza por responsabilidad, no por número de fase
(de ahí el nombre `governance`, no `r1` ni `claims`, que envejecerían).

```
policy/governance/
  claims.py       # schema puro (Pydantic), sin I/O
  loaders.py       # I/O de config ESTÁTICA de policy: predicates.yaml,
                    # closed_vocabulary.yaml, render_templates.yaml + hash
  validator.py     # I/O de ESTADO VIVO del sistema (config.toml de
                    # las_manos/motor_registry, filesystem) + resolvers
  vocab_sweep.py    # barrido léxico, puro — recibe vocabulario ya cargado
  renderer.py       # motor de plantillas, puro — recibe templates ya cargados
policy/tools/
  template_hash.py  # nuevo — mismo algoritmo que corpus_hash.py, hashea
                     # templates/render_templates.yaml
```

**Dos módulos de I/O, no uno, y con roles distintos.** `loaders.py` lee
config de policy versionada por commit (YAML + hash) — si falla, la
política está corrupta y el subsistema **no arranca**. `validator.py` lee
estado vivo que cambia sin commits (`config.toml`, filesystem) — si falla
un resolver individual, es un rechazo de ese claim, no una falla de
arranque. Mezclar las dos cosas en un módulo le daría dos motivos
distintos para cambiar.

Los cuatro módulos restantes (`claims.py`, `vocab_sweep.py`, `renderer.py`,
la lógica interna de los resolvers en `validator.py`) son puros: reciben
datos ya cargados, nunca abren un archivo. Mismo principio que
`las_manos/envelope.py` ya documenta ("Este módulo es PURO: sin I/O, sin
red, testeable en aislamiento").

## 2. `claims.py` — schema estructural

Dos capas de validación, igual que `IntentEnvelope`: estructural
(Pydantic, este módulo) y semántica (`validator.py`, §3).

```python
class Claim(BaseModel):
    predicate: str
    args: dict[str, str]
    authority: Literal["EJECUTADO", "OBSERVADO", "RECUPERADO", "INFERIDO"]
    provenance_ref: str
    evidence_pointer: str
    scope: str
```

- `authority` es `Literal` hardcodeado (como `verification_label` en
  `IntentEnvelope`): son 4 valores fijos con significado embebido en la
  lógica del validador (`INFERIDO` prohibido en canal `claim` es un
  invariante de código).
- `predicate` es `str`, no `Literal`: la lista de predicados es config
  que crece por enmienda (`predicates.yaml`); un `Literal` hardcodeado
  obligaría a tocar código en cada ampliación, duplicando la fuente de
  verdad.
- `args: dict[str, str]` — **decisión consciente, no definitiva.** Los
  dos resolvers reales de esta ronda (`CAPABILITY_AVAILABLE`:
  name/mode; `FILE_EXISTS`: path/hash) usan solo strings. De los 6
  predicados restantes, `MEMORY_ENTRY_EXISTS` (`memory_id`) podría
  necesitar un entero. Se anota aquí para que el tercer resolver que
  necesite un tipo no-string sea una decisión explícita, no un
  descubrimiento a mitad de implementación.

## 3. `loaders.py` — I/O de config estática

```python
def load_predicates() -> dict[str, PredicateSpec]      # predicates.yaml
def load_vocabulary() -> ClosedVocabulary                # closed_vocabulary.yaml
def load_templates() -> dict[str, TemplateSpec]           # render_templates.yaml + hash
```

`ClosedVocabulary` expone dos formas de los mismos datos, sin parsear el
YAML dos veces: `.flattened: frozenset[str]` (todas las categorías —
capabilities, ops, facetas, motores, config_paths, commands — para
`vocab_sweep.py`) y `.config_paths: frozenset[str]` (solo la sección
`config_paths`, para la allowlist de `FILE_EXISTS`).

**`load_templates()` es fail-closed sin excepción.** Recalcula el sha256
de `templates/render_templates.yaml` (byte a byte, mismo algoritmo que
`policy/tools/corpus_hash.py`) y lo compara contra la línea
`templates_sha256:` de `policy/VERSION` (nueva, escrita por
`policy/tools/template_hash.py`, que se agrega mirando exactamente
`corpus_hash.py`). Si no coincide: excepción, nada se carga, el
subsistema no arranca. No hay carga parcial ni degradación silenciosa —
es el mismo patrón de falla que ya causó tres incidentes independientes
en este ecosistema (ver memoria `fail-open-pattern-tres-casos`: retención
de backups, `_HTTP_FACETS`, `output_validator.py`); este es el módulo que
existe específicamente para no ser el cuarto.

`policy/VERSION` pasa a tener tres líneas:
```
version: 0.1.0
sha256: <hash del corpus de rules/, sin cambios>
templates_sha256: <nuevo, hash de render_templates.yaml>
```

## 4. `validator.py` — validación semántica y resolvers

Recibe un `Claim` ya válido estructuralmente, más `predicates: dict[str,
PredicateSpec]` (de `loaders.load_predicates()`). Primer paso, antes de
cualquier resolver — sin tocar disco ni red:

1. `predicate` no está en `predicates` → `UNKNOWN_PREDICATE`.
2. Claves de `args` no coinciden exactamente con las declaradas para ese
   predicado → `ARGS_MISMATCH`.
3. `authority == "INFERIDO"` → `AUTHORITY_INVALID` (prohibido en canal
   `claim` por §3.1.4).

Solo si las tres pasan, despacha al resolver del predicado.

### `Verdict` — ocho estados, ninguno es "warn"

```python
class Verdict(BaseModel):
    status: Literal[
        "VALID", "UNKNOWN_PREDICATE", "ARGS_MISMATCH",
        "RESOLVER_NOT_IMPLEMENTED", "FACT_MISMATCH", "AUTHORITY_INVALID",
        "SOURCE_CONFLICT", "PATH_NOT_ALLOWED",
    ]
    predicate: str
    detail: str
```

P07 ("no existe bypass en producción") aplicado al tipo de retorno: no
hay forma de expresar "falló pero seguí" porque el tipo no lo permite.
`RESOLVER_NOT_IMPLEMENTED` es un rechazo, no un paso silencioso —
mensaje con razón, no solo "no implementado":

```python
raise NotImplementedError(
    "ENGINE_STATUS: sin fuente de verdad en el dominio de jax. "
    "La tabla 'model' de jax-platform tiene semántica distinta "
    "(disponibilidad del provider, no salud del motor). "
    "Ver REFORMAS-v3 §3.1.3 y este spec, sección 'Por qué solo dos "
    "resolvers reales'."
)
```

### Resolver `CAPABILITY_AVAILABLE` ({name, mode})

Consulta dos fuentes reales, ambas en `jax`, ambas síncronas (config ya
en memoria, sin red):

- `las_manos/policy.py`: `name in config['ops']`; si está, `mode` se
  deriva de `MUTATING_CAPABILITIES` (`envelope.py:28`) — mismo criterio
  ya usado en el resto de LAS MANOS, no uno nuevo.
- `las_manos/motor_registry/catalog.py`: `catalog.get_capability(name)
  is not None`. Este registro no tiene concepto de `mode` (solo
  `risk_level`, campo distinto) — **limitación declarada:** si la única
  fuente que encuentra el nombre es este catálogo, el `mode` del claim se
  acepta sin poder contradecirlo.

Verificado contra `config.toml` (2026-08-17): los namespaces `[ops.*]`
(11 nombres) y `[capabilities.*]` (12 nombres) son **completamente
disjuntos hoy** — cero solapamiento.

| Presente en | Veredicto |
|---|---|
| Ninguna fuente | `FACT_MISMATCH` |
| Solo una fuente | `VALID` (mode verificado si la fuente es `ops`; aceptado sin verificar si es solo el catálogo) |
| Ambas fuentes | `SOURCE_CONFLICT`, siempre — sin importar si coincide el mode |

`SOURCE_CONFLICT` no es "el claim no coincide con la realidad": es "el
sistema no tiene una única fuente de verdad para este nombre" — conecta
directamente con la tensión P04 ya documentada en el corpus (dos sistemas
de autorización paralelos, hallazgo de Fase 1). Hoy nunca dispara (0
solapamiento); queda listo para el día que lo haga. **`detail` debe
nombrar ambas fuentes explícitamente** (p.ej. `"presente en ops y en
capabilities"`) — es un veredicto que va a aparecer una vez cada mucho
tiempo, y cuando aparezca tiene que explicarse solo, sin que alguien
tenga que ir a buscar dónde.

### Resolver `FILE_EXISTS` ({path, hash})

Orden estricto, allowlist antes que filesystem:

1. `path` contra `ClosedVocabulary.config_paths` — match exacto para
   entradas de archivo (`las_manos/config.toml`, `/etc/jax/.env`,
   `las_manos/logs/audit.jsonl`), match de prefijo para entradas de
   directorio (`jax/policy/`, distinguible por el `/` final ya presente
   en el YAML). Relativos se normalizan contra la raíz del repo.
2. Fuera de la allowlist → `PATH_NOT_ALLOWED`, **sin tocar el
   filesystem**. Si se hiciera `stat()` primero y se filtrara después, un
   path fuera de la lista igual revelaría si el archivo existe por el
   tiempo de respuesta — chequear permiso antes que hecho es el orden
   que corresponde. **`PATH_NOT_ALLOWED` es explícitamente no
   informativo:** el veredicto no dice si el archivo existe, ni su hash,
   ni nada derivado de haberlo mirado — solo que el path no está
   permitido. Se escribe así para que sobreviva a una refactorización
   futura que no recuerde el razonamiento del punto 1.
3. Dentro de la allowlist: no existe → `FACT_MISMATCH`. Existe con hash
   distinto → `FACT_MISMATCH`. Existe con hash igual (sha256, mismo
   algoritmo que el resto de `policy/`) → `VALID`.

No es alcance extra sobre lo pedido por R5: una allowlist restringe la
capacidad que ya se está construyendo, es más barata que la alternativa
(el resolver de todos modos tiene que decidir qué path lee). Sin ella,
en el sub-proyecto 2 el `path` viene de texto generado por un modelo —
un resolver sin allowlist ahí es lectura y hasheo de cualquier archivo
legible por el proceso, dirigido por el modelo. Si una carga futura
necesita otro path, se agrega a `closed_vocabulary.yaml` — cambio
versionado y auditable, no una excepción en código.

## 5. `vocab_sweep.py` — barrido léxico

Puro, recibe el vocabulario ya cargado:

```python
def sweep(text: str, vocabulary: frozenset[str]) -> list[str]
```

Devuelve los términos de `vocabulary` encontrados en `text`. Sin
coincidencias → pasa. Con coincidencia → el llamador decide si el bloque
se reformula como `claim` o se rechaza (§3.1.5) — eso es sub-proyecto 2,
fuera de alcance acá.

## 6. `renderer.py` — plantillas versionadas

Puro, recibe templates ya cargados y verificados por `loaders.py`:

```python
def render(claim: Claim, templates: dict[str, TemplateSpec]) -> str
```

Rechaza (excepción) si `claim.predicate` no tiene entrada `status:
definida` en `templates`. `render_templates.yaml` ya tiene 3 templates
`definida` (`CAPABILITY_AVAILABLE`, `AUDIT_EVENT_EXISTS`, `FILE_EXISTS`) —
ninguno para `ENGINE_STATUS`. Que `AUDIT_EVENT_EXISTS` tenga template sin
resolver esta ronda **no es inconsistencia**: `renderer.py` es
deliberadamente agnóstico de qué predicados tienen resolver real — esa
desconexión es lo que permite probarlo con claims sintéticos sin esperar
al resolver. En producción (sub-proyecto 2), un claim de
`AUDIT_EVENT_EXISTS` nunca llegaría a renderizarse porque su resolver
devuelve `RESOLVER_NOT_IMPLEMENTED` antes.

## 7. Testing

Por módulo, TDD en la implementación (`superpowers:test-driven-development`,
fuera de este spec):

- `claims.py`, `vocab_sweep.py`, `renderer.py` — fixtures sintéticas
  puras, sin filesystem.
- `validator.py`, `loaders.py` — únicos con fixtures de filesystem
  (`config.toml` de prueba, YAML de prueba).
- **El test que más importa de los cuatro:** hash roto a propósito en
  `load_templates()`, verificando que el subsistema levanta excepción y
  no carga nada — es la clase de prueba que faltó en
  `backup-hall9000.sh` (ver memoria `backup-hall9000-fail-open-fixed`) y
  que este subsistema existe específicamente para no repetir.

## Siguiente paso

`superpowers:writing-plans` para convertir este spec en plan de
implementación TDD, tarea por tarea — mismo formato que
`docs/superpowers/plans/2026-08-15-reformas-fase1-desbloqueo-lectura.md`.
