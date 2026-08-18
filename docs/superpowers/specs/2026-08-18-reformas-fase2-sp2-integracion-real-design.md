# REFORMAS-v3 Fase 2, Sub-proyecto 2 — Contrato de facetas, shadow validation e integración real — Design

## Contexto

Sub-proyecto 1 (PR #5) construyó `policy/governance/` — schema de claim,
carga de config estática fail-closed, validador con resolvers, barrido
de vocabulario cerrado y renderer — probado íntegramente contra claims
**sintéticos**. Este sub-proyecto conecta ese núcleo a tráfico real de
la Mesa web (`/api/chat`), en modo shadow (REFORMAS-v3.md §3.1.8): mide,
no bloquea.

## Prerrequisito duro

**PR #5 mergeado a `master`.** No es una tarea de este plan, es la
condición para que el plan exista. La extensión de `load_vocabulary()`/
`sweep()` (tarea 1) es un cambio aditivo sobre código ya en `master` —
apilar esta rama sobre el PR #5 sin mergear reintroduce el mismo
solapamiento sin mergear que ya se aceptó una vez entre PR #4 y PR #5, y
esta vez no hay razón funcional para repetirlo: PR #5 ya pasó revisión
final con 34/34 tests.

## Alcance

### La brecha de Q14 no se cierra en este sub-proyecto

De los 8 predicados del vocabulario cerrado, solo 2 (`CAPABILITY_AVAILABLE`,
`FILE_EXISTS`) tienen resolver real. La mayoría de lo que una faceta
declare va a caer en `analysis`/`judgment`, gobernados solo por el
barrido léxico — no por el validador determinista. Se decidió **no**
construir más resolvers todavía (eso sería adivinar casos de uso sin
tráfico real — el mismo patrón que R5 existe para evitar) ni inventar
una tercera forma de gobernanza para `analysis`/`judgment` a ciegas.

En cambio, este sub-proyecto **instrumenta** lo que hoy no se mide:
distribución de tráfico por canal (`claim`/`analysis`/`judgment`) y qué
predicados/categorías de vocabulario intentan las facetas. Esa
instrumentación es casi gratis (contar, no validar) y es lo único que
puede convertir "más resolvers" de una adivinanza en una lista ordenada
por frecuencia real, para la ronda siguiente.

### Orden de tareas

1. Extender `load_vocabulary()`/`sweep()` con categoría (`policy/governance/`).
2. Fix del issue #6 (`_resolve_file_exists` explota con `IsADirectoryError`
   sin capturar cuando `path` coincide con una entrada de directorio de
   la allowlist). Agrupada con la tarea 1 por tocar el mismo módulo, pero
   ambas tienen que estar antes de que la `BackgroundTask` (tarea 5)
   llame al validador con un `path` que viene de texto generado por un
   modelo — `path="policy"` es trivial de producir.
3. Wrapper de contrato en `chat.py`.
4. Migración de las tres tablas en `jax_memory`.
5. `BackgroundTask` de validación shadow.
6. Footnote en `jax-platform-frontend`.

## Fuera de alcance, explícitamente

- **El reenrutamiento de Jacobs (`_HTTP_FACETS`/`_MOTOR_FACETS` en
  `jacobs/executor.py`) no cambia.** `Muscle.invoke()` sigue devolviendo
  `str` para todo invocador — el wrapper de contrato vive en `chat.py`,
  no en `Muscle.invoke()`. El riesgo dormido de que ada/thot reenrutados
  se salten `allowed_callers`/`requires_human_gate`/`sandbox_only` queda
  exactamente donde Fase 1 lo dejó: no tocado. **Precisión importante:**
  esto es distinto de decir que ada y thot no participan de este
  sub-proyecto — sí participan, como facetas de la Mesa web, igual que
  las otras cinco (ver "Facetas incluidas" abajo). Lo que queda fuera es
  el camino de Jacobs, no las facetas en sí.
- REPL y la carga #1 (audit log) — ya excluidos por la decisión de
  alcance de Fase 2 en general.
- Pasar de shadow a enforcement. El criterio C3 de REFORMAS-v3.md (100
  rechazos adjudicados) es la señal para decidirlo, informada por los
  datos de este sub-proyecto — la decisión en sí es la ronda siguiente.
- Construir resolvers nuevos — ídem, informado por los datos, no
  adelantado acá.

## 1. Contrato de salida y wrapper en `chat.py`

`Muscle.invoke(prompt, model=None, history=None) -> str` no cambia. El
wrapper vive enteramente en `chat.py`: pide `{claim: [...], analysis:
str, judgment: str | None}` en su propio prompt de sistema, y parsea su
propia respuesta. Esto es una sola llamada al modelo (no hay una
segunda ronda de extracción posterior sobre texto libre — eso violaría
§0.1 de REFORMAS-v3, que rechaza explícitamente un extractor separado).

**El parseo del contrato es real, no shadow — se decide y se muestra ya:**

- Si el JSON parsea → sigue el flujo normal, `contract_parsed=True`.
- Si el JSON **no** parsea (Kimi trunca a 488 bytes hoy, en producción,
  sin que nadie haya tocado nada — este bug ya existe): todo el texto
  recuperado cae a `analysis` sin claims, `contract_parsed=False`,
  `degradation_reason` (texto libre, no enum — todavía no se conocen
  todas las formas de incumplimiento: truncamiento, markdown fences,
  comillas raras, claves faltantes) y el fragmento crudo quedan
  auditados en `shadow_messages`.

**Tres garantías sobre esta degradación**, para que el fallback no sea
una puerta trasera fail-open:

1. **Se audita** — cada degradación queda registrada con faceta, motivo
   y fragmento crudo (`shadow_messages`).
2. **Se marca al usuario** — nota sobria al pie del bloque en el
   frontend (ver sección 4), no un banner de error. El objetivo es que
   sea distinguible, no alarmante.
3. **JSON-que-no-parsea ≠ JSON-válido-sin-claims.** Un contrato bien
   formado con `claim: []` es una respuesta legítima sin afirmaciones —
   `contract_parsed=True`. Un JSON roto a mitad de un claim es
   incumplimiento de contrato — `contract_parsed=False`. Si se
   mezclaran en el mismo estado, el bug de Kimi quedaría enmascarado
   detrás de "no tuvo claims esta vez".

## 2. Shadow validation — dos objetos distintos

**El parseo del contrato (arriba) es real y presente, no shadow** —
§3.1.8 no aplica ahí: un JSON que no parsea es incumplimiento
verificable de forma determinista, no algo que haya que calibrar contra
falsos rechazos.

**El shadow (§3.1.8) aplica solo al veredicto sobre los claims** — si lo
que la faceta afirmó es `VALID`/`FACT_MISMATCH`/etc. contra
`policy/governance/validator.py`. Eso no se le muestra a nadie todavía;
se audita para medir la tasa de falso rechazo antes de activar
enforcement (criterio C3).

## 3. Ejecución — sincrónico vs. background

- **Sincrónico, en el camino crítico:** el parseo del JSON. De eso
  depende qué se le muestra al usuario ya.
- **Asincrónico, vía `BackgroundTasks` de FastAPI, fuera del camino
  crítico:** la validación de claims y el barrido de vocabulario. La
  Mesa responde apenas tiene el JSON parseado; la medición corre después
  sin que el usuario espere I/O de resolvers (`CAPABILITY_AVAILABLE`
  consulta dos registros, `FILE_EXISTS` lee y hashea archivos).

El shadow es un instrumento de medición estadística (el criterio C3
pide 100 rechazos adjudicados, no "todos los rechazos que hubo") — la
pérdida ocasional de mediciones no invalida el resultado, siempre que
sea detectable y no sistemáticamente sesgada. Ver `queued_at`/
`validated_at` abajo: no hace falta un contador aparte de
encoladas-vs-completadas, la ausencia de `validated_at` en una fila que
tiene `queued_at` **es** la medición de pérdida.

## 4. Esquema de datos (`jax_memory`, MariaDB)

Tres tablas nuevas, sin FK dura entre ellas (comparten `conversation_id`
+ `message_id` para cruzarlas cuando haga falta, pero un fallo en una no
bloquea a las otras — se escriben desde el mismo proceso, y una FK dura
convertiría un fallo parcial en fallo total).

### `shadow_messages`

Una fila por respuesta de la Mesa evaluada — insertada **al encolar**
(antes de que la `BackgroundTask` corra), actualizada al completar. Es
el denominador: sin esta tabla, "¿qué % cae en judgment?" no tiene total
contra el que dividir, y el caso más grave (JSON truncado sin ningún
claim recuperable ni ningún término de vocabulario en lo que
sobrevivió) no dejaría rastro en ninguna tabla.

| columna | tipo | nota |
|---|---|---|
| `conversation_id` | | índice |
| `message_id` | | |
| `facet` | | índice |
| `contract_parsed` | bool | |
| `degradation_reason` | `TEXT` (no `VARCHAR(n)`) | no enum — formas de incumplimiento todavía no catalogadas, y un `VARCHAR` corto trunca justo la información que hace falta |
| `has_claim` | bool | |
| `has_analysis` | bool | |
| `has_judgment` | bool | |
| `queued_at` | timestamp | se escribe al encolar |
| `validated_at` | timestamp, NULL | NULL permanente = worker murió antes de completar; con `queued_at` da latencia cuando no es NULL |

### `shadow_claim_verdicts`

Una fila por claim validado.

| columna | tipo | nota |
|---|---|---|
| `conversation_id` | | índice |
| `message_id` | | |
| `predicate` | | |
| `status` | | VALID / FACT_MISMATCH / RESOLVER_NOT_IMPLEMENTED / etc. |
| `detail` | | |
| `args` | `JSON` nativo de MariaDB (no `TEXT`) | qué afirmó exactamente el modelo (ej. `path="policy"`) — necesario para adjudicar los 100 rechazos de C3 con contexto, no solo saber que falló; con `JSON` nativo se puede consultar `args->>'$.path'` directo, sin parsear en el cliente |

### `shadow_vocab_hits`

Una fila por **par término-categoría** — no una lista en una columna,
para que `GROUP BY category` funcione directo sin parsear nada.

| columna | tipo | nota |
|---|---|---|
| `conversation_id` | | índice |
| `message_id` | | |
| `channel` | | analysis / judgment |
| `term` | | |
| `category` | | índice — capabilities/ops/facets_jax/facets_las_manos/motors/commands |

Un término puede pertenecer a más de una categoría (`ada` está en
`facets_las_manos`, `facets_jax` y `motors`) — de ahí fila-por-par en vez
de mapeo 1:1.

## 5. Extensión de `load_vocabulary()`/`sweep()`

Hoy `load_vocabulary()` aplana todo a un `frozenset[str]` sin categoría,
y `sweep()` solo recibe/devuelve ese set aplanado — no hay forma de
responder "¿esto era un comando o un motor?", que es justo la pregunta
que mide la fuga del Apéndice B (`jax_local` mencionando sintaxis
inventada del sistema en `judgment`).

Se resuelve **dentro de `policy/governance/`**, no con un segundo lector
de `closed_vocabulary.yaml` en Sub-proyecto 2: `loaders.py` es el único
módulo que toca config estática de policy, y duplicar su lógica de
indexado se salta la verificación de hash contra `VERSION` (la garantía
fail-closed que ya existe para `load_templates()`).

- `load_vocabulary()` devuelve categoría junto con término.
- `sweep()` cambia su firma a `list[tuple[str, frozenset[str]]]` —
  término + **todas** sus categorías de origen, no una elegida.
- `vocab_sweep.py` sigue siendo puro: cambia qué recibe y qué devuelve,
  no su naturaleza (ciego a política, solo detección léxica).

## 6. Facetas incluidas: las 7, desde el día uno

No se filtra por política de `grounding` (`off`/`auto`/`required_web`/
`local_context_only`) — eso sería filtrar la muestra según la hipótesis
de que grounding predice producción de claims, que es justo lo que el
shadow existe para verificar. Una faceta que nunca produce claims es un
dato, no una ausencia de dato. El costo de incluir las 7 es acotado
(prompt más largo, parseo de JSON — no I/O ni latencia de red extra), y
sacar después una faceta que no aporta señal es trivial comparado con
tener que esperar otra ventana de medición si se hubiera excluido de
entrada.

Dos casos de interés esperados desde el primer día:

- **`jax_local`** — la faceta del Apéndice B, la que originó "trae a
  hyde". Si el barrido detecta menciones a `commands`/`capabilities` en
  su canal `judgment` con frecuencia, es la fuga medida con precisión —
  no "mencionó vocabulario", sino "inventó sintaxis del sistema".
- **Kimi** — va a producir señal de degradación desde el arranque por el
  truncamiento a 488 bytes. Con `degradation_reason` como columna
  propia, en pocos días se sabe si degrada siempre o solo en respuestas
  largas — el dato que falta para diagnosticar el bug de una vez.

## 7. Frontend (`jax-platform-frontend`)

Nota sobria al pie del bloque cuando `contract_degraded=True`: texto
secundario, tamaño reducido, sin rojo de error — la respuesta no
cumplió el formato esperado, no "algo se rompió". Sin esto, la garantía
#2 de la sección 1 se cumple solo estructuralmente (el dato existe) pero
no en la experiencia real — el mismo patrón que el sello cosmético
`origin_of_authority` que §5 de REFORMAS-v3 señaló sin que existiera en
código: un campo correcto que nadie ve no es gobernanza, es teatro.

Cumple la política global del proyecto sin excepción: string en
`es.js`/`en.js` (cero hardcoding de texto visible), CSS variables para
el color (respeta dark/light).

## 8. Rollback

En orden:

1. **Frontend** — sacar el footnote. Cosmético, sin efecto en datos ni
   en el comportamiento de la Mesa.
2. **`chat.py`** — sacar el wrapper. La Mesa vuelve al comportamiento
   actual (texto libre, sin `BackgroundTask`).
3. **Las tres tablas quedan.** No molestan a nadie, sin consumidores
   fuera de las queries de este sub-proyecto.
4. **La extensión de `load_vocabulary()`/`sweep()` (tarea 1) también
   queda**, aunque haya llegado a `master` vía PR #5 antes que el resto.
   Es aditiva y `vocab_sweep.py` sigue funcionando exactamente igual
   para cualquier llamador que no use la categoría — simplemente devuelve
   información extra que nadie consume. Explícito para que nadie la
   revierta "por las dudas" y rompa lo que sí quedó bien.

## 9. Testing

TDD (`superpowers:test-driven-development`), mismo patrón que
Sub-proyecto 1. Casos que el plan debe cubrir explícitamente:

- `sweep()` con categoría — incluyendo un término en múltiples
  categorías (`ada`).
- Wrapper de `chat.py`: JSON válido con claims, JSON válido sin claims
  (`contract_parsed=True`, `has_claim=False`), JSON truncado a mitad de
  un claim (`contract_parsed=False`).
- `_resolve_file_exists` contra una entrada de directorio real de la
  allowlist (issue #6) — antes de que cualquier claim con `path`
  generado por modelo llegue al validador.
- `shadow_messages` con `validated_at` NULL tras fallo simulado del
  worker — confirma que la ausencia es visible, no silenciosa.
- Footnote en frontend: presente cuando `contract_degraded=True`,
  ausente en el caso normal, en ambos modos de tema.

## Siguiente paso

Spec para revisión de Fernando. Aprobado → `superpowers:writing-plans`
para el plan TDD, siguiendo el orden de tareas de la sección "Alcance".
