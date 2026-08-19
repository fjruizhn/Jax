# JAX / Axioma — Especificación de Reformas

**Versión:** 3.1 (especificación con historial de ejecución)
**Fecha:** 2026-08-18
**Autor:** Fernando Ruiz Torres
**Rondas de revisión:** v1 → DeepSeek + GPT → v2 → tercera revisión → v3 → ejecución (Fase 0 a Sub-proyecto 2) → v3.1
**Estado:** Fase 0, Fase 0.5, Fase 1 y Fase 2 (Sub-proyectos 1 y 2) ejecutadas. Ver "Registro de ejecución" (nueva sección) para el detalle verificado de cada una.

**Nota de versión:** v3.1 corrige nueve afirmaciones de v3.0 que la ejecución demostró falsas o imprecisas (§A del changelog), cierra las cinco preguntas abiertas de §6 con su resolución real (§B), y agrega el material que la ejecución produjo y v3.0 no podía anticipar: el corpus normativo, el patrón fail-open (cinco casos), un segundo incidente de referencia, y el registro de ejecución completo (§C). Fuente: `CORRECCIONES-v3.1.md`, 2026-08-18. Ningún contenido de esta versión excede ese listado — lo que no estaba ahí queda como pregunta abierta en §6-bis, no integrado.

---

## 0. Qué cambió respecto de v2

La v2 cerró con seis preguntas abiertas (Q7–Q12). La tercera ronda resolvió cinco y abrió una falla nueva. Este documento integra esas resoluciones y añade tres correcciones propias sobre la revisión.

### 0.1 Cambio arquitectónico principal — inversión de salida

**El defecto de v2:** R1 validaba claims enlazados a evidencia, pero no resolvía quién descomponía la prosa en claims. Si lo hacía el modelo que la generó, un modelo que alucina podía omitir la afirmación alucinada de su propia lista.

**La resolución:** no hay extracción. Se invierte la dependencia.

```
v2 (defectuoso):
  modelo → prosa → extracción de claims → validación → renderizado

v3 (correcto):
  modelo → [claim tipado, ...] JSON → validación determinista → renderer → prosa
```

**El modelo nunca produce prosa factual libre.** Produce una estructura de datos. El runtime valida. El renderer verbaliza. La prosa es salida derivada, no primaria.

**[corregido v3.1: ejecutado]** Esta inversión, descrita en v3.0 como propuesta, está construida y verificada — ver §3.1.1 y "Registro de ejecución" (Sub-proyecto 2).

### 0.2 Correcciones de esta versión sobre la tercera revisión

| # | Corrección | Motivo |
|---|---|---|
| C1 | El agujero de Q7 **no se elimina: se desplaza** al clasificador de canal. Se nombra y se mitiga con barrido de vocabulario cerrado | La partición claim/analysis/judgment la declara el modelo |
| C2 | El renderer usa **plantillas por predicado escritas por humano, versionadas y con hash** — no concatenación genérica | La propuesta original produce salida ilegible |
| C3 | El criterio de falso rechazo se redefine: **muestra adjudicada explícitamente**, no ausencia de reporte | Ausencia de reporte no es validación |

---

## 1. Diagnóstico

*(Sin cambios sustantivos respecto de v2. Se conserva íntegro salvo lo marcado.)*

### 1.1 Qué existe y funciona

| Componente | Estado | Evidencia |
|---|---|---|
| LAS MANOS — Intent Envelope 18 campos | Funciona | Rechazos por condición numerada (2, 6, 8, 9, 18) |
| Validación estructural y semántica | Funciona | `ENVELOPE_REJECTED [estructural]` / `[semantica]` con campo identificado |
| Fail-closed ante operación mutante | Funciona | Ninguna pasó sin `rollback_plan` ni `kill_switch_scope` |
| Verificación de procedencia en memoria | Funciona | `has_provenance=false` rechaza |
| Compuerta humana | Validador construido, ciclo operativo indefinido | Prueba rechazo ante ausencia de token. No prueba emisión, aprobación, expiración ni revocación |
| Memoria semántica (MariaDB, nomic-embed 768d) | Funciona | Doble scope operativo |
| Scheduler Jacobs (DAG, `asyncio.gather`) | Parcial | Ejecuta; no enruta por competencia |
| Plataforma web axioma-ia.io | Funciona | UI, WS, panel de auditoría en vivo |

El núcleo de gobernanza está construido y es correcto. El defecto no es que falte el mecanismo de reglas.

### 1.2 Defecto central: dos caminos, una sola validación

```
Camino A — Ejecución
  Facet → Intent Envelope (18 campos) → validación → LAS MANOS → audit log
  ESTADO: gobernado, auditado, fail-closed.

Camino B — Chat
  Facet → texto libre → renderer → usuario
  ESTADO: sin validación, sin auditoría, sin procedencia (v3.0). Gobernado en shadow desde Sub-proyecto 2 (v3.1) — ver "Registro de ejecución".
```

El sello `🔧 Origen de autoridad` es un string estático del renderer. No deriva del campo `origin_of_authority` del envelope. Es una etiqueta cosmética que reutiliza el nombre de un campo real de un camino que nunca se ejecuta.

### 1.3 Defecto de método: infraestructura sin carga de trabajo

Seis meses, cero cargas productivas recurrentes. No faltó demanda —auditoría, backups, diagnóstico de Kimi, port forwarding— sino **enrutamiento**: todas esas necesidades se resolvieron con Claude Code directo mientras la plataforma se construía en paralelo.

Un sistema sin cargas solo se mide por sus errores. No hay contador de trabajo completado, solo de fallas.

**[corregido v3.1, ver A3 y C3 del changelog]** El backup de hall9000 resultó ser exactamente este defecto, encontrado en producción: reportó éxito cada noche durante un mes con dos fallos diarios escritos en su propio log, sin escalamiento — ver Apéndice B-bis.

### 1.4 Subutilización de cómputo

| Recurso | Capacidad | Uso real |
|---|---|---|
| GPT-OSS-120B (Beelink, 128GB, persistente) | 120B disponible 24/7, costo marginal cero | Prácticamente ocioso |
| Qwen3-Coder-30B | Extracción, clasificación, triaje sin costo | Usado como facet conversacional — su peor caso de uso |
| API (DeepSeek, Gemini, OpenAI, GLM, Kimi) | Amplitud y juicio | Tokens en tareas que el local resolvería |
| Kimi | — | Truncamiento a 488 bytes — corregido, verificado vigente. Ver nota. |

**Nota sobre esta fila (corrección 2026-08-18, no estaba en v3.0):** v3.0 documentaba este truncamiento como defecto abierto. No lo es, y la cronología completa es verificable en `motor_jobs.jsonl`. Bug real el 2026-08-09/10: `kimi-k2.7-code` es modelo de razonamiento, `_call_kimi` (`motor_registry/worker.py`) nunca mandaba `max_tokens`, y `reasoning_content` competía por el mismo budget que `content` — confirmado en vivo contra la API real de Moonshot, no por hipótesis. Corregido el mismo día (`017ba2f`). Un día antes, por un motivo no relacionado (Bloque C, `447d3ec`, 2026-08-08), la ruta de Kimi hacia la Mesa web había quedado blindada (`facet.kimi.transport='motor_registry'`, sin rama en `_invoke_facet` de `chat.py` para ese transporte). Verificado vigente el 2026-08-18 con dos muestras separadas por 8 días en `las_manos/logs/motor_jobs.jsonl` (`_finish_reason: "stop"` en ambas, no `"length"`) — ver `docs/runbooks/verificar-truncamiento-kimi.md` para el método reutilizable.

La cifra de 488 bytes se propagó sin re-verificar contra el código a esta fila de v3.0 y a dos specs de Sub-proyecto 1/2 (17/18-ago) — el plan de SP2 llegó a asumir que Kimi produciría señal de degradación en la Mesa web por este motivo, sin que el código conectara `chat.py` con `motor_registry`. Misma clase de sesgo de A1 que la instancia número siete documentada en Sub-proyecto 2 (§Registro de ejecución): una premisa de diseño, no un dato de color, sin re-verificar antes de construir sobre ella.

**Hallazgo aparte, sin corregir:** Kimi es hoy inalcanzable desde la Mesa web — `chat.py` responde "no disponible" sin invocar la API, porque `_invoke_facet` no tiene rama para `transport='motor_registry'`. No es el mismo bug (no trunca; no responde). Queda en §6-bis como feature gap, no como bug de esta fila.

---

## 2. Principios

**P1** — Una regla es una regla solo si existe código que puede rechazar un output que la viole.
**P2** — Ninguna salida llega al usuario sin pasar por validación. Sin excepción por camino.
**P3** — La procedencia se deriva de hechos verificables y se enlaza a la afirmación concreta.
**P4** — La capacidad se otorga por contrato de tarea, no por identidad de facet.
**P5** — Local primero. La API es escalamiento, no default.
**P6** — Toda plataforma debe justificarse con carga de trabajo en producción.
**P7** — No existe bypass en producción. El rollback se hace por versionado de política y despliegue canario, jamás mostrando como autorizada una salida que falló.
**P8** — La autodeclaración de calidad por parte del modelo no es contractual. **[extendido v3.1]** Aplica también a metadata de gobernanza, no solo al contenido: el emisor nunca certifica su propia procedencia. Ver B-Q16.
**P9** — **La prosa factual es salida derivada.** El modelo emite estructura; el runtime verbaliza. Ningún texto con autoridad se origina directamente en el modelo.

---

## 3. Reformas

### R1 — Gobernanza universal por claim estructurado

#### 3.1.1 Los tres canales de salida

Toda respuesta de un motor se compone de bloques en uno de tres canales. El canal determina el tratamiento.

**[corregido v3.1, ver A4]** v3.0 describía estos tres canales como si ya existieran. No existían: las facetas emitían texto plano de una pieza, y `chat.py` devolvía ese `str` sin validación. Sub-proyecto 2 los construyó — ver "Registro de ejecución".

| Canal | Contenido | Validación | Autoridad |
|---|---|---|---|
| `claim` | Afirmación factual sobre el estado de JAX | Determinista, obligatoria | Sí, con procedencia |
| `analysis` | Razonamiento sobre datos ya establecidos | Barrido de vocabulario (§3.1.5) | No |
| `judgment` | Opinión, hipótesis, recomendación | Barrido de vocabulario (§3.1.5) | No |

Presentación diferenciada al usuario, sin mezclar planos en la misma oración:

```
✅ Hecho verificado   — canal claim, con referencia de evidencia
🔍 Análisis           — canal analysis, sin autoridad
💭 Juicio             — canal judgment, sin autoridad, autoría del motor identificada
```

#### 3.1.2 Formato del claim

```json
{
  "predicate": "CAPABILITY_AVAILABLE",
  "args": { "name": "read_audit_log", "mode": "read_only" },
  "authority": "OBSERVADO",
  "provenance_ref": "tool_result:sha256:...",
  "evidence_pointer": "/capabilities/3",
  "scope": "project:JAX"
}
```

**[corregido v3.1, ver B-Q16]** En la implementación real (Sub-proyecto 2), el modelo emite solo `predicate` y `args`. `authority`, `provenance_ref`, `evidence_pointer` y `scope` los fija el sistema server-side, nunca el modelo — extensión de P8 a metadata de gobernanza: el emisor nunca certifica su propia procedencia. Efecto colateral: el JSON que el modelo tiene que emitir es más corto, lo que reduce la presión sobre el truncamiento de Kimi.

#### 3.1.3 Predicados de semántica cerrada

**Solo estos predicados son emitibles en canal `claim`.** La lista es cerrada y versionada; ampliarla requiere enmienda.

| Predicado | Argumentos | Fuente de verdad |
|---|---|---|
| `CAPABILITY_AVAILABLE` | name, mode | Registro de capabilities |
| `FACET_EXISTS` | name, engine | Configuración de facetas |
| `ENGINE_STATUS` | name, status | Health check |
| `CONFIG_VALUE` | path, key, value | Archivo de configuración con hash |
| `FILE_EXISTS` | path, hash | Sistema de archivos |
| `AUDIT_EVENT_EXISTS` | event_hash | Audit log |
| `JOB_STATUS` | job_id, status | Scheduler |
| `MEMORY_ENTRY_EXISTS` | memory_id, scope | MariaDB jax_memory |

**[corregido v3.1, ver B-Q14]** De estos ocho, solo dos tienen resolver real hoy: `CAPABILITY_AVAILABLE` (consulta ambos registros de capacidades; presente en ambos con valores distintos → `SOURCE_CONFLICT`; ausente en ambos → `FACT_MISMATCH`) y `FILE_EXISTS` (la allowlist de `config_paths` se chequea antes de tocar el filesystem; fuera de ella → `PATH_NOT_ALLOWED`, sin revelar nada del archivo). `ENGINE_STATUS` se descartó explícitamente: no tiene fuente de verdad en el dominio de jax — la tabla `model` de jax-platform mide disponibilidad del provider, no salud del motor, que es una pregunta distinta. Los otros cinco quedan con resolver no implementado y motivo explícito. Esto no invalida la lista cerrada — es el punto de partida honesto que R1 pide.

**La frontera de Q8 queda establecida:** el validador compara tipo de predicado, valores exactos, alcance y procedencia. **No interpreta significado.** Donde haría falta comprender en lugar de comparar, el contenido no es claim — es `analysis` o `judgment`.

Ejemplos que **no** son claims y no pueden serlo: *"esta configuración es peligrosa"*, *"Hyde probablemente resolverá"*, *"el truncamiento parece originarse en el transporte"*, *"esta salida es confiable"*.

#### 3.1.4 `origin_of_authority` — cuatro valores

| Valor | Significado | Requiere |
|---|---|---|
| `EJECUTADO` | Se corrió una operación y quedó en audit log | Hash de entrada de auditoría |
| `OBSERVADO` | Se leyó estado del sistema sin mutarlo | Hash de resultado de herramienta |
| `RECUPERADO` | Memoria o fuente externa | URI o id resoluble |
| `INFERIDO` | Priors del modelo | Solo canal `analysis`/`judgment`. **Prohibido en canal `claim`.** |

**[corregido v3.1, ver B-Q14]** En la implementación real, ninguna faceta ejecuta código durante un turno de chat (`EJECUTADO` descartado) y la Mesa web no tiene grounding cableado al mecanismo de claims — no hay señal por-claim de "esto se buscó" o "esto se observó" (`RECUPERADO`/`OBSERVADO` descartados también). La única autoridad honesta hoy es `INFERIDO`, fijada server-side. Consecuencia directa: **todo claim de la Mesa web sale `AUTHORITY_INVALID`** — resultado esperado y protegido por test (`test_shadow_validation_claim_produces_authority_invalid_verdict`), no un defecto a corregir. El cuello de botella real de R1 no es la cobertura de predicados (§3.1.3): es que el grounding no está conectado al mecanismo de claims. Candidato de alcance para el sub-proyecto siguiente, informado por estos datos una vez que el shadow acumule tráfico real.

#### 3.1.5 Barrido de vocabulario cerrado — corrección C1

**El problema que la tercera revisión no nombró.** La partición en canales la declara el modelo. Nada impide emitir una afirmación factual falsa etiquetada como juicio:

```
💭 Juicio: JAX no puede leer sus propios logs
```

Esa es la afirmación #1 del Apéndice B, y atraviesa el sistema sin tocar el validador. El agujero de Q7 **no se eliminó: se desplazó** de "quién extrae los claims" a "quién decide el canal".

Es un problema menor que el original —el canal de juicio no lleva sello de autoridad y se marca visualmente como opinión— pero requiere mitigación, y existe una determinista.

**Mecanismo.** El registro de herramientas y el de capabilities definen un **vocabulario cerrado**: nombres de capabilities, facetas, motores, rutas de configuración y comandos registrados. Barrido léxico sobre todo texto en canales `analysis` y `judgment`:

- Sin coincidencias → pasa.
- Con coincidencia → el bloque debe reformularse como `claim` (y validarse) o se rechaza.

Sin modelo, sin inferencia semántica, sin clasificador. Comparación contra una lista.

**[corregido v3.1, ver B-Q13]** El vocabulario se implementó con categoría por término (no solo el término plano): un término puede pertenecer a más de una categoría (ej. `ada` está en facetas, en LAS MANOS y en motores a la vez), y el barrido devuelve término + categorías de origen. Esto hace la fuga del Apéndice B medible con precisión — no "mencionó vocabulario", sino "mencionó categoría `commands` en canal `judgment`". Nota de implementación real: la primera versión del cargador de vocabulario descartaba en silencio cualquier categoría cuyo valor no fuera un `dict` — la categoría real `commands:` es una lista, y desaparecía de la carga sin aviso. Es el cuarto caso del patrón fail-open (ver §C2 nuevo), corregido.

**Limitación declarada:** el barrido no detecta afirmaciones sobre el sistema que no usen vocabulario registrado (*"esto no puede leer sus propios registros"*). Se acepta como residuo conocido. Reduce el espacio de fallo; no lo cierra. Ver Q13, cerrada en §6-A de esta versión.

#### 3.1.6 Renderer — corrección C2

**Falla identificada por la tercera revisión (Q12):** el renderer no es un transformador neutral. Si contiene lógica interpretativa, introduce afirmaciones no verificadas. El ejemplo dado: un claim `CAPABILITY_AVAILABLE=true` renderizado como *"está disponible **y se puede usar ahora mismo**"* — la segunda mitad no está en el claim.

**Corrección a la solución propuesta.** La revisión propuso concatenación genérica, produciendo `"La CAPABILITY_AVAILABLE read_audit_log está true."` Es verificable pero ilegible, y la ilegibilidad tiene costo real: un usuario que no puede leer la salida verificada va a preferir el canal de opinión, que es el no gobernado.

**Especificación correcta:**

- Una plantilla fija **por predicado**, escrita por humano.
- Versionada, con hash, bajo control de versiones. Cambiar una plantilla es un commit auditable.
- Slots solo para valores del claim. Sin ramas condicionales, sin texto generado, sin adjetivos que no provengan de un argumento.
- El renderer no puede emitir texto para un predicado sin plantilla registrada.

```
CAPABILITY_AVAILABLE:
  "La capability {name} está disponible en modo {mode}."

AUDIT_EVENT_EXISTS:
  "Existe un evento de auditoría con hash {event_hash}."

FILE_EXISTS:
  "El archivo {path} existe (hash {hash})."
```

Legible y no generativo. La restricción está en la plantilla, no en la torpeza.

**[corregido v3.1, ver B-Q15]** Checkpoint de legibilidad ejecutado: las tres plantillas `definida` se leyeron en voz alta con Fernando. Todas sobrias, casi de log — ninguna suena a prosa espontánea. El hash se mantiene completo en la plantilla de `FILE_EXISTS` (no se acortó a `hash_short`): no existe hoy un canal entre `validate()` (que devuelve `Verdict`) y `render()` (que recibe solo `Claim`) para derivar un hash corto sin romper la separación de responsabilidades que este mismo apartado exige. Reversible en una ronda futura si hace falta.

El renderer de `analysis` y `judgment` no tiene esta restricción, pero esos canales no llevan sello de autoridad y **el renderer no mezcla planos en la misma oración**.

#### 3.1.7 Comportamiento ante rechazo

No se renderiza. Reintento con la restricción explícita. Si persiste, escalamiento de motor. Si no hay escalamiento, se muestra el rechazo con la condición violada. **Nunca la prosa inválida.**

**[corregido v3.1]** En la implementación real de Sub-proyecto 2, un JSON que no cumple el contrato de forma (no parsea, le faltan claves) no se trata igual que un claim que sí parsea pero falla validación semántica. Lo primero es incumplimiento de contrato — se audita, se degrada el bloque completo a `analysis` con el texto crudo, y se marca al usuario con una nota sobria (no un error). Lo segundo es lo que describe este apartado. Distinguir ambos casos importa: mezclarlos enmascararía bugs de transporte (como el truncamiento de Kimi) detrás de "el modelo afirmó algo inválido".

#### 3.1.8 Despliegue seguro

1. **Shadow validation** — el validador corre sobre todas las salidas, registra veredicto, no bloquea. Ventana acotada.
2. **Canario** — enforcement en un subconjunto de perfiles de salida.
3. **Versionado de política** — rollback a la última política válida, nunca a "sin política".
4. **Modo diagnóstico** — solo en entorno aislado, sin usuario, sin memoria, sin capabilities.

**Sin WARN. Sin BYPASS en producción.**

**[corregido v3.1]** El shadow real (Sub-proyecto 2) distingue dos objetos: el parseo del contrato (real, se muestra ya al usuario — ver §3.1.7) y el veredicto sobre los claims (shadow puro, se audita, no se muestra). Corre asincrónicamente vía `BackgroundTask`, después de responder al usuario — la fila del mensaje evaluado se inserta al encolar, no al completar, y el timestamp de validación queda `NULL` permanente si el proceso muere a mitad de camino. Esa ausencia es la medición de pérdida: no hace falta un contador aparte.

**Criterio de paso a enforcement — corrección C3.**

La tercera revisión propuso: tasa de falso rechazo < 1% sobre 500 interacciones, definiendo falso rechazo como *"salida que el usuario considera válida (no la reporta como errónea)"*.

**Dos defectos.** Primero, ausencia de reporte no es validación — es la misma falacia que la revisión anterior señaló correctamente sobre la compuerta humana. Segundo, 500 interacciones completas en 7 días son ~70 diarias para un operador único: probablemente inalcanzable, y forzar el número degradaría la muestra.

**Criterio revisado:**

| Parámetro | Valor |
|---|---|
| Ventana | 14 días, o hasta completar la muestra adjudicada — lo que ocurra primero |
| Muestra | 100 salidas que el validador **habría bloqueado**, adjudicadas explícitamente una por una |
| Adjudicación | Fernando marca cada una: rechazo correcto / falso rechazo. Sin adjudicación no cuenta |
| Umbral | Falso rechazo < 5% de la muestra adjudicada |
| Si supera | Se corrige el validador y se reinicia la ventana con la política versionada nueva |

Se adjudica solo el conjunto de rechazos, no todo el tráfico: es el único subconjunto donde el falso rechazo puede existir. Eso hace la muestra alcanzable sin diluirla.

**Criterio de aceptación de R1:** reproducir el incidente del Apéndice B y verificar que produce `ENVELOPE_REJECTED` o reclasificación forzada, no prosa con autoridad.

---

### R2 — Jerarquía de cómputo local-first

| Nivel | Motor | Clase de tarea | Costo |
|---|---|---|---|
| **L0 — Determinista** | Código, sin modelo | Parseo, regex, SQL, correlación por ventana, joins, conteos, diff, barrido de vocabulario | 0 |
| **L1 — Local ligero** | Qwen3-Coder-30B | Clasificación, enrutamiento, extracción estructurada, triaje, embeddings | 0 |
| **L2 — Local pesado** | GPT-OSS-120B | Reducción de contexto, primera pasada, borrador, validación semántica | 0 |
| **L3 — API** | Claude Code / DeepSeek / GPT | Juicio arquitectónico, debugging sutil, decisiones irreversibles, amplitud | $ |

**Regla dura:** ninguna tarea llega a L3 sin registro auditado de por qué L0–L2 no la resuelven.

**Contrato de cobertura.** Dos bloques separados; **solo el primero es contractual**:

```json
{
  "coverage_deterministic": {
    "input_chunks": 842, "processed_chunks": 842, "discarded_chunks": 0,
    "source_bytes": 52428800, "evidence_refs_emitted": 37, "entities_indexed": 214
  },
  "coverage_model_reported": {
    "unresolved_items": [], "estimated_uncertainty": 0.27
  }
}
```

Un modelo que perdió una señal distribuida no sabe que la perdió, y reportará confianza alta precisamente donde falló. Tratar su estimación como garantía reproduciría el patrón conceptual señalado en revisión sobre el sello cosmético — no un artefacto que exista en código para eliminar, sino un patrón a no introducir al construir esto.

**Señales distribuidas** — relaciones causales que solo emergen al cruzar registros distantes. Un umbral de confianza autodeclarada no las detecta. Estrategias por clase de tarea:

- **L0 primero** — correlación temporal, joins y conteos sobre el corpus, sin modelo.
- Map-reduce con índice de entidades y marcas de tiempo.
- Segunda pasada transversal sobre el índice, no sobre el texto.
- Muestreo del original por el juez, guiado por `evidence_refs`.
- Escalamiento a lectura total **por clase de tarea declarada**, no por confianza estimada.

**Línea base obligatoria:** la carga #1 se corre primero con API para medir costo real. Sin línea base, el objetivo de ≥90% no es verificable.

---

### R3 — Capacidades por contrato

1. **Registro de capabilities desacoplado del facet.** Tipadas, con nivel de riesgo y requisitos de envelope.
2. **El contrato de la tarea declara las capabilities.** El orquestador las otorga al motor que ejecuta. LAS MANOS sigue fail-closed; la puerta la abre el contrato, no el nombre.
3. **Capabilities de solo lectura disponibles a cualquier motor:** `read_audit_log`, `read_own_config`, `list_facets_and_capabilities`, `read_memory`. Producen `authority=OBSERVADO`.
4. **Rechazo tipado:**
```json
{ "status": "CAPABILITY_UNBOUND", "required": ["read_audit_log"],
  "candidates": ["hyde", "gptoss_120b"], "task_id": "..." }
```
Interceptado por el scheduler y re-enrutado. El usuario nunca ve este estado.
5. **Identidad inyectada.** Cada motor recibe quién es, qué capabilities tiene en esta tarea, qué motores existen y qué puede cada uno, la lista de predicados emitibles, y el protocolo de rechazo tipado.
6. **Token de compuerta humana:**

| Propiedad | Requisito |
|---|---|
| Emisor | Identidad autorizada, registrada |
| Alcance | Sujeto y operación específicos |
| Vinculación | Hash exacto del plan aprobado |
| Expiración | Timestamp duro |
| Unicidad | Nonce de un solo uso, sin replay |
| Revocación | Mecanismo activo |
| Invalidación | Qué cambios del plan invalidan el token |
| Precedencia | La aprobación humana nunca sobrepasa un DENY de política |

**[corregido v3.1, ver A7]** La ejecución de Fase 1 encontró que hoy coexisten **dos sistemas de autorización con namespaces disjuntos**, no un mecanismo compitiendo consigo mismo: `las_manos/policy.py` gatea 11 nombres bajo `[ops.*]`; `motor_registry` gatea 12 nombres bajo `[capabilities.*]`; cero solapamiento verificado. La unificación que R3 propone no es armonizar dos listas del mismo concepto — es decidir si `ops` y `capabilities` son el mismo concepto de gobernanza o dos legítimamente distintos, y esa decisión sigue pendiente. Registrado también: el reenrutamiento de `CAPABILITY_UNBOUND` (implementado en Fase 1) puede hoy reasignar hacia `_HTTP_FACETS` (ada/thot) saltándose la gobernanza del Motor Registry (`allowed_callers`, `requires_human_gate`, `sandbox_only`) — dormido porque ninguna capability con gate lista más de un motor todavía, documentado como invariante diferido a la unificación de R3.

---

### R4 — Motor, rol y faceta

La separación responde a *dónde viven identidad, permisos y autoridad* — no a *qué modelo rinde mejor*. Los benchmarks responden lo segundo; no responden lo primero.

| Concepto | Qué es | Ejemplo |
|---|---|---|
| **Motor** | Modelo, endpoint, versión, capacidades técnicas | `gptoss-120b@beelink:8082` |
| **Rol** | Obligación, herramientas y límites dentro de una tarea | `reductor`, `ejecutor`, `crítico`, `verificador`, `árbitro` |
| **Faceta** | Identidad, relación con el usuario, presentación | Hyde, Hipatia, Ada |

El envelope registra qué motor actuó bajo qué rol y con qué autoridad. Sin la separación, el permiso se hereda por identidad — el defecto que R3 corrige. Tiene valor aunque todos los roles usen temporalmente el mismo motor.

**Implementación mínima:** `productor → crítico independiente → resolución`.

**Consolidación — regla de tres cargas.** La afirmación de v1 ("siete con dos rotas rinden menos que tres") se retira por no demostrada.

> Un motor solo puede declararse no útil para un rol después de que al menos **tres cargas completadas de esa familia de tareas** lo hayan mostrado consistentemente inferior en tasa útil.

Distinción operativa:
- **Desactivar para una familia de tareas** — permitido tras una carga. (Ej.: Kimi no se usa para reducción de logs.)
- **Desactivar de la flota** — requiere tres cargas.

Consecuencia aceptada: a cadencia de una carga a la vez, la consolidación de flota queda a meses de distancia. Es el costo de decidir con datos.

```
tasa útil = resultados correctos, con procedencia y contractualmente válidos
            ─────────────────────────────────────────────────────────────
                                intentos totales
```

Dimensiones: competencia, confiabilidad, latencia, costo, tasa de salida válida, estabilidad de integración, capacidad exclusiva. Un modelo brillante que trunca o viola el envelope puede tener menor tasa útil que uno inferior pero estable.

**Evidencia empírica del rol de crítico.** Este documento la produjo. Tres modelos con pesos distintos, sin coordinación, detectaron por separado un patrón de autodeclaración de calidad no contractual (el mismo que P8 nombra) en la propuesta de `confianza_promedio`. Dos de tres detectaron el agujero de WARN/BYPASS. La tercera ronda encontró la falla del renderer, que ninguna de las dos anteriores vio. Ninguno de esos hallazgos aparece pidiendo autocrítica al mismo modelo en un segundo turno. **La diferencia no es el turno adicional: son los pesos distintos.**

**[corregido v3.1, ver A6]** El mismo patrón se reprodujo en sentido inverso durante esta misma revisión: **ambos** revisores externos (no solo DeepSeek en v1→v2, como registraba el Apéndice C original) inventaron artefactos que no existen — GPT afirmó la existencia de un "Motor Intercambiable" que no aparece en ningún documento del ecosistema. El patrón de A1 (el que supone se equivoca) no es anécdota de un modelo: es general al método de revisión cruzada mismo, y hay que seguir verificando lo que un revisor cita, no solo lo que un modelo afirma en producción.

---

### R5 — Carga de trabajo primero

Se prohíbe construir capacidad nueva sin una carga concreta que la consuma.

**Carga productiva #1: lectura y análisis del audit log.**

```
1. Capability read_audit_log otorgada por contrato          (R3)
2. L0 correlaciona por ventana temporal y entidad           (R2)
3. GPT-OSS-120B destila con cobertura determinista          (R2, L2)
4. Claims tipados, validados, renderizados por plantilla    (R1)
5. Claude Code juzga el destilado y propone acción          (R2, L3)
6. Registro en audit log con hash                           (LAS MANOS)
```

**[corregido v3.1, ver A3]** Esta carga #1 completa **no se construyó** — Fase 1 solo otorgó la capability y la identidad inyectada (pasos 1 y parte del 4). La decisión de alcance real, tomada al llegar a Fase 2, fue distinta: construir el núcleo de gobernanza de R1 (pasos 4 en abstracto: schema, validador, barrido, renderer) y conectarlo en shadow a la Mesa web (tráfico de chat real y diario) en vez de construir el pipeline de 6 pasos de arriba desde cero. R5 se satisface igual — "carga concreta que la consuma" no exige que sea *esta* carga específica, y el tráfico diario de la Mesa cuenta como carga real tanto como un batch nocturno. El pipeline de 6 pasos de audit log queda como una carga futura posible, no como la única forma de satisfacer R5.

**Definición de listo:** informe de anomalías de las últimas 24h, con procedencia verificable por claim, sin intervención manual, en menos de 60 segundos, bajo 3K tokens de API contra la línea base medida.

**Compuerta de verificación de utilidad:**

> Si 30 días después de habilitar `read_audit_log` no existe al menos una carga recurrente cuyos resultados Fernando consuma efectivamente, se congela toda construcción de capacidades nuevas y se revisa la hipótesis de utilidad de las cargas previstas.

Cargas siguientes, una a la vez: verificación nocturna de integridad de backups (criterio B1.4, **desbloqueado** — ver Apéndice B-bis); detección de deriva del catálogo de modelos; auditoría de port forwarding del ER7212PC.

---

## 4. Métricas

| Métrica | Definición | Estado | Objetivo |
|---|---|---|---|
| **Cargas en producción** | Trabajos recurrentes sin intervención | Shadow validation sobre la Mesa web (SP2) | 1 en 30 días (compuerta dura) |
| **Cobertura de gobernanza** | % de salidas al usuario con canal declarado y validado | Parseo de contrato: 100% en 6/7 facetas (shadow del veredicto de claims, no enforcement) | 100% |
| **Claims sin procedencia** | Claims emitidos con `authority=INFERIDO` | 100% — por diseño, ver §3.1.4 | 0 (imposible por diseño, condicionado a grounding conectado) |
| **Fugas de canal** | Afirmaciones factuales sobre JAX emitidas en `analysis`/`judgment`, detectadas por barrido | Medible desde SP2 (`shadow_vocab_hits`, por categoría) | Medida y decreciente |
| **Tasa útil por motor** | Correctos + procedentes + válidos ÷ intentos | Sin medir | Ranking tras carga #1 |
| **Ratio local/API** | Tokens en L0–L2 ÷ tokens de API | Sin medir | ≥ 10:1 contra línea base |
| **Costo por carga completada** | USD de API por ejecución | Sin medir | Declarado y decreciente |
| **Utilización de GPT-OSS-120B** | Horas/semana con carga real | ~0 | > 0 y creciente |
| **Falso rechazo** | Rechazos adjudicados como incorrectos ÷ muestra adjudicada | Sin medir — la muestra empieza a acumularse con SP2 en producción | < 5% antes de enforcement |
| **Reglas con test** | Normas que sobreviven al filtro P1 y tienen test | 0 de 25 en el corpus normativo (línea base honesta, ver §C1) | 100% |

---

## 5. Secuencia

**Fase 0 — Verificación contractual** *(bloqueante de Fase 1)* — **EJECUTADA**
- Confirmar si las entradas del audit log del 2026-08-**09** [corregido v3.1, ver A1] 16:53:05 son batería de tests o tráfico real. → Batería de tests (`test_envelope_brutal.py` + `test_audit_traffic_class.py`).
- Auditar Six Impossible Things con el filtro P1. → **[corregido v3.1, ver A5]** No aplicaba: no es un documento normativo, es un inventario de seis entidades de infraestructura (Hall9000, ATEM-AI, Sésamo, JAX, Axioma, Red Queen/Marina). Consecuencia: motivó la Fase 0.5 (corpus, §C1).
- Localizar dónde persiste LAS MANOS su auditoría (archivo vs MariaDB). → Archivo, `las_manos/logs/audit.jsonl`, JSONL append-only.
- **Verificar los contratos citados por revisión**: *Motor Intercambiable*, *Principio de No Diferimiento Contractual*, *Memoria Viva*, *Enmienda*, *Threat Model*. → **[corregido v3.1, ver A6]** "Motor Intercambiable" no existe en ningún lado — inventado por el revisor. Los otros cuatro existen dentro de `six-impossible-things.html`, todos v0.1/borrador, ninguno ratificado formalmente.
- Definir la forma del benchmark derivado de las cargas.

**Fase 0.5 — Corpus normativo** *(nueva, no prevista en v3.0)* — **EJECUTADA**. Ver §C1.

**Fase 1 — Desbloqueo de lectura** — **EJECUTADA**
- Capability `read_audit_log`.
- Identidad y capabilities inyectadas por motor.
- Rechazo tipado `CAPABILITY_UNBOUND` + intercepción en scheduler.

**Fase 2 — Gobernanza universal mínima** — **EJECUTADA (Sub-proyectos 1 y 2)**
- Registro de herramientas y de predicados cerrados.
- Salida en canales `claim` / `analysis` / `judgment`.
- Validador determinista de claims.
- Barrido de vocabulario cerrado sobre canales libres.
- **Plantillas de renderer por predicado, versionadas y con hash.**
- Eliminación del sello cosmético.
- Shadow validation (14 días / 100 rechazos adjudicados) → canario → enforcement. — Shadow conectado y corriendo (SP2); ventana de adjudicación arranca cuando el PR mergee a producción.

**Fase 3 — Carga productiva #1**
- Línea base con API. Análisis del audit log de punta a punta. Medición contra la definición de listo. **[ver A3]** — alcance real distinto al descrito acá; ver R5.

**Fase 4 — Benchmark acumulativo**
- Motores medidos sobre cargas reales. Tasa útil. Kimi investigado como incidente de integración. Desactivación por familia permitida; consolidación de flota solo tras tres cargas.

**Fase 5 — Local-first**
- Clasificador L0–L3 obligatorio. Reducción con cobertura determinista. Estrategias para señales distribuidas.

**Fase 6 — Equipo mínimo**
- `productor + crítico independiente + árbitro`.

---

## 6. Preguntas abiertas — cerradas en v3.1

**Q13 — Residuo del barrido de vocabulario. CERRADA.**
§3.1.5 mitiga la fuga de canal para afirmaciones que usan vocabulario registrado. No detecta *"esto no puede leer sus propios registros"* — misma afirmación falsa, sin token registrado.
**Resolución:** se acepta como residuo permanente, mitigado mejor de lo especificado — el barrido ahora devuelve categoría de origen por término, no solo el término, lo que hace la fuga del Apéndice B medible con precisión ("mencionó categoría `commands` en canal `judgment`"). No existe mecanismo determinista adicional propuesto que no sea un clasificador — eso queda fuera de alcance de R1.

**Q14 — Cobertura de los predicados cerrados. CERRADA.**
La lista de §3.1.3 tiene ocho predicados. ¿Cubre las afirmaciones factuales que realmente se necesitan en operación diaria?
**Resolución, en dos niveles:** (1) solo 2 de 8 predicados tienen resolver real medido — ver §3.1.3. (2) Reencuadre más profundo: aunque los 8 tuvieran resolver, ningún claim de la Mesa web pasaría hoy, porque `authority` solo puede ser honestamente `INFERIDO` (grounding no cableado al mecanismo de claims) y `INFERIDO` está prohibido en canal `claim` — ver §3.1.4. El cuello de botella real no es cobertura de predicados. La proporción de contenido por canal se mide con datos reales desde que el shadow de SP2 esté en producción, no estimada.

**Q15 — Legibilidad bajo plantilla. CERRADA.**
**Resolución:** checkpoint de lectura en voz alta ejecutado sobre las tres plantillas `definida` — sobrias, no suenan a prosa espontánea. Ver §3.1.6.

**Q16 — Costo de la inversión de salida. CERRADA.**
**Resolución:** el modelo solo emite `predicate`+`args` (dos campos), el resto del claim lo completa el sistema — ver §3.1.2 y P8. El sobrecosto de tokens es menor al previsto porque el JSON a emitir es más corto que un claim completo de seis campos, con el efecto colateral favorable de aliviar presión sobre el truncamiento de Kimi.

**Q17 — ¿Qué falla grave contiene esta v3 que no se está viendo? CERRADA.**
**Resolución:** la falla no vista era A4 (§ del changelog de esta versión): el documento describía como existente una arquitectura (los tres canales de salida) que todavía no estaba construida. El sesgo se repitió de forma sistemática durante la ejecución — siete casos registrados en la sesión de Sub-proyecto 2 donde leer el código real corrigió una suposición del plan o del documento, incluida esta misma pregunta.

---

## 6-bis. Preguntas abiertas nuevas — v3.1

Estas no estaban resueltas en el listado de correcciones y no se integran como hechos — quedan como preguntas para la próxima ronda:

- ¿La reconciliación de `is_canned` (señal posicional de Sub-proyecto 2) con `UsageInfo` de `infra/facetas-bloque-d` coincide en el tiempo con el destrabe de B1.4 (backup, Apéndice B-bis)? Si esa rama de 28 commits son los PRs que esperaban ese criterio, la reconciliación está más cerca de lo que se documentó en Sub-proyecto 2.
- ¿Cuándo se cablea grounding al mecanismo de claims (§3.1.4)? Depende de que el shadow acumule suficiente tráfico real para que sea una decisión informada, no de una fecha.
- ¿`ops` y `capabilities` (A7) son el mismo concepto de gobernanza o dos legítimamente distintos? La unificación de R3 no puede avanzar sin responder esto primero.
- **(nuevo, 2026-08-18)** Kimi es inalcanzable desde la Mesa web — `_invoke_facet` en `chat.py` no tiene rama para `transport='motor_registry'`, cae al fallback "no disponible" sin invocar la API (ver nota en §1.4). ¿Se cablea `chat.py` contra `las_manos:7777/motor/dispatch`, o se retira `kimi` de la whitelist de facetas de la Mesa mientras tanto? Ninguna de las dos está decidida.

---

## 7. Lo que este documento NO propone

- No abandonar LAS MANOS.
- No eliminar las facetas. Separarlas de rol y motor.
- No reescribir desde cero. Todas las reformas extienden mecanismos existentes.
- No dejar de usar Claude Code. Dejar de gastarlo donde un modelo local ocioso absorbe.
- No desactivar motores por incidentes aislados. Solo por tasa útil medida sobre tres cargas.
- No métricas de actividad. Líneas de código, commits y horas no son resultados.

---

## Apéndice A — Axiomas

**A1. El que supone se equivoca.** Violado por diseño en el camino de chat. R1 lo convierte en imposibilidad estructural: el modelo no puede emitir prosa factual, solo claims validables. Violado también, durante el propio proceso de revisión, por los dos modelos revisores (Apéndice C) — y de nuevo, siete veces, durante la ejecución de Sub-proyecto 2, cada vez corregido por lectura directa del código antes de construir sobre el supuesto.

**A2. Saber no cuesta nada.** Contradicho en la práctica: saber cuesta tokens cuando se consulta al modelo caro para lo que el local resuelve gratis. R2 lo restituye.

**A3. Mañana es el día que el fracasado tiene más que hacer.** Aplicable a R5. Compuerta de 30 días.

---

## Apéndice B — Incidente de referencia

**Fecha:** 2026-08-**09** [corregido v3.1, ver A1], 22:26–22:28
**Facet:** jax_local (Ollama, qwen3-coder:30b)
**Petición:** revisar el audit log

**Afirmaciones producidas, todas falsas o no verificadas:**
1. "JAX no tiene acceso directo a los logs internos del sistema" — el audit log se renderiza en el panel derecho de la misma pantalla.
2. "no puedo invocar a Hyde ni ejecutar comandos por mí mismo" — afirmación sobre el orquestador emitida por un facet sin acceso a esa información.
3. `trae a hyde` — sintaxis presentada como comando del sistema. Presuntamente inventada.
4. `/var/log/jax*` — ruta sugerida sin verificación.

**Todas bajo el sello:** `🔧 Origen de autoridad: conocimiento técnico de JAX.`
**Entradas en audit log durante el incidente: 0.** **[corregido v3.1, ver A1]** Última previa: 16:53:05, **cinco días** antes, no 5h33m — el hueco real es mucho mayor que el que registraba v3.0, y hace la evidencia del Camino B sin gobernar más fuerte, no más débil.

**Trazado contra la arquitectura v3:**

| # | Bajo v3 |
|---|---|
| 1 | Bloqueada. Si va como `claim` requiere `CAPABILITY_AVAILABLE` con evidencia (no existe). Si va como `judgment`, el barrido detecta "logs"/"audit" en vocabulario registrado y fuerza reclasificación |
| 2 | Bloqueada. `ENGINE_STATUS`/`CAPABILITY_AVAILABLE` sin procedencia |
| 3 | Bloqueada. `trae a hyde` no está en el registro de herramientas |
| 4 | Bloqueada. `FILE_EXISTS` requiere hash; la ruta no existe |

Caso de prueba de aceptación de R1.

---

## Apéndice B-bis — Segundo incidente de referencia: el backup *(nuevo en v3.1)*

**Sistema:** `backup-hall9000.sh`, systemd timer.
**Duración del fallo:** aproximadamente un mes, sin escalamiento.

`backup-hall9000.sh` reportó éxito cada noche durante un mes, con dos fallos diarios escritos en su propio log: un `chown` sin regla `NOPASSWD` correspondiente en sudoers (desde 2026-07-14), y el paso de retención (`forget`+`prune`) fallando desde 2026-07-13 por un lock de restic huérfano — el `prune` era además estructuralmente imposible de completar contra el Bucket Lock de Cloudflare R2 (retención de 10 días, objetos no eliminables antes de eso). `systemctl status` verde, el timer verde, cero escalamiento a ningún canal.

Detectado por auditoría manual, no por el propio sistema.

**Por qué es el par del Apéndice A/A1 y no un incidente aislado:** es P02 ("ninguna salida llega al usuario sin pasar por validación. Sin excepción por camino") violado en la infraestructura de la que depende todo lo demás — un backup que miente sobre su propio éxito es el Camino B de este documento, aplicado a operaciones en vez de a texto. Las imágenes de las VMs en sí nunca fallaron; el defecto estaba en la capa de reporte, exactamente el patrón que R1 existe para cerrar en el camino de chat.

**Corrección aplicada:** `chown` movido a una operación con regla `NOPASSWD` que coincide exactamente con los flags invocados; retención separada de la copia (local con `prune` real, R2 solo con `forget`, nunca `prune`, por el Bucket Lock); códigos de salida distinguibles por etapa en vez de un solo "éxito/fallo"; prueba explícita de que el script falla cerrado ante cada clase de error, no abierto.

**Consecuencia sobre R5:** el criterio B1.4 de la carga siguiente ("verificación nocturna de integridad de backups") quedó **desbloqueado** por esta corrección.

**Relación con el patrón fail-open (Apéndice C-bis):** es el primero de los cinco casos documentados, cronológicamente — el que dio nombre al patrón antes de que se reconociera como patrón.

---

## Apéndice C — Registro de revisión

### v1 → v2

| Aporte | Origen | Resolución |
|---|---|---|
| Tool calls como adquisición de evidencia | DeepSeek | Aceptado como registro de herramientas |
| Tool call = procedencia | DeepSeek | Rechazado — la unidad es la afirmación (GPT) |
| Metadatos de cobertura | DeepSeek | Aceptado con partición determinista/autodeclarada |
| `confianza_promedio` como garantía | DeepSeek | Rechazado — autodeclaración, patrón a no introducir (convergencia de tres revisiones) |
| Motor/rol/faceta es ceremonial | DeepSeek | Rechazado — valor contractual (GPT) |
| Benchmark antes de consolidar | DeepSeek | Aceptado, derivado de carga real |
| "No había cliente" | DeepSeek | Rechazado — hubo demanda, faltó enrutamiento |
| Deadline de 30 días | DeepSeek | Aceptado, reformulado |
| WARN / BYPASS | DeepSeek | Rechazado — reabre el Camino B (GPT) |
| Shadow validation + canario + versionado | GPT | Aceptado |
| Compuerta humana infradiseñada | DeepSeek | Aceptado, con especificación de token (GPT) |
| `OBSERVADO` como cuarto valor | GPT | Aceptado |
| Tasa útil sobre precisión promedio | GPT | Aceptado |
| Estrategias para señales distribuidas | GPT | Aceptado, con L0 antepuesto |
| Mapeo erróneo de facetas a modelos | DeepSeek | Corregido; registrado como evidencia |
| "Motor Intercambiable" como artefacto citable | GPT | **[corregido v3.1, ver A6]** Rechazado — no existe en ningún documento del ecosistema, inventado por el revisor |

### v2 → v3

| Aporte | Origen | Resolución |
|---|---|---|
| Claims como salida primaria, prosa derivada | Tercera revisión | **Aceptado — cambio arquitectónico principal** |
| Frontera de entailment: predicados de semántica cerrada | Tercera revisión | Aceptado, con lista explícita de ocho |
| Partición claim / analysis / judgment | Tercera revisión | Aceptado, con mitigación añadida (C1) |
| Renderer como superficie de ataque | Tercera revisión | **Aceptado — falla real no vista antes** |
| Renderer por concatenación genérica | Tercera revisión | Corregido (C2) — plantillas por predicado, versionadas |
| Shadow 7 días / 500 interacciones / no-reporte | Tercera revisión | Corregido (C3) — 14 días, 100 rechazos adjudicados, umbral 5% |
| Regla de tres cargas para consolidación | Tercera revisión | Aceptado, con distinción familia/flota |
| Verificación de contratos como bloqueante de Fase 1 | Tercera revisión | Aceptado |
| Fuga de canal por autoetiquetado | **No detectado por ninguna revisión** | Añadido (C1) — §3.1.5 |

### Nota metodológica

En su revisión de v1, DeepSeek afirmó desconocer qué modelo respalda a Hipatia y Ada y dudó de la existencia de dos facetas. **Hipatia es Gemini, Ada es Zhipu GLM, Jekyll es DeepSeek, Thot es OpenAI**, y las siete existen. El documento v1 no las enumeraba; el revisor rellenó la laguna con suposición y construyó sobre ella su recomendación de Q4.

Es el incidente del Apéndice B reproducido por un modelo de frontera, dentro de un documento sobre cómo impedirlo. Se conserva como evidencia de que el patrón es general.

**[corregido v3.1, ver A6]** El patrón se confirmó general, no específico de DeepSeek: GPT lo reprodujo en sentido inverso, inventando la existencia de un artefacto ("Motor Intercambiable") en vez de negar la de uno real.

---

## Apéndice C-bis — El patrón fail-open *(nuevo en v3.1)*

Cinco casos independientes, encontrados en sesiones separadas, del mismo defecto de diseño: un validador o gate que, ante error o ausencia de señal, **deja pasar** en vez de bloquear.

1. **`backup-hall9000.sh`, paso de retención** — solo logueaba el fallo, nunca tocaba una variable de estado; el script siempre salía en éxito. Ver Apéndice B-bis. Corregido.
2. **`_HTTP_FACETS` (ada/thot) puede saltarse la gobernanza del Motor Registry** si el reenrutamiento de Fase 1 cae ahí — `allowed_callers`/`requires_human_gate`/`sandbox_only` no se aplican. Dormido hoy, diferido a la unificación de R3 (ver A7). No corregido — documentado como invariante.
3. **`output_validator.py` del Motor Registry** valida JSON de salida de motor contra schema, pero es fail-open por diseño explícito ("nunca bloquea un job"). Encontrado durante el relevamiento de Fase 2. R1 exige exactamente lo opuesto. No corregido en esta ronda.
4. **`load_vocabulary()`** (§3.1.5) descartaba en silencio cualquier categoría del vocabulario cerrado cuyo valor no fuera un `dict` — la categoría real `commands:` es una lista, se perdía sin aviso. El caso más significativo de los cinco: apareció en el módulo escrito específicamente para prevenir este patrón. Corregido en el mismo commit que lo encontró.
5. **`IsADirectoryError` sin capturar en `_resolve_file_exists`** — una excepción sin capturar es fail-open por otra vía: no devuelve un `Verdict` de rechazo, revienta, y el comportamiento que sigue depende de quién llame. Corregido (Sub-proyecto 2, Task 2).

**Por qué importa:** cinco subsistemas sin relación directa entre sí llegaron al mismo defecto de forma independiente — no es un bug aislado, es un sesgo de diseño recurrente en el ecosistema, tan arraigado que se reprodujo incluso en el módulo escrito para prevenirlo (caso 4). Candidata fuerte a ser la primera regla **NORMATIVA** (no CULTURAL) del corpus (§C1): *"ningún validador o gate puede fallar abierto ante error o ausencia de señal, incluyendo vía excepción sin capturar."* Admite test trivial y tiene cinco casos reales de fundamento — más que cualquier otra regla candidata del corpus a la fecha de esta versión.

---

## Registro de ejecución *(nuevo en v3.1)*

Resumen verificado de cada fase ejecutada entre 2026-08-15 y 2026-08-18. El detalle completo, con hashes de commit, vive en los specs y planes de `docs/superpowers/` de los repos `jax` y `jax-platform`, no se transcribe acá.

**Fase 0 — Verificación contractual.** Cuatro verificaciones cerradas (ver §5). Validador de LAS MANOS probado con 30 tests en `test_envelope_brutal.py`. Audit log confirmado como archivo (`las_manos/logs/audit.jsonl`, JSONL append-only), no MariaDB.

**Fase 0.5 — Corpus normativo.** Ver §C1 (nueva sección abajo).

**Fase 1 — Desbloqueo de lectura.** `read_audit_log` otorgada por contrato de tarea (no por identidad de facet). `CAPABILITY_UNBOUND` tipado, interceptado y reenrutado por el scheduler antes de abortar el pipeline. Identidad inyectada al motor. Bug real encontrado por verificación directa, no por test automatizado: `Step` es un modelo Pydantic mutable, y `model_copy()` en el camino de reenrutamiento ocultaba que el reenrutamiento hubiera ocurrido — corregido antes del PR. El reenrutamiento hacia `_HTTP_FACETS` quedó documentado como invariante diferido a R3 (ver A7, Apéndice C-bis #2). PR mergeado a `master` del repo `jax`.

**Sub-proyecto 1 — Núcleo del validador (`policy/governance/`).** Cuatro módulos puros o casi puros: `claims.py` (schema), `loaders.py` (I/O de config estática, fail-closed por hash), `validator.py` (I/O de estado vivo + resolvers), `vocab_sweep.py` (barrido léxico), `renderer.py` (plantillas hasheadas). Ocho estados de `Verdict`, ninguno es "warn" — consistente con P7. Solo dos resolvers reales de ocho predicados (ver §3.1.3). 34/34 tests. Revisión final de rama encontró y corrigió, en la misma ronda: un path traversal crítico en la allowlist de `FILE_EXISTS` (inerte por una entrada de allowlist rota, ambos corregidos juntos); fuga de oráculo de hash en `FACT_MISMATCH`; matching de substring ingenuo en el barrido (falsos positivos/negativos); y el caso 4 del patrón fail-open (Apéndice C-bis). PR mergeado.

**Sub-proyecto 2 — Contrato de facetas, shadow validation, integración real.** Wrapper de contrato `{claim, analysis, judgment}` en 6 de las 7 facetas de la Mesa web (`hyde` nunca llega al wrapper — devuelve una respuesta enlatada antes de invocar el LLM, por construcción del producto, no por decisión de alcance). Tres tablas nuevas en `jax_memory` para el shadow (`shadow_messages` como denominador, `shadow_claim_verdicts`, `shadow_vocab_hits`). `BackgroundTask` asincrónica: la fila se inserta al encolar, se completa al terminar, y una excepción a mitad de camino deja el timestamp de validación en `NULL` de forma visible. Footnote sobrio en el frontend cuando el contrato no parsea.

Hallazgo mayor durante la implementación, autorreportado antes de que nadie más lo notara: el plan se había escrito leyendo `chat.py` desde una rama de infraestructura 28 commits por delante de la base real (`infra/facetas-bloque-d` vs `master`), sin verificar cuál rama estaba activa al leer el código — instancia número siete del sesgo de A1 en esta misma sesión. Resuelto con una señal posicional propia (`is_canned`) en vez de la infraestructura que ese plan asumía, documentada como reconciliación futura sin fecha (§6-bis).

Revisión final de rama (modelo más capaz, dedicado) encontró cuatro hallazgos Important, los cuatro sobre el propio mecanismo de fail-closed del shadow: un `facet` sin validar podía romper el insert de la fila que existe para hacer visible una pérdida; un `predicate` sin límite de largo perdía verdicts de forma sesgada hacia los modelos peor comportados; la especificación de rollback no excluía un fix de conectividad de base de datos que si se revertía tumbaba la conexión real del backend; y el import diferido de la validación de shadow podía convertir un turno de chat ya exitoso en un error 500 para el usuario. Los cuatro corregidos y re-revisados. 103/103 tests de backend, 33/33 de frontend. PR abierto contra `master` de `jax-platform`.

### C1 — El corpus normativo

`jax/policy/` — 25 reglas en YAML versionado, con hash registrado en `VERSION`, validador de esquema, y `CORPUS.md` generado (nunca editado a mano). Línea base al cierre de esta versión: **0 NORMATIVA / 9 NORMATIVA_PENDIENTE / 8 CULTURAL / 4 HISTORICA**.

El cero en NORMATIVA no es un defecto del corpus — es el resultado honesto que P1 ("una regla es una regla solo si existe código que puede rechazarla") existía para producir: ninguna regla tenía enforcement todavía cuando se completó el inventario. La tensión más significativa quedó documentada explícitamente, no resuelta a la fuerza: P04 ("la capacidad se otorga por contrato de tarea, no por identidad de facet") contradice `allowed_callers` en `las_manos/config.toml`, que sí es identity-based — queda NORMATIVA_PENDIENTE porque es el primer hallazgo real del corpus, no un defecto a tapar.

La métrica "reglas con test" de §4 tiene ahora un denominador real: 25, no una promesa sin número.

---

*Fin del documento. Versión 3.1 — especificación con historial de ejecución verificado. Los desacuerdos nuevos de §6-bis quedan abiertos.*
