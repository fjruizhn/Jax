# Rediseño de Jacobs — Orquestador Autónomo
## Plan de fases. Examen de admisión: generar el contrato Las Manos modular, solo.

> Estado verificado (no supuesto):
> - Jacobs YA orquesta: Pipeline/Step/executor/estado/dry-run/API (puerto 7777, dentro de las_manos).
> - Jacobs YA descompone objetivo→steps (PlanBuilder.build) — pero con qwen3:14b y prompt plano.
> - Jacobs YA encadena contexto entre steps (_build_context_input/_enrich_prompt) — pero trunca a 500 chars.
> - Jacobs YA persiste todo en MariaDB (jacobs_pipelines/steps/events).
> - Axioma (web) YA consume Jacobs (backend :8080 → :7777, panel "Director Jacobs", StepCard, WebSocket).
> - El core (jax CLI) y la web comparten el MISMO PlanBuilder → mejorarlo beneficia a ambos sin tocar la web.
>
> Conclusión: NO se reconstruye Jacobs. Se le cambia el cerebro y se le enseña a pensar en módulos.
> Meta de fondo (Fernando): reducir progresivamente la dependencia de modelos de terceros —
> que el modelo LOCAL aprenda estas descomposiciones vía ejemplos de oro capturados de Ada.

---

## FASE A — Cerebro: descomponer con Ada + enrutamiento por dificultad
**Archivo:** `~/jax/jacobs/plan.py` (PlanBuilder)
**Problema:** descompone con qwen3:14b (débil para trabajo formal).
**Cambio:**
- PlanBuilder enruta por dificultad del objetivo:
  - trivial/repetitivo (pocas fases, patrón conocido) → jax_local (qwen) — gratis, soberano.
  - formal/complejo (módulos, dependencias, tipos) → Ada (glm-5.2, vía el worker ya arreglado: 128K + streaming).
- El enrutador es explícito y auditable (no una heurística mágica): un clasificador simple
  (heurística por longitud/keywords + opcionalmente una llamada corta a jax_local que diga
  "trivial|formal"). Declarar incertidumbre: la clasificación es PROPUESTA, ajustable.
- Ada descompone con el patrón modular (Fase B define el prompt).
**Resultado:** el trabajo formal se planifica con el mejor cerebro; lo trivial sigue local.

## FASE B — Prompt modular: enseñarle el patrón LLM-as-Compiler
**Archivo:** `~/jax/jacobs/plan.py` (_PLAN_SYSTEM + el prompt de _llm_plan)
**Problema:** el prompt pide "lista de N pasos" — planes planos, sin dependencias ni tipos comunes.
**Cambio:** reescribir el system prompt de planificación para que produzca planes que:
  - pongan **common_types PRIMERO** (definir tipos/enums una vez; los demás los referencian).
  - declaren **dependencias** explícitas por step (`depends_on: [step_ids]`).
  - ordenen por dependencia (capabilities → schemas → decision → invariants al final).
  - marquen un **step de validación de consistencia** y uno de **ensamble** al cierre.
  - sigan el principio de evidencia ya presente ("el que supone se equivoca").
**Nota de diseño:** esto requiere que el modelo de Step admita `depends_on`. Verificar si
  models.py ya lo soporta; si no, agregarlo (campo opcional, no rompe nada existente).
**Resultado:** Jacobs planifica trabajo formal como un compilador, no como una lista de tareas.

## FASE C — Contexto completo entre steps (para trabajo formal)
**Archivo:** `~/jax/jacobs/executor.py` (línea ~87, `summary = str(result_text)[:500]`)
**Problema:** cada step recibe solo 500 chars de los anteriores — insuficiente para que el
  step "decision_function" vea el texto COMPLETO de "common_types".
**Cambio:** que el truncado dependa de la dependencia declarada:
  - si el step actual `depends_on` un step previo → pasar su output COMPLETO (no 500 chars).
  - steps no relacionados → mantener resumen corto (eficiencia).
**Resultado:** los módulos realmente "importan" el texto de sus dependencias (corazón del patrón).

## FASE D — Captura de ejemplos de oro (el mecanismo de "aprender")
**Archivos:** `~/jax/jacobs/store.py` (+ tabla), `~/jax/jacobs/plan.py` (few-shot)
**Problema:** las buenas descomposiciones de Ada se ejecutan y se olvidan.
**Cambio:**
  - agregar campo `quality`/`is_golden` a `jacobs_pipelines` (o tabla `jacobs_golden_plans`).
  - cuando un pipeline complejo termina bien (y opcionalmente Fernando lo marca), su
    {objective → plan} se guarda como ejemplo de oro.
  - PlanBuilder, al planificar con jax_local, **inyecta los ejemplos de oro relevantes
    como few-shot** en el prompt. El local aprende por imitación, sin fine-tuning.
  - Métrica de soberanía: % de planes resueltos por local vs Ada, en el tiempo.
**Resultado:** el sistema migra solo de "Ada descompone" → "local descompone, Ada audita" →
  "local solo". Dependencia de terceros decreciente, medible. (Fine-tuning real: fase futura,
  con el dataset de oro acumulado.)

## FASE E — Verificar doble frente core + web (mínima, ya está casi todo)
**Cambio:** NO construir integración (ya existe). Solo verificar tras A-D que:
  - `jax --task` puede invocar Jacobs para un objetivo grande (CLI).
  - Axioma (panel "Director Jacobs") sigue reflejando bien los nuevos steps con dependencias
    (StepCard puede necesitar mostrar `depends_on` — ajuste menor de UI si se quiere).
  - el WebSocket sigue emitiendo el estado correcto.

---

## EXAMEN DE ADMISIÓN (tras A-E)
Dar a Jacobs UN objetivo: *"Generá el paquete modular del contrato Las Manos:
manifest + common_types + capabilities + envelope_keys + plan_authorization + provenance +
decision_function + invariants, en orden de dependencia, cada módulo coherente con los tipos comunes."*
- Jacobs descompone SOLO (con Ada), ejecuta los steps, cada módulo ve sus dependencias completas,
  valida consistencia, ensambla.
- Si el paquete sale coherente sin que Fernando pique nada → Jacobs está listo para RICH→Looking Glass.
- Si falla → sabremos en qué fase exacta (descomposición, contexto, validación o ensamble).

## ORDEN DE EJECUCIÓN
A → B → C → (probar examen parcial) → D → E → (examen completo).
Cada fase = misión Hyde acotada, con backup y gate. Verificar en disco, no al reporte.

## RIESGO LATENTE ANOTADO (no de este plan, pero pendiente)
Jacobs existe DUPLICADO byte a byte en ~/jax/jacobs y ~/jax/las_manos/jacobs.
Antes o durante el rediseño hay que decidir cuál es la fuente de verdad y consolidar,
o los cambios se aplicarán a una copia y la otra divergirá. Tratar en Fase A (definir
el path canónico antes de editar).
