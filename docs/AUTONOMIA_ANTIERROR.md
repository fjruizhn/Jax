# Sistema de Autonomía Anti-Error — Jacobs / Director de Orquesta

> **Estado:** Diseño aprobado por Fernando. Pendiente de implementación por fases.
> **Rama base:** `feat/director-orquesta-waves` (commit `01eacc4`)
> **Objetivo de negocio (palabras de Fernando):** "Que el sistema trabaje por mí, no conmigo. Que se auto-audite en cada etapa, que no dependa de mí y un *yes*. Que las cosas verdaderamente importantes sí me notifiquen; lo demás, para eso están los backups y los sandbox."
> **Caso de uso de aceptación:** Fernando codea con el Mundial de fondo. El sistema aguanta los 90 minutos del partido sin necesitar su intervención, y solo le toca el hombro (Telegram) si algo es rojo de verdad.

---

## 0. Principio rector

La palabra que define este sistema es **anti-error**, no "sin frenos". Autonomía no significa quitar las barreras de seguridad — significa **cambiar quién las ejerce**: de Fernando (un *yes* manual) a barreras automáticas (sandbox, validación de schema, auditoría clean-room, snapshot+rollback).

Regla firmada por los cinco, aplicada a arquitectura:
> **"El que supone se equivoca."** Ninguna etapa aplica su output sin que otra etapa lo verifique. Ningún cambio irreversible ocurre sin red de seguridad. Toda incertidumbre que el sistema no pueda resolver solo, se escala.

El sistema reemplaza el *yes* de Fernando por **tres verificadores automáticos** (schema, clean-room, post-check) y **una red de seguridad** (snapshot+rollback). Fernando solo entra al loop cuando esos cuatro no alcanzan.

---

## 1. Diagnóstico de raíz (por qué existe este trabajo)

El bug que detonó este diseño NO es un payload mal escrito. Es una **asimetría arquitectónica de validación de contrato**, expuesta en `CAPABILITIES_CONTRACT.md`:

1. El planner (Jacobs) emite `capability` como **texto libre** del LLM — sin restricción a ningún catálogo. Solo `facet` está restringido a `VALID_FACETS`.
2. **Solo el facet `kimi`** valida esa capability contra un catálogo cerrado (`_invoke_motor` → `policy.check`). Los otros cinco facets (ada, thot, hipatia, jekyll, jax_local) la ignoran por completo.
3. El planner está **sembrado en sus propios prompts** para emitir capabilities que no existen en el catálogo ni en el mapa de traducción: `design`, `validate_consistency`, `reconcile`, `critique`, `generate`, `reason`.

**Consecuencia latente (bomba de tiempo):** cualquier plan autónomo que asigne a `kimi` una capability fuera del catálogo → `policy.check` rechaza → el step falla → si `skip_on_fail=false`, la ola entera registra `failed` → `PIPELINE_ABORTED`. Hoy salió con `generate` en un test. Mañana saldría con `reconcile` o `validate_consistency` en un examen formal real, que son capabilities **centrales** de la ruta `_PLAN_SYSTEM_MODULAR` (penúltimo y antepenúltimo step).

**Decisión de diseño de Fernando:** No desperdiciar a kimi. Es el motor con mejor relación $/resultado. La corrección debe **ampliar** lo que kimi puede recibir, no restringirlo — y hacer que Jacobs autónomo pueda explotarlo a fondo, con barreras automáticas en lugar de gate humano.

---

## 2. Arquitectura: las cuatro capas

```
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 1 — Contrato de capabilities cerrado y total               │
│  (sin esto nada corre: los planes explotan)                      │
├─────────────────────────────────────────────────────────────────┤
│  CAPA 2 — Auto-auditoría bloqueante por etapa                    │
│  (reemplaza el "yes" de Fernando con verificadores automáticos)  │
├─────────────────────────────────────────────────────────────────┤
│  CAPA 3 — Snapshot + rollback automático                         │
│  (hace que "no dependa de tu yes" sea SEGURO: errores reversibles)│
├─────────────────────────────────────────────────────────────────┤
│  CAPA 4 — Gate de severidad + notificación Telegram              │
│  (escalamiento por excepción: solo lo ROJO te molesta)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. CAPA 1 — Contrato de capabilities cerrado y total

**Meta:** que ninguna capability emitida por el planner pueda ser aceptada por una ruta y rechazada por otra. Cerrar el contrato en la fuente (planner), completar el mapa, y validar uniforme antes del dispatch.

### 3.1 — Catálogo ampliado (`las_manos/config.toml`)

Añadir entradas de catálogo para las capabilities de razonamiento que hoy pasan crudas. Decisión de Fernando: kimi debe poder ejecutarlas (aprovecharlo al máximo), pero **respetando la naturaleza de cada motor**. Las nuevas entradas con `allowed_callers` incluyendo `jacobs`:

| capability nueva | allowed_motors | allowed_callers | requires_human_gate | output_schema | severidad |
|---|---|---|---|---|---|
| `generate` | kimi, ada | jacobs, hyde, ada | false | `generate.v1` | 🟢 verde |
| `reason` | ada, kimi | jacobs, hyde, ada, thot | false | `reason.v1` | 🟢 verde |
| `design` | ada, kimi | jacobs, hyde, ada | false | `design.v1` | 🟢 verde |
| `validate_consistency` | thot, ada | jacobs, hyde, thot | false | `validation.v1` | 🟢 verde |
| `reconcile` | ada, kimi | jacobs, hyde, ada | false | `reconcile.v1` | 🟢 verde |
| `critique` | thot, ada | jacobs, hyde, thot | false | `critique.v1` | 🟢 verde |

> **Nota de naturaleza:** `validate_consistency` y `critique` listan `thot/ada` primero porque la auditoría independiente (clean-room) las prefiere en facets de razonamiento. kimi sigue disponible como motor, pero el planner las dirige a auditores naturales.

Las 6 capabilities de código existentes (`code_swarm`, `refactor`, `bug_hunt`, `pipeline_analysis`, `implementation`, `architecture_review`) se mantienen, con un ajuste de `allowed_callers` para que `jacobs` pueda invocar las que hoy le están vedadas:

| capability existente | cambio en allowed_callers | severidad |
|---|---|---|
| `refactor` | (ya incluye jacobs) | 🟢 verde |
| `pipeline_analysis` | (ya incluye jacobs) | 🟢 verde |
| `implementation` | (ya incluye jacobs) | 🟢 verde |
| `code_swarm` | **añadir jacobs** | 🟡 amarillo (mantiene gate→ ver Capa 4) |
| `bug_hunt` | **añadir jacobs** | 🟡 amarillo (mantiene gate→ ver Capa 4) |
| `architecture_review` | **añadir jacobs** | 🟢 verde |

### 3.2 — Mapa total (`jacobs/executor.py`, `_CAPABILITY_MAP`)

El mapa deja de ser parcial. Toda capability semántica que el planner emite tiene destino conocido. Lo que ya no esté en el mapa, en lugar de pasar crudo con un warning, se valida contra el catálogo y se degrada de forma segura si no existe (ver 3.4). El objetivo: **cero pass-through silencioso**.

### 3.3 — Restricción del planner (`jacobs/plan.py`, `_parse_plan_json`)

Hoy: `capability = str(item.get("capability","reason"))[:50]` — libre.

Cambio: introducir `VALID_CAPABILITIES` (equivalente a `VALID_FACETS`). Si el planner emite una capability fuera del conjunto conocido, se mapea a la más cercana o se degrada a `reason` (capability genérica segura), **nunca se deja pasar cruda**. Esto convierte "el planner puede inventar cualquier cosa" en "el planner solo usa vocabulario que el sistema garantiza poder ejecutar".

### 3.4 — Validación uniforme pre-dispatch (`jacobs/executor.py`, `_dispatch_step`)

Cerrar la asimetría: la validación de capability ocurre en un punto común **antes** de rutear por facet, aplicando a TODOS los facets, no solo kimi. Una función `validate_capability(step)` que verifica:
- La capability existe en el catálogo o en el mapa.
- El facet asignado es compatible con la capability (`allowed_motors`).
- El caller (`jacobs`) está en `allowed_callers`.

Si falla → el step se marca con error claro **antes** de ejecutar, no a mitad de la ruta del motor. Defensa en profundidad: aunque el planner (3.3) deje pasar algo, aquí se atrapa.

---

## 4. CAPA 2 — Auto-auditoría bloqueante por etapa

**Meta:** reemplazar el *yes* humano con verificadores automáticos que corren en cada etapa.

### 4.1 — Clean-room de warning a gate real (`jacobs/plan.py` + `jacobs/executor.py`)

Hoy `_check_cleanroom` solo emite `logger.warning`. Cambio: la violación clean-room **bloquea** el ensamble. Ningún output de un step se considera "aprobado" hasta que un facet **distinto al productor** lo auditó. Regla: *quien produce no aprueba.*

- El planner ya está sembrado para generar un step auditor con facet distinto.
- Si un plan viola clean-room (auditor = productor), el sistema lo detecta en construcción y lo corrige (reasigna auditor) o lo escala como 🔴 si no puede.

### 4.2 — Validación de output contra schema (`motor_registry/worker.py:220`)

El `output_schema` por capability deja de ser un "landmine" y se vuelve el verificador automático principal. Cada output de step se valida contra su schema declarado (`generate.v1`, `reconcile.v1`, etc.) **antes** de persistirse como completado y antes de alimentar a los steps dependientes.

- Output que no cumple schema → step falla limpio (no propaga basura a la siguiente ola).
- Esto es lo que reemplaza el ojo humano de Fernando revisando cada output.

> **Acción de implementación:** definir los schemas nuevos (`generate.v1`, `reason.v1`, `design.v1`, `validation.v1`, `reconcile.v1`, `critique.v1`). Empezar laxos (estructura mínima) y endurecer con evidencia de uso real — no bloquear de más al inicio.

### 4.3 — Reconciliación obligatoria antes de ensamble

La etapa `reconcile` (ada) que ya existe en la ruta formal se vuelve obligatoria como penúltimo paso: recibe el contexto completo de N dependencias, verifica coherencia entre módulos, y produce patches. Es la última auditoría automática antes del ensamble mecánico (`_assemble_mechanical`).

---

## 5. CAPA 3 — Snapshot + rollback automático

**Meta:** que aplicar un cambio a producción sin el *yes* de Fernando sea SEGURO, porque cualquier error es reversible automáticamente.

### 5.1 — Snapshot previo (Restic / Sésamo)

Antes de cualquier operación clasificada 🟡 amarillo (aplica a producción), el sistema toma un snapshot Restic automático del estado relevante. Reusa la infraestructura existente: repo Restic en Sésamo, rclone a R2.

### 5.2 — Post-check tras aplicar

Después de aplicar un cambio amarillo, el sistema corre una verificación automática:
- ¿El servicio afectado sigue vivo? (`systemctl is-active`)
- ¿El output cumplió su schema? (Capa 2)
- ¿Los tests relevantes pasan? (si aplica)

### 5.3 — Rollback automático

Si el post-check (5.2) falla → el sistema **restaura solo** del snapshot tomado en 5.1, registra el evento, y NO requiere a Fernando para curarse. Si el rollback **también** falla → escala a 🔴 (Capa 4): ahí sí, Fernando debe saberlo.

> Esta capa es la que convierte "el sistema trabaja sin tu yes" en algo seguro en vez de una ruleta. Un error reversible se deshace solo; solo lo irreversible-y-no-recuperable molesta a Fernando.

---

## 6. CAPA 4 — Gate de severidad + notificación Telegram

**Meta:** escalamiento por excepción. El sistema clasifica cada operación y solo interrumpe + notifica cuando algo es 🔴 rojo.

### 6.1 — Clasificador de severidad

Cada step, antes de ejecutar, se clasifica:

#### 🟢 VERDE — corre y aplica libre, cero fricción
- `refactor`, `pipeline_analysis`, `implementation`, `generate`, `reason`, `design`, `architecture_review`, análisis
- Todo lo reversible y verificable por schema
- **Comportamiento:** ejecuta → auto-audita (Capa 2) → si pasa, aplica y continúa. Sin notificación.

#### 🟡 AMARILLO — sandbox + snapshot + rollback, sin molestar a Fernando
- `code_swarm`, `bug_hunt` que pasaron sandbox + schema + clean-room
- Aplicación a producción reversible
- **Comportamiento:** corre en sandbox → auto-audita → snapshot previo → aplica → post-check → si falla, rollback automático. NO notifica (salvo resumen opcional post-partido). Solo escala a 🔴 si el rollback falla.

#### 🔴 ROJO — interrumpe el pipeline + Telegram (Fernando mira en el medio tiempo)
- Toca `forbidden_paths` (`.env`, `secrets/`, `credentials/`, `private_keys/`)
- Afecta infra de uptime familiar: VM de producción (.11), mail, DNS, vector DB de JAX
- La auto-auditoría falló Y el rollback automático también falló
- Un swarm excede límites duros (tiempo / recursión / recursos)
- **Comportamiento:** el pipeline se detiene en `PipelineStatus.interrupted`, registra el motivo, y manda Telegram. El trabajo verde/amarillo previo ya aplicado NO se revierte; solo se detiene el avance. Fernando revisa cuando quiere (medio tiempo, fin del partido, o desde el celular).

> **Clave de la visión:** ni siquiera 🔴 obliga a Fernando a estar presente en tiempo real. El pipeline *espera* interrumpido. Lo verde y amarillo avanzó solo. Solo lo potencialmente destructivo lo espera, en vez de explotar sin avisar.

### 6.2 — Canal de notificación

Telegram, reusando el bot ya configurado para alertas de backup (probado en vivo, ver historial de infraestructura). Mensaje 🔴 incluye: pipeline_id, step, motivo de la clasificación roja, estado (interrumpido), y link/comando para revisar.

### 6.3 — Gate de Jacobs para capabilities con gate (code_swarm / bug_hunt)

Estas mantienen `requires_human_gate=true` en el catálogo PERO el flujo cambia: en lugar de que jacobs nunca pueda invocarlas (porque no envía token), Jacobs las clasifica 🟡 amarillo y las corre bajo el régimen sandbox+snapshot+rollback de la Capa 3+5. El "gate" se satisface automáticamente por las barreras, no por un token humano — salvo que crucen un criterio 🔴 (forbidden_paths, infra familiar), en cuyo caso sí escalan a Fernando.

---

## 7. Orden de implementación por fases

> Cada fase se verifica con evidencia antes de pasar a la siguiente. Protocolo Hyde: ningún paso se cierra sin output conocido.

**FASE A — Capa 1 (contrato cerrado).** Sin esto nada más funciona.
1. Ampliar `config.toml` con las 6 capabilities nuevas + ajustar `allowed_callers`.
2. Completar `_CAPABILITY_MAP`.
3. `VALID_CAPABILITIES` + restricción en `_parse_plan_json`.
4. `validate_capability` pre-dispatch.
5. **Verificación:** re-correr el test de integración (Fase 2 que quedó pendiente) — ahora con catálogo coherente. El fan-out ada+kimi+thot debe completar sin rechazo de policy.

**FASE B — Capa 2 (auto-auditoría).**
1. Definir los 6 schemas nuevos (laxos al inicio).
2. Clean-room de warning a gate bloqueante.
3. Conectar validación de schema al flujo de completado de step.
4. **Verificación:** un pipeline con output deliberadamente malo debe fallar limpio en la etapa de validación, sin propagar a la ola siguiente.

**FASE C — Capa 3 (snapshot + rollback).**
1. Hook de snapshot Restic pre-aplicación amarilla.
2. Post-check automático.
3. Rollback automático + escalamiento a rojo si rollback falla.
4. **Verificación:** simular un cambio amarillo que falla post-check; confirmar que el rollback restaura solo y registra.

**FASE D — Capa 4 (severidad + Telegram).**
1. Clasificador de severidad.
2. Integración Telegram (reusar bot de backups).
3. Gate de Jacobs para code_swarm/bug_hunt bajo régimen amarillo.
4. **Verificación:** un step que toca forbidden_paths debe interrumpir + notificar; un step verde no debe notificar nada.

---

## 8. Criterio de aceptación (la prueba del Mundial)

El sistema se considera listo para "ver el partido sin mirar el monitor" cuando, observado por Fernando con evidencia (no asumido):

1. Un pipeline autónomo con mezcla de steps verdes/amarillos corre completo sin un solo *yes*.
2. Un fallo reversible inducido se auto-revierte vía rollback, sin notificación.
3. Una condición roja inducida (tocar forbidden_paths) interrumpe el pipeline y llega el Telegram — y NO llega Telegram por nada verde/amarillo.
4. Fernando observa los tres comportamientos al menos un par de ciclos con sus propios ojos.

> **Nota de método (firmada por los cinco):** la confianza en la autonomía se gana con evidencia, igual que todo lo demás. La primera vez que esto corra de verdad, el partido es la prueba *con Fernando cerca*. Una vez que el sistema se mostró auto-curándose y escalando correctamente varias veces, la confianza es ganada, no supuesta. "El que supone se equivoca" aplica también a confiar en el propio sistema.

---

## 9. Lo que este sistema NO hace (límites explícitos)

- **No quita las barreras automáticas.** Autonomía = barreras automáticas en lugar de gate humano, no ausencia de barreras.
- **No aplica cambios irreversibles sin red.** Todo lo amarillo tiene snapshot+rollback. Lo que no se puede revertir ni verificar es rojo por definición.
- **No oculta fallos.** Todo se registra en `jacobs_events`. El silencio de notificación significa "se resolvió solo", no "no pasó nada".
- **No reemplaza el criterio de Fernando en lo verdaderamente importante.** Lo rojo escala. Fernando sigue siendo el dueño de las decisiones irreversibles sobre infra crítica.

---

*Documento de diseño — ecosistema JAX / Axioma. Inversiones Diamante Negro / RICH S. de R.L.*
*Principios fundacionales: I) No suponer nunca. II) Saber no cuesta nada. III) Mañana es el día que el fracasado tiene más que hacer. — Marina.*
