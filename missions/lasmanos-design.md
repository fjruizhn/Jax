# Diseño — Modelo de Autoridad de Las Manos (blueprint)

> Estado: **diseño, no ejecución.** Este documento se implementa después, en su
> propia misión, con Ada (formaliza el contrato e invariantes) + Kimi (implementa)
> + Thot (audita). No correr como `--task` todavía.
>
> Origen: síntesis de la revisión de GPT sobre autoridad de operadores, cruzada
> con la arquitectura existente de LAS MANOS (Intent Envelope, human gate,
> forensic audit) y la constitución de Neverland.

---

## 1. Principio raíz

> **Las Manos autoriza *capabilities*, no *personalidades*.**

Hoy el riesgo es que la ejecución dependa de un solo operador (Hyde). El problema
no es "qué modelo es lo bastante listo para ejecutar", sino "qué operación puede
solicitar cada faceta, en qué ambiente, con qué permiso y con qué gate".

Una faceta no tiene autoridad por ser inteligente; la tiene por **capacidad de
control, trazabilidad, reversibilidad y contención.**

## 2. Reglas constitucionales (a firmar por las 7 + Jacobs)

1. **Ninguna faceta posee autoridad inherente sobre Las Manos.** Toda operación
   se autoriza por: capability + ambiente + riesgo + caller + política +
   auditabilidad + posibilidad de reversión.
2. **Ninguna intención conversacional activa una faceta ejecutora.** Toda
   ejecución real exige modo Comando, invocación explícita o autorización política.
3. **La ejecución privilegiada no se concede por inteligencia del modelo**, sino
   por control, trazabilidad, reversibilidad y contención.
4. **Nadie se autoaprueba.** Quien propone no es quien vetea; quien implementa no
   es quien audita.
5. Reformulación de la constitución: *muchos pueden pensar, varios proponer,
   algunos escribir, pocos ejecutar, nadie saltarse el contrato.*

## 3. Niveles de ambiente (de menor a mayor privilegio)

| Nivel | Ambiente | Qué se puede |
|---|---|---|
| 0 | Conversación | pensar, proponer (todas las facetas) |
| 1 | Lectura controlada | leer contexto, docs, logs filtrados, estado de jobs |
| 2 | Plan / dry-run | planes, simulaciones, validaciones formales |
| 3 | Sandbox | código, tests, archivos temporales, ramas experimentales |
| 4 | Staging | cambios reales reversibles, con auditoría |
| 5 | Producción | cerrado: gate humano + veto de Thot obligatorios |

## 4. Capabilities (forma del contrato)

Las Manos no chequea "¿es Hyde?", chequea "¿esta capability permite este caller en
este ambiente?". Forma propuesta (TOML/registry):

```toml
[capability."formal.spec"]
allowed_callers   = ["ada", "jacobs"]
environment       = ["plan"]
read_only         = false
writes_to         = ["specs", "plans", "test_cases"]
requires_gate     = false

[capability."implementation.sandbox"]
allowed_callers   = ["kimi", "ada", "jacobs"]
environment       = ["sandbox"]
requires_gate     = false
requires_audit    = true

[capability."implementation.staging"]
allowed_callers   = ["kimi", "hyde", "jacobs"]
environment       = ["staging"]
requires_gate     = "conditional"   # según riesgo
requires_audit    = true
requires_revert   = true

[capability."production.deploy"]
allowed_callers   = ["hyde", "jacobs"]
environment       = ["production"]
requires_gate     = true             # humano, siempre
requires_thot_review = true          # veto de Thot obligatorio
requires_audit    = true
requires_revert   = true

[capability."audit.review"]
allowed_callers   = ["thot", "jacobs"]
read_only         = true

[capability."research.web"]
allowed_callers   = ["hipatia", "jacobs"]
read_only         = true

[capability."docs.write"]
allowed_callers   = ["jekyll", "kimi", "jacobs"]
environment       = ["sandbox", "staging"]
```

## 5. Matriz de autoridad por faceta

| Faceta | Rol | Puede operar Las Manos en | NO puede |
|---|---|---|---|
| **Jacobs** | Director / despachador | crear jobs, dividir tareas, despachar capabilities | saltarse policy, shell arbitrario, producción sin gate |
| **Ada** | Formalización | `formal.spec`: contratos, pre/postcondiciones, invariantes, validar planes, test cases formales | tocar producción, infra, comandos destructivos |
| **Kimi** | Implementación | `implementation.sandbox` y `.staging`: código, refactor, tests, parches, diffs/PRs | producción directa, borrar datos, autoaprobarse |
| **Thot** | Auditor / veto | `audit.review` (read-only): leer planes/diffs/logs, dry-run, emitir GO/NO-GO, **bloquear** | implementar, ejecutar, autoaprobar su recomendación |
| **Hipatia** | Investigación | `research.web` (read-only): fuentes, versiones, grounding | modificar archivos, infra, decidir deploy |
| **Jekyll** | Humanidades | `docs.write`: documentación, manifiestos, tono | comandos técnicos, infra, código crítico |
| **JAX local** | Local ligero | tareas locales de bajo riesgo, clasificación, resumen, comandos *sugeridos* | producción, root, acciones peligrosas sin gate |
| **Hyde** | Ejecutor privilegiado | `implementation.staging`, `production.deploy` — **solo** con modo Comando + capability + policy + gate + audit | ser el **único** operador; ejecutar sin contrato |

Regla de oro: **Hyde es una mano fuerte, no Las Manos completas.**

## 6. Encadenamiento inter-agente (lo que GPT señaló y faltaba)

Hyde sale del auto-routing **de usuario**, pero **sigue siendo invocable como
destino de cadena** desde otros agentes, bajo capability + gate:

- Thot detecta una falla → encadena a Hyde para aplicar el parche (`staging`, con audit).
- Kimi genera código validado → encadena a Hyde para el deploy (`production`, con gate humano + veto Thot).
- Ada formaliza el contrato → Kimi implementa en sandbox → Thot audita → Hyde aplica.

El usuario nunca dispara Hyde por conversación; los agentes sí lo encadenan, con contrato.

## 7. Cómo extiende lo que YA tenés

No es construir desde cero. LAS MANOS ya tiene **Intent Envelope**, **human gate**
y **forensic audit** (vía Thot). Esto los generaliza:

- El **Intent Envelope** pasa a llevar `capability` + `caller` + `environment`.
- El **human gate** se vuelve condicional por nivel de ambiente (auto en sandbox,
  obligatorio en producción).
- El **forensic audit** registra `caller → capability → environment → resultado →
  reversible?` en cada operación.
- Se agrega un **policy engine** que resuelve la tabla de §4 antes de actuar.

## 8. Decisiones abiertas (para vos, antes de implementar)

1. **¿Kimi en `production.deploy`?** GPT lo deja fuera de producción directa (solo
   staging). Mi voto: igual — Kimi implementa, Hyde despliega. Confirmá.
2. **¿`requires_gate` condicional en staging** según qué señal de riesgo? Hay que
   definir el clasificador de riesgo (¿toca prod? ¿borra? ¿migra? → gate).
3. **¿Veto de Thot bloqueante o advisory** en producción? Propongo bloqueante.
4. **¿JAX local ejecuta algo** o queda solo en "comandos sugeridos" (sin tocar)?
5. Nombres y granularidad de las capabilities (esto lo formaliza **Ada**).

## 9. Plan de implementación (cuando lo arranquemos)

Pipeline natural, usando las facetas para su propio gobierno:

1. **Ada** — formaliza el contrato: el esquema de capabilities, invariantes
   (ej. "ninguna `production.*` sin gate"), pre/postcondiciones del policy engine.
2. **Kimi** — implementa el policy engine + el registry de capabilities + el
   Intent Envelope extendido, en sandbox.
3. **Thot** — audita: busca formas de saltarse el contrato, threat model.
4. **Hyde** — aplica a staging, luego a producción con gate.

Misión Hyde a redactar cuando se apruebe este blueprint y se cierren las §8.
