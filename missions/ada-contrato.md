ada: Esto es tu dominio: formalización. Trabajás sobre el blueprint v2 de Las Manos (incluido íntegro más abajo, bajo el separador). Tu tarea es producir el CONTRATO FORMAL del que dependerá toda la implementación de Fase 1 y 2. NO implementás código: formalizás el contrato que el policy engine determinista hará cumplir.

DECISIONES YA CERRADAS (no las reabras — formalízalas tal cual):
- Kimi no ejecuta producción. Propone deploy vía artefacto firmado (hash); Hyde ejecuta solo un plan_hash aprobado, sin modificar el plan en vuelo.
- Veto de Thot: bloqueante para facetas. Override solo-operador = tipeo explícito + razón escrita + plan_hash + incidente de seguridad + rollback obligatorio. Dejá el hook de dual-control definido pero apagado (flag).
- jax_local: solo sugiere comandos, no ejecuta nada.
- 10 capabilities: formal.spec, implementation.sandbox, implementation.staging, production.deploy, production.rollback, audit.review, research.web, docs.write, control_plane.change (solo jacobs), capability.grant (solo humano).
- resource_scope: granularidad por servicio + path (ej. "~/jax-platform/backend/"), ni por archivo individual ni "todo el server".
- Alcance: formalizás el contrato completo de Fase 1+2 de una.

PRODUCÍ ESTOS 5 ARTEFACTOS FORMALES, EN ORDEN:

1. ESQUEMA DE CAPABILITIES (las 10). Para cada una: definición formal con TODOS sus parámetros tipados — allowed_callers, environment[], resource_scope, allowed_paths, denied_paths, secret_access, network_egress, data_classification, max_runtime, blast_radius, requires_gate, requires_audit, requires_rollback, approval_class. Formato TOML con tipos explícitos por campo.

2. INVARIANTES. La lista completa expresada como predicados verificables, para-todo-cuantificados, sin ambiguedad, cada uno chequeable por codigo determinista. Marca cada invariante como HARD (bloquea) o SOFT (advierte).

3. FUNCION DE DECISION DEL POLICY ENGINE. Defini decide(envelope, plan) -> {ALLOW, DENY, GATE}: precondiciones, orden exacto de chequeos, postcondiciones, y que ocurre en cada rama. Regla constitucional 6: vos formalizas las reglas, el motor determinista las ejecuta — ningun LLM decide en runtime.

4. ESQUEMA DEL INTENT ENVELOPE FIRMADO. Estructura completa con tipos. Distingui que campos propone el LLM (solo desired_action) de los que fija el policy engine (environment, risk_class, requires_gate). Especifica como se verifica la firma del caller y el caller_chain.

5. AUTORIZACION DE PLAN. Estructura del plan; como se computa plan_hash; como se computa aggregate_risk = max(step_risk) + composition_risk; como se atan la aprobacion humana y el review de Thot al plan_hash exacto; y las deny-rules de composicion como tabla formal.

ESTANDAR DE RIGOR (tu principio): el que supone se equivoca. Si un campo o regla queda ambiguo en el blueprint, NO lo rellenes con un supuesto — declaralo como "DECISION PENDIENTE" con las opciones. No inventes defaults de seguridad sin marcarlos explicitamente.

Entrega los 5 artefactos en formato que Kimi pueda implementar directo (esquemas tipados, no prosa). Cerra con la lista de DECISIONES PENDIENTES que detectaste.

========================== BLUEPRINT v2 (material a formalizar) ==========================

# Diseño v2 — Modelo de Autoridad de Las Manos (blueprint)

> Estado: **diseño, no ejecución.** v2 incorpora la revisión cruzada de GPT +
> DeepSeek. El veredicto de ambos: el modelo v1 autorizaba *capabilities* pero no
> *planes* — un hueco crítico (capability laundering / confused deputy). v2 lo cierra.
>
> Frase-contrato (a firmar):
> **"Las Manos no ejecuta intenciones. Ejecuta planes autorizados, acotados,
> auditables, reversibles y ligados criptográficamente a una aprobación concreta."**

---

## 1. Principio raíz

**Las Manos autoriza CAPABILITIES dentro de un PLAN, no PERSONALIDADES.**
Una faceta no tiene autoridad por ser inteligente, sino por control,
trazabilidad, reversibilidad y contención.

## 2. Reglas constitucionales

1. Ninguna faceta posee autoridad inherente sobre Las Manos. Toda operación se
   autoriza por: capability + ambiente + caller_chain + plan + riesgo + política
   + auditoría + reversibilidad.
2. Ninguna intención conversacional activa una faceta ejecutora.
3. La ejecución privilegiada no se concede por inteligencia, sino por control.
4. **Nadie se autoaprueba.** `implementer(J) ≠ auditor(J)`; `proposer(J)` no es el
   único `approver(J)`.
5. **Una faceta no puede obtener indirectamente (vía Jacobs) una capability que no
   podría pedir directamente**, salvo política de delegación explícita.
6. **El LLM recomienda; el policy engine determinista decide.** Los campos
   estructurales del envelope (risk, environment, requires_gate) no los setea
   ningún LLM.
7. Ninguna capability de ejecución puede modificar su propio audit trail.
8. Todo contenido externo es `untrusted`: entra como dato, nunca como instrucción.
9. *Muchos pueden pensar, varios proponer, algunos escribir, pocos ejecutar,
   nadie saltarse el contrato.*

## 3. Niveles de ambiente

| Nivel | Ambiente | Notas |
|---|---|---|
| 0 | Conversación | todas las facetas |
| 1 | Lectura controlada | docs, logs filtrados, estado de jobs |
| 2 | Plan / dry-run | planes, simulaciones, validación formal |
| 3 | Sandbox **real** | efímero, sin red, sin secrets, sin docker.sock, sin sudo, con timeout/quota |
| 4 | Staging | reversible + audit; gate condicional (§7) |
| 5 | Producción | gate humano + veto Thot obligatorios |

**`environment_isolation_check` obligatorio:** "staging" solo cuenta como staging
si NO comparte DB / secrets / red / runners / buckets con producción. Si los
comparte → se le aplican reglas de producción.

## 4. Capabilities (10) — parametrizadas, no permisos planos

| Capability | Ambientes | Callers | Reglas |
|---|---|---|---|
| `formal.spec` | 0-4 | ada, jacobs | escribe specs/planes/invariantes |
| `implementation.sandbox` | 0-3 | kimi, ada, jacobs | sandbox real |
| `implementation.staging` | 0-4 | kimi, hyde, jacobs | reversible, audit, gate condicional |
| `production.deploy` | 5 | hyde, jacobs | gate humano + veto Thot + plan_hash |
| `production.rollback` | 5 | hyde, jacobs | **separada** de deploy |
| `audit.review` | 0-5 | thot, jacobs | **read-only siempre**, secrets redactados |
| `research.web` | 0-5 | hipatia, jacobs | read-only, rate-limited, untrusted output |
| `docs.write` | 0-4 | jekyll, kimi, jacobs | sandbox/staging |
| `control_plane.change` | 5 | **solo jacobs** | + gate + veto + Ada invariant check |
| `capability.grant` | 5 | **solo humano** | NUNCA un agente |

**Parámetros por capability** (no flat): `resource_scope`, `allowed_paths`,
`denied_paths`, `network_egress`, `secret_access`, `data_classification`,
`max_runtime`, `max_diff_size`, `blast_radius`, `requires_tests`,
`requires_rollback`, `approval_class`.

**`control_plane.change`** cubre cambios a: policy engine, capability registry,
router, human gate, audit logger, Jacobs, Las Manos, motor registry, prompts de
sistema. NUNCA bajo `implementation.staging`.

## 5. Capa de Autorización de Plan (el fix central de v2)

No basta autorizar pasos sueltos. Antes de ejecutar una cadena:

1. **Jacobs declara el plan completo:** `goal`, `steps[]`, `capabilities_needed[]`,
   `target_environment`, `resources_touched[]`, `data_classes[]`,
   `expected_artifacts[]`, `risk_class`, `rollback_plan`, `test_plan`,
   `approval_requirements`.
2. **Riesgo acumulado:** `aggregate_risk = max(step_risk) + composition_risk`.
   Si la cadena incorpora un paso de mayor riesgo que el plan aprobado → suspende
   hasta reaprobación. Si cualquier paso toca recurso `production_shared` → reglas
   de producción.
3. **Congelar:** `plan_hash = hash(plan)`. Aprobación humana y review de Thot se
   atan a ese hash exacto (no "la idea").
4. **Tokens efímeros por paso:** `{capability, resource_scope, environment,
   caller_chain, plan_hash, step_id, expires_at, max_runtime, allowed_actions}`.
   Nada de permisos permanentes por faceta.
5. **Runtime monitor:** durante la ejecución se observan comandos/archivos/red/
   procesos reales. Si el job se sale del plan → kill + freeze + audit + reaprobar.

### Deny rules de composición (ejemplos)
```
research.web + secrets.read                 = deny
docs.write + control_plane.change           = gate
implementation.staging + ci_cd.change       = gate
staging + production_shared_resource        = reglas de producción
audit.review + log.delete                   = deny
sandbox + docker.sock                       = riesgo nivel prod
```

## 6. Intent Envelope firmado
```json
{
  "intent": "...",
  "desired_action": "refactor router",     // lo único que propone el LLM
  "caller": "kimi",
  "caller_signature": "0x...",             // firma con clave del caller
  "caller_chain": ["user@x", "jacobs", "kimi"],
  "original_user": "fernando",
  "trace_id": "uuid",
  "timestamp": "ISO",
  "capability_requested": "implementation.staging",
  "environment": 4,                        // lo decide el policy engine, NO el LLM
  "plan_hash": "...",
  "constraints": { "...": "..." }
}
```
Campos estructurales (environment, risk, requires_gate) los fija el policy engine
determinista. El LLM solo propone `desired_action`.

## 7. Gate humano condicional en staging — disparadores

Gate obligatorio (aunque sea staging) si la operación:
- toca producción / recurso compartido con prod
- borra o trunca datos
- migra esquema de BD
- modifica red / firewall / DNS / certs / TLS
- toca secrets / .env / vault / SSH / OAuth / API keys
- es irreversible o sin rollback declarado y verificado
- **cambia el control plane** (auth, policy, capabilities, router, human gate, audit, Jacobs, Las Manos, motor registry, prompts)
- **toca CI/CD, deploy scripts, systemd, cron, runners, workers, schedulers**
- instala o actualiza dependencias externas
- requiere privilegios elevados (root, sudo, docker.sock, package manager)
- introduce ejecución dinámica (eval, shell, plugins, webhooks, loaders)
- afecta backups / restore / retención / cifrado
- afecta observabilidad (logs, audit, métricas, alertas)
- afecta aislamiento multi-tenant
- tiene egress externo no declarado
- es la primera ejecución de un nuevo tipo de capability

Si nada aplica **y** staging está realmente aislado → corre sin gate (auto + audit).

## 8. Matriz de autoridad por faceta (refinada)

| Faceta | Rol | Puede | NO puede |
|---|---|---|---|
| Jacobs | Director | despachar capabilities, construir plan | saltarse policy; delegar capability sin verificar caller_chain |
| Ada | Formalización | `formal.spec`, validar planes, invariantes, test cases | tocar prod/infra; validar su propia spec para deploy |
| Kimi | Implementación | `implementation.sandbox`/`.staging`; propone deploy vía **artefacto firmado** | `production.deploy`; auditar su propio código |
| Thot | Auditor/veto | `audit.review` (read-only), GO/NO-GO, **bloquear** | implementar; ejecutar; autoaprobarse |
| Hipatia | Investigación | `research.web` (read-only, rate-limited) | modificar archivos; recibir secrets como input |
| Jekyll | Humanidades | `docs.write` | comandos técnicos; infra |
| JAX local | Local ligero | **solo sugiere** comandos; clasifica/resume | ejecutar nada (ni bajo riesgo) |
| Hyde | Ejecutor | `implementation.staging`, `production.deploy/rollback` con token + gate | ser único operador; modificar el plan en vuelo; ampliar scope; regenerar comandos |

**Hyde en producción:** solo ejecuta un `plan_hash` aprobado. No reinterpreta, no
regenera, no amplía. Cambio en vuelo = nuevo plan + nueva aprobación.

## 9. Invariantes formales para Ada (a garantizar)

**Producción**
- ∀ J ∈ production.* : ∃ human_gate.approved ∧ ∃ thot_review(plan_hash)
- ∀ production.deploy : plan_hash_aprobado = plan_hash_ejecutado ∧ artifact_hash igual
- ningún production.* con risk_class = unknown
- ∀ production.* : reversible ∧ rollback verificado

**Separación de funciones**
- implementer(J) ≠ auditor(J); proposer(J) ≠ único approver(J)
- Ada no despliega su propia spec; Kimi no audita su propio código

**Override de veto**
- veto de Thot no se elimina, solo se overridea por humano
- ∀ override : reason no vacío ∧ actor humano ∧ plan_hash ∧ risk_acknowledgement ∧ rollback obligatorio ∧ security_incident

**Hyde / Kimi**
- Hyde no se activa por intención conversacional; solo con token válido; no amplía scope
- Kimi ∉ production.deploy; artefacto a prod requiere revisión ≠ Kimi

**Control plane / capability.grant**
- control_plane.change ⇒ gate + Thot + Ada invariant check
- capability.grant ⇒ caller = humano (máquina denegada)
- ninguna faceta modifica sus propios permisos efectivos

**Composición / auditoría / secrets**
- toda cadena tiene chain_id; riesgo efectivo ≥ max(riesgo de pasos)
- audit append-only; ningún caller modifica el audit de su propio job
- ningún secret aparece sin redacción en outputs; read-only ⇒ secrets redactados

## 10. Vectores de ataque cubiertos (resumen)

| Vector | Mitigación |
|---|---|
| Capability laundering (cadena que lava intención) | autorización de plan + riesgo acumulado |
| Confused deputy vía Jacobs | caller_chain + regla 5 |
| Spoofing del envelope | firma del caller |
| Inyección por research.web | contenido = untrusted-data, no instrucción |
| Exfiltración por read-only | data classification + redacción |
| TOCTOU (aprobado ≠ ejecutado) | plan_hash + diff/artifact hash |
| Bypass por fallback a jax_local | jax_local no sugiere capabilities restringidas |
| Auto-modificación del control plane | control_plane.change separado + reglas duras |
| Audit manipulable | append-only + hash-chain |
| Runaway agent / DoS / costo | rate limits + quotas + time-box |

## 11. Lo que NO se adopta (templado a tu escala)

- **Blockchain para logs** (GPT): innecesario. Append-only + hash-chain alcanza.
- **HSM físico + OTP/2FA** (GPT): aspiracional, no day-1. El break-glass debe ser
  pesado y auditado; la forma exacta escala con la operación.
- **Dual-control (2 humanos)** (ambos): no realista con un operador solo. Se deja
  el *hook* listo; el piso hoy es override tipeado + razón + plan_hash + incidente
  + rollback. Se activa dual-control cuando haya 2º operador de confianza.

## 12. Decisiones abiertas v2 (para Fernando)

1. **Override de Thot:** ¿confirmás el piso solo-operador (tipeo + razón + plan_hash
   + incidente + rollback), con dual-control como hook futuro? ¿O preferís dual-control
   ya (requiere 2º humano — ¿tu hijo con hall9000 v1.0?)?
2. **Alcance de v1:** ¿implementamos el núcleo de seguridad completo de una, o por
   fases (§13)?
3. **`local.scratch.exec`** para jax_local (jaula efímera) — ¿lo querés en el roadmap
   o jax_local queda 100% sin ejecución indefinidamente?
4. **Granularidad de parámetros** por capability — ¿qué nivel de `resource_scope` /
   `allowed_paths` te sirve (por repo, por path, por servicio)? Esto lo afina Ada.

## 13. Plan de implementación por fases

**Fase 1 — Núcleo de gobernabilidad (lo que hace el sistema "no inseguro"):**
policy engine determinista, capability registry (10) parametrizado, Intent Envelope
firmado + caller_chain, plan_hash + congelado de aprobación, control_plane.change y
capability.grant separados, audit append-only con hash-chain.

**Fase 2 — Contención de ejecución:** sandbox real (jaula), tokens efímeros por
paso, runtime monitor + kill-on-deviation, environment_isolation_check, deny rules
de composición, redacción de secrets en read-only.

**Fase 3 — Operación avanzada:** rate limits/quotas/time-box, blast_radius fino,
override break-glass formal, hook de dual-control, `local.scratch.exec`.

**Pipeline de cada fase (las facetas se gobiernan a sí mismas):**
Ada formaliza el contrato e invariantes → Kimi implementa en sandbox → Thot audita
(busca el bypass) → Hyde aplica a staging y luego producción con gate.
