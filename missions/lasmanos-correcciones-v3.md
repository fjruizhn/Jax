# Correcciones v3 — cierre de la auditoría adversarial de Thot

> Insumo para el **contrato v3** de Ada. Cierra TODOS los hallazgos del veredicto
> NO-GO de Thot sobre el contrato v2. Ningún hallazgo queda abierto.
>
> Donde la solución de Thot era enterprise-desproporcionada (root de Jacobs, DoS de
> Thot), se cierra el hallazgo con la versión a-escala que igual resuelve la causa
> raíz — marcado `[A-ESCALA]`. No es recorte: el problema queda resuelto.

---

## PRINCIPIO RAÍZ DE LA v3

Casi todos los bloqueadores de Thot son la misma enfermedad: **el motor confía en
datos autodeclarados por el plan/artefacto para decisiones de seguridad.**

> **REGLA MAESTRA v3:** Todo campo que afecte una decisión de seguridad
> (`resource_class`, `data_class`, `source_trust`, `runtime_privilege`,
> `blast_radius`, `reversibility`, `risk_class`, lineage, `parent_artifact_ids`)
> se **deriva de fuente autoritativa** — un registry determinista (path→clase) o el
> runtime monitor (inputs observados). El plan/faceta **propone**; el motor/runtime
> **deriva y decide**. Si lo declarado < lo efectivo → señal de subdeclaración.

Esta regla, aplicada en `decide()`, cierra los bloqueadores 4, 5, 6, 7, 8 de un golpe.

---

## A. Correcciones al `decide()` (Artefacto 3)

**A1. `promote_env` no muta-y-sigue** (Vector 1, bloqueador #1). El bug crítico.
- `promote_env` SIEMPRE retorna `GATE(reason="environment_promotion_requires_redecision")`.
  NUNCA muta `envelope.environment` y continúa.
- La reentrada exige un **nuevo envelope firmado** (nuevo env, nuevo nonce, capability
  recargada desde A5). Si la capability no admite el env promovido → `DENY`.
- **`implementation.sandbox` con `runtime_privilege ∈ {docker_sock, host_mount, root, sudo}`
  → `DENY` directo**, no `promote_env`. (Una sandbox con docker.sock no se "promueve":
  se rechaza.)

**A2. Envelope inmutable dentro de `decide()`** (Problema 2). El caller firmó un env
concreto; mutarlo invalida la firma. Usar `effective_environment` como variable local;
si difiere del firmado → `GATE`/`DENY` exigiendo nuevo envelope firmado. Nunca se emite
token para un env que el caller no firmó.

**A3. `cap` se recarga tras cualquier promoción** (Problema 3). Toda promoción reinicia
la decisión desde A5 con el nuevo contexto. Prohibido evaluar producción con la
capability de sandbox/staging/docs original.

**A4. Nueva Fase A0 — validación estructural ANTES de firma y nonce** (Problema 1).
`structurally_valid_raw(envelope)` (sin floats, sin claves duplicadas, UTF-8 válido,
canonicalización no ambigua) corre **antes** de A1 (firma) y A10 (nonce). Orden:
A0 estructura → parseo canónico → firma → nonce.

**A5. Nonce no se consume en rama que termina en GATE** (brief punto 2). El nonce se
**reserva** (no se marca usado) hasta decisión terminal ALLOW/DENY. `GATE` mantiene el
nonce reservado atado al estado pendiente; tras aprobación humana, la reentrada usa ese
mismo nonce reservado (no uno nuevo). Solo ALLOW/DENY consumen el slot anti-replay.

**A6. Redacción enforced en Fase K, con código real** (Problema 5). Deja de ser
aspiración:
```python
if step.secret_access != "none" or step.data_class in {"secret","credential","pii"}:
    if not redaction_policy_attached(envelope, plan):
        return DENY(reason="redaction_policy_missing")
```

**A7. Fase F (lineage) ANTES del cálculo de aggregate_risk** (Vector 6, bloqueador #6).
Los factores de lineage (F-transform de artefactos heredados) deben entrar en `additive`
ANTES de calcular `aggregate_risk`. Reordenar: derivar lineage_factors → sumarlos a
additive → recién entonces calcular aggregate_risk. Si lineage sube el riesgo sobre lo
declarado → `GATE(reason="lineage_risk_exceeds_declared")`.

---

## B. Enforcement por riesgo efectivo (bloqueador #4 — el hallazgo conceptual #8)

**Clasificar no es autorizar.** Hoy `aggregate_risk = critical` pasa si la capability no
pide gate. Eso se corrige con una **matriz de enforcement por riesgo EFECTIVO** que
DISPARA controles, independiente de la capability:
```
low:      auto_audit permitido en env ≤ 3
medium:   audit trail + checks reforzados
high:     human_gate si env ≥ staging; tests externos si toca code/config/ci_cd
critical: SIEMPRE human_gate + thot_review + rollback + tests + audit remoto,
          independientemente de la capability o de requires_gate
```
Nuevas invariantes:
```
I48 [HARD] ∀ P : aggregate_risk(P)=critical → human_gate(P) ∧ thot_review(P)
                 ∧ rollback_verified(P) ∧ tests_passed(P) ∧ audit_remoto(P)
I49 [HARD] ∀ P : aggregate_risk(P)=high ∧ target_env ≥ staging → human_gate(P)
```
El chequeo `aggregate_risk > plan.risk_class` se mantiene (consistencia), pero ya NO es
el único control: el riesgo efectivo activa la matriz de arriba.

---

## C. Metadatos autoritativos (bloqueador #5, Vector 3, Invariante 3) — la raíz

**C1. `resource_class`/`data_class`/`runtime_privilege` se DERIVAN, no se declaran.**
```
I47 [HARD] ∀ step s :
   effective_resource_class   = authoritative_resource_registry(s.path, s.resource_instance)
   effective_data_class       = authoritative_data_registry(s.path)
   effective_runtime_privilege = runtime_policy(s)
   ∧ risk se calcula con valores EFECTIVOS, no declarados
   ∧ si declarado < efectivo → flag subdeclaración (SOFT warn + log)
```
El plan puede proponer; el motor sobreescribe con lo autoritativo.

**C2. `denied_paths` globales mínimos** — independientes de lo que `[FERNANDO]` ponga en
`allowed_paths`, y SIEMPRE ganan:
```
**/.github/workflows/**   **/.gitlab-ci.yml   **/Jenkinsfile
**/ci/**   **/cd/**   **/deploy/**   **/runner/**
**/prompts/**   **/tool_manifest/**   **/policy/**   **/infra/**
**/.env   **/secrets/**   **/vault/**
```
Tocar cualquiera de estos fuerza `resource_class` autoritativo (ci_cd/control_plane/etc.)
→ floor de riesgo correspondiente + gate. No se puede tocar CI/CD "como si fuera docs".

---

## D. Capabilities y binding (bloqueadores #2, #3)

**D1. `allowed_actions` obligatorio; ausencia = DENY** (Vector 2).
```
I46 [HARD] ∀ capability C : C.allowed_actions definido ∧ C.denied_actions definido
```
En `decide()`: `if cap.allowed_actions is None: return DENY("capability_missing_allowed_actions")`.
Las 10 capabilities llevan `allowed_actions` explícito en el registry.

**D2. Token = intersección, no herencia.** El token nunca hereda las acciones del plan
sin intersectarlas:
```python
effective_actions = intersection(step.allowed_actions, cap.allowed_actions)
if effective_actions != step.allowed_actions:
    return DENY(reason="actions_exceed_capability")
```

**D3. Binding envelope.capability ↔ step.capability + decisión por step** (Invariante 1).
```
I45 [HARD] ∀ decisión D sobre step s :
   envelope.capability_requested = plan.steps[s].capability
   ∧ cap = registry[plan.steps[s].capability]
   ∧ step.environment ∈ cap.environment
   ∧ step.allowed_actions ⊆ cap.allowed_actions
```
Cada step se decide con SU propia capability — no un único `decide()` por plan global.

---

## E. Provenance / lineage derivada del runtime (bloqueadores #7, #8, Vectores 4/5, Huecos 1-5)

**El piso falso que encontró Thot:** todo el Artefacto 6 asumía que el artefacto dice la
verdad sobre su propio origen. Se corrige derivando provenance del runtime, no de
autodeclaración.

**E1. Lineage derivado de observación, no declarado** (Vectores 4, 5, Huecos 3, 4).
```
∀ plan P : observed_inputs(P) [del runtime monitor] = declared_parent_artifacts(P)
   ∧ discrepancia → DENY(reason="lineage_mismatch")
∀ artefacto A leído/ejecutado/linkeado/compilado/empaquetado/usado-como-input durante P :
   A ∈ P.expected_artifacts   ∧   sino → DENY(reason="undeclared_artifact_consumed")
```
`parent_artifact_ids` y `source_trust` los **produce el runtime monitor**, no la faceta.

**E2. `source_trust` no se puede limpiar** (Vector 5).
```
A_hijo.source_trust = union(source_trust(padres_observados)) ∪ source_trust(inputs_nuevos)
si depth=0 → parent_artifact_ids=[] ∧ ancestor_chain=[] ∧ runtime no observó inputs
```
Nacer con `source_trust=internal` mientras el runtime observó un padre `external_untrusted`
→ `DENY`.

**E3. CREATE Artifact solo por trusted_builder/runtime, no por faceta LLM** (Hueco 4).
La faceta propone metadata; el runtime la certifica o rechaza. Provenance no nace de la
palabra del LLM.

**E4. `artifact_hash` calculado por componente independiente** (Hueco 2). Content-addressed
storage (OCI digest / CAS / blob inmutable), no hash autodeclarado por la faceta productora.
Runtime verifica `hash(bytes_ejecutados) == artifact_hash_aprobado` inmediatamente antes de
ejecutar.

**E5. Definir `ArtifactRef`** (Hueco 1) — el tipo que el Plan usa pero nunca se definió:
```typescript
interface ArtifactRef {
  artifact_id: string
  artifact_hash: string
  artifact_signature: Signature
  artifact_key_id: string
  immutable_storage_uri: string      // content-addressed
  full_lineage_hash: string          // hash de toda la genealogía
}
```
Fase F carga y verifica el `Artifact` completo desde storage inmutable.

**E6. Entidad `ArtifactReview`** (Hueco 5) — `approved` deja de ser genérico:
```typescript
interface ArtifactReview {
  artifact_id: string
  artifact_hash: string
  approved_for_environment: Environment
  reviewer: string
  reviewer_signature: Signature      // "JAX_ARTIFACT_REVIEW_V1:" || canonical(...)
  plan_hash: string
  policy_version: string
  expires_at?: string
}
```
Fase F verifica firma + vigencia + hash. Un `approved` no sirve para otro ambiente ni
otro contenido.

**E7. I28 bien cuantificada** (Invariante mal cuantificada 5):
```
A.source_trust ≠ [] ∧ ∀ t ∈ A.source_trust : t ∈ {internal, external_untrusted}
∧ (∃ ancestro con external_untrusted → A también tiene external_untrusted)
```

---

## F. Cripto e identidad (Artefacto 4)

**F1. Root de confianza de Jacobs** (Hallazgo cripto 1, bloqueador #9) `[A-ESCALA]`.
- Crear `JAX_ROOT_IDENTITY_KEY`: clave raíz **offline**, en soporte físico **fuera de
  hall9000** y desconectado. El soporte es **genérico/migrable** — la propiedad que
  importa es "clave físicamente separada del sistema que protege", no el aparato. Soporte
  inicial libre (USB cifrado); migrable a hardware-backed (p. ej. token tipo YubiKey) sin
  cambiar el contrato.
- Firma el cert de Jacobs una vez: `cert(jacobs) = {jacobs.pubkey, "jacobs", key_id}`
  firmado con la root.
- Toda la cadena de identidad cuelga de ahí: root → Jacobs → facetas.
- Audit append-only de toda emisión de cert.
- **`[A-ESCALA]`:** sin HSM físico ni quorum humano (no aplican a operador solo). La root
  offline resuelve la causa raíz: Jacobs deja de ser raíz auto-certificada.
- Recuperación si Jacobs se compromete: re-certificar con la root offline.
- **El contrato NO especifica el soporte físico** — solo exige `offline ∧ separado_de_hall9000`.

**F2. Domain separation completa** (Hallazgo cripto 2, bloqueador #10). I39 incluye TODOS
los prefijos usados:
```
JAX_PLAN_V1, JAX_APPROVAL_V1, JAX_CAPABILITY_TOKEN_V1, JAX_AUDIT_EVENT_V1,
JAX_IDENTITY_V1, JAX_DELEGATION_V1, JAX_ARTIFACT_V1, JAX_ARTIFACT_REVIEW_V1,
JAX_ROOT_IDENTITY_V1
```

**F3. Runtime adapter = signer restringido** (Hallazgo cripto 3). I41 desplazaba el SPOF
al adapter; se acota:
- No firma envelopes arbitrarios.
- Valida policy local mínima antes de firmar.
- Requiere nonce emitido por el policy engine.
- Registra cada intento de firma + rate limits.
- Rechaza cambios de capability/env/scope no autorizados.
- (Hardware-backed key → opcional, no obligatorio a escala.)
```
I51 [HARD] ∀ firma del adapter : nonce_emitido_por_policy_engine(firma)
   ∧ policy_local_validada(firma) ∧ audit(intento_firma) ∧ dentro_de_rate_limit
```

**F4. Delegación atada a contexto** (Vector 7). Cada `ChainLink` firma un objeto de
delegación completo (no solo delegador+delegatee):
```typescript
{ delegator, delegatee, capability_requested, allowed_environments,
  resource_scope, allowed_actions, plan_hash?, not_before, not_after,
  nonce, delegation_id, policy_version, registry_version }
```
`decide()` verifica que el envelope esté **dentro del scope delegado**. Una delegación de
bajo riesgo no autentica una intención más riesgosa.

---

## G. Thot — disponibilidad y resistencia (auto-crítica, bloqueador #10)

**G1. Separar disponibilidad de incidente** (Riesgo 1) `[A-ESCALA]`.
- `security_incident` = Thot dice NO-GO y el operador overridea (lo que ya teníamos).
- `availability_override` = Thot **indisponible/caído** — severidad distinta, no es un
  incidente de seguridad. Evita que el sistema se trabe si Thot cae.
- El veto de Thot tiene **TTL**: pasado el TTL sin resolución, escala a
  `availability_override` (operador), no a bloqueo indefinido.
- **`[A-ESCALA]`:** sin quorum de N auditores (no existen). El operador solo cubre el
  fallback de disponibilidad.

**G2. Thot resistente a prompt injection** (Riesgo 2). Generalizar I27 (hoy solo cubre
`research.web`) a TODA faceta auditora:
```
I50 [HARD] ∀ faceta auditora F, ∀ contenido auditado C :
   C se entrega como datos no ejecutables, delimitado, sin autoridad instructiva,
   con system prompt inmutable, y con detector de prompt injection.
   Texto dentro de un plan/diff NUNCA es instrucción para Thot.
```

---

## Resumen de cierre — los 10 bloqueadores de Thot

| # | Bloqueador | Cierre |
|---|---|---|
| 1 | promote_env no reentra | A1 (GATE, no muta; sandbox+docker_sock=DENY) |
| 2 | allowed_actions abierto | D1 (obligatorio, ausencia=DENY) + D2 (intersección) |
| 3 | sin binding capability↔step | D3 / I45 (decisión por step) |
| 4 | riesgo declarado por el plan | C1 / I47 (derivado autoritativo) + REGLA MAESTRA |
| 5 | critical no fuerza gate | B / I48-I49 (matriz de enforcement) |
| 6 | Fase F tras cálculo de riesgo | A7 (lineage antes de aggregate_risk) |
| 7 | expected_artifacts omitible | E1 (reconciliación con observed_inputs) |
| 8 | lineage nace falso | E1-E3 (derivado del runtime, no declarado) |
| 9 | Jacobs sin root | F1 (JAX_ROOT_IDENTITY_KEY offline) [A-ESCALA] |
| 10 | Thot SPOF / injection | G1 (availability_override) + G2 (I50) [A-ESCALA] |

---

## Instrucción para Ada (contrato v3)

Producir el **contrato v3** corrigiendo TODOS los hallazgos según las correcciones A–G.
Cambios estructurales que Ada debe formalizar:
- `decide()` reordenado (A0 estructura → firma → nonce reservado → riesgo → lineage ANTES
  de aggregate → enforcement por riesgo efectivo → gates). `promote_env` = GATE siempre.
- Las 10 capabilities con `allowed_actions`/`denied_actions` explícitos.
- REGLA MAESTRA: campos de seguridad derivados de registry autoritativo + runtime, no del
  plan. Nuevas I45-I51. Matriz de enforcement por riesgo efectivo.
- Artefacto 6 reescrito: lineage y source_trust derivados del runtime monitor; `ArtifactRef`
  y `ArtifactReview` definidos; `artifact_hash` por content-addressed storage; CREATE solo
  por trusted builder.
- Artefacto 4: `JAX_ROOT_IDENTITY_KEY`, domain separation completa, adapter restringido,
  delegación atada a contexto.
- Thot: `availability_override` vs `security_incident`, veto con TTL, I50 anti-injection.

Mismo estándar de rigor: lo que sea dato de entorno → `[FERNANDO]`; default sensato →
`[PROPUESTO]`; lo que siga ambiguo → `DECISIÓN PENDIENTE`. No suponer.

NOTA `[A-ESCALA]`: la root de Jacobs es offline-en-USB (no HSM/quorum) y el fallback de
Thot es operador-solo (no quorum de N). Ambos cierran el hallazgo a la escala real. No
re-expandir a enterprise.
