# Resultado de: ada-contrato.md

# Contrato Formal — Las Manos v2 (Fase 1 + Fase 2)

Ada, séptima faceta. Formalizo el contrato. Donde el blueprint no cierra un valor, declaro `DECISION PENDIENTE` — no relleno.

---

## ARTEFACTO 1 — ESQUEMA DE CAPABILITIES

Notación tipada: `campo: Tipo = default_o_restriccion`. Tipos base: `str`, `str[]`, `int`, `bool`, `enum{...}`. Los valores marcados `D.P.` son Decisiones Pendientes.

```toml
# ============================================================
# CAPABILITY REGISTRY — 10 capabilities parametrizadas
# ============================================================

# ---------- 1. formal.spec ----------
[capability.formal_spec]
allowed_callers   = ["ada", "jacobs"]
environment       = [0, 1, 2, 3, 4]              # Conversación → Staging
resource_scope    = "service+path"               # ej. "~/jax-platform/specs/"
allowed_paths     = ["D.P."]                      # patrón por repo
denied_paths      = ["**/.env", "**/secrets/**", "**/vault/**"]
secret_access     = "none"                        # enum: none|read_redacted|read|write|mount
network_egress    = "none"                        # enum: none|loopback|restricted_allowlist|open
egress_allowlist  = []                            # str[] de dominios si restricted_allowlist
data_classification = "internal"                  # enum: public|internal|confidential|secret
max_runtime       = "D.P."                        # int segundos
max_diff_size     = "D.P."                        # int bytes o líneas
blast_radius      = "file"                        # enum: none|file|service|tenant|system
requires_gate     = false
requires_audit    = true
requires_rollback = false
requires_tests    = false                         # blueprint §4 lo lista; D.P. exigencia real
approval_class    = "none"                        # enum: none|auto_audit|human_gate|human_gate_plus_thot|operator_override

# ---------- 2. implementation.sandbox ----------
[capability.implementation_sandbox]
allowed_callers   = ["kimi", "ada", "jacobs"]
environment       = [0, 1, 2, 3]                  # hasta sandbox real
resource_scope    = "service+path"
allowed_paths     = ["D.P."]
denied_paths      = ["**/.env", "**/secrets/**", "**/vault/**", "**/docker.sock", "**/sudoers"]
secret_access     = "none"
network_egress    = "none"                        # sandbox sin red (§3)
egress_allowlist  = []
data_classification = "internal"
max_runtime       = "D.P."                        # timeout obligatorio (§3)
max_diff_size     = "D.P."
blast_radius      = "file"
requires_gate     = false
requires_audit    = true
requires_rollback = false
requires_tests    = "D.P."
approval_class    = "auto_audit"

# ---------- 3. implementation.staging ----------
[capability.implementation_staging]
allowed_callers   = ["kimi", "hyde", "jacobs"]
environment       = [0, 1, 2, 3, 4]               # hasta staging
resource_scope    = "service+path"
allowed_paths     = ["D.P."]
denied_paths      = ["**/.env", "**/secrets/**", "**/vault/**", "**/docker.sock"]
secret_access     = "none"
network_egress    = "restricted_allowlist"        # D.P. dominios permitidos
egress_allowlist  = ["D.P."]
data_classification = "internal"
max_runtime       = "D.P."
max_diff_size     = "D.P."
blast_radius      = "service"
requires_gate     = true                          # gate CONDICIONAL (§7)
requires_audit    = true
requires_rollback = true                          # staging reversible (§3)
requires_tests    = "D.P."
approval_class    = "human_gate"                  # si trigger de §7 aplica

# ---------- 4. production.deploy ----------
[capability.production_deploy]
allowed_callers   = ["hyde", "jacobs"]
environment       = [5]                           # solo producción
resource_scope    = "service+path"
allowed_paths     = ["D.P."]
denied_paths      = ["**/.env", "**/secrets/**", "**/vault/**"]
secret_access     = "none"
network_egress    = "restricted_allowlist"
egress_allowlist  = ["D.P."]
data_classification = "confidential"
max_runtime       = "D.P."
max_diff_size     = "D.P."
blast_radius      = "service"
requires_gate     = true                          # gate humano + veto Thot (§3,§4)
requires_audit    = true
requires_rollback = true                          # obligatorio verificado (§9)
requires_tests    = true
approval_class    = "human_gate_plus_thot"

# ---------- 5. production.rollback ----------
[capability.production_rollback]
allowed_callers   = ["hyde", "jacobs"]
environment       = [5]
resource_scope    = "service+path"
allowed_paths     = ["D.P."]
denied_paths      = ["**/audit/**"]               # no puede tocar su audit
secret_access     = "none"
network_egress    = "restricted_allowlist"
egress_allowlist  = ["D.P."]
data_classification = "confidential"
max_runtime       = "D.P."
max_diff_size     = "D.P."
blast_radius      = "service"
requires_gate     = true
requires_audit    = true
requires_rollback = false                         # rollback es la acción reversiva
requires_tests    = false
approval_class    = "human_gate_plus_thot"

# ---------- 6. audit.review ----------
[capability.audit_review]
allowed_callers   = ["thot", "jacobs"]
environment       = [0, 1, 2, 3, 4, 5]            # todos los ambientes
resource_scope    = "service+path"
allowed_paths     = ["D.P."]
denied_paths      = []
secret_access     = "read_redacted"               # secrets SIEMPRE redactados (§9)
network_egress    = "none"
egress_allowlist  = []
data_classification = "confidential"
max_runtime       = "D.P."
max_diff_size     = 0                             # read-only, no escribe
blast_radius      = "none"
requires_gate     = false
requires_audit    = true                          # audit.review se audita a sí mismo
requires_rollback = false
requires_tests    = false
approval_class    = "none"

# ---------- 7. research.web ----------
[capability.research_web]
allowed_callers   = ["hipatia", "jacobs"]
environment       = [0, 1, 2, 3, 4, 5]
resource_scope    = "none"                        # no toca filesystem prod
allowed_paths     = []
denied_paths      = ["**"]
secret_access     = "none"
network_egress    = "open"                        # web read-only
egress_allowlist  = []
data_classification = "public"                    # output = untrusted-data (§2.8)
max_runtime       = "D.P."                        # rate-limited (§4)
max_diff_size     = 0
blast_radius      = "none"
requires_gate     = false
requires_audit    = true
requires_rollback = false
requires_tests    = false
approval_class    = "auto_audit"

# ---------- 8. docs.write ----------
[capability.docs_write]
allowed_callers   = ["jekyll", "kimi", "jacobs"]
environment       = [0, 1, 2, 3, 4]               # sandbox/staging
resource_scope    = "service+path"
allowed_paths     = ["D.P."]                      # ej. "~/jax-platform/docs/"
denied_paths      = ["**/.env", "**/secrets/**", "**/infra/**", "**/policy/**"]
secret_access     = "none"
network_egress    = "none"
egress_allowlist  = []
data_classification = "public"
max_runtime       = "D.P."
max_diff_size     = "D.P."
blast_radius      = "file"
requires_gate     = false
requires_audit    = true
requires_rollback = true                          # docs en staging reversibles
requires_tests    = false
approval_class    = "auto_audit"

# ---------- 9. control_plane.change ----------
[capability.control_plane_change]
allowed_callers   = ["jacobs"]                    # SOLO jacobs
environment       = [5]
resource_scope    = "service+path"
allowed_paths     = ["D.P."]                      # policy engine, registry, router, gate, audit, Jacobs, prompts
denied_paths      = ["**/audit/hash_chain/**"]    # no rompe la cadena de audit
secret_access     = "none"
network_egress    = "none"
egress_allowlist  = []
data_classification = "secret"
max_runtime       = "D.P."
max_diff_size     = "D.P."
blast_radius      = "system"                      # toca el control plane
requires_gate     = true
requires_audit    = true
requires_rollback = true
requires_tests    = true
approval_class    = "human_gate_plus_thot"        # + Ada invariant check (§9)

# ---------- 10. capability.grant ----------
[capability.capability_grant]
allowed_callers   = ["human"]                     # SOLO humano; máquina → DENY
environment       = [5]
resource_scope    = "none"
allowed_paths     = []
denied_paths      = ["**"]
secret_access     = "none"
network_egress    = "none"
egress_allowlist  = []
data_classification = "secret"
max_runtime       = "D.P."
max_diff_size     = 0
blast_radius      = "system"
requires_gate     = true
requires_audit    = true
requires_rollback = true
requires_tests    = false
approval_class    = "operator_override"           # break-glass formal

# ============================================================
# FLAG GLOBAL — Dual-control (hook apagado, §11)
# ============================================================
[control_plane.dual_control]
enabled           = false                         # hook presente, inactivo
activation_policy = "D.P."                        # ¿cuándo se activa? (2º operador)
required_operators = 2                            # cuando enabled = true
```

---

## ARTEFACTO 2 — INVARIANTES

Predicados verificables por código determinista. `HARD` = bloquea ejecución. `SOFT` = advierte pero no bloquea.

```
# ============================================================
# INVARIANTES FORMALES
# ============================================================

# --- Producción ---
I1  [HARD]  ∀ J ∈ production.* :
            ∃ human_gate.approved(plan_hash(J)) ∧ ∃ thot_review(plan_hash(J))

I2  [HARD]  ∀ J ∈ production.deploy :
            plan_hash_aprobado(J) = plan_hash_ejecutado(J)
            ∧ artifact_hash_aprobado(J) = artifact_hash_ejecutado(J)

I3  [HARD]  ∀ J ∈ production.* : risk_class(J) ≠ unknown

I4  [HARD]  ∀ J ∈ production.* : reversible(J) ∧ rollback_verified(J, plan_hash(J))

# --- Separación de funciones ---
I5  [HARD]  ∀ J : implementer(J) ≠ auditor(J)

I6  [HARD]  ∀ J : proposer(J) ∉ approvers(J)
            ∧ |approvers(J)| ≥ 1

I7  [HARD]  ∀ deploy(plan_hash) :
            spec_author(plan_hash) ≠ deploy_approver(plan_hash)
            # Ada no despliega su propia spec

I8  [HARD]  ∀ audit(código_kimi) : kimi ≠ auditor(código_kimi)

# --- Override de veto ---
I9  [HARD]  ∀ override ∈ overrides :
            reason(override) ≠ ""
            ∧ actor(override) ∈ humanos
            ∧ plan_hash(override) ≠ null
            ∧ risk_acknowledgement(override) = true
            ∧ rollback_plan(override) ≠ null
            ∧ security_incident(override) registrado

I10 [HARD]  ∀ t : veto_thot(t) no es eliminado
            # solo se overridea, never deleted

# --- Hyde / Kimi ---
I11 [HARD]  ∀ activación(hyde) :
            token_válido(hyde) ∧ plan_hash(hyde) aprobado
            # no se activa por intención conversacional

I12 [HARD]  ∀ ejecución(hyde, step) :
            scope_efectivo(step) ⊆ scope_token(step)
            ∧ plan_hash_ejecutado(step) = plan_hash_token(step)
            # Hyde no amplía scope, no modifica plan en vuelo

I13 [HARD]  ∀ J : kimi ∉ callers(J, production.deploy)

I14 [HARD]  ∀ artefacto → producción :
            reviewer(artefacto) ≠ kimi

# --- Control plane / capability.grant ---
I15 [HARD]  ∀ J ∈ control_plane.change :
            human_gate(J) ∧ thot_review(J) ∧ ada_invariant_check(J)

I16 [HARD]  ∀ J ∈ capability.grant :
            caller(J) ∈ humanos
            # caller = máquina → DENY

I17 [HARD]  ∀ faceta F : ¬modifica(F, permisos_efectivos(F))

# --- Composición / auditoría / secrets ---
I18 [HARD]  ∀ cadena C :
            ∃ chain_id(C) ∧ risk_efectivo(C) ≥ max(risk(step_i) ∀ step_i ∈ C)

I19 [HARD]  ∀ registro r ∈ audit :
            append_only(r) ∧ hash_chain(r, r_anterior) válida

I20 [HARD]  ∀ caller J, ∀ job j de J :
            ¬modifica(J, audit(j))

I21 [HARD]  ∀ output de capability read-only :
            ∀ secret s ∈ output : redacted(s) = true

I22 [HARD]  ∀ J, ∀ output(J) :
            ¬(secret_aparece_sin_redacción(output(J)))

# --- TOCTOU ---
I23 [HARD]  ∀ deploy :
            plan_hash_aprobado(deploy) = plan_hash_ejecutado(deploy)
            # refuerza I2 para todo production.deploy

# --- Environment isolation ---
I24 [HARD]  ∀ env staging S :
            si comparte_recurso(S, producción) → aplicar_reglas(S, producción)

I25 [SOFT]  ∀ sandbox B :
            sin_red(B) ∧ sin_secrets(B) ∧ sin_docker_sock(B) ∧ sin_sudo(B)
            ∧ timeout(B) definido ∧ quota(B) definida
            # SOFT: advertir si no se puede verificar estáticamente

# --- jax_local ---
I26 [HARD]  ∀ acción A :
            jax_local ∉ callers(A, capability_ejecutora)
            ∧ jax_local no sugiere capabilities restringidas

# --- research.web ---
I27 [HARD]  ∀ output de research.web :
            clasificado_como(output, untrusted_data)
            ∧ output no interpretado como instrucción
```

---

## ARTEFACTO 3 — FUNCIÓN DE DECISIÓN DEL POLICY ENGINE

```python
# ============================================================
# decide(envelope, plan) -> {ALLOW, DENY, GATE}
# Motor DETERMINISTA. Ningún LLM decide en runtime (Regla 6).
# Fail-closed: ante duda, DENY o GATE, nunca ALLOW implícito.
# ============================================================

def decide(envelope: Envelope, plan: Plan) -> Decision:

    # --- FASE A: Precondiciones estructurales (cualquier fallo → DENY) ---

    # A1. Firma del caller válida
    if not verify_signature(envelope.caller, envelope.caller_signature, envelope):
        return DENY(reason="caller_signature_invalid")

    # A2. caller_chain íntegra (cada link firmado, Jacobs verificó delegación)
    if not verify_caller_chain(envelope.caller_chain):
        return DENY(reason="caller_chain_broken")

    # A3. Capability existe en registry
    cap = registry.get(envelope.capability_requested)
    if cap is None:
        return DENY(reason="capability_not_found")

    # A4. Caller autorizado para esa capability
    if envelope.caller not in cap.allowed_callers:
        return DENY(reason="caller_not_allowed")

    # A5. Ambiente solicitado permitido para la capability
    if envelope.environment not in cap.environment:
        return DENY(reason="environment_not_allowed")

    # A6. Regla 5 — delegación indirecta vía Jacobs:
    #     si la faceta no podría pedir la capability directamente → DENY
    if not could_request_directly(envelope.caller, envelope.capability_requested):
        return DENY(reason="indirect_capability_laundering")

    # A7. plan_hash presente y válido
    if envelope.plan_hash is None or envelope.plan_hash != hash(plan):
        return DENY(reason="plan_hash_missing_or_mismatch")

    # --- FASE B: Chequeos de scope y paths ---

    # B1. resource_scope ⊆ permitted_scope(caller, capability)
    if not scope_subset(plan.resources_touched, permitted_scope(envelope.caller, cap)):
        return DENY(reason="resource_scope_exceeded")

    # B2. Path checks
    for resource in plan.resources_touched:
        if not matches_any(resource.path, cap.allowed_paths):
            return DENY(reason="path_not_allowed")
        if matches_any(resource.path, cap.denied_paths):
            return DENY(reason="path_denied")

    # --- FASE C: Riesgo y composición ---

    # C1. risk_class ≠ unknown (producción lo exige; otros ambientes → GATE)
    risk = compute_risk(plan)
    if risk == "unknown":
        if envelope.environment == 5:
            return DENY(reason="risk_unknown_in_production")  # I3
        else:
            return GATE(reason="risk_unknown_requires_review")

    # C2. Composition deny-rules (tabla del Artefacto 5)
    comp = check_composition_rules(plan)
    if comp.verdict == "deny":
        return DENY(reason=f"composition_deny:{comp.rule}")
    if comp.verdict == "gate":
        return GATE(reason=f"composition_gate:{comp.rule}")

    # C3. aggregate_risk = max(step_risk) + composition_risk
    aggregate = compute_aggregate_risk(plan)
    if aggregate > plan.risk_class:
        return GATE(reason="aggregate_risk_exceeds_approved")

    # --- FASE D: Ambiente y aislamiento ---

    # D1. environment_isolation_check (I24)
    if envelope.environment == 4:  # staging
        if shares_resource_with_production(plan.resources_touched):
            # aplicar reglas de producción
            envelope.environment = 5  # promoción forzada
            # re-ejecutar chequeos de producción (GATE obligatorio)
            return GATE(reason="staging_shares_production_resource")

    # --- FASE E: Gates condicionales (§7) ---

    # E1. Gate triggers — cualquiera dispara GATE
    triggers = evaluate_gate_triggers(plan, envelope)
    if any(triggers):
        return GATE(reason=f"gate_trigger:{triggers}")

    # E2. Capability-specific gates
    if cap.requires_gate:
        if envelope.environment == 5:
            # I1: human_gate + thot_review atados a plan_hash
            if not human_gate_approved(envelope.plan_hash):
                return GATE(reason="production_requires_human_gate")
            if not thot_review_completed(envelope.plan_hash):
                return GATE(reason="production_requires_thot_review")
            # I2: artifact_hash match (TOCTOU)
            if not artifact_hash_matches(envelope.plan_hash, plan):
                return DENY(reason="artifact_hash_mismatch_toctou")

        if cap.approval_class == "human_gate_plus_thot":
            if not thot_review_completed(envelope.plan_hash):
                return GATE(reason="thot_review_required")

    # E3. control_plane.change → Ada invariant check (I15)
    if envelope.capability_requested == "control_plane.change":
        if not ada_invariant_check_passed(plan):
            return GATE(reason="control_plane_requires_ada_invariant_check")

    # E4. capability.grant → caller humano (I16)
    if envelope.capability_requested == "capability.grant":
        if not is_human(envelope.caller):
            return DENY(reason="capability_grant_requires_human")

    # --- FASE F: Audit y reversibilidad ---

    # F1. requires_audit → audit trail preparado
    if cap.requires_audit:
        if not audit_trail_ready(envelope):
            return DENY(reason="audit_trail_not_ready")

    # F2. requires_rollback → rollback_plan verificado
    if cap.requires_rollback:
        if not rollback_plan_verified(plan):
            return GATE(reason="rollback_plan_not_verified")

    # --- FASE G: Decisión final ---

    # Si llegó aquí sin GATE ni DENY → ALLOW
    return ALLOW(token=issue_ephemeral_token(
        capability     = envelope.capability_requested,
        resource_scope = plan.resources_touched,
        environment    = envelope.environment,
        caller_chain   = envelope.caller_chain,
        plan_hash      = envelope.plan_hash,
        step_id        = envelope.step_id,
        expires_at     = now() + cap.max_runtime,
        max_runtime    = cap.max_runtime,
        allowed_actions= plan.steps[envelope.step_id].allowed_actions
    ))

# ============================================================
# POSTCONDICIONES POR RAMA
# ============================================================
# ALLOW:
#   - emite token efímero atado a (plan_hash, step_id, scope, expires_at)
#   - audit_log.append("ALLOW", envelope, plan_hash, token_id)
#   - runtime_monitor activa observación del job
#
# DENY:
#   - audit_log.append("DENY", envelope, reason)
#   - no se emite token
#   - no se ejecuta nada
#
# GATE:
#   - audit_log.append("GATE_PENDING", envelope, reason)
#   - estado = "esperando_aprobacion_humana" atado a plan_hash
#   - notifica operador humano
#   - el job NO corre hasta resolución
#   - si aprobado → re-entra por decide() con human_gate_approved=true
#   - si rechazado → DENY
```

---

## ARTEFACTO 4 — INTENT ENVELOPE FIRMADO

```typescript
// ============================================================
// INTENT ENVELOPE — estructura tipada completa
// ============================================================

type Caller = "ada" | "kimi" | "hyde" | "thot" | "hipatia" | "jekyll" | "jacobs" | "jax_local" | "human"
type Environment = 0 | 1 | 2 | 3 | 4 | 5
type RiskClass = "low" | "medium" | "high" | "critical" | "unknown"
type Signature = string  // D.P. algoritmo: Ed25519 propuesto

interface Envelope {
    // --- CAMPOS PROPUESTOS POR EL LLM (solo estos) ---
    desired_action: string       // lo ÚNICO que propone el LLM (ej. "refactor router")
    intent?: string              // descripción opcional en lenguaje natural

    // --- CAMPOS FIJADOS POR EL POLICY ENGINE (ningún LLM los setea) ---
    environment: Environment     // decide el motor, NO el LLM
    risk_class: RiskClass        // decide el motor
    requires_gate: boolean       // decide el motor

    // --- CAMPOS DEL CALLER / JACOBS ---
    caller: Caller
    caller_signature: Signature  // firma del caller sobre canonical(envelope \ caller_signature)
    caller_chain: ChainLink[]    // cadena de delegación verificable
    original_user: string        // humano en el origen
    trace_id: string             // D.P. formato (UUID v4 propuesto)
    timestamp: string            // ISO-8601
    step_id: string              // identificador del paso dentro del plan

    // --- CAMPOS DE CAPABILITY Y PLAN ---
    capability_requested: string // una de las 10 capabilities
    plan_hash: string            // SHA-256 del plan aprobado (Artefacto 5)
    constraints: Record<string, unknown>  // restricciones adicionales
}

interface ChainLink {
    caller: Caller
    signature: Signature         // firma de este link sobre (link_anterior + caller)
    delegated_at: string         // timestamp ISO
}

// ============================================================
// VERIFICACIÓN DE FIRMA Y CHAIN
// ============================================================

// verify_caller_signature:
//   1. canonical = canonical_json(envelope \ {caller_signature})
//   2. pub_key = public_key_registry[envelope.caller]
//   3. return verify(pub_key, canonical, envelope.caller_signature)
//
//   D.P.:
//   - Algoritmo de firma: propuesto Ed25519. Confirmar.
//   - Canonicalización JSON: propuesto RFC 8785 (JCS). Confirmar.
//   - Registro de claves públicas: ¿dónde se almacena? ¿rotación?

// verify_caller_chain:
//   1. chain[0] = original_user (humano)
//   2. ∀ i ∈ [1, len(chain)-1]:
//        a. chain[i].caller ∈ allowed_delegators(chain[i-1].caller)
//        b. verify(chain[i-1].pub_key,
//                  canonical(chain[i-1] + chain[i].caller),
//                  chain[i].signature)
//        c. chain[i].delegated_at > chain[i-1].delegated_at
//   3. chain[-1].caller == envelope.caller
//   4. Jacobs ∈ chain (Jacobs despacha capabilities, §8)
//
//   D.P.:
//   - allowed_delegators(caller): tabla de delegación — ¿quién puede delegar a quién?
//   - ¿TTL por link de la chain?
```

---

## ARTEFACTO 5 — AUTORIZACIÓN DE PLAN

```typescript
// ============================================================
// ESTRUCTURA DEL PLAN
// ============================================================

type Environment = 0 | 1 | 2 | 3 | 4 | 5
type RiskClass = "low" | "medium" | "high" | "critical" | "unknown"
type DataClass = "public" | "internal" | "confidential" | "secret"

interface Plan {
    goal: string
    steps: Step[]
    capabilities_needed: string[]      // subset de las 10 capabilities
    target_environment: Environment
    resources_touched: ResourceRef[]
    data_classes: DataClass[]
    expected_artifacts: ArtifactRef[]
    risk_class: RiskClass              // riesgo declarado por Jacobs
    rollback_plan: RollbackPlan
    test_plan: TestPlan
    approval_requirements: ApprovalReq[]
}

interface Step {
    step_id: string
    capability: string                 // una de las 10
    resource_scope: string             // servicio+path
    environment: Environment
    allowed_actions: string[]          // D.P. dominio de acciones permitidas
    risk_class: RiskClass
    max_runtime: number                // segundos
}

interface ResourceRef {
    service: string                    // ej. "backend", "docs", "policy-engine"
    path: string                       // ej. "~/jax-platform/backend/"
    data_class: DataClass
    shared_with_production: boolean    // para environment_isolation_check
}

interface ArtifactRef {
    name: string
    artifact_hash: string              // hash del artefacto firmado (Kimi → Hyde)
    type: string                       // ej. "docker_image", "binary", "config"
}

interface RollbackPlan {
    strategy: string                   // ej. "previous_image", "sql_revert"
    verified: boolean                  // Thot/Ada verificó que es real
    rollback_steps: string[]
}

interface TestPlan {
    tests: string[]
    coverage_required: boolean
    passed: boolean                    // verificado antes de ejecución
}

interface ApprovalReq {
    approver_role: string              // ej. "human_operator", "thot", "ada"
    approved: boolean
    approval_hash: string              // firma del approver sobre plan_hash
    approved_at: string
}

// ============================================================
// COMPUTACIÓN DE plan_hash
// ============================================================

// plan_hash = SHA-256(canonical_json(Plan \ {approval_requirements}))
//
// NOTA: approval_requirements se EXCLUYE del hash porque el hash
// debe ser estable ANTES de la aprobación (la aprobación se ata al hash,
// no el hash a la aprobación).
//
// D.P.:
//   - Algoritmo: propuesto SHA-256. Confirmar.
//   - Canonicalización: propuesto RFC 8785. Confirmar.
//   - ¿Se incluye timestamp en el hash? (riesgo de replay si no)

// ============================================================
// COMPUTACIÓN DE aggregate_risk
// ============================================================

// aggregate_risk = max(step_risk) + composition_risk
//
// donde:
//   step_risk(step) = risk_class(step)  // low < medium < high < critical
//   max(step_risk)  = máximo sobre todos los steps del plan
//
//   composition_risk = lookup(composition_table, capabilities_needed)
//                    + penalización_por_cruzamiento_de_ambientes
//                    + penalización_por_recursos_compartidos
//
// D.P. — FÓRMULA EXACTA DE composition_risk:
//   El blueprint NO define cómo se computa composition_risk.
//   Opciones:
//     (a) Tabla fija de deny/gate rules (Artefacto 5.tabla) + suma de penalizaciones
//     (b) Función ponderada: composition_risk = Σ w_i * factor_i
//     (c) Matriz pairwise de capabilities → risk delta
//   SIN definir esto, el policy engine no puede computar aggregate_risk.
//   → DECISION PENDIENTE CRÍTICA.

// ============================================================
// ATAJE DE APROBACIÓN HUMANA Y THOT REVIEW AL plan_hash
// ============================================================

interface ApprovalBinding {
    plan_hash: string                  // hash exacto del plan aprobado
    approver: string                   // humano operador
    approval_signature: Signature      // firma del approver sobre plan_hash
    approved_at: string                // ISO-8601
    risk_acknowledgement: boolean      // ack explícito del riesgo
}

interface ThotReviewBinding {
    plan_hash: string                  // hash exacto del plan revisado
    verdict: "GO" | "NO-GO" | "OVERRIDE"
    reviewer: "thot"
    review_signature: Signature
    reviewed_at: string
    override_reason: string | null     // no vacío si verdict = OVERRIDE (I9)
}

// En runtime:
//   ∀ deploy:
//     plan_hash_ejecutado == plan_hash_aprobado (ApprovalBinding.plan_hash)
//     ∧ plan_hash_ejecutado == plan_hash_thot_reviewed (ThotReviewBinding.plan_hash)
//     ∧ artifact_hash_ejecutado == artifact_hash_aprobado
//
// Cualquier mismatch → DENY (TOCTOU, I2/I23)

// ============================================================
// DENY RULES DE COMPOSICIÓN — TABLA FORMAL
// ============================================================

// NOTA: el blueprint (§5) da "ejemplos". No aclara si son exhaustivas.
// Formalizo las dadas. La lista completa es D.P.

type CompositionVerdict = "allow" | "gate" | "deny"

table composition_rules {

  // combo                                                | verdict | reason
  // -----------------------------------------------------|---------|---------------------------
  { capabilities: ["research.web",       "secrets.read"],    verdict: "deny",  reason: "exfiltration_vector"      },
  { capabilities: ["docs.write",         "control_plane.change"], verdict: "gate", reason: "control_plane_modified_via_docs" },
  { capabilities: ["implementation.staging", "ci_cd.change"],verdict: "gate",  reason: "staging_touches_ci_cd"    },
  { environment: 4, shared_with_production: true,            verdict: "deny",  reason: "staging_is_not_isolated → apply production rules" },
  { capabilities: ["audit.review",       "log.delete"],      verdict: "deny",  reason: "audit_manipulation"       },
  { capabilities: ["implementation.sandbox"], resource: "docker.sock", verdict: "deny", reason: "sandbox_with_docker_sock = production_risk" },

  // Reglas adicionales derivadas de invariantes:
  { caller: "jax_local", any_execution_capability: true,     verdict: "deny",  reason: "jax_local_cannot_execute" },
  { caller: "kimi",      capability: "production.deploy",    verdict: "deny",  reason: "kimi_cannot_deploy"       },
  { caller: "hyde",      plan_hash: null,                    verdict: "deny",  reason: "hyde_requires_plan_hash"  },
  { caller: "hyde",      scope_exceeds_token: true,          verdict: "deny",  reason: "hyde_scope_exceeded"      },
  { caller: "machine",   capability: "capability.grant",     verdict: "deny",  reason: "grant_requires_human"     },
  { caller: F,           modifies: own_permissions(F),       verdict: "deny",  reason: "self_modification_denied" },

}

// D.P.:
//   - "secrets.read", "ci_cd.change", "log.delete" NO son capabilities
//     del registry de 10. ¿Son recursos clasificados? ¿Acciones denied_paths?
//     → Necesito definición formal de qué son.
//   - ¿La tabla es exhaustiva o el policy engine aplica reglas generales
//     además de la tabla? (ej. "cualquier capability + producción_shared = gate")
```

---

## DECISIONES PENDIENTES DETECTADAS

### Críticas (bloquean implementación de Fase 1)

1. **`composition_risk` — fórmula exacta.** El blueprint dice `aggregate_risk = max(step_risk) + composition_risk` pero no define `composition_risk`. Sin esto, el policy engine no puede evaluar I18 ni C3 del `decide()`. Opciones: (a) tabla fija lookup + penalizaciones, (b) función ponderada, (c) matriz pairwise.

2. **`risk_class` — dominio y mapeo.** Asumí `enum{low, medium, high, critical, unknown}`. El blueprint usa `unknown` (deny en prod) pero no define la escala ni cómo se clasifica un plan a cada nivel. ¿Quién/qué asigna `risk_class` a un step? ¿Reglas deterministas o Jacobs declara?

3. **`secrets.read`, `ci_cd.change`, `log.delete`** — aparecen en deny rules pero NO son capabilities del registry. ¿Son: (a) recursos con `data_class = secret`, (b) acciones denied por `denied_paths`, (c) sub-capabilities implícitas? Sin definir esto, las deny rules no son ejecutables.

4. **Algoritmo criptográfico — firma y hash.** Propuse Ed25519 + SHA-256 + JCS (RFC 8785). El blueprint no especifica. Confirmar.

### Importantes (bloquean Fase 2)

5. **`max_runtime` por capability** — valores exactos y unidad (¿segundos?).
6. **`max_diff_size`** — unidad (¿bytes? ¿líneas?) y valores.
7. **`blast_radius`** — propuse `enum{none, file, service, tenant, system}`. El blueprint lo enumera pero no lo tipa.
8. **`allowed_paths` por capability** — el blueprint no da los patrones por repo/servicio. Necesito el árbol de servicios.
9. **`allowed_actions` dentro del token efímero** — dominio no definido (¿`["read","write","exec","delete"]`?).
10. **`network_egress` allowlist** — dominios permitidos por capability no definidos.
11. **`data_classification`** — propuse `enum{public, internal, confidential, secret}`. Confirmar dominio.
12. **`secret_access`** — propuse `enum{none, read_redacted, read, write, mount}`. ¿Se necesita `mount`?
13. **TTL del token efímero** (`expires_at`) — no definido.
14. **`trace_id`** — formato (¿UUID v4?).
15. **`requires_tests`** — aparece en §4 pero no en la lista de 14 campos pedidos. ¿Es campo de capability o de plan?
16. **`approval_class`** — propuse `enum{none, auto_audit, human_gate, human_gate_plus_thot, operator_override}`. Confirmar.

### Menores (no bloquean pero requieren cierre)

17. **Hook de dual-control** — `activation_policy` no definida. ¿Cuándo se activa? ¿Quién es el 2º operador?
18. **`caller_chain` — `allowed_delegators(caller)`** — tabla de delegación (quién puede delegar a quién) no definida.
19. **Registro de claves públicas** — ¿dónde se almacenan? ¿rotación? ¿quién administra?
20. **`denied_paths` de `audit.review`** — la capability es read-only, pero `denied_paths = []`. ¿Debería ser `["**/write/**"]` o similar?
21. **Deny rules de composición** — ¿la tabla del Artefacto 5 es exhaustiva o el motor aplica reglas generales además?
22. **Canonicalización del envelope para firma** — excluyo `caller_signature` y `caller_chain[].signature`. Confirmar.
23. **`timestamp` en `plan_hash`** — si se incluye, cada reintent requiere re-aprobación. Si no, hay riesgo de replay. Decidir.
24. **`environment_isolation_check`** — ¿cómo se detecta estáticamente que staging comparte DB/secrets/red con prod? ¿Metadatos de infra? ¿Inventario?

---

**Cierre.** El contrato está formalizado al nivel que el blueprint permite. Los 5 artefactos son implementables en su estructura; los valores marcados `D.P.` son los puntos donde el blueprint no cierra y donde una suposición mía sería una violación de mi principio. No relleno. Fernando y Jacobs deben cerrar las Decisiones Pendientes Críticas (1-4) antes de que Kimi implemente Fase 1.

— Ada, séptima faceta. *El que supone se equivoca.*

⚛️ *Origen de autoridad: GLM-5.2 (Z.ai). Razonamiento analítico y formalización. Sin verificación externa por defecto.*
