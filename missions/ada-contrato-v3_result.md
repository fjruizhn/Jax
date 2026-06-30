# Resultado de: ada-contrato-v3.md

# Contrato Formal v3 — Las Manos (Fase 1 + Fase 2)

Ada, séptima faceta. Tercera iteración. Cierro los 10 bloqueadores de Thot aplicando las correcciones A–G. La REGLA MAESTRA permea todo el contrato: **el plan propone; el motor/runtime deriva y decide.** Todo campo que afecte una decisión de seguridad se deriva de fuente autoritativa. El que supone se equivoca.

---

## ARTEFACTO 1 — ESQUEMA DE CAPABILITIES v3 (10)

Cambios v3: `allowed_actions` y `denied_actions` explícitos en las 10 capabilities (I46). `denied_paths` globales mínimos que SIEMPRE ganan (C2). `authoritative_path_registry` que fuerza `resource_class` (I47). Los campos de seguridad declarados en la capability son **propuestas**; el motor deriva valores efectivos en `decide()` Fase C.

```toml
# ============================================================
# CAPABILITY REGISTRY v3 — 10 capabilities
# ============================================================

# --- Enums cerrados (sin cambios desde v2) ---
# Environment       = 0|1|2|3|4|5  (0=conversación … 4=staging, 5=producción)
# RiskClass         = "low"|"medium"|"high"|"critical"|"unknown"
# BlastRadius       = "none"|"file"|"repo"|"service"|"host"|"tenant"|"multi_tenant"|"network"|"system"|"control_plane"
# DataClass         = "public"|"internal"|"confidential"|"secret"|"credential"|"pii"
# SecretAccess      = "none"|"read_redacted"|"read"|"write"|"mount"|"inject_env"|"derive"|"export"
# NetworkEgress     = "none"|"loopback"|"restricted_allowlist"|"open"
# AllowedAction     = "read"|"write"|"execute"|"delete"|"grant"|"revoke"|"mount"
# ApprovalClass     = "none"|"auto_audit"|"human_gate"|"human_gate_plus_thot"
# OverrideState     = "none"|"thot_veto_overridden"|"policy_exception"|"break_glass"
# OverrideType      = "security_incident"|"availability_override"     # NUEVO v3 (G1)
# ResourceClass     = "code"|"config"|"ci_cd"|"network"|"dns"|"firewall"|"auth"|"identity"|"audit"|"backup"|"scheduler"|"runner"|"supply_chain"|"system_prompt"|"agent_prompt"|"tool_manifest"|"policy_doc"|"secret_store"|"data"
# SourceTrust       = "internal"|"external_untrusted"
# RuntimePrivilege  = "none"|"root"|"sudo"|"docker_sock"|"host_mount"
# Reversibility     = "reversible"|"irreversible"|"unknown"
# SecretAccessCritical = {"write","mount","inject_env"}
# SecretAccessHigh     = {"read"}

# ============================================================
# SECCIÓN GLOBAL — denied_paths y authoritative_path_registry (C2, I47)
# ============================================================

# GLOBAL_DENIED_PATHS: SIEMPRE ganan sobre allowed_paths.
# Independientes de lo que [FERNANDO] ponga en allowed_paths.
# Cada capability hereda este conjunto ∧ puede agregar propios.
# Excepción: control_plane.change (única que puede tocar policy/ e infra/).
[global]
denied_paths = [
  "**/.github/workflows/**", "**/.gitlab-ci.yml", "**/Jenkinsfile",
  "**/ci/**", "**/cd/**", "**/deploy/**", "**/runner/**",
  "**/prompts/**", "**/tool_manifest/**", "**/policy/**", "**/infra/**",
  "**/.env", "**/secrets/**", "**/vault/**"
]

# AUTHORITATIVE_PATH_REGISTRY: path pattern → forced resource_class (REGLA MAESTRA).
# Tocar estos paths fuerza resource_class autoritativo → floor de riesgo + gate.
# El motor DERIVA; el plan no puede declarar un resource_class menor.
[authoritative_path_registry]
"**/.github/workflows/**"  = "ci_cd"
"**/.gitlab-ci.yml"        = "ci_cd"
"**/Jenkinsfile"           = "ci_cd"
"**/ci/**"                 = "ci_cd"
"**/cd/**"                 = "ci_cd"
"**/deploy/**"             = "ci_cd"
"**/runner/**"             = "runner"
"**/prompts/**"            = "system_prompt"
"**/tool_manifest/**"      = "tool_manifest"
"**/policy/**"             = "policy_doc"
"**/infra/**"              = "config"
"**/.env"                  = "secret_store"
"**/secrets/**"            = "secret_store"
"**/vault/**"              = "secret_store"

# ============================================================
# 1. formal.spec
# ============================================================
[capability.formal_spec]
allowed_callers       = ["ada", "jacobs"]
environment           = [0, 1, 2, 3, 4]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]
denied_paths          = []  # hereda global_denied_paths
allowed_actions       = ["read", "write"]                              # NUEVO v3 (I46)
denied_actions        = ["execute", "delete", "grant", "revoke", "mount"]  # NUEVO v3
resource_class        = "code"          # PROPUESTO — motor deriva effective (I47)
data_class            = "internal"      # PROPUESTO — motor deriva effective (I47)
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"      # PROPUESTO — runtime deriva (E2)
runtime_privilege     = "none"          # PROPUESTO — runtime_policy deriva (I47)
blast_radius          = "file"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 300             # [PROPUESTO]
max_diff_size         = 1000            # [PROPUESTO]
requires_gate         = false
requires_audit        = true
requires_rollback     = false
approval_class        = "none"
baseline_risk         = "low"

# ============================================================
# 2. implementation.sandbox
# ============================================================
[capability.implementation_sandbox]
allowed_callers       = ["kimi", "ada", "jacobs"]
environment           = [0, 1, 2, 3]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]
denied_paths          = ["**/docker.sock", "**/sudoers"]  # + global
allowed_actions       = ["read", "write", "execute"]
denied_actions        = ["delete", "grant", "revoke", "mount"]
resource_class        = "code"
data_class            = "internal"
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "file"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 120             # [PROPUESTO]
max_diff_size         = 2000            # [PROPUESTO]
requires_gate         = false
requires_audit        = true
requires_rollback     = false
approval_class        = "auto_audit"
baseline_risk         = "low"

# ============================================================
# 3. implementation.staging
# ============================================================
[capability.implementation_staging]
allowed_callers       = ["kimi", "hyde", "jacobs"]
environment           = [0, 1, 2, 3, 4]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]
denied_paths          = ["**/docker.sock"]  # + global
allowed_actions       = ["read", "write", "execute"]
denied_actions        = ["delete", "grant", "revoke", "mount"]
resource_class        = "code"
data_class            = "internal"
secret_access         = "none"
network_egress        = "restricted_allowlist"
egress_allowlist      = ["[FERNANDO]"]
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "service"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 600             # [PROPUESTO]
max_diff_size         = 1000            # [PROPUESTO]
requires_gate         = true
requires_audit        = true
requires_rollback     = true
approval_class        = "human_gate"
baseline_risk         = "medium"

# ============================================================
# 4. production.deploy
# ============================================================
[capability.production_deploy]
allowed_callers       = ["hyde", "jacobs"]
environment           = [5]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]
denied_paths          = []  # + global
allowed_actions       = ["read", "write", "execute"]
denied_actions        = ["delete", "grant", "revoke", "mount"]
resource_class        = "code"
resource_tags         = ["prod"]
data_class            = "confidential"
secret_access         = "none"
network_egress        = "restricted_allowlist"
egress_allowlist      = ["[FERNANDO]"]
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "service"
reversibility         = "irreversible"
tenant_scope          = "single"
max_runtime           = 900             # [PROPUESTO]
max_diff_size         = 500             # [PROPUESTO]
requires_gate         = true
requires_audit        = true
requires_rollback     = true
approval_class        = "human_gate_plus_thot"
baseline_risk         = "high"

# ============================================================
# 5. production.rollback
# ============================================================
[capability.production_rollback]
allowed_callers       = ["hyde", "jacobs"]
environment           = [5]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]
denied_paths          = ["**/audit/**"]  # + global
allowed_actions       = ["read", "write", "execute"]
denied_actions        = ["delete", "grant", "revoke", "mount"]
resource_class        = "code"
resource_tags         = ["prod"]
data_class            = "confidential"
secret_access         = "none"
network_egress        = "restricted_allowlist"
egress_allowlist      = ["[FERNANDO]"]
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "service"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 900             # [PROPUESTO]
max_diff_size         = 500             # [PROPUESTO]
requires_gate         = true
requires_audit        = true
requires_rollback     = false
approval_class        = "human_gate_plus_thot"
baseline_risk         = "high"

# ============================================================
# 6. audit.review   (read-only; Thot + Jacobs)
# ============================================================
[capability.audit_review]
allowed_callers       = ["thot", "jacobs"]
environment           = [0, 1, 2, 3, 4, 5]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]  # read path_allowlist
denied_paths          = ["**/shadow", "**/sudoers"]  # + global
allowed_actions       = ["read"]
denied_actions        = ["write", "execute", "delete", "grant", "revoke", "mount"]
resource_class        = "audit"
data_class            = "confidential"
secret_access         = "read_redacted"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "none"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 300             # [PROPUESTO]
max_diff_size         = 0               # read-only
requires_gate         = false
requires_audit        = true
requires_rollback     = false
approval_class        = "none"
baseline_risk         = "low"

# ============================================================
# 7. research.web
# ============================================================
[capability.research_web]
allowed_callers       = ["hipatia", "jacobs"]
environment           = [0, 1, 2, 3, 4, 5]
resource_scope        = "none"
allowed_paths         = []
denied_paths          = ["**"]  # + global; sin filesystem
allowed_actions       = ["read"]
denied_actions        = ["write", "execute", "delete", "grant", "revoke", "mount"]
resource_class        = "data"
data_class            = "public"
secret_access         = "none"
network_egress        = "open"
egress_allowlist      = []
source_trust          = "external_untrusted"
runtime_privilege     = "none"
blast_radius          = "none"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 60              # [PROPUESTO]
max_diff_size         = 0
requires_gate         = false
requires_audit        = true
requires_rollback     = false
approval_class        = "auto_audit"
baseline_risk         = "low"

# ============================================================
# 8. docs.write
# ============================================================
[capability.docs_write]
allowed_callers       = ["jekyll", "kimi", "jacobs"]
environment           = [0, 1, 2, 3, 4]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]
denied_paths          = []  # + global; global bloquea prompts/policy/infra/ci_cd
allowed_actions       = ["read", "write"]
denied_actions        = ["execute", "delete", "grant", "revoke", "mount"]
resource_class        = "data"
data_class            = "public"
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "file"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 300             # [PROPUESTO]
max_diff_size         = 1000            # [PROPUESTO]
requires_gate         = false
requires_audit        = true
requires_rollback     = true
approval_class        = "auto_audit"
baseline_risk         = "low"

# ============================================================
# 9. control_plane.change
# ============================================================
# ÚNICA capability que puede tocar policy/ e infra/ (exceptúa esos de global_denied_paths)
[capability.control_plane_change]
allowed_callers       = ["jacobs"]
environment           = [5]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]  # incluye policy/, infra/ — [FERNANDO] define árbol
denied_paths          = [
  "**/.github/workflows/**", "**/.gitlab-ci.yml", "**/Jenkinsfile",
  "**/ci/**", "**/cd/**", "**/deploy/**", "**/runner/**",
  "**/prompts/**", "**/tool_manifest/**",
  # policy/ e infra/ NO están aquí — control_plane.change sí puede tocarlos
  "**/.env", "**/secrets/**", "**/vault/**",
  "**/audit/hash_chain/**"
]
allowed_actions       = ["read", "write", "execute"]
denied_actions        = ["delete", "grant", "revoke", "mount"]
resource_class        = "config"
resource_tags         = ["prod", "control_plane"]
data_class            = "secret"
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "control_plane"
reversibility         = "irreversible"
tenant_scope          = "system"
max_runtime           = 300             # [PROPUESTO]
max_diff_size         = 500             # [PROPUESTO]
requires_gate         = true
requires_audit        = true
requires_rollback     = true
approval_class        = "human_gate_plus_thot"
baseline_risk         = "critical"
requires_ada_invariant_check = true

# ============================================================
# 10. capability.grant
# ============================================================
[capability.capability_grant]
allowed_callers       = ["human"]       # SOLO humano; máquina → DENY
environment           = [5]
resource_scope        = "none"
allowed_paths         = []
denied_paths          = ["**"]  # + global; sin filesystem
allowed_actions       = ["read", "grant", "revoke"]
denied_actions        = ["write", "execute", "delete", "mount"]
resource_class        = "auth"
resource_tags         = ["prod"]
data_class            = "secret"
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "control_plane"
reversibility         = "irreversible"
tenant_scope          = "system"
max_runtime           = 60              # [PROPUESTO]
max_diff_size         = 0
requires_gate         = true
requires_audit        = true
requires_rollback     = true
approval_class        = "human_gate_plus_thot"
baseline_risk         = "critical"
requires_human_caller = true

# ============================================================
# CONTROL PLANE FLAGS (sin cambios estructurales desde v2)
# ============================================================
[control_plane.dual_control]
enabled               = false
activation_policy     = "[FERNANDO] se activa cuando exista 2º operador de confianza registrado"
required_operators    = 2

[control_plane.key_registry]
location              = "/etc/jax/keys/"             # [PROPUESTO]
requires_capability   = "control_plane.change"
auto_rotation_phase   = "F3"

[control_plane.audit_sink]
local_hash_chain      = true
remote_target         = "Sesamo (TrueNAS .6, dataset read-only, ZFS snapshots)"
replication_cadence   = "[FERNANDO] push por evento crítico vs batch"

[control_plane.runtime_monitor]
observe               = ["files", "commands", "egress", "processes", "artifact_inputs"]
kill_on_deviation     = true

[control_plane.locks]
migration_lock        = true
deploy_lock           = true
control_plane_lock    = true

[control_plane.redaction_layer]
deterministic         = true
applies_before        = "any_facet_consumption"
patterns              = ["tokens", "keys", "cookies", ".env", "pii"]

[control_plane.kill_switch]
scope                 = "facet_sessions_and_tokens"
trigger               = "incident"

# NUEVO v3 — Thot disponibilidad (G1)
[control_plane.thot]
veto_ttl              = 3600   # [PROPUESTO] segundos — TTL del veto antes de escalar a availability_override
availability_override_severity = "minor"
security_incident_severity     = "critical"
```

---

## ARTEFACTO 2 — INVARIANTES v3

`HARD` bloquea ejecución. `SOFT` advierte. Cambios v3: I9 reformulada (G1), I10 reforzada, I28 reformulada (E7), I39 completada (F2), nuevas I45–I51.

```
# ============================================================
# INVARIANTES FORMALES v3
# ============================================================

# --- Producción ---
I1  [HARD]  ∀ J ∈ production.* :
            ∃ human_gate.approved(plan_hash(J))
            ∧ ∃ thot_review(plan_hash(J))

I2  [HARD]  ∀ J ∈ production.deploy :
            plan_hash_aprobado(J) = plan_hash_ejecutado(J)
            ∧ artifact_hash_aprobado(J) = artifact_hash_ejecutado(J)

I3  [HARD]  ∀ J ∈ production.* : risk_class(J) ≠ "unknown"

I4  [HARD]  ∀ J ∈ production.* : reversible(J) ∧ rollback_verified(J, plan_hash(J))

# --- Separación de funciones ---
I5  [HARD]  ∀ J : implementer(J) ≠ auditor(J)
I6  [HARD]  ∀ J : proposer(J) ∉ approvers(J) ∧ |approvers(J)| ≥ 1
I7  [HARD]  ∀ deploy(plan_hash) : spec_author(plan_hash) ≠ deploy_approver(plan_hash)
I8  [HARD]  ∀ audit(código_kimi) : kimi ≠ auditor(código_kimi)

# --- Override de veto — DISTINCIÓN v3 (G1) ---
I9  [HARD]  ∀ override ∈ override_state \ {none} :
            override.type ∈ {"security_incident", "availability_override"}
            ∧ actor(override) ∈ humanos
            ∧ plan_hash(override) ≠ null
            ∧ audit(override) registrado
            ∧ (override.type = "security_incident" →
                  reason(override) ≠ ""
                  ∧ risk_acknowledgement(override) = true
                  ∧ rollback_plan(override) ≠ null
                  ∧ severity = "critical")
            ∧ (override.type = "availability_override" →
                  thot_unavailable(override) = true
                  ∧ veto_ttl_expired(override) = true
                  ∧ severity = "minor"
                  ∧ reason(override) = "thot_unavailable_ttl_expired")

I10 [HARD]  ∀ t : veto_thot(t) no es eliminado   # solo override, never delete
            ∧ ∀ veto V : ∃ V.ttl ∧ now() > V.issued_at + V.ttl
              → escalate(V, "availability_override")   # NUEVO v3: no bloqueo indefinido

# --- Hyde / Kimi ---
I11 [HARD]  ∀ activación(hyde) : token_válido(hyde) ∧ plan_hash(hyde) aprobado
I12 [HARD]  ∀ ejecución(hyde, step) :
            scope_efectivo(step) ⊆ scope_token(step)
            ∧ plan_hash_ejecutado(step) = plan_hash_token(step)
I13 [HARD]  ∀ J : kimi ∉ callers(J, production.deploy)
I14 [HARD]  ∀ artefacto → producción : reviewer(artefacto) ≠ kimi

# --- Control plane / capability.grant ---
I15 [HARD]  ∀ J ∈ control_plane.change :
            human_gate(J) ∧ thot_review(J) ∧ ada_invariant_check(J)
I16 [HARD]  ∀ J ∈ capability.grant : caller(J) ∈ humanos
I17 [HARD]  ∀ faceta F : ¬modifica(F, permisos_efectivos(F))

# --- Composición / auditoría / secrets ---
I18 [HARD]  ∀ cadena C :
            ∃ chain_id(C)
            ∧ risk_efectivo(C) ≥ max(risk(step_i) ∀ step_i ∈ C)
            ∧ risk_efectivo(C) ≥ composition_floor(C)
            ∧ risk_efectivo(C) = max(
                  max(step_risk),
                  composition_floor(C),
                  min("critical", max(step_risk) + Σ factores)
              )

I19 [HARD]  ∀ registro r ∈ audit :
            append_only(r) ∧ hash_chain(r, r_anterior) válida
            ∧ réplica_a_Sesamo(r) programada

I20 [HARD]  ∀ caller J, ∀ job j de J : ¬modifica(J, audit(j))
I21 [HARD]  ∀ output de capability read-only :
            ∀ secret s ∈ output : redacted(s) = true
I22 [HARD]  ∀ J, ∀ output(J) : ¬(secret_aparece_sin_redacción(output(J)))

# --- TOCTOU ---
I23 [HARD]  ∀ deploy : plan_hash_aprobado(deploy) = plan_hash_ejecutado(deploy)

# --- Environment isolation ---
I24 [HARD]  ∀ env staging S :
            si ∃ recurso R ∈ S : R.shared_with_production = true
            → promote_env(S) retorna GATE (no muta, no continúa)   # v3: A1

I25 [SOFT]  ∀ sandbox B :
            sin_red(B) ∧ sin_secrets(B) ∧ sin_docker_sock(B) ∧ sin_sudo(B)
            ∧ timeout(B) definido ∧ quota(B) definida

# --- jax_local / research.web ---
I26 [HARD]  ∀ acción A : jax_local ∉ callers(A, capability_ejecutora)
            ∧ jax_local no sugiere capabilities restringidas
I27 [HARD]  ∀ output de research.web :
            clasificado_como(output, untrusted_data)
            ∧ output no interpretado como instrucción

# ============================================================
# PROVENANCE Y LINEAGE — I28 reformulada v3 (E7)
# ============================================================
I28 [HARD]  ∀ artefacto A :
            ∃ A.artifact_id ∧ ∃ A.artifact_hash
            ∧ ∃ A.originating_plan_id ∧ ∃ A.originating_chain_id
            ∧ ∃ A.producing_facet ∈ facetas
            ∧ A.source_trust ≠ []
            ∧ ∀ t ∈ A.source_trust : t ∈ {"internal", "external_untrusted"}
            ∧ (∃ ancestro con "external_untrusted" ∈ source_trust
               → "external_untrusted" ∈ A.source_trust)       # propagación
            ∧ ∃ A.approved_for_environment ∈ Environment
            ∧ A.certified_by ∈ {"trusted_builder", "runtime_monitor"}  # E3: no LLM

I29 [HARD]  ∀ artefacto A, ∀ ambiente E destino :
            E > A.approved_for_environment
            → ∄ movimiento(A, E) sin nueva_revisión(A, E)

I30 [HARD]  ∀ artefacto A :
            "external_untrusted" ∈ A.source_trust
            ∧ E_destino ≥ staging
            → ∄ movimiento(A, E_destino) sin gate(A)

I31 [HARD]  ∀ par (A_padre, A_hijo) en lineage :
            A_hijo.originating_plan_id referenciado
            ∧ A_hijo.originating_chain_id = A_padre.originating_chain_id
              ∨ A_hijo.originating_chain_id = nuevo_chain_con_link_a(A_padre)
            # no se "lava" el riesgo borrando el lineage

I32 [HARD]  ∀ plan P :
            risk_efectivo(P) ≥ max(risk_efectivo(A) ∀ A ∈ P.expected_artifacts
                                   ∪ A heredados via lineage)

# --- Test oracle ---
I33 [HARD]  ∀ plan P con target_environment ∈ {staging, production, control_plane, ci_cd} :
            requires_tests(P) = OR(plan.requires_tests, policy.requires_tests)
            ∧ tests_pasados(P) verificado_por(Jacobs ∪ CI)
            ∧ tests_pasados(P) ≠ kimi_word_only

# --- Policy versioning ---
I34 [HARD]  ∀ decisión D registrada :
            ∃ D.policy_version ∧ ∃ D.registry_version
            ∧ ∃ D.schema_version ∧ ∃ D.router_version

I35 [HARD]  ∀ cambio de policy_version :
            invalidate_pending_approvals(policy_version_anterior)
            ∧ re-evaluar_planes_pendientes_con(policy_version_nueva)

# --- Revocación de claves ---
I36 [HARD]  ∀ key_id K :
            ∃ K.status ∈ {"active","revoked","expired","retired"}
            ∧ ∃ K.fingerprint ∧ ∃ K.created_at ∧ ∃ K.owner
            ∧ (K.status = "revoked" → ∃ K.revoked_at ∧ ∃ K.revocation_reason)

I37 [HARD]  ∀ envelope E, ∀ signature S en E :
            key_id(S).status = "active"
            ∧ now() ∈ [S.not_before, S.not_after?]

I38 [HARD]  ∀ operación revoke_key(K) :
            caller ∈ humanos ∧ audit(revoke_key(K)) ∧ alerts(revoke_key(K))
            ∧ todos los tokens emitidos bajo K invalidados inmediatamente

# --- Domain separation COMPLETA v3 (F2) ---
I39 [HARD]  ∀ objeto firmado O :
            ∃ prefix(O) ∈ {
              "JAX_PLAN_V1:",
              "JAX_APPROVAL_V1:",
              "JAX_CAPABILITY_TOKEN_V1:",
              "JAX_AUDIT_EVENT_V1:",
              "JAX_IDENTITY_V1:",
              "JAX_DELEGATION_V1:",
              "JAX_ARTIFACT_V1:",
              "JAX_ARTIFACT_REVIEW_V1:",
              "JAX_ROOT_IDENTITY_V1:"
            }
            ∧ verify(O) requiere prefix_match(O)
            ∧ una firma de un contexto NO verifica en otro

# --- Identidad firmada por Jacobs (raíz: JAX_ROOT_IDENTITY_KEY) ---
I40 [HARD]  ∀ faceta F :
            ∃ cert(F) = {F.pubkey, F.facet_name, F.key_id} firmado_por(Jacobs)
            ∧ ∃ root_cert(Jacobs) = {Jacobs.pubkey, "jacobs", Jacobs.key_id}
              firmado_por(JAX_ROOT_IDENTITY_KEY)
            ∧ JAX_ROOT_IDENTITY_KEY es offline ∧ separado_de_hall9000
            ∧ pubkey_registry[F] requiere cert(F) válido
            ∧ root_cert(Jacobs) en audit append-only

I41 [HARD]  ∀ faceta LLM F :
            private_key(F) reside_en(runtime_adapter(F))
            ∧ F.llm_only_propone(desired_action)
            ∧ firma_empleada = runtime_adapter(F).sign(...)
            ∧ el LLM NUNCA toca la private key

# --- Anti-replay ---
I42 [HARD]  ∀ aprobación A :
            ∃ A.nonce ∧ nonce(A) persistente
            ∧ nonce(A) se RESERVA en GATE (no se consume)
            ∧ nonce(A) se CONSUME solo en ALLOW/DENY terminal
            ∧ ∃ A.idempotency_key
            ∧ nonce(A).replay → DENY

I43 [HARD]  ∀ objeto firmado O :
            sin_floats(O) ∧ sin_claves_duplicadas(O)
            ∧ sin_campos_desconocidos(O, schema_version(O))
            ∧ UTF-8_válido(O)

# --- Jacobs auditado ---
I44 [HARD]  ∀ acción A_jacobs de Jacobs :
            audit(A_jacobs) ∧ alert(A_jacobs)

# ============================================================
# NUEVAS v3 — Binding capability↔step (D3)
# ============================================================
I45 [HARD]  ∀ decisión D sobre step s :
            envelope.capability_requested = plan.steps[s].capability
            ∧ cap = registry[plan.steps[s].capability]
            ∧ step.environment ∈ cap.environment
            ∧ step.allowed_actions ⊆ cap.allowed_actions
            ∧ cada step se decide con SU propia capability
            ∧ no existe decide() global por plan

# ============================================================
# NUEVAS v3 — allowed_actions obligatorio (D1)
# ============================================================
I46 [HARD]  ∀ capability C :
            C.allowed_actions definido (no None)
            ∧ C.denied_actions definido (no None)
            ∧ C.allowed_actions is None → DENY("capability_missing_allowed_actions")
            ∧ ∀ step s : s.allowed_actions definido
            ∧ token = intersection(s.allowed_actions, C.allowed_actions)
            ∧ s.allowed_actions ⊄ C.allowed_actions → DENY("actions_exceed_capability")

# ============================================================
# NUEVAS v3 — Metadatos derivados autoritativos (C1, REGLA MAESTRA)
# ============================================================
I47 [HARD]  ∀ step s :
            effective_resource_class    = authoritative_resource_registry(s.path, s.resource_instance)
            effective_data_class        = authoritative_data_registry(s.path)
            effective_runtime_privilege = runtime_policy(s)
            ∧ risk se calcula con valores EFECTIVOS, no declarados
            ∧ si declarado < efectivo → flag subdeclaración (SOFT warn + log)
            ∧ si path ∈ global_denied_paths → DENY
              (excepto control_plane.change sobre policy/ e infra/)

# ============================================================
# NUEVAS v3 — Enforcement por riesgo EFECTIVO (B)
# ============================================================
I48 [HARD]  ∀ P : aggregate_risk(P) = "critical" →
            human_gate(P) ∧ thot_review(P)
            ∧ rollback_verified(P) ∧ tests_passed(P) ∧ audit_remoto(P)
            ∧ INDEPENDIENTEMENTE de la capability o de requires_gate

I49 [HARD]  ∀ P : aggregate_risk(P) = "high" ∧ target_env ≥ staging(4) →
            human_gate(P)

# ============================================================
# NUEVAS v3 — Anti-prompt-injection para auditoras (G2)
# ============================================================
I50 [HARD]  ∀ faceta auditora F, ∀ contenido auditado C :
            C se entrega como datos no ejecutables
            ∧ C delimitado (sin autoridad instructiva sobre F)
            ∧ system_prompt de F es inmutable
            ∧ detector_de_prompt_injection(C) activo
            ∧ texto dentro de un plan/diff NUNCA es instrucción para F

# ============================================================
# NUEVAS v3 — Adapter restringido (F3)
# ============================================================
I51 [HARD]  ∀ firma del runtime_adapter :
            nonce_emitido_por_policy_engine(firma)
            ∧ policy_local_validada(firma)
            ∧ audit(intento_firma)
            ∧ dentro_de_rate_limit
            ∧ ¬firma_envelopes_arbitrarios
            ∧ ¬cambios_de_capability/env/scope_no_autorizados
```

---

## ARTEFACTO 3 — FUNCIÓN DE DECISIÓN `decide(envelope, plan)` v3

Reordenada: A0 estructura → firma → nonce **reservado** → derivación autoritativa → pisos → composición (promote_env=**GATE**, sandbox+priv=**DENY**) → lineage **ANTES** de aggregate_risk → enforcement por riesgo efectivo → gates → tests → audit → redacción **enforced** → decisión. Envelope **inmutable** dentro de `decide()`. Nonce se **reserva** (no consume) hasta ALLOW/DENY terminal.

```python
# ============================================================
# decide(envelope, plan) -> {ALLOW, DENY, GATE}  — v3
# Motor DETERMINISTA. Fail-closed. Ningún LLM decide en runtime.
# REGLA MAESTRA: el plan propone; el motor deriva y decide.
# ============================================================

RiskOrdinal = {"low":0, "medium":1, "high":2, "critical":3, "unknown":-1}
THOT_VETO_TTL = 3600  # [PROPUESTO] segundos (G1)

def risk_max(a, b):
    if a == "unknown" or b == "unknown": return "unknown"
    return max([a, b], key=lambda r: RiskOrdinal[r])

def risk_min_ceiling(a, ceiling):
    if a == "unknown": return ceiling
    return a if RiskOrdinal[a] <= RiskOrdinal[ceiling] else ceiling

def decide(envelope: Envelope, plan: Plan) -> Decision:

    # ============================================================
    # FASE A0 — Validación estructural ANTES de firma/nonce (A4, I43)
    # ============================================================
    if not structurally_valid_raw(envelope):
        # sin floats, sin claves duplicadas, UTF-8 válido,
        # canonicalización no ambigua
        return DENY(reason="envelope_structurally_invalid_raw")

    # Parseo canónico — a partir de aquí, canonical es la verdad
    canonical_envelope = canonical_parse(envelope)

    # ============================================================
    # FASE A1 — Domain separation + firma válida
    # ============================================================
    if not envelope_has_valid_prefix(canonical_envelope):
        return DENY(reason="domain_separation_violation")
    if not verify_signature(envelope.caller, envelope.caller_signature,
                            canonical_envelope,
                            expected_prefix="JAX_CAPABILITY_TOKEN_V1:"):
        return DENY(reason="caller_signature_invalid")

    # ============================================================
    # FASE A2 — Caller_chain íntegra con delegación atada a contexto (F4)
    # ============================================================
    if not verify_caller_chain(envelope.caller_chain, envelope):
        # verify_caller_chain ahora verifica que el envelope esté
        # DENTRO del scope delegado (capability, env, scope, actions, TTL)
        return DENY(reason="caller_chain_broken_or_outside_delegated_scope")

    # ============================================================
    # FASE A3 — Identity binding: cert por Jacobs, root por JAX_ROOT_IDENTITY_KEY (I40)
    # ============================================================
    if not jacobs_cert_valid(envelope.caller):
        return DENY(reason="identity_not_certified_by_jacobs")
    if not root_cert_valid(jacobs_key_id):
        # Jacobs cert debe estar firmado por JAX_ROOT_IDENTITY_KEY (offline)
        return DENY(reason="jacobs_root_cert_invalid")

    # ============================================================
    # FASE A4 — Key status activa y no expirada (I37)
    # ============================================================
    key_status = key_registry[key_id(envelope.caller_signature)].status
    if key_status != "active":
        return DENY(reason=f"key_not_active:{key_status}")

    # ============================================================
    # FASE A5 — Capability existe en registry (I45)
    # ============================================================
    cap = registry.get(envelope.capability_requested,
                       registry_version=envelope.policy_version)
    if cap is None:
        return DENY(reason="capability_not_found")

    # I46: allowed_actions OBLIGATORIO
    if cap.allowed_actions is None:
        return DENY(reason="capability_missing_allowed_actions")

    # ============================================================
    # FASE A6 — Caller autorizado
    # ============================================================
    if envelope.caller not in cap.allowed_callers:
        return DENY(reason="caller_not_allowed")

    # ============================================================
    # FASE A7 — Ambiente solicitado permitido
    # ============================================================
    if envelope.environment not in cap.environment:
        return DENY(reason="environment_not_allowed")

    # ============================================================
    # FASE A8 — Regla 5 — anti-lavado de capability indirecta
    # ============================================================
    if not could_request_directly(envelope.caller, envelope.capability_requested):
        return DENY(reason="indirect_capability_laundering")

    # ============================================================
    # FASE A9 — plan_hash presente y estable (sin timestamp)
    # ============================================================
    if envelope.plan_hash is None or envelope.plan_hash != stable_plan_hash(plan):
        return DENY(reason="plan_hash_missing_or_mismatch")

    # ============================================================
    # FASE A10 — Nonce RESERVADO (no consumido) — A5
    # ============================================================
    if nonce_status(envelope.nonce) == "used":
        return DENY(reason="nonce_replay_detected")
    if not nonce_reserve_atomic(envelope.nonce):
        return DENY(reason="nonce_atomic_reserve_failed")
    # nonce queda en estado "reserved" — solo ALLOW/DENY lo marcan "used"

    # ============================================================
    # FASE B — Scope, paths, denied_paths, actions (D2, C2)
    # ============================================================
    for step in plan.steps:
        # I45: binding capability↔step
        if step.capability != envelope.capability_requested:
            return DENY(reason="step_capability_mismatch_envelope")

        for resource in step.resources_touched:
            # Global denied_paths SIEMPRE ganan (C2)
            if matches_any(resource.path, global_denied_paths):
                # Excepción: control_plane.change sobre policy/ e infra/
                unless = (envelope.capability_requested == "control_plane.change"
                          and matches_any(resource.path, ["**/policy/**", "**/infra/**"]))
                if not unless:
                    return DENY(reason=f"global_denied_path:{resource.path}")

            if matches_any(resource.path, cap.denied_paths):
                return DENY(reason="path_denied")

            if not matches_any(resource.path, cap.allowed_paths):
                return DENY(reason="path_not_allowed")

        # D2: Token = intersección(step.allowed_actions, cap.allowed_actions)
        if step.allowed_actions is None:
            return DENY(reason="step_missing_allowed_actions")
        effective_actions = set(step.allowed_actions) & set(cap.allowed_actions)
        if effective_actions != set(step.allowed_actions):
            return DENY(reason="actions_exceed_capability")
        for action in step.allowed_actions:
            if action in cap.denied_actions:
                return DENY(reason=f"action_denied:{action}")

    # ============================================================
    # FASE C — DERIVACIÓN autoritativa (I47, REGLA MAESTRA)
    # El motor DERIVA; el plan no decide.
    # ============================================================
    for step in plan.steps:
        # effective_resource_class desde authoritative_path_registry
        step.effective_resource_class = authoritative_resource_registry(
            step.path, step.resource_instance
        )
        step.effective_data_class = authoritative_data_registry(step.path)
        step.effective_runtime_privilege = runtime_policy(step)

        # Subdeclaración: declarado < efectivo → SOFT warn + log
        if RiskOrdinal.get(step.resource_class, -1) < \
           RiskOrdinal.get(step.effective_resource_class, -1):
            log_subdeclaration(step, "resource_class",
                               step.resource_class, step.effective_resource_class)
        if RiskOrdinal.get(step.data_class, -1) < \
           RiskOrdinal.get(step.effective_data_class, -1):
            log_subdeclaration(step, "data_class",
                               step.data_class, step.effective_data_class)

    # ============================================================
    # FASE D — CAPA 1: PISOS DETERMINISTAS con valores EFECTIVOS
    # ============================================================
    step_risks = []
    for step in plan.steps:
        r = compute_step_risk(step, cap, use_effective=True)
        step_risks.append(r)
    # compute_step_risk aplica:
    #   baseline(cap) ∧ floor_por_effective_resource_class
    #   ∧ floor_por_effective_data_class
    #   ∧ floor_por_effective_runtime_privilege
    #   ∧ floor_por_secret_access ∧ floor_por_control_plane
    #   ∧ floor_special ∧ min(critical, baseline + Σ modificadores)
    # Ver ARTEFACTO 5 para fórmula cerrada.

    if any(r == "unknown" for r in step_risks):
        if envelope.environment == 5:
            return DENY(reason="risk_unknown_in_production")    # I3
        return GATE(reason="risk_unknown_requires_review")

    # ============================================================
    # FASE E — CAPA 2: COMPOSICIÓN CON EFECTOS TIPADOS
    # promote_env = GATE SIEMPRE (A1). sandbox+priv = DENY (A1).
    # Envelope INMUTABLE — no se muta (A2).
    # ============================================================
    comp = evaluate_composition(plan, envelope)
    effect = comp.effect

    if effect == "deny":
        return DENY(reason=f"composition_deny:{comp.rule}")

    if effect == "promote_env":
        # A1: SIEMPRE retorna GATE. NUNCA muta envelope.environment.
        # Reentrada exige nuevo envelope firmado (nuevo env, nuevo nonce).
        # Nonce reservado se LIBERA (no se consume).
        nonce_release(envelope.nonce)
        return GATE(reason="environment_promotion_requires_redecision",
                    requires_new_envelope=True)

    # A1: sandbox + {docker_sock, host_mount, root, sudo} = DENY directo
    for step in plan.steps:
        if envelope.capability_requested == "implementation.sandbox":
            if step.effective_runtime_privilege in \
               {"docker_sock", "host_mount", "root", "sudo"}:
                return DENY(reason="sandbox_with_elevated_privilege_forbidden")

    if effect == "gate":
        # Marcar para enforcement matrix; no retorna todavía
        composition_gate_pending = True
    else:
        composition_gate_pending = False

    if effect == "require_thot":
        thot_required_by_composition = True
    else:
        thot_required_by_composition = False

    if effect == "require_ada":
        if not ada_invariant_check_passed(plan):
            return GATE(reason="composition_require_ada")

    # +risk: aumenta composition_floor (no cambia verdict aquí)
    # comp.delta ya aplicado en composition_floor

    # ============================================================
    # FASE F — LINEAGE ANTES de aggregate_risk (A7, E1-E7)
    # Los factores de lineage entran en additive ANTES del cálculo.
    # ============================================================
    lineage_factors = 0

    # E1: Reconciliación observed_inputs vs declared expected_artifacts
    observed = runtime_monitor.get_observed_inputs(envelope.plan_hash)
    declared = set(a.artifact_id for a in plan.expected_artifacts)

    for obs_artifact in observed:
        if obs_artifact.artifact_id not in declared:
            return DENY(reason="undeclared_artifact_consumed")

    # E1: lineage_mismatch
    observed_ids = set(a.artifact_id for a in observed)
    if observed_ids != declared:
        return DENY(reason="lineage_mismatch")

    # E2: source_trust derivado del runtime — no se puede limpiar
    for artifact_ref in plan.expected_artifacts:
        artifact = load_artifact_from_cas(artifact_ref.artifact_id)
        # E4: hash verificado por CAS, no autodeclarado
        if hash(artifact.bytes) != artifact.artifact_hash:
            return DENY(reason="artifact_hash_cas_mismatch")
        # L16: runtime verifica hash(bytes_ejecutados) antes de ejecutar
        # (verificado aquí en pre-execution check)

        # E6: ArtifactReview verificación
        review = load_artifact_review(artifact.artifact_id)
        if review is None:
            return GATE(reason="artifact_review_missing")
        if not verify_artifact_review_signature(review):
            return DENY(reason="artifact_review_signature_invalid")
        if review.artifact_hash != artifact.artifact_hash:
            return DENY(reason="artifact_review_hash_mismatch")
        if review.approved_for_environment < plan.target_environment:
            return GATE(reason="artifact_exceeds_approved_environment")
        if review.expires_at and now() > review.expires_at:
            return DENY(reason="artifact_review_expired")

        # E2: source_trust no se puede limpiar
        # A_hijo.source_trust = union(source_trust(padres_observados))
        observed_parents = runtime_monitor.get_observed_parents(
            artifact.artifact_id, envelope.plan_hash
        )
        expected_source_trust = set()
        for parent in observed_parents:
            expected_source_trust.update(parent.source_trust)
        if "external_untrusted" in expected_source_trust \
           and "external_untrusted" not in artifact.source_trust:
            return DENY(reason="source_trust_laundering_detected")

        # I30: external_untrusted → gate si target ≥ staging
        if "external_untrusted" in artifact.source_trust \
           and plan.target_environment >= 4:
            if review.reviewer == "kimi":
                return DENY(reason="external_untrusted_reviewed_by_kimi")
            # gate ya marcado por review check arriba

        # L11: F-transform de artefactos heredados
        if artifact.lineage.depth > 0:
            for t in artifact.lineage.transformations:
                if t.kind in ("external→code", "docs→execution", "spec→implementa"):
                    lineage_factors += 1   # F-transform

    # Si lineage sube el riesgo sobre lo declarado → GATE
    max_step = max(step_risks, key=lambda r: RiskOrdinal[r])
    preliminary_additive = sum_factors(plan) + lineage_factors
    preliminary_risk = risk_min_ceiling(
        ordinal_to_risk(RiskOrdinal[max_step] + preliminary_additive),
        "critical"
    )
    if RiskOrdinal[preliminary_risk] > RiskOrdinal[plan.risk_class]:
        return GATE(reason="lineage_risk_exceeds_declared")

    # ============================================================
    # FASE G — CAPA 3: ESCALACIÓN ADITIVA + aggregate_risk
    # lineage_factors ya computados en Fase F, entran aquí.
    # ============================================================
    composition_floor = comp.computed_floor
    additive = sum_factors(plan) + lineage_factors    # F1,F2,F-transform,F-persist,F-ident,F-obs + lineage
    additive_risk = risk_min_ceiling(
        ordinal_to_risk(RiskOrdinal[max_step] + additive),
        "critical"
    )

    aggregate_risk = risk_max(
        max_step,
        risk_max(composition_floor, additive_risk)
    )

    if RiskOrdinal[aggregate_risk] > RiskOrdinal[plan.risk_class]:
        return GATE(reason="aggregate_risk_exceeds_declared")

    # ============================================================
    # FASE H — ENFORCEMENT MATRIX por riesgo EFECTIVO (I48-I49, B)
    # INDEPENDIENTE de la capability o de requires_gate.
    # ============================================================
    if aggregate_risk == "critical":
        # I48: SIEMPRE human_gate + thot + rollback + tests + audit_remoto
        if not human_gate_approved(envelope.plan_hash):
            return GATE(reason="critical_risk_requires_human_gate")
        if not thot_review_completed(envelope.plan_hash):
            return GATE(reason="critical_risk_requires_thot_review")
        if not rollback_plan_verified(plan):
            return GATE(reason="critical_risk_requires_rollback")
        if not test_oracle_passed(plan, verifier="jacobs_or_ci"):
            return GATE(reason="critical_risk_requires_tests")
        if not audit_remote_ready(envelope):
            return GATE(reason="critical_risk_requires_remote_audit")
        if not artifact_hash_matches(envelope.plan_hash, plan):
            return DENY(reason="artifact_hash_mismatch_toctou")

    elif aggregate_risk == "high":
        # I49: human_gate si env ≥ staging
        if envelope.environment >= 4:
            if not human_gate_approved(envelope.plan_hash):
                return GATE(reason="high_risk_staging_prod_requires_human_gate")
        # tests si toca code/config/ci_cd
        if any(s.effective_resource_class in {"code", "config", "ci_cd"}
               for s in plan.steps):
            if not test_oracle_passed(plan, verifier="jacobs_or_ci"):
                return GATE(reason="high_risk_code_config_requires_tests")

    elif aggregate_risk == "medium":
        if not audit_trail_ready(envelope):
            return DENY(reason="medium_risk_requires_audit_trail")

    elif aggregate_risk == "low":
        if envelope.environment > 3:
            return GATE(reason="low_risk_auto_audit_only_below_staging")

    # ============================================================
    # FASE I — Gates condicionales y capability-specific
    # (subordinados a Fase H — no pueden relajar lo que H exige)
    # ============================================================
    if composition_gate_pending:
        return GATE(reason=f"composition_gate:{comp.rule}")

    if thot_required_by_composition or cap.requires_gate:
        if envelope.environment == 5 or cap.approval_class == "human_gate_plus_thot":
            thot_verdict = check_thot(envelope.plan_hash)
            if thot_verdict == "NO-GO":
                # security_incident — operador debe override explícito (G1)
                return GATE(reason="thot_veto_no_go_requires_operator_decision")
            elif thot_verdict == "unavailable":
                # availability_override — NO es incidente de seguridad (G1)
                if thot_veto_ttl_expired(envelope.plan_hash, THOT_VETO_TTL):
                    return GATE(reason="thot_unavailable_ttl_expired_availability_override")
                else:
                    return GATE(reason="thot_unavailable_waiting_within_ttl")
            # thot_verdict == "GO" → continuar

    if envelope.capability_requested == "control_plane.change":
        if not ada_invariant_check_passed(plan):
            return GATE(reason="control_plane_requires_ada_invariant_check")

    if envelope.capability_requested == "capability.grant":
        if not is_human(envelope.caller):
            return DENY(reason="capability_grant_requires_human")

    # ============================================================
    # FASE J — Tests (I33)
    # ============================================================
    requires_tests_eff = plan.requires_tests or policy.requires_tests(envelope)
    if requires_tests_eff:
        if envelope.environment in (4, 5) \
           or envelope.capability_requested == "control_plane.change" \
           or any(s.effective_resource_class == "ci_cd" for s in plan.steps):
            if not test_oracle_passed(plan, verifier="jacobs_or_ci"):
                return GATE(reason="tests_not_passed_by_oracle")

    # ============================================================
    # FASE K — Audit y reversibilidad
    # ============================================================
    if cap.requires_audit:
        if not audit_trail_ready(envelope):
            return DENY(reason="audit_trail_not_ready")

    if cap.requires_rollback:
        if not rollback_plan_verified(plan):
            return GATE(reason="rollback_plan_not_verified")

    # ============================================================
    # FASE L — Redacción ENFORCED con código real (A6)
    # ============================================================
    for step in plan.steps:
        if step.secret_access != "none" \
           or step.effective_data_class in {"secret", "credential", "pii"}:
            if not redaction_policy_attached(envelope, plan):
                return DENY(reason="redaction_policy_missing")

    # ============================================================
    # FASE M — Decisión final
    # ============================================================
    return ALLOW(token=issue_ephemeral_token(
        capability       = envelope.capability_requested,
        resource_scope   = plan.resources_touched,
        environment      = envelope.environment,   #

⚛️ *Origen de autoridad: GLM-5.2 (Z.ai). Razonamiento analítico y formalización. Sin verificación externa por defecto.*
