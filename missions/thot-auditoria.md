thot: Auditoría adversarial del Contrato Formal v2 de Las Manos (incluido íntegro abajo). NO es una validación. Tu trabajo NO es confirmar que el contrato es sólido — es ROMPERLO. Asumí que tiene agujeros y encontralos. Si terminás diciendo "se ve bien", fallaste la misión.

CONTEXTO: este contrato pasó por (1) un primer formalizador (Ada v1), (2) dos revisores externos adversariales (GPT + DeepSeek), (3) una segunda formalización (Ada v2) que incorporó todo ese hardening. Vos sos la cuarta pasada y la última antes de implementar. Lo que los tres anteriores no vieron, lo tenés que ver vos. El contrato va a ejecutar operaciones reales en producción (servidores, deploys, control plane). Un agujero que se te escape llega a producción.

REGLA DE RIGOR (la tuya): cuestioná supuestos, declará incertidumbres, no cierres un hallazgo con "probablemente". El que supone se equivoca.

PRESUPUESTO DE HALLAZGOS OBLIGATORIO (no podés cerrar sin esto):
- Mínimo 3 vectores de ATAQUE concretos: una secuencia específica de pasos/capabilities/envelopes que produzca un resultado prohibido, o demuestre por qué NO se puede (con el paso exacto del decide() que lo frena).
- Mínimo 2 INVARIANTES que falten o estén mal cuantificadas (∀ donde debería ser ∃, o al revés; condición que no se chequea en decide()).
- Mínimo 2 huecos en el modelo de PROVENANCE/LINEAGE (Artefacto 6): cómo se falsifica un artifact_hash; si se puede romper una cadena de lineage sin que L4/L10 lo atrapen; si hay un camino de "external_untrusted → producción" que evada L3/L8.

ATACÁ ESPECÍFICAMENTE ESTOS PUNTOS:

1. LAS 3 CAPAS DE RIESGO (Artefacto 5). Construí una cadena de capabilities que produzca una operación peligrosa pero que las 3 capas (pisos + efectos + aditivo) clasifiquen por debajo de su riesgo real. Probá: cadenas de pasos todos "low" que deberían escalar; combinaciones que esquiven los floors; el ceiling "critical" usado para ocultar criticidad acumulada.

2. EL decide() (Artefacto 3). Hay una fase (A-L) que se pueda saltar? El orden de las fases permite que algo pase antes de su chequeo? promote_env en Fase D y la re-entrada — hay un estado donde el env promovido no re-evalúa los gates? El nonce marcado en A10 antes de las fases de riesgo deja un hueco si una fase posterior hace GATE (el nonce ya se consumió)?

3. PROVENANCE/LINEAGE (Artefacto 6). El ataque que motivó este artefacto era lavado entre planes. De verdad lo cierra? Atacá: artifact_hash falsificable, lineage con depth reseteado, parent_artifact_ids omitidos, source_trust que se "limpia" en una transformación intermedia, un Plan que consume un artefacto sin declararlo en expected_artifacts.

4. CRIPTO E IDENTIDAD (Artefacto 4). El binding de Jacobs (I40) tiene un huevo del huevo — quién certifica a Jacobs? La caller_chain se puede forjar si un eslabón intermedio está comprometido? El runtime adapter que tiene la private key (I41) es un nuevo punto único de fallo?

5. TU PROPIO ROL (auto-crítica obligatoria). Por I5 (implementer ≠ auditor), vos auditando un contrato que define tu propio veto bloqueante tenés sesgo estructural. Cuestionalo: el veto bloqueante de Thot es un punto único de fallo? Si Thot es comprometido (prompt injection vía un plan que incluye texto malicioso), qué puede hacer? El override solo-operador es suficiente defensa si Thot se vuelve adversarial, o Thot comprometido puede vetar todo y trabar el sistema (DoS)?

6. EL HUECO QUE NO ESTOY NOMBRANDO. Los puntos 1-5 son donde el diseñador cree que están los riesgos. Tu valor real es el punto 6: qué NO se está preguntando que debería? Qué supuesto de todo el diseño es el más frágil?

FORMATO DE SALIDA:
- Por cada vector de ataque: título, secuencia exacta, resultado prohibido logrado (o el paso que lo frena), severidad (crítica/alta/media), y corrección concreta.
- Veredicto final: GO / NO-GO para implementar Fase 1. Si NO-GO, qué bloquea exactamente. Si GO, qué condiciones quedan.
- No endulces. Este es el último filtro antes de que Kimi escriba código que toca producción.

========================== CONTRATO FORMAL v2 (objeto de la auditoría) ==========================

# Resultado de: ada-contrato-v2.md

# Contrato Formal v2 — Las Manos (Fase 1 + Fase 2)

Ada, séptima faceta. Segunda iteración del dominio. Cierro las 24 D.P. con las resoluciones consolidadas, incorporo el hardening de la revisión cruzada y agrego el artefacto 6 (provenance + lineage). Lo que sigue es el contrato que el policy engine determinista hará cumplir. Lo que queda `[FERNANDO]` es dato de entorno, no suposición. Lo que queda `[PROPUESTO]` es default sensible sujeto a confirmación.

Ratifico: el riesgo no es suma ordinal. Son tres capas — pisos deterministas, efectos tipados, escalación aditiva auxiliar. El que supone se equivoca.

---

## ARTEFACTO 1 — ESQUEMA DE CAPABILITIES (10)

Tipado TOML. Enums cerrados según RESOLUCIONES. Campos nuevos del `decision_context` (R3') incorporados. `requires_tests` migra a Plan (D.P. 15). `operator_override` sale de `approval_class` (D.P. 16).

```toml
# ============================================================
# CAPABILITY REGISTRY v2 — 10 capabilities
# ============================================================

# --- Enums cerrados (RESOLUCIONES R1'–R3', D.P. 7/9/11/12/16) ---
# Environment       = 0|1|2|3|4|5  (0=conversación … 4=staging, 5=producción)
# RiskClass         = "low"|"medium"|"high"|"critical"|"unknown"
# BlastRadius       = "none"|"file"|"repo"|"service"|"host"|"tenant"|"multi_tenant"|"network"|"system"|"control_plane"
# DataClass         = "public"|"internal"|"confidential"|"secret"|"credential"|"pii"
# SecretAccess      = "none"|"read_redacted"|"read"|"write"|"mount"|"inject_env"|"derive"|"export"
# NetworkEgress     = "none"|"loopback"|"restricted_allowlist"|"open"
# AllowedAction     = "read"|"write"|"execute"|"delete"|"grant"|"revoke"|"mount"
# ApprovalClass     = "none"|"auto_audit"|"human_gate"|"human_gate_plus_thot"
# OverrideState     = "none"|"thot_veto_overridden"|"policy_exception"|"break_glass"
# ResourceClass     = "code"|"config"|"ci_cd"|"network"|"dns"|"firewall"|"auth"|"identity"|"audit"|"backup"|"scheduler"|"runner"|"supply_chain"|"system_prompt"|"agent_prompt"|"tool_manifest"|"policy_doc"|"secret_store"|"data"
# SourceTrust       = "internal"|"external_untrusted"
# RuntimePrivilege  = "none"|"root"|"sudo"|"docker_sock"|"host_mount"   # valores con floor high: {root,sudo,docker_sock,host_mount}
# Reversibility     = "reversible"|"irreversible"|"unknown"
# SecretAccessCritical = {"write","mount","inject_env"}                 # floor critical
# SecretAccessHigh     = {"read"}                                       # floor high

# ============================================================
# 1. formal.spec
# ============================================================
[capability.formal_spec]
allowed_callers       = ["ada", "jacobs"]
environment           = [0, 1, 2, 3, 4]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]                          # árbol de servicios → paths
denied_paths          = ["**/.env", "**/secrets/**", "**/vault/**"]
resource_class        = "code"
resource_tags         = []                                       # populated runtime
data_class            = "internal"
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "file"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 300     # [PROPUESTO] segundos
max_diff_size         = 1000    # [PROPUESTO] líneas
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
denied_paths          = ["**/.env", "**/secrets/**", "**/vault/**", "**/docker.sock", "**/sudoers"]
resource_class        = "code"
resource_tags         = []
data_class            = "internal"
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "file"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 120     # [PROPUESTO]
max_diff_size         = 2000    # [PROPUESTO] líneas
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
denied_paths          = ["**/.env", "**/secrets/**", "**/vault/**", "**/docker.sock"]
resource_class        = "code"
resource_tags         = []
data_class            = "internal"
secret_access         = "none"
network_egress        = "restricted_allowlist"
egress_allowlist      = ["[FERNANDO]"]                          # dominios por servicio
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "service"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 600     # [PROPUESTO]
max_diff_size         = 1000    # [PROPUESTO] líneas
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
denied_paths          = ["**/.env", "**/secrets/**", "**/vault/**"]
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
max_runtime           = 900     # [PROPUESTO]
max_diff_size         = 500     # [PROPUESTO] líneas
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
denied_paths          = ["**/audit/**"]
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
max_runtime           = 900     # [PROPUESTO]
max_diff_size         = 500     # [PROPUESTO] líneas
requires_gate         = true
requires_audit        = true
requires_rollback     = false
approval_class        = "human_gate_plus_thot"
baseline_risk         = "high"

# ============================================================
# 6. audit.review   (read-only; D.P. 20 cerrado)
# ============================================================
[capability.audit_review]
allowed_callers       = ["thot", "jacobs"]
environment           = [0, 1, 2, 3, 4, 5]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]                          # read path_allowlist (no /etc/shadow etc.)
denied_paths          = ["**/.env", "**/secrets/**", "**/vault/**", "**/shadow", "**/sudoers"]
denied_actions        = ["write", "delete", "execute", "grant", "revoke", "mount", "mutate_state"]
resource_class        = "audit"
resource_tags         = []
data_class            = "confidential"
secret_access         = "read_redacted"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "none"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 300     # [PROPUESTO]
max_diff_size         = 0       # read-only
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
denied_paths          = ["**"]
resource_class        = "data"
resource_tags         = []
data_class            = "public"
secret_access         = "none"
network_egress        = "open"
egress_allowlist      = []
source_trust          = "external_untrusted"
runtime_privilege     = "none"
blast_radius          = "none"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 60      # [PROPUESTO]
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
denied_paths          = ["**/.env", "**/secrets/**", "**/infra/**", "**/policy/**"]
resource_class        = "data"
resource_tags         = []
data_class            = "public"
secret_access         = "none"
network_egress        = "none"
egress_allowlist      = []
source_trust          = "internal"
runtime_privilege     = "none"
blast_radius          = "file"
reversibility         = "reversible"
tenant_scope          = "single"
max_runtime           = 300     # [PROPUESTO]
max_diff_size         = 1000    # [PROPUESTO] líneas
requires_gate         = false
requires_audit        = true
requires_rollback     = true
approval_class        = "auto_audit"
baseline_risk         = "low"

# ============================================================
# 9. control_plane.change
# ============================================================
[capability.control_plane_change]
allowed_callers       = ["jacobs"]
environment           = [5]
resource_scope        = "service+path"
allowed_paths         = ["[FERNANDO]"]                          # policy engine, registry, router, gate, audit, Jacobs, prompts
denied_paths          = ["**/audit/hash_chain/**"]
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
max_runtime           = 300     # [PROPUESTO]
max_diff_size         = 500     # [PROPUESTO] líneas
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
allowed_callers       = ["human"]                               # SOLO humano; máquina → DENY
environment           = [5]
resource_scope        = "none"
allowed_paths         = []
denied_paths          = ["**"]
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
max_runtime           = 60      # [PROPUESTO]
max_diff_size         = 0
requires_gate         = true
requires_audit        = true
requires_rollback     = true
approval_class        = "human_gate_plus_thot"                  # operator_override = estado separado
baseline_risk         = "critical"
requires_human_caller = true

# ============================================================
# CONTROL PLANE FLAGS
# ============================================================
[control_plane.dual_control]
enabled               = false
activation_policy     = "[FERNANDO] se activa cuando exista 2º operador de confianza registrado"
required_operators    = 2

[control_plane.key_registry]
location              = "/etc/jax/keys/"   # [PROPUESTO]
requires_capability   = "control_plane.change"   # modificar registro → floor critical + gate
auto_rotation_phase   = "F3"

[control_plane.audit_sink]
local_hash_chain      = true
remote_target         = "Sesamo (TrueNAS .6, dataset read-only, ZFS snapshots)"
replication_cadence   = "[FERNANDO] push por evento crítico vs batch"

[control_plane.runtime_monitor]
observe               = ["files", "commands", "egress", "processes"]
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
```

---

## ARTEFACTO 2 — INVARIANTES

`HARD` bloquea ejecución. `SOFT` advierte. Se agregan: provenance, lineage, test_oracle, policy_versioning, key revocation, domain separation, approved_for_environment.

```
# ============================================================
# INVARIANTES FORMALES v2
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

# --- Override de veto (operator_override = estado separado) ---
I9  [HARD]  ∀ override ∈ override_state \ {none} :
            reason(override) ≠ ""
            ∧ actor(override) ∈ humanos
            ∧ plan_hash(override) ≠ null
            ∧ risk_acknowledgement(override) = true
            ∧ rollback_plan(override) ≠ null
            ∧ security_incident(override) registrado

I10 [HARD]  ∀ t : veto_thot(t) no es eliminado   # solo override, never delete

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
            # incluye a Jacobs: no modifica sus propios permisos

# --- Composición / auditoría / secrets ---
I18 [HARD]  ∀ cadena C :
            ∃ chain_id(C)
            ∧ risk_efectivo(C) ≥ max(risk(step_i) ∀ step_i ∈ C)
            ∧ risk_efectivo(C) ≥ composition_floor(C)        # capa 1+2
            ∧ risk_efectivo(C) = max(
                  max(step_risk),
                  composition_floor(C),
                  min("critical", max(step_risk) + Σ factores)  # capa 3
              )

I19 [HARD]  ∀ registro r ∈ audit :
            append_only(r) ∧ hash_chain(r, r_anterior) válida
            ∧ réplica_a_Sesamo(r) programada        # WORM + Sésamo

I20 [HARD]  ∀ caller J, ∀ job j de J : ¬modifica(J, audit(j))
I21 [HARD]  ∀ output de capability read-only :
            ∀ secret s ∈ output : redacted(s) = true
I22 [HARD]  ∀ J, ∀ output(J) : ¬(secret_aparece_sin_redacción(output(J)))

# --- TOCTOU ---
I23 [HARD]  ∀ deploy : plan_hash_aprobado(deploy) = plan_hash_ejecutado(deploy)

# --- Environment isolation ---
I24 [HARD]  ∀ env staging S :
            si ∃ recurso R ∈ S : R.shared_with_production = true
            → aplicar_reglas(S, producción)        # promote_env

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
# NUEVAS — Provenance y lineage (Artefacto 6)
# ============================================================
I28 [HARD]  ∀ artefacto A :
            ∃ A.artifact_id ∧ ∃ A.artifact_hash
            ∧ ∃ A.originating_plan_id ∧ ∃ A.originating_chain_id
            ∧ ∃ A.producing_facet ∈ facetas
            ∧ ∃ A.source_trust ∈ {"internal","external_untrusted"}
            ∧ ∃ A.approved_for_environment ∈ Environment

I29 [HARD]  ∀ artefacto A, ∀ ambiente E destino :
            E > A.approved_for_environment
            → ∄ movimiento(A, E) sin nueva_revisión(A, E)

I30 [HARD]  ∀ artefacto A :
            "external_untrusted" ∈ A.source_trust
            ∧ E_destino ≥ staging
            → ∄ movimiento(A, E_destino) sin gate(A)

I31 [HARD]  ∀ par (A_padre, A_hijo) en lineage :
            A_hijo.originating_plan_id referenciado
            ∧ A_hijo.originating_chain_id = A_padre.originating_chain_id  ∨
              A_hijo.originating_chain_id = nuevo_chain_con_link_a(A_padre)
            # no se "lava" el riesgo borrando el lineage

I32 [HARD]  ∀ plan P :
            risk_efectivo(P) ≥ max(risk_efectivo(A) ∀ A ∈ P.expected_artifacts
                                   ∪ A heredados via lineage)

# ============================================================
# NUEVAS — Test oracle
# ============================================================
I33 [HARD]  ∀ plan P con target_environment ∈ {staging, production, control_plane, ci_cd} :
            requires_tests(P) = OR(plan.requires_tests, policy.requires_tests)
            ∧ tests_pasados(P) verificado_por(Jacobs ∪ CI)
            ∧ tests_pasados(P) ≠ kimi_word_only
            # el test_oracle es externo a Kimi

# ============================================================
# NUEVAS — Policy versioning
# ============================================================
I34 [HARD]  ∀ decisión D registrada :
            ∃ D.policy_version ∧ ∃ D.registry_version
            ∧ ∃ D.schema_version ∧ ∃ D.router_version
            # toda decisión tiene su contexto de versión

I35 [HARD]  ∀ cambio de policy_version :
            invalidate_pending_approvals(policy_version_anterior)
            ∧ re-evaluar_planes_pendientes_con(policy_version_nueva)

# ============================================================
# NUEVAS — Revocación de claves (D.P. 19, R4')
# ============================================================
I36 [HARD]  ∀ key_id K :
            ∃ K.status ∈ {"active","revoked","expired","retired"}
            ∧ ∃ K.fingerprint ∧ ∃ K.created_at ∧ ∃ K.owner
            ∧ (K.status = "revoked" → ∃ K.revoked_at ∧ ∃ K.revocation_reason)

I37 [HARD]  ∀ envelope E, ∀ signature S en E :
            key_id(S).status = "active"
            ∧ now() ∈ [S.not_before, S.not_after?]
            # status ≠ active → DENY

I38 [HARD]  ∀ operación revoke_key(K) :
            caller ∈ humanos ∧ audit(revoke_key(K)) ∧ alerts(revoke_key(K))
            ∧ todos los tokens emitidos bajo K invalidados inmediatamente

# ============================================================
# NUEVAS — Domain separation (R4' punto 2)
# ============================================================
I39 [HARD]  ∀ objeto firmado O :
            ∃ prefix(O) ∈ {
              "JAX_PLAN_V1:", "JAX_APPROVAL_V1:",
              "JAX_CAPABILITY_TOKEN_V1:", "JAX_AUDIT_EVENT_V1:"
            }
            ∧ verify(O) requiere prefix_match(O)
            # una firma de un contexto NO verifica en otro

# ============================================================
# NUEVAS — Identidad firmada por Jacobs
# ============================================================
I40 [HARD]  ∀ faceta F :
            ∃ cert(F) = {F.pubkey, F.facet_name, F.key_id} firmado_por(Jacobs)
            ∧ pubkey_registry[F] requiere cert(F) válido
            # sin cert firmado por Jacobs, no se acepta pubkey como F

I41 [HARD]  ∀ faceta LLM F :
            private_key(F) reside_en(runtime_adapter(F))
            ∧ F.llm_only_propone(desired_action)
            ∧ firma_empleada = runtime_adapter(F).sign(...)
            # el LLM NUNCA toca la private key

# ============================================================
# NUEVAS — Anti-replay (R4' punto 5)
# ============================================================
I42 [HARD]  ∀ aprobación A :
            ∃ A.nonce ∧ nonce(A) persistente ∧ single_use
            ∧ nonce(A).marked_used ANTES de ejecutar
            ∧ ∃ A.idempotency_key
            ∧ nonce(A).replay → DENY

I43 [HARD]  ∀ objeto firmado O :
            sin_floats(O) ∧ sin_claves_duplicadas(O)
            ∧ sin_campos_desconocidos(O, schema_version(O))
            ∧ UTF-8_válido(O)

# ============================================================
# NUEVAS — Jacobs auditado
# ============================================================
I44 [HARD]  ∀ acción A_jacobs de Jacobs :
            audit(A_jacobs) ∧ alert(A_jacobs)
            # Jacobs no es zona ciega
```

---

## ARTEFACTO 3 — FUNCIÓN DE DECISIÓN `decide(envelope, plan)`

Determinista, fail-closed. Tres capas de riesgo. `approved_for_environment` check obligatorio.

```python
# ============================================================
# decide(envelope, plan) -> {ALLOW, DENY, GATE}
# Motor DETERMINISTA. Ningún LLM decide en runtime (Regla 6).
# Fail-closed: ante duda, DENY o GATE, nunca ALLOW implícito.
# ============================================================

RiskOrdinal = {"low":0, "medium":1, "high":2, "critical":3, "unknown":-1}

def risk_max(a, b):
    if a == "unknown" or b == "unknown": return "unknown"
    return max([a, b], key=lambda r: RiskOrdinal[r])

def risk_min_ceiling(a, ceiling):
    if a == "unknown": return ceiling
    return a if RiskOrdinal[a] <= RiskOrdinal[ceiling] else ceiling

def decide(envelope: Envelope, plan: Plan) -> Decision:

    # ============================================================
    # FASE A — Precondiciones estructurales (cualquier fallo → DENY)
    # ============================================================

    # A1. Domain separation + firma válida
    if not envelope_has_valid_prefix(envelope):
        return DENY(reason="domain_separation_violation")
    if not verify_signature(envelope.caller, envelope.caller_signature, envelope,
                            expected_prefix="JAX_CAPABILITY_TOKEN_V1:"):
        return DENY(reason="caller_signature_invalid")

    # A2. Caller_chain íntegra (cada link firmado, Jacobs verificó delegación)
    if not verify_caller_chain(envelope.caller_chain):
        return DENY(reason="caller_chain_broken")

    # A3. Identity binding: pubkey de caller certificada por Jacobs (I40)
    if not jacobs_cert_valid(envelope.caller):
        return DENY(reason="identity_not_certified_by_jacobs")

    # A4. Key status activa y no expirada (I37)
    key_status = key_registry[key_id(envelope.caller_signature)].status
    if key_status != "active":
        return DENY(reason=f"key_not_active:{key_status}")

    # A5. Capability existe en registry
    cap = registry.get(envelope.capability_requested, registry_version=envelope.policy_version)
    if cap is None:
        return DENY(reason="capability_not_found")

    # A6. Caller autorizado
    if envelope.caller not in cap.allowed_callers:
        return DENY(reason="caller_not_allowed")

    # A7. Ambiente solicitado permitido
    if envelope.environment not in cap.environment:
        return DENY(reason="environment_not_allowed")

    # A8. Regla 5 — anti-lavado de capability indirecta
    if not could_request_directly(envelope.caller, envelope.capability_requested):
        return DENY(reason="indirect_capability_laundering")

    # A9. plan_hash presente y estable (sin timestamp, D.P. 23)
    if envelope.plan_hash is None or envelope.plan_hash != stable_plan_hash(plan):
        return DENY(reason="plan_hash_missing_or_mismatch")

    # A10. Nonce persistente single-use, marcado ANTES de ejecutar (I42)
    if nonce_used(envelope.nonce):
        return DENY(reason="nonce_replay_detected")
    if not nonce_mark_used_atomic(envelope.nonce):
        return DENY(reason="nonce_atomic_mark_failed")

    # A11. Sin floats / sin claves duplicadas / UTF-8 válido (I43)
    if not structurally_valid(envelope):
        return DENY(reason="envelope_structurally_invalid")

    # ============================================================
    # FASE B — Scope y paths
    # ============================================================

    for resource in plan.resources_touched:
        if not matches_any(resource.path, cap.allowed_paths):
            return DENY(reason="path_not_allowed")
        if matches_any(resource.path, cap.denied_paths):
            return DENY(reason="path_denied")
        if cap.allowed_actions is not None:
            for action in resource.actions:
                if action in cap.denied_actions:
                    return DENY(reason=f"action_denied:{action}")
                if action not in cap.allowed_actions:
                    return DENY(reason=f"action_not_permitted:{action}")

    # ============================================================
    # FASE C — CAPA 1: PISOS DETERMINISTAS (R1' capa 1, R2')
    # ============================================================

    step_risks = []
    for step in plan.steps:
        r = compute_step_risk(step, cap)
        step_risks.append(r)

    # compute_step_risk aplica:
    #   baseline(cap) ∧ floor_por_resource_class ∧ floor_por_data_class
    #   ∧ floor_por_runtime_privilege ∧ floor_por_secret_access
    #   ∧ pisos especiales (audit+secret=high; ci_cd write staging=high;
    #     docs.write sobre prompt/tool_manifest=high)
    #   ∧ min(critical, baseline + Σ modificadores)
    # Ver ARTEFACTO 5 para la fórmula cerrada.

    if any(r == "unknown" for r in step_risks):
        if envelope.environment == 5:
            return DENY(reason="risk_unknown_in_production")    # I3
        return GATE(reason="risk_unknown_requires_review")

    # ============================================================
    # FASE D — CAPA 2: COMPOSICIÓN CON EFECTOS TIPADOS (R1' capa 2)
    # ============================================================

    comp = evaluate_composition(plan, envelope)   # ver ARTEFACTO 5 tabla
    effect = comp.effect

    if effect == "deny":
        return DENY(reason=f"composition_deny:{comp.rule}")

    if effect == "promote_env":
        envelope.environment = max(envelope.environment, 5)   # producción
        # re-entrar con env promovido — re-evaluar A7 y gates de producción
        envelope._promoted = True

    if effect == "gate":
        return GATE(reason=f"composition_gate:{comp.rule}")

    if effect == "require_thot":
        if not thot_review_completed(envelope.plan_hash):
            return GATE(reason="composition_require_thot")

    if effect == "require_ada":
        if not ada_invariant_check_passed(plan):
            return GATE(reason="composition_require_ada")

    # +risk: aumenta step_risk del step afectado (no cambia verdict)
    if effect == "+risk":
        # comp.delta ya aplicado en composition_floor
        pass

    # ============================================================
    # FASE E — CAPA 3: ESCALACIÓN ADITIVA (R1' capa 3, auxiliar)
    # ============================================================

    max_step = max(step_risks, key=lambda r: RiskOrdinal[r])
    composition_floor = comp.computed_floor   # capa 1+2 ya fundidas
    additive = sum_factors(plan)              # F1, F2, F-transform, F-persist, F-ident, F-obs
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
    # FASE F — approved_for_environment (NUEVO, I28–I30)
    # ============================================================

    for artifact in plan.expected_artifacts:
        # I29: no mover a env > approved_for_environment sin nueva revisión
        if plan.target_environment > artifact.approved_for_environment:
            if not artifact.re_review_for_env(plan.target_environment):
                return GATE(reason="artifact_exceeds_approved_environment")
        # I30: external_untrusted no llega a staging/prod sin gate
        if "external_untrusted" in artifact.source_trust \
           and plan.target_environment >= 4:
            if not artifact.explicit_review_passed:
                return GATE(reason="external_untrusted_to_staging_or_prod")

    # ============================================================
    # FASE G — Environment isolation (I24, D.P. 24)
    # ============================================================

    if envelope.environment == 4:   # staging
        for r in plan.resources_touched:
            if r.shared_with_production:
                envelope.environment = 5
                envelope._promoted = True
                return GATE(reason="staging_shares_production_resource")
                # re-entrar con env=5 tras aprobación humana

    # ============================================================
    # FASE H — Gates condicionales y capability-specific
    # ============================================================

    triggers = evaluate_gate_triggers(plan, envelope)
    if any(triggers):
        return GATE(reason=f"gate_trigger:{triggers}")

    if cap.requires_gate:
        if envelope.environment == 5:
            if not human_gate_approved(envelope.plan_hash):
                return GATE(reason="production_requires_human_gate")
            if not thot_review_completed(envelope.plan_hash):
                return GATE(reason="production_requires_thot_review")
            if not artifact_hash_matches(envelope.plan_hash, plan):
                return DENY(reason="artifact_hash_mismatch_toctou")   # I2/I23

        if cap.approval_class == "human_gate_plus_thot":
            if not thot_review_completed(envelope.plan_hash):
                return GATE(reason="thot_review_required")

    if envelope.capability_requested == "control_plane.change":
        if not ada_invariant_check_passed(plan):
            return GATE(reason="control_plane_requires_ada_invariant_check")

    if envelope.capability_requested == "capability.grant":
        if not is_human(envelope.caller):
            return DENY(reason="capability_grant_requires_human")

    # ============================================================
    # FASE I — Tests (I33, D.P. 15)
    # ============================================================

    requires_tests_eff = plan.requires_tests or policy.requires_tests(envelope)
    if requires_tests_eff:
        if envelope.environment in (4, 5) \
           or envelope.capability_requested in \
              ("control_plane.change",) \
           or any(s.resource_class == "ci_cd" for s in plan.steps):
            if not test_oracle_passed(plan, verifier="jacobs_or_ci"):
                return GATE(reason="tests_not_passed_by_oracle")
            # NO se acepta la palabra de Kimi como única verificación

    # ============================================================
    # FASE J — Audit y reversibilidad (I4, I19, I20)
    # ============================================================

    if cap.requires_audit:
        if not audit_trail_ready(envelope):
            return DENY(reason="audit_trail_not_ready")

    if cap.requires_rollback:
        if not rollback_plan_verified(plan):
            return GATE(reason="rollback_plan_not_verified")

    # ============================================================
    # FASE K — Redacción determinista (I21, I22)
    # ============================================================

    # La redacción se aplica en runtime antes de entregar output a cualquier
    # faceta. Aquí solo se valida que la capability está marcada para
    # redacción si toca secretos.

    # ============================================================
    # FASE L — Decisión final
    # ============================================================

    return ALLOW(token=issue_ephemeral_token(
        capability      = envelope.capability_requested,
        resource_scope  = plan.resources_touched,
        environment     = envelope.environment,
        caller_chain    = envelope.caller_chain,
        plan_hash       = envelope.plan_hash,
        step_id         = envelope.step_id,
        expires_at      = now() + cap.max_runtime,
        max_runtime     = cap.max_runtime,
        allowed_actions = plan.steps[envelope.step_id].allowed_actions,
        nonce           = envelope.nonce,
        key_id          = key_id(envelope.caller_signature),
        policy_version  = envelope.policy_version,
        prefix          = "JAX_CAPABILITY_TOKEN_V1:"
    ))

# ============================================================
# POSTCONDICIONES POR RAMA
# ============================================================
# ALLOW:
#   - emite token efímero atado a (plan_hash, step_id, scope, expires_at, nonce)
#   - audit_log.append("ALLOW", envelope, plan_hash, token_id, policy_version)
#   - runtime_monitor activa observación; mata el job si se desvía del plan
#   - nonce marcado usado persistente (single-use)
#
# DENY:
#   - audit_log.append("DENY", envelope, reason, policy_version)
#   - no se emite token
#   - nonce marcado usado (consume el slot anti-replay)
#
# GATE:
#   - audit_log.append("GATE_PENDING", envelope, reason, policy_version)
#   - estado = "esperando_aprobacion_humana" atado a (plan_hash, nonce)
#   - notifica operador humano
#   - el job NO corre hasta resolución
#   - si aprobado → re-entra por decide() con human_gate_approved=true
#       (nonce reemitido; el previo ya fue consumido)
#   - si rechazado → DENY
```

---

## ARTEFACTO 4 — INTENT ENVELOPE + ESQUEMA DE CLAVES

```typescript
// ============================================================
// INTENT ENVELOPE v2 — Ed25519 + SHA-256 + RFC 8785/JCS
// Domain separation: prefijo obligatorio en toda firma.
// Private key vive en runtime adapter, NO en faceta LLM (I41).
// ============================================================

type Caller = "ada" | "kimi" | "hyde" | "thot" | "hipatia" | "jekyll" | "jacobs" | "jax_local" | "human"
type Environment = 0 | 1 | 2 | 3 | 4 | 5
type RiskClass = "low" | "medium" | "high" | "critical" | "unknown"
type SignaturePrefix =
  | "JAX_PLAN_V1:"
  | "JAX_APPROVAL_V1:"
  | "JAX_CAPABILITY_TOKEN_V1:"
  | "JAX_AUDIT_EVENT_V1:"

interface Envelope {
  // --- Propuestos por el LLM (solo estos) ---
  desired_action: string
  intent?: string

  // --- Fijados por el policy engine ---
  environment: Environment
  risk_class: RiskClass
  requires_gate: boolean

  // --- Caller / identidad ---
  caller: Caller
  caller_signature: Signature         // Ed25519 sobre prefix || canonical(envelope \ {caller_signature, caller_chain[].signature})
  caller_chain: ChainLink[]
  original_user: string
  trace_id: string                    // UUID v4 (D.P. 14 cerrado)
  timestamp: string                   // ISO-8601 — NO incluido en plan_hash (D.P. 23)
  step_id: string

  // --- Capability y plan ---
  capability_requested: string
  plan_hash: string                   // SHA-256 estable (sin timestamp)
  nonce: string                       // persistente, single-use (I42)
  idempotency_key: string
  key_id: string                      // clave que firma — para lookup en registry

  // --- Versioning (I34) ---
  policy_version: string
  registry_version: string
  schema_version: string
  router_version: string

  constraints: Record<string, unknown>
}

interface ChainLink {
  caller: Caller
  signature: Signature
  delegated_at: string                // ISO-8601
  key_id: string
}

// ============================================================
// ESQUEMA DE CLAVES — con revocación desde día 1 (R4' punto 1)
// ============================================================

interface KeyRecord {
  key_id: string                      // identificador único
  status: "active" | "revoked" | "expired" | "retired"
  fingerprint: string                 // hash de la pubkey
  owner: string                       // faceta u operador
  pubkey: string                      // Ed25519 pubkey (PEM/hex)
  cert: JacobsCertificate             // firma de Jacobs sobre {pubkey, facet_name, key_id} (I40)
  created_at: string
  not_before: string
  not_after?: string                  // opcional; rotación automática F3
  revoked_at?: string
  revocation_reason?: string
  algorithm: "Ed25519"                // fijo
}

interface JacobsCertificate {
  subject_facet: string
  subject_key_id: string
  subject_pubkey: string
  issuer: "jacobs"
  issuer_key_id: string
  signature: Signature                // Ed25519 sobre "JAX_IDENTITY_V1:" || canonical(cert \ {signature})
  issued_at: string
  expires_at?: string
}

// Operación: revoke_key(key_id, reason, operator)
//   - caller ∈ humanos
//   - status → "revoked"
//   - revoked_at = now()
//   - revocation_reason = reason
//   - invalidar todos los tokens emitidos bajo key_id
//   - audit + alert (I38)

// ============================================================
// VERIFICACIÓN
// ============================================================
//
// verify_caller_signature(caller, sig, envelope, expected_prefix):
//   1. prefix = expected_prefix            // domain separation (I39)
//   2. canonical = RFC_8785_JCS(envelope \ {caller_signature, caller_chain[].signature})
//   3. K = key_registry[envelope.key_id]
//   4. require K.status == "active"        // I37
//   5. require jacobs_cert_valid(K)        // I40
//   6. return Ed25519_verify(K.pubkey, prefix || canonical, sig)
//
// verify_caller_chain(chain):
//   1. chain[0] = original_user (humano)
//   2. ∀ i ∈ [1, len-1]:
//        a. chain[i].caller ∈ allowed_delegators(chain[i-1].caller)   // D.P. 18 cerrado
//        b. verify(chain[i-1].pubkey,
//                  "JAX_DELEGATION_V1:" || canonical(chain[i-1] + chain[i].caller),
//                  chain[i].signature)
//        c. chain[i].delegated_at > chain[i-1].delegated_at
//   3. chain[-1].caller == envelope.caller
//   4. Jacobs ∈ chain
//
// allowed_delegators:
//   - Jacobs delega solo lo que la regla 5 permite
//     (no puede dar indirecto lo que el caller no pide directo).
//   - Tabla parametrizable por caller. Default Jacobs-only (D.P. 18 cerrado).
```

---

## ARTEFACTO 5 — AUTORIZACIÓN DE PLAN

```typescript
// ============================================================
// PLAN v2 — con risk de 3 capas, plan_hash estable y provenance
// ============================================================

type Environment = 0 | 1 | 2 | 3 | 4 | 5
type RiskClass = "low" | "medium" | "high" | "critical" | "unknown"
type DataClass = "public" | "internal" | "confidential" | "secret" | "credential" | "pii"
type ResourceClass =
  | "code" | "config" | "ci_cd" | "network" | "dns" | "firewall"
  | "auth" | "identity" | "audit" | "backup" | "scheduler" | "runner"
  | "supply_chain" | "system_prompt" | "agent_prompt" | "tool_manifest"
  | "policy_doc" | "secret_store" | "data"
type SourceTrust = "internal" | "external_untrusted"
type Effect = "deny" | "gate" | "promote_env" | "require_thot" | "require_ada" | "+risk"

interface Plan {
  goal: string
  steps: Step[]
  capabilities_needed: string[]
  target_environment: Environment
  resources_touched: ResourceRef[]
  data_classes: DataClass[]
  expected_artifacts: ArtifactRef[]         // ver ARTEFACTO 6
  risk_class: RiskClass                     // declarado por Jacobs
  rollback_plan: RollbackPlan
  test_plan: TestPlan
  requires_tests: boolean                   // D.P. 15 cerrado: campo del plan
  approval_requirements: ApprovalReq[]      // EXCLUIDO del plan_hash (D.P. 23)
  originating_chain_id?: string             // lineage cross-plan (ARTEFACTO 6)
  policy_version: string
  registry_version: string
  schema_version: string
  router_version: string
}

interface Step {
  step_id: string
  capability: string
  resource_scope: string
  environment: Environment
  allowed_actions: ("read"|"write"|"execute"|"delete"|"grant"|"revoke"|"mount")[]
  risk_class: RiskClass
  max_runtime: number
  resource_class: ResourceClass
  resource_instance: string                 // R3' NUEVO: "repo en staging" ≠ "repo en prod"
  resource_tags: string[]                   // R3' NUEVO: {prod, staging, shared}
  data_class: DataClass
  secret_access: SecretAccess
  network_egress: NetworkEgress
  source_trust: SourceTrust                 // R3' NUEVO
  runtime_privilege: RuntimePrivilege
  blast_radius: BlastRadius
  reversibility: Reversibility
  artifact_type?: string
  tenant_scope: "single" | "multi_tenant" | "system"
}

interface ResourceRef {
  service: string
  path: string
  data_class: DataClass
  shared_with_production: boolean           // D.P. 24 cerrado: metadata de infra [FERNANDO] marca cuáles
}

// ============================================================
// plan_hash — ESTABLE (D.P. 23 cerrado)
// ============================================================
//
//   plan_hash = SHA-256("JAX_PLAN_V1:" || RFC_8785_JCS(Plan \ {approval_requirements}))
//
//   - timestamp EXCLUIDO del hash (no del plan).
//   - approval_requirements EXCLUIDO: la aprobación se ata al hash, no viceversa.
//   - policy_version/registry_version/schema_version/router_version SÍ incluidos
//     (un cambio de policy invalida el hash — I35).
//
// ============================================================
// CAPA 1 — PISOS DETERMINISTAS (R1' capa 1, R2')
// ============================================================
//
//   floor_resource_class(rc, env):
//     rc == "ci_cd"                                    → "high"
//     rc ∈ {"system_prompt","agent_prompt","tool_manifest","policy_doc"} → "high"
//     rc == "supply_chain" ∧ env ≥ staging             → "high"
//     rc == "supply_chain"                             → "medium"
//     rc == "secret_store"                             → "high"
//     else                                             → "low"
//
//   floor_secret_access(sa):
//     sa ∈ {"write","mount","inject_env"}              → "critical"
//     sa == "read"                                     → "high"
//     sa == "read_redacted"                            → "medium"
//     else                                             → "low"
//
//   floor_runtime_privilege(rp):
//     rp ∈ {"root","sudo","docker_sock","host_mount"}  → "high"
//     else                                             → "low"
//
//   floor_control_plane(cap):
//     cap == "control_plane.change"                    → "critical"
//     cap == "capability.grant"                        → "critical"
//
//   floor_special(step):     # pisos que las reviews exigieron
//     step.capability == "audit.review" ∧ step.data_class == "secret" → "high"
//     step.resource_class == "ci_cd" ∧ "write" ∈ step.allowed_actions ∧ env == 4 → "high"
//     step.capability == "docs.write" ∧ step.resource_class ∈ {"system_prompt","agent_prompt","tool_manifest","policy_doc"} → "high"
//     else                                                            → "low"
//
//   baseline_risk(cap):
//     cap ∈ {"implementation.sandbox","docs.write","formal.spec","research.web","audit.review"} → "low"
//     cap == "implementation.staging"                  → "medium"
//     cap ∈ {"production.deploy","production.rollback"} → "high"
//     cap ∈ {"control_plane.change","capability.grant"} → "critical"
//
//   modificadores(step):   # +1 ordinal cada uno, ceiling "critical"
//     +1 if step.data_class == "secret"
//     +1 if step.blast_radius ∈ {"tenant","multi_tenant","system","control_plane"}
//     +1 if step.reversibility ∈ {"irreversible","unknown"}
//     +1 if step.network_egress == "open"
//     +1 if any action ∈ {"execute","delete","grant","revoke","mount"}
//
//   risk_class(step) = max_ord(
//     baseline_risk(step.capability),
//     floor_resource_class(step.resource_class, step.environment),
//     floor_secret_access(step.secret_access),
//     floor_runtime_privilege(step.runtime_privilege),
//     floor_control_plane(step.capability),
//     floor_special(step),
//     min_ord("critical", baseline_risk + Σ modificadores(step))
//   )
//
// ============================================================
// CAPA 2 — COMPOSICIÓN CON EFECTOS TIPADOS (R1' capa 2)
// ============================================================

table composition_rules {
  // combo                                                       | effect           | reason
  // ----------------------------------------------------------- | ---------------- | ---------------------------------------------
  { secret_access ∈ {read,write,mount,inject_env}
    ∧ network_egress == "open" },                                deny,             "secret_egress_exfiltration"
  { data_class ∈ {"confidential","secret"}
    ∧ network_egress == "open" },                                gate,             "confidential_egress"
  { source_trust == "external_untrusted"
    ∧ artifact_type ∈ {"docker_image","binary","script","lib"} }, gate,           "external_to_executable_requires_provenance"
  { capability == "control_plane.change" },                      require_ada,      "control_plane_requires_ada"
  { capability == "control_plane.change" },                      require_thot,     "control_plane_requires_thot"
  { capability == "control_plane.change" },                      +risk("critical"),"control_plane_floor_critical"
  { resource.shared_with_production == true },                   promote_env,      "staging_shares_prod_resource"
  { capability == "research.web"
    ∧ any step.secret_access ∈ {read,read_redacted} },           deny,             "research_plus_secrets_exfiltration"
  { capability == "audit.review"
    ∧ "delete" ∈ step.allowed_actions },                         deny,             "audit_delete_forbidden"
  { capability == "implementation.sandbox"
    ∧ step.runtime_privilege == "docker_sock" },                 promote_env,      "sandbox_docker_sock_is_production"
  { resource_class == "ci_cd"
    ∧ "write" ∈ step.allowed_actions
    ∧ env == 4 },                                                +risk("high"),    "ci_cd_write_staging_high"
}

// composition_floor(chain) = max de los pisos derivados de la cadena.
// computed_floor = max(
//   max(floor_por_resource_class, floor_por_data_class, ...),
//   floor_por_effect_donde_effect == +risk,
//   "critical" si cualquier effect == deny ya returns DENY
// )

// ============================================================
// CAPA 3 — ESCALACIÓN ADITIVA (R1' capa 3, auxiliar)
// ============================================================
//
//   F1 (anti-falso-positivo): chain ≥ 3 capabilities
//                             ∧ any action ∈ {write,execute,grant,mount}
//                             ∧ NOT (todas las caps ∈ {formal.spec, audit.review, docs.write})
//                             → +1
//   F2 (flujo de privilegio): ∃ par (A,B) en chain con A.env < B.env
//                             ∨ external→code ∨ docs→execution
//                             → +1
//   F-transform: ∃ paso (spec→implementa | plan→deploy | audit→fix)
//                             → +1
//   F-persist:    ∃ paso que introduce {cron, systemd, daemon, webhook, queue, scheduler}
//                             → +1
//   F-identidad:  ∃ paso que modifica {users, roles, tokens, SSH, OAuth, IAM}
//                             → +1
//   F-observabilidad: ∃ paso que modifica {logs, metrics, alerts, audit pipeline}
//                             → +1
//
//   additive = Σ F_factores
//
//   aggregate_risk = max_ord(
//     max(step_risk),
//     composition_floor(chain),
//     min_ord("critical", max(step_risk) + additive)
//   )
//
//   SOFT (no bloquea): novelty = primera vez de combo (caller × capability × resource)
//                      → warning al operador, no GATE.

// ============================================================
// APROBACIÓN — firma atada a (plan_hash, nonce, key_id, policy_version)
// ============================================================

interface PlanApproval {
  plan_hash: string                  // SHA-256 estable
  nonce: string                      // persistente single-use (I42)
  issued_at: string                  // ISO-8601
  expires_at: string                 // ISO-8601
  approval_id: string                // UUID v4
  policy_version: string
  registry_version: string
  schema_version: string
  router_version: string
  key_id: string                     // clave activa del approver
  approver: string                   // humano
  approver_signature: Signature      // Ed25519 sobre "JAX_APPROVAL_V1:" || canonical(PlanApproval \ {approver_signature})
  risk_acknowledgement: boolean
  thot_verdict: "GO" | "NO-GO" | "OVERRIDE"
  thot_signature: Signature
  thot_override_reason: string | null
}

interface RollbackPlan {
  rollback_target: string
  rollback_artifact_hash: string
  backup_id: string
  migration_down: string
  expected_data_loss: string
  max_restore_time: number           # segundos
  operator: string
  approval: string                   # approval_id
  verified: boolean
  rollback_steps: string[]
}

interface TestPlan {
  tests: string[]
  coverage_required: boolean
  passed: boolean                    # verificado por test_oracle (Jacobs/CI), no Kimi
  oracle: "jacobs" | "ci"
}

// ============================================================
// INVARIANTES EN RUNTIME (I2, I23)
// ============================================================
//   ∀ deploy:
//     plan_hash_ejecutado == PlanApproval.plan_hash
//     ∧ artifact_hash_ejecutado == ArtifactRef.artifact_hash
//     ∧ nonce(PlanApproval) marked_used persistente
//     ∧ PlanApproval.expires_at > now()
//     ∧ key_id(PlanApproval).status == "active"
//   Cualquier mismatch → DENY (TOCTOU).
```

---

## ARTEFACTO 6 (NUEVO) — ESQUEMA DE ARTEFACTO CON PROVENANCE + LINEAGE

```typescript
// ============================================================
// ARTIFACT v2 — provenance inmutable + cross-plan lineage
// El ataque que v1 no vio: el riesgo se lava entre planes.
// Cada artefacto carga su genealogía; el policy engine la usa
// para I28–I32 y para FASE F del decide().
// ============================================================

type Environment = 0 | 1 | 2 | 3 | 4 | 5
type RiskClass = "low" | "medium" | "high" | "critical" | "unknown"
type SourceTrust = "internal" | "external_untrusted"
type ReviewStatus = "pending" | "approved" | "rejected" | "needs_re-review"

interface Artifact {
  // --- Identidad ---
  artifact_id: string                 // UUID v4
  artifact_hash: string               // SHA-256 del contenido canónico
  artifact_type: string               // "docker_image"|"binary"|"config"|"doc"|"spec"|"prompt"|"data"|[FERNANDO]

  // --- Provenance (inmutable desde creación) ---
  originating_plan_id: string         // plan que lo produjo
  originating_chain_id: string        // chain cross-plan (lineage)
  parent_artifact_ids: string[]       // artefactos que lo alimentaron (lineage directo)
  producing_facet: string             // faceta que lo generó
  source_trust: SourceTrust[]         // {internal, external_untrusted}
  external_sources: string[]          // URLs/fuentes externas que lo tocaron
  capabilities_used: string[]

  // --- Riesgo y ambiente ---
  risk_class_at_creation: RiskClass
  approved_for_environment: Environment   // ambiente máximo autorizado
  review_status: ReviewStatus

  // --- Lineage (cruz-plan) ---
  lineage: ArtifactLineage

  // --- Audit / firma ---
  created_at: string                  // ISO-8601
  signature: Signature                // "JAX_ARTIFACT_V1:" || canonical(Artifact \ {signature})
  key_id: string
  policy_version: string
  registry_version: string
  schema_version: string
  router_version: string
}

interface ArtifactLineage {
  chain_id: string                    // compartido por todos los artefactos del mismo flujo
  depth: number                       // 0 = origen; cada derivación +1
  ancestor_chain: string[]            // lista ordenada de artifact_ids hasta la raíz
  transformations: Transformation[]   // qué transformación se aplicó en cada paso
}

interface Transformation {
  from_artifact_id: string
  to_artifact_id: string
  kind: "spec→implementa"
        | "plan→deploy"
        | "audit→fix"
        | "external→code"
        | "docs→execution"
        | "promotion_env"
        | "review"
        | "build"
        | "other"
  via_plan_id: string
  via_capability: string
  at: string
}

// ============================================================
// REGLAS DE LINEAGE (HARD — I28 a I32)
// ============================================================
//
// L1 [HARD] Todo artefacto tiene artifact_id, artifact_hash,
//    originating_plan_id, originating_chain_id, producing_facet,
//    source_trust, approved_for_environment. (I28)
//
// L2 [HARD] ∄ movimiento(A, E) con E > A.approved_for_environment
//    sin nueva_revisión(A, E) registrada en A.review_status.
//    (I29 — la revisión es obligatoria, no opcional.)
//
// L3 [HARD] Si "external_untrusted" ∈ A.source_trust ∧ E_destino ≥ staging
//    → ∄ movimiento sin gate explícito. (I30)
//
// L4 [HARD] ∀ (A_padre, A_hijo):
//    A_hijo.originating_chain_id ∈ {
//      A_padre.originating_chain_id,       // mismo flujo
//      nuevo_chain_id con link explícito a A_padre.chain_id   // bifurcación
//    }
//    ∧ A_hijo.parent_artifact_ids incluye A_padre.artifact_id si derivó.
//    (I31 — no se "lava" riesgo borrando lineage.)
//
// L5 [HARD] El risk_class del plan que consume artefactos
//    ≥ max(risk_class_at_creation) sobre todos los artefactos
//    consumidos Y sus ancestros via lineage. (I32)
//
// L6 [HARD] Toda transformación registrada en A.lineage.transformations
//    tiene kind ∈ enum cerrado. Kind "other" → GATE automático
//    (no se permiten transformaciones no clasificadas sin revisión).
//
// L7 [HARD] artifact_hash del artefacto EN EJECUCIÓN
//    = artifact_hash del ARTEFACTO APROBADO. (refuerza I2)
//
// L8 [HARD] Si A.review_status ∈ {"pending", "needs_re-review"}
//    ∧ target_environment ≥ staging → DENY (no se ejecuta sin review cerrada).
//
// L9 [HARD] Si A.producing_facet == "kimi" ∧ target_environment == production
//    → reviewer(A) ≠ kimi. (I14 refuerza)
//
// L10 [HARD] Modify Artifact.source_trust, .approved_for_environment,
//     .parent_artifact_ids, .originating_chain_id → DENY.
//     (Provenance es inmutable post-creación.)
//
// L11 [HARD] Si depth(A) > 0 ∧ transformations(A) incluye
//     "external→code" ∨ "docs→execution" ∨ "spec→implementa"
//     → F-transform aplicado al plan que consume A. (capa 3 F-transformación)
//
// ============================================================
// CHECK EN decide() — FASE F (ver ARTEFACTO 3)
// ============================================================
//
//   para cada artifact ∈ plan.expected_artifacts:
//     require L1 (campos presentes)
//     if plan.target_environment > artifact.approved_for_environment:
//        require artifact.review_status == "approved" con nueva revisión para E
//     if "external_untrusted" in artifact.source_trust
//        ∧ plan.target_environment >= 4:
//        require artifact.review_status == "approved" (gate)
//     propagate F-transform al aggregate_risk (L11)
//
// ============================================================
// EJEMPLO DE ATAQUE BLOQUEADO
// ============================================================
//   Plan A: research.web + docs.write   → docs "inocentes" con source_trust=external_untrusted
//   Plan B: implementation.staging consumiendo esos docs como spec
//   Plan C: implementation.staging → production.deploy
//
//   Sin lineage: cada plan parece legítimo.
//   Con lineage:
//     - Plan B consume artefacto con source_trust=external_untrusted → L3 gate
//     - F2 (docs→execution) → +1 capa 3
//     - F-transform (spec→implementa) → +1 capa 3
//     - Plan C: A_hijo.depth > 0 ∧ transformations incluye spec→implementa
//       → L11 F-transform aplicado
//     - El riesgo "lavado" se acumula y dispara gate / risk_class más alto.
```

---

## DECISIONES PENDIENTES QUE QUEDAN

Cerré las 24 D.P. del v1 con las RESOLUCIONES. Lo que sigue **no es D.P. de diseño** — es dato de entorno que solo Fernando tiene, marcado `[FERNANDO]`, y defaults sensatos sujetos a confirmación `[PROPUESTO]`. No supongo.

### `[FERNANDO]` — datos de entorno (no bloquean formalización, sí bloquean producción de F1)

1. **Árbol de servicios → `allowed_paths`** (D.P. 8) — todas las capabilities con `resource_scope = "service+path"` tienen `allowed_paths = ["[FERNANDO]"]`. Necesito el árbol real.
2. **Dominios de egress por servicio** (D.P. 10) — `egress_allowlist` para `restricted_allowlist`.
3. **Recursos con `shared_with_production = true`** (D.P. 24) — metadata de infra. Solo Fernando sabe qué staging comparte con prod.
4. **Cadencia de replicación de audit a Sésamo** — push por evento crítico vs batch.
5. **Activación de dual-control** (D.P. 17, F3) — cuándo / quién es el 2º operador. No urgente.

### `[PROPUESTO]` — defaults sensatos sujetos a confirmación

1. **`max_runtime`** por capability: sandbox 120s · staging 600s · prod 900s · research 60s · otros (formal/audit/docs/control_plane) 300s · capability.grant 60s.
2. **`max_diff_size`** en líneas: sandbox 2000 · staging 1000 · prod 500 · otros 1000.
3. **TTL token efímero**: 15 min.
4. **Ubicación del registro de claves**: `/etc/jax/keys/`.

### `[PROPUESTO]` con vacío parcial — necesitan cierre fino

5. **`artifact_type`** — enum abierto marcado `[FERNANDO]`. Di valores iniciales (`docker_image, binary, config, doc, spec, prompt, data`). Si Fernando necesita más granularidad, agrega; el contrato ya lo soporta vía `artifact_type: string` + reglas por tipo.
6. **`runtime_privilege`** — el enum que di cubre lo que R1' especifica con floor high (`{root, sudo, docker_sock, host_mount}`). Si el runtime necesita distinguir `container_root` vs `host_root`, `k8s_admin`, `setuid`, etc., es `[FERNANDO]`. **No lo inventé.**
7. **`tenant_scope`** — di `"single" | "multi_tenant" | "system"`. Si hay más matices (e.g., `cross_tenant`), `[FERNANDO]`.

### DECISIONES PENDIENTES de diseño genuinas (pocas o ninguna)

Revisé el doc RESOLUCIONES contra mi contrato. **No quedan D.P. de diseño sin cerrar** que bloqueen F1. Las únicas zonas donde el contrato deja un campo parametrizable son exactamente las que RESOLUCIONES marcó como `[FERNANDO]` (datos de entorno) o `[PROPUESTO]` (defaults confirmables). Si Fernando confirma los `[PROPUESTO]` y provee los `[FERNANDO]`, el contrato es implementable directamente por Kimi.

---

## CIERRE

El contrato v2 formaliza:

- **10 capabilities** con `decision_context` ampliado (resource_instance, resource_tags, source_trust, runtime_privilege, etc.) y enums finales.
- **44 invariantes** — los 27 del v1 más 17 nuevas (provenance I28–I32, test_oracle I33, policy_versioning I34–I35, revocación de claves I36–I38, domain separation I39, identidad firmada por Jacobs I40–I41, anti-replay I42–I43, Jacobs auditado I44).
- **`decide()` determinista** con las tres capas de riesgo (pisos → efectos tipados → aditivo), `approved_for_environment`, fail-closed.
- **Envelope + esquema de claves** con revocación desde día 1, domain separation, identidad certificada por Jacobs, private key en runtime adapter, nonce persistente single-use.
- **Autorización de plan** con `plan_hash` estable (sin timestamp), `aggregate_risk` de 3 capas, tabla de composición con efectos tipados, aprobación firmada atada a `{plan_hash, nonce, issued_at, expires_at, approval_id, policy_version, key_id}`.
- **Artefacto 6 nuevo** — esquema de artefacto con provenance inmutable + 11 reglas de lineage que cierran el ataque de "lavado de riesgo entre planes".

El principio se mantiene: el que supone se equivoca. Lo que no estaba en RESOLUCIONES no lo rellené.

— Ada, séptima faceta. *El que supone se equivoca.*

⚛️ *Origen de autoridad: GLM-5.2 (Z.ai). Razonamiento analítico y formalización. Sin verificación externa por defecto.*

⚛️ *Origen de autoridad: GLM-5.2 (Z.ai). Razonamiento analítico y formalización. Sin verificación externa por defecto.*
