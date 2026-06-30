ada: Tercera iteración. El contrato v2 (incluido abajo como DOCUMENTO 2) pasó por auditoría adversarial de Thot, que dio veredicto NO-GO con 10 bloqueadores + hallazgos. Las correcciones están consolidadas en el DOCUMENTO 1 (CORRECCIONES v3). Tu tarea: producir el CONTRATO v3 que cierra TODOS los hallazgos según esas correcciones.

PRINCIPIO RAÍZ de esta iteración (REGLA MAESTRA): casi todos los bloqueadores eran la misma enfermedad — el motor confiaba en datos autodeclarados por el plan/artefacto para decisiones de seguridad. La cura: TODO campo que afecte una decisión de seguridad (resource_class, data_class, source_trust, runtime_privilege, blast_radius, reversibility, risk_class, lineage, parent_artifact_ids) se DERIVA de fuente autoritativa (registry determinista path→clase, o runtime monitor). El plan PROPONE; el motor/runtime DERIVA y DECIDE.

Reescribí estos artefactos del v2 aplicando las correcciones A–G del DOCUMENTO 1:

- ARTEFACTO 3 (decide): reordenado. Nueva Fase A0 (validación estructural ANTES de firma/nonce). promote_env SIEMPRE retorna GATE (nunca muta y sigue); sandbox+{docker_sock,host_mount,root,sudo} = DENY directo. Envelope inmutable dentro de decide(). cap se recarga tras promoción. Nonce se RESERVA (no consume) hasta ALLOW/DENY terminal. Fase F (lineage) ANTES del cálculo de aggregate_risk. Redacción enforced con código real. Matriz de enforcement por riesgo EFECTIVO (critical SIEMPRE gate+thot+rollback+tests+audit_remoto, independiente de la capability).

- ARTEFACTO 1 (capabilities): las 10 con allowed_actions Y denied_actions explícitos (ausencia = DENY). Token = intersección(step.allowed_actions, cap.allowed_actions). denied_paths globales mínimos que SIEMPRE ganan (ci_cd, prompts, policy, infra, secrets).

- ARTEFACTO 2 (invariantes): agregá I45 (binding capability↔step, decisión por step), I46 (allowed_actions obligatorio), I47 (metadatos derivados autoritativos), I48-I49 (enforcement por riesgo efectivo), I50 (anti-prompt-injection para auditoras), I51 (adapter restringido). Corregí I28 (source_trust array bien cuantificado + propagación de external_untrusted a descendientes).

- ARTEFACTO 4 (cripto): JAX_ROOT_IDENTITY_KEY offline (soporte físico genérico/migrable, fuera de hall9000 y desconectado — el contrato NO especifica el aparato, solo exige offline ∧ separado_de_hall9000) que firma el cert de Jacobs — raíz de confianza. Domain separation COMPLETA en I39 (incluí JAX_IDENTITY_V1, JAX_DELEGATION_V1, JAX_ARTIFACT_V1, JAX_ARTIFACT_REVIEW_V1, JAX_ROOT_IDENTITY_V1). Runtime adapter = signer restringido (nonce del policy engine, policy local, rate limit, audit). Delegación (ChainLink) atada a contexto completo (capability, env, scope, actions, TTL, nonce).

- ARTEFACTO 6 (provenance): REESCRITO. Lineage y source_trust DERIVADOS del runtime monitor (observed_inputs), no autodeclarados. expected_artifacts reconciliado contra inputs observados (input no declarado = DENY). source_trust no se puede "limpiar" (union de padres observados). CREATE Artifact solo por trusted_builder/runtime, no por faceta LLM. artifact_hash por content-addressed storage. Definí ArtifactRef y ArtifactReview (que faltaban). 

- THOT: separá availability_override (Thot caído, severidad menor) de security_incident (Thot NO-GO). Veto con TTL. I50 anti-injection.

NOTA [A-ESCALA] (respetar, NO re-expandir): la root de Jacobs es offline-en-USB, NO HSM ni quorum. El fallback de Thot es operador-solo, NO quorum de N auditores. Ambos cierran el hallazgo a la escala real de un operador solo.

Estándar de rigor: dato de entorno → [FERNANDO]; default sensato → [PROPUESTO]; ambigüedad → DECISIÓN PENDIENTE. El que supone se equivoca. NO rellenar huecos con supuestos.

Entregá los 6 artefactos completos y tipados, implementables directo por Kimi. Cerrá con las DECISIONES PENDIENTES que queden (deberían ser pocas; las de entorno son [FERNANDO]).
