# C5 Solo Órdenes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let C5 audit cloud missions that touch customer data while sending the cloud provider only mission objective, instructions, machine identity, and commands, never captured outputs or claims.

**Architecture:** Add a fail-closed boolean configuration key. Carry an explicit audit mode with the resolved auditor choice, project the outbound request from an allowlist, and reject claim verdicts in `SOLO_ORDENES`. Record the chosen facet/mode in the mission event stream and the append-only startup record. Keep platform admin declaration in a separate PR if required.

**Tech Stack:** Python 3, pytest, httpx, append-only C3 registry.

**Spec:** `docs/superpowers/specs/2026-09-18-auditor-local-opcion.md` (amend before production code).

## Global Constraints

- New key: `ejecutor.c5_auditor_nube_solo_ordenes`, strict bool, required in config, default false in seed.
- Only `auditor_faceta` (configured cloud auditor, today thot) can run sensitive missions in `SOLO_ORDENES` when the key is true.
- Outbound fields are an explicit allowlist; no line, context, output, stderr, claim text, step input, or error output.
- Any assertions returned in `SOLO_ORDENES` are invalid; claims remain marked as not audited by C5 for human supervision.
- No production database, `/etc`, `/opt`, or `PENDIENTES.md` changes.

## Review Focus

- Nested secrets in tool inputs, stderr, output, and claim fields: assert no marker byte appears in the serialized HTTP request.
- Missing/invalid new config: reject before resolving or using a cloud auditor.
- Sensitive mission with the new key off: preserve current local auditor choice.
- Cloud auditor bound to the wrong provider or mode: preserve provider and locality checks.
- Malformed/claim-bearing provider response in `SOLO_ORDENES`: fail closed and do not approve claims.

---

### Task 1: Specify the privacy and authority contract

**Files:**
- Modify: `docs/superpowers/specs/2026-09-18-auditor-local-opcion.md`

- [x] Add a dated decision section defining `SOLO_ORDENES`, exactly which data leaves, what stays local, superadmin `/admin/config` authorization, audit logging, and false default.
- [x] Review that the section states claims are not audited and that identity reads required by the executor contract remain in scope.

### Task 2: Implement mode selection, projection, and interpretation

**Files:**
- Modify: `jax/ejecutor/contratos/eleccion_c5.py`
- Modify: `jax/ejecutor/contratos/auditor.py`
- Modify: `jax/ejecutor/contratos/auditor_cliente.py`
- Modify: `jax/ejecutor/contratos/auditor_instrucciones.md`
- Modify: `jax/ejecutor/contratos/arranque.py`
- Modify: `jax/ejecutor/mision.py`
- Modify: `jax/ejecutor/mision_servicio.py`
- Modify: `jax/ejecutor/contratos/vigia_servicio.py`
- Modify: `jax/ejecutor/proxy_carril.py`
- Add: `jax/ejecutor/contratos/c3_control.py`
- Test: `tests/test_ejecutor_contratos_eleccion_c5.py`
- Test: `tests/test_ejecutor_contratos_auditor.py`
- Test: `tests/test_ejecutor_contratos_auditor_cliente.py`
- Test: `tests/test_ejecutor_mision.py`

- [x] Add failing tests for strict config parsing, sensitive-mode selection and validation, allowlist-only request projection, body-level secret absence, and claim verdict rejection.
- [x] Run those tests against the unchanged baseline and confirm they fail for the intended behavior.
- [x] Implement the smallest changes consistent with the spec, including startup and per-mission C3 audit records through the proxy's sole writer.
- [x] Filter SOLO_ORDENES canaries to projectable Bash commands and exercise the real client/projection through a mock HTTP transport.
- [x] Run affected tests and policy checks.

### Task 3: Config administration split check

**Files:**
- Inspect separately: `/home/fruiz/jax-platform`

- [ ] Check whether `/admin/config` requires a managed-key declaration. If it does, prepare a separate platform worktree/PR only after the JAX implementation; otherwise record that no platform change is needed.

### Task 4: Verification and handoff

- [x] Measure changed-test count delta and update `docs/ci/pisos.md`.
- [x] Make one real minimal thot call through the system resolver/client without exposing credentials or putting them in argv; report exact outcome.
- [ ] Review the exact implementation SHA with an independent Tier 3 adversarial reviewer.
- [ ] Report PR, SHA, test results, test-floor delta, and provider-call result. Do not merge.
