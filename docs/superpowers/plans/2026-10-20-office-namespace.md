# Office Namespace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a fail-closed confidential project namespace and transactional creation hook for Virtual Office.

**Architecture:** Preserve classic JAX authority. Add a namespace boundary and trusted same-transaction authorization adapter; JXP owns office identity and grants.

**Tech Stack:** Existing Python, aiomysql and MariaDB12.3.3; no new dependency.

**Spec:** docs/superpowers/specs/2026-10-20-office-namespace.md

## Global Constraints

- CLASSIC/OFFICE_CONFIDENTIAL; existing rows remain CLASSIC.
- No policy/Faro/workflow edits, live DB, services or user provisioning.
- Namespaced idempotency reauthorizes and binds actor+tenant+payload.
- Test DB only isolated33991, async existing pool, no permission cache.
- Codex parent alone publishes; commits carry Co-Authored-By Codex.

## Review Focus

- Actor changes while replaying an idempotency key: conflict, no leaked ID.
- Permission revoked before replay: authorization fails, no cached success.
- Legacy query lacking namespace filter: confidential project invisible.
- Failure between project and scope inserts: rollback all rows.
- Two concurrent identical creations: one project and stable replay.

---

### Task 1: Namespace and atomic namespaced creation

**Files:** Modify jax/memory/project_authority_migrations.py, scope_authority.py, project_queries.py, project_authority.py, __init__.py. Tests in tests/test_office_namespace.py and tests/test_office_namespace_mariadb.py; retain current tests/test_project_scope_authority.py, tests/test_b9_scope_authority.py and tests/test_project_authority_input_validation.py (baseline27 passing).

**Interfaces:** Consumes existing ProjectAuthorityAdmin/pool/CreatedProject. Produces ProjectNamespace, NamespaceMutationAuthorizer, create_namespaced_project with signatures in spec; existing API signatures remain compatible.

- [ ] Write failing tests for schema/idempotent migration, classic/confidential resolver matrix, callback transaction, replay binding, rollback and concurrent creation.
- [ ] Run focused tests; capture expected failures before code.
- [ ] Implement guarded DDL, namespace checks and shared private row creation. Trace existing project consumers for bypasses; do not alter unrelated paths.
- [ ] Run focused unit tests and real MariaDB tests in disposable33991. Add EXPLAIN evidence and bounded contention measurement. Check classic project regressions.
- [ ] Parent independently verifies, commits and pushes with updated TRASPASO. Sol reviews exact SHA before JXP consumes new capability.

## Plan self-review / ledger

Only one implementation owner touches all shared files. Migration defines the enum column consumed by resolvers; creation consumes the same enum and transaction protocol. Tests above cover the five review-focus cases. No downstream capability is enabled in this slice; subsequent JXP schema/API/MFA/UI plans consume this exact contract.

### Task 2: Paired namespace contract marker

**Files:** jax/memory/project_authority.py and tests/test_office_namespace.py only.

**Interfaces:** Export `OFFICE_NAMESPACE_CONTRACT_VERSION = "office-namespace.1"` from project_authority, identifying the reviewed CLASSIC/Office resolver, transactional creation and legacy LAS MANOS isolation contract. This is a compatibility marker, not a permission or an alternative authorization path. JXP startup compares the exact marker before enabling Office features.

- [ ] Write a failing unit test importing and requiring the exact marker; implement the constant without changing authority/migrations or existing APIs.
- [ ] Run focused namespace unit tests and classic input-validation regressions.
- [ ] Parent verifies and commits; independent exact-SHA scoped review before JXP consumes the marker.

### Task 3: Same-transaction Office membership adapter

**Files:** jax/memory/project_authority.py, project_authority_migrations.py, relatednamespace tests. Specification docs/superpowers/specs/2026-10-20-office-membership.md is binding. Root owns docs.

**Interfaces:** typedgrant/revokecallback and immutable result in spec; existingcreation/classic APIs unchanged. Marker office-namespace.3, membershiporiginOFFICE, idempotencyrecord/eventsin callertransaction. Grantactiveonly, revokecan cleaninactive/archivedscope withcurrentexplicitadminauthority.

- [ ] Read specification; add focused RED tests for the new adapter and regression matrix.
- [ ] Implement minimal transactional methods and migration; no nested transaction, rawJXPSQL bypass, permissionscache or genericProjectRoleinput.
- [ ] Run unit+realMariaDB33991 tests, rollback/concurrency/idempotency/upgrade/classicregressions, EXPLAIN actualqueries.
- [ ] Report task-3-report.md; parent independently verifies, commits, pushes and exactSHA Solreview before JXP consumes newmethod/version.

## Task 4: Opaque Office execution on Jacobs

Binding spec docs/superpowers/specs/2026-10-20-office-execution.md, Solarchitecture+roottransportreview. Deliver bounded sequentialslices4Acontracts/store/cupo/classicvisibility,4Bno-tools+opaqueerrors/probes,4Ccallbacks/executor/route. Reuseallcanonicalgovernance; no extraqueue/executor, no rawOfficecontentdurableclassic, disableduntilpairedJXPgates. Explicitownership per slice and exactSHAaudit. Doesnotmodify namespaceTask1-3authority or enableproduction.

Task4Bdeliverysplit sameTier2owner4B1directinvokers/scopedlogs then4B2probe/preflight/coalescing; bindingoffice-runtime-policy.md, exactSHArevieweach.4Cstillmandatory/unexposed.

Task4C se entrega4C1 DTO/storeatómico/aislamientoclassic sinrutaOffice ni llamadas, luego4C2 callback/config/route/runner/recovery. Contratovinculante adicional docs/superpowers/specs/2026-10-21-office-callback-boundary.md. Cada checkpoint revisiónexacta independiente; mismaobligación4C,no capacidadhabilitadaantesdelparJXP.
