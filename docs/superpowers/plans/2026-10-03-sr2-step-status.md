# SR2 — STEP_STATUS implementation plan

## Contract and scope

Add only the human-authorized `STEP_STATUS(step_id, status)` predicate, resolved from one canonical Jacobs read joining the step to its pipeline owner. Use the closed `jacobs.models.StepStatus` enum and observation time from a successful read. Require a confirmed pipeline owner; missing, orphaned, hidden, corrupt, or inaccessible source state fails closed. The Platform presentation aliases (`waiting_gate`, `completed`) are never accepted as canonical step status. No `project_id` mapping is inferred.

## Files and changes

JAX:

- `jacobs/store.py`: server-owned async snapshot query joining `jacobs_steps` and `jacobs_pipelines` in one statement; include canonical owner acknowledgment and pipeline visibility state.
- `policy/vocabulary/predicates.yaml`: add the explicitly authorized closed predicate and arguments.
- `policy/governance/resolution.py`: new adapter kind, exact-scope binding, source identity, and status dispatch registration.
- `policy/governance/runtime_status.py`: resolver, source configuration identity/digest, exact argument contract, registry entry, resolver and binding version cut.
- `policy/governance/governed_domain.py`: add STEP_STATUS grammar and bounded structured-status detection from the exact Jacobs enum; bump domain specification version.
- focused governance tests: canonical enum tripwire; single-query ownership snapshot; every enum value; missing/orphan/ownerless/hidden/corrupt/read failure; wrong owner; unsupported project scope; requested-status mismatch; stale observation; F2-B receipt and F2-C claim integration; preserve existing SR-01..04 coverage.
- `docs/superpowers/plans/2026-10-03-sr2-step-status.md`: this implementation record.

Platform:

- `backend/jax_engine/status_resolution.py`: map the exact-pair STEP_STATUS bridge to canonical JAX evidence; never use UI aliases or an alternate status source.
- focused bridge and exact-pair integration tests: exact source SHA, F2-B receipt, F2-C rendering, F2-D transport bytes, owner-scope failure, unknown-version fail-closed behavior.
- `.github/workflows/policy.yml`: a separate SR2 exact-pair job may be required. This is coordinator-reserved; do not edit until jax-14 confirms the shared-file window. The job must pin the frozen JAX PR SHA and execute source, resolver, renderer, and lifecycle integration against that same checkout.

## Ordered execution

1. Freeze and verify the approved F2-D JAX feature SHA; verify both current master deltas and worktree cleanliness.
2. Add failing JAX tests for the source snapshot and governance behavior; run them to demonstrate the missing predicate.
3. Implement the canonical query and JAX F2-B/F2-C registration; run focused tests and existing F2-E-SR regressions.
4. Add and run enum drift tripwire and structured decoder regressions without widening existing decoder bounds.
5. Freeze a JAX candidate SHA and run required CI.
6. Implement the Platform exact-pair bridge and integration tests against that exact JAX SHA; do not edit reserved workflow files before coordination.
7. With coordinator clearance, add the exact-pair CI job, pin both exact PR identities, run the complete job, and record its artifact.
8. Run a narrow independent Tier-3 audit on the exact JAX/Platform pair; fix findings and repeat the audit on any changed SHA.
9. Report green PRs and audit to Fernando/jax-14. Do not merge or deploy.

## Version and performance constraints

STEP_STATUS changes the closed predicate vocabulary and registry identity. Cut the runtime-status binding/resolver version and F2-C domain version explicitly so old receipts/configurations cannot be reused across the changed contract. Keep renderer API and envelope schema unchanged unless code proves their contracts change. The resolver performs one indexed step primary-key lookup joined to the pipeline primary/unique key; inspect `EXPLAIN` against the test schema. Do not introduce a cache.

## Acceptance gates

- Exactly seven canonical statuses match `StepStatus`; enum drift fails CI.
- The authoritative row and owner metadata come from one SQL statement and one observation timestamp.
- No owner acknowledgment, orphan, hidden pipeline, unknown status, source/config error, or scope mismatch produces a resolved receipt.
- Platform aliases never enter the accredited claim.
- F2-B receipt verification, F2-C renderability, and F2-D exact transport lifecycle all execute in exact-pair CI against the same JAX SHA.
- Existing F2-E-SR invariants remain passing; production, Faro, F2-F, and general F2-E remain untouched.
