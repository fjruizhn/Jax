# Office project creation · canonical cross-tenant lock order

2026-10-22 · OP343/345 · Codex technical ruling accepts read-only Sol office_3c_boundary source findings. This is a minimal prerequisite for planned JXP3C1, not a change to namespace authority or the already approved opaque execution contract.

Current create_namespaced_project owns its store.mutation transaction and locks target tenant before invoking the JXP authorizer. The authorizer's joined actor/profile/tenant FOR UPDATE then locks the actor tenant; this can invert the established ascending affected-tenant order used by identity administration. Core's bounded1213/1205 retry cannot guarantee safety of the competing transaction selected as victim. MariaDBB9Store.mutation only begin/commit/rollback, no retry. Do not treat retry as a lock-order exemption.

## Private implementation, unchanged public API

Inside each fresh creation transaction:

1. Plain indexed canonical jax_users PK lookup discovers actor tenant as a non-authorizing hint. Missing actor denies before authorizer/effects.
2. Lock each distinct actor-tenant hint and target tenant individually in numeric ascending order. Same tenant locks once; no caller-supplied hint/global mutex.
3. Lock canonical actor row and recheck exact tenant equality. Missing actor denies; drift rolls back before authorizer/effect. Never acquire the newly observed tenant later in that transaction.
4. Invoke unchanged trusted authorizer, which still determines all active identity/profile/MFA/Office authority. Continue existing idempotent namespace project insertion without Classic grants/events.

Use a private drift sentinel to restart through existing bounded fresh transaction loop; exhaustion raises existing ProjectAuthorityRetryable code reintentar with stable content-free text. Keep residual deadlock/timeout retry for unrelated contention. No change to public signature, authorizer protocol, namespace marker .2, permission source, role/class eligibility or request digest. Do not return earlier success on replay before current authorization.

## Verification

Trace lower/higher/same tenant order and exact actor row recheck. Missing actor/drift deny or bounded retry before any authorizer/effect; drift retry uses a genuinely fresh transaction. Real MariaDB reversed cross-tenant concurrent creates/compatible correctly ordered competitor demonstrate no1213/1205 and expected authorized results; track exact duplicate/idempotent replay and Classic preservation. EXPLAIN real new PK selects indexed; no new DDL/cache. Separate private C1 schema only, no JXP shared test lease/production changes. Audit exact new producer SHA independently before JXP3C1 consumes it. Root may keep bootstrap/schema/admin work progressing independently.
