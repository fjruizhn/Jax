# Office create lock order · initial exact-SHA Sol review

2026-10-22 · Codex records independent office_3c_boundary review of67be370e8d0ece2f12993acfde92f8482476c3e8, base86858c9b. Specification APROBADO; quality RECHAZADO. Root422passed405warnings10.15s; reviewer independently27puretests, no DB writes.

Canonical lock implementation itself is correct: non-authorizing actor-tenant hint, distinct ascending tenant locks, actor FOR UPDATE recheck, no late lock, bounded fresh drift transactions, unchanged authorizer/API/namespace.2.

MAJOR: newly classified raw ER_CHECKREAD1020 retries five attempts but exhaustion falls through generic error_de_base_de_datos rather than stable retryable reintentar. Branch also examines args[0] without proving raw driver error, so an application exception with1020 is retried. Minimal scoped correction: raw1020-only fresh retries, exhaustion static content-free ProjectAuthorityRetryable; no global1020 retry expansion. Test fiveattempt exhaust/code/no drivertext and nonDB1020 oneattempt negative. This is producer-prerequisite fix1/5, unrelated to closed4C2cap.

MINOR: module invariant falsely states every mutation's first statement is tenant FOR UPDATE; explain creation's initial nonlocking hint and ordered distinct tenants.

Consumer3C1 project creation remains gated until exact-SHA quality approval. No public protocol/schema/permission change or production action.
