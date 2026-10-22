# F2-E Tranche 2 — structured DTO projection

## Authorization and scope

Fernando authorized an additive, explicitly versioned and immutable JSON projection. Server-owned DTO slots may bind only to claims resolved through F2-B. Other fields remain typed untrusted data and retain their per-field origin. SR-03 structured-status protection remains active. F2-D binds the projection to the exact canonical bytes. Existing API/client schemas remain unchanged; no new predicate is accredited.

## First surface

Use jax-platform `GET /api/pipelines` as the first DTO seam. `get_current_user` supplies the authenticated `AuthUser`; the handler scopes its SQL query by both `user_id` and `tenant_id`. The response contains multiple pipeline rows but no step DTOs, so it can bind each top-level `status` to its own `PIPELINE_STATUS` claim under the same authenticated response scope without depending on SR2 `STEP_STATUS`. Each claim resolver independently checks the canonical Jacobs owner and status. Never derive scope from response/body fields. `/jacobs/pipeline` is excluded because its current request path uses service identity and body-supplied user/tenant; the signed delegated-identity design needed for that path is recorded separately for 2026-10-11.

## Bounded implementation

1. Add a closed server-owned structured schema registry and immutable F2-C structured renderer API with exact version matching.
2. Bind each `/pipelines/*/status` field to exactly one assertable `PIPELINE_STATUS` claim in the sealed envelope; derive the emitted value from that claim. The incoming producer DTO must omit status slots. Reject duplicate, missing, unbound, or producer-supplied status fields.
3. Keep every other field typed and origin-tagged as untrusted data. Reject nested/encoded canonical status objects that are not claim slots.
4. Add a separate additive F2-D structured transport API that hashes and carries the exact canonical JSON bytes; apply the existing lifecycle states and reject unauthenticated acknowledgements.
5. Wire only the authenticated platform pipeline-list response in the exact-pair integration. Preserve exact public keys and use a server-owned static fallback on governance failure. The JAX branch implements the reusable F2-C/F2-D APIs and their real-path exact-pair seam; the platform branch consumes the APIs using `AuthUser` scope.

## Tests and gates

Use TDD. Exercise the real jax-platform authenticated `GET /api/pipelines` producer with F2-B `PIPELINE_STATUS` resolution, F2-C projection and F2-D byte-bound response. Prove immutable canonical bytes, each list row status bound to one F2-B claim, untrusted typed data and field-level provenance, SR-03 rejection for nested status payloads, rejected-candidate exclusion, prepared/committing/committed transitions, acknowledgement rejection, fail-closed behavior, exact-version rejection, and exact external schema compatibility. Remeasure all changed CI floors on the branch; do not modify unrelated CI comparator helpers or `pisos-no-bajan`.

## Exclusions

No change to F2-A/B semantics, F2-C text API semantics, F2-D existing text lifecycle semantics, F2-P, STEP_STATUS/SR2, OCR/SR3, tool capability, Faro, production, F2-F or F2-G.
