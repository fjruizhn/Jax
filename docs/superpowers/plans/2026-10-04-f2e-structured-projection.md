# F2-E Tranche 2 — structured DTO projection

## Authorization and scope

Fernando authorized an additive, explicitly versioned and immutable JSON projection. Server-owned DTO slots may bind only to claims resolved through F2-B. Other fields remain typed untrusted data and retain their per-field origin. SR-03 structured-status protection remains active. F2-D binds the projection to the exact canonical bytes. Existing API/client schemas remain unchanged; no new predicate is accredited.

## First surface

Use the small Jacobs `POST /jacobs/pipeline` creation response (`pipeline_id`, `status`, `mode`, `step_count`, `message`) as the first DTO seam. The existing response is flat and has an existing producer, while the current pipeline-status resolver already accredits `PIPELINE_STATUS`. Exclude all step DTOs so this tranche does not depend on the separate SR2 `STEP_STATUS` merge. Before implementation, prove the route has a server-derived scope; request-body identity must never manufacture its F2-B scope.

## Bounded implementation

1. Add a closed server-owned structured schema registry and immutable F2-C structured renderer API with exact version matching.
2. Bind the `status` field to the matching assertable `PIPELINE_STATUS` claim in the sealed envelope and the F2-C rendering result; derive its value from that claim. Reject producer-supplied status as untrusted data.
3. Keep every other field typed and origin-tagged as untrusted data. Reject nested/encoded canonical status objects that are not claim slots.
4. Add a separate additive F2-D structured transport API that hashes and carries the exact canonical JSON bytes; apply the existing lifecycle states and reject unauthenticated acknowledgements.
5. Wire only the selected Jacobs creation response. Preserve exact public keys and use a server-owned static fallback on governance failure.

## Tests and gates

Use TDD. Exercise the actual Jacobs route, F2-B receipt path, F2-C projection and F2-D response-byte boundary together. Prove immutable canonical bytes, slot/claim binding, untrusted typed data and provenance, SR-03 rejection, rejected-candidate exclusion, prepared/committing/committed transitions, acknowledgement rejection, fail-closed behavior, exact version rejection, and schema compatibility. Remeasure all changed CI floors on the branch; do not modify unrelated CI comparator helpers or pisos-no-bajan.

## Exclusions

No change to F2-A/B semantics, F2-C text API semantics, F2-D existing text lifecycle semantics, F2-P, STEP_STATUS/SR2, OCR/SR3, tool capability, Faro, production, F2-F or F2-G.
