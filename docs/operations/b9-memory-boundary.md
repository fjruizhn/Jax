# B9 shared-memory boundary

Memory is operational history. It is not authority, evidence, execution
authorization, or current state. Consumers use the B9 Memory API and receive
`MemoryEnvelope` objects; model-facing callers construct `PromptMemoryContext`
instead of concatenating database memory strings.

Production scheduled memory jobs are owned by the corresponding systemd units.
The B9 migration SQL is additive and must be applied through the reviewed
database migration workflow, never automatically by importing an application.

## Current-source boundary

No memory content is current truth by itself. A typed B4--B8 or operational
reference is only current when its designated read-only resolver returns a
successful `ResolutionResult`; unavailable references remain unavailable.
Similarity, confidence, verification, and provider metadata cannot promote
memory content to a current claim.

## Lifecycle boundary

Revisions plus append-only events are canonical history. Current projections
are rebuildable caches and sensitive mutation stops with reconciliation
required when they diverge. Persistent B9 mutation bundles commit revision,
payload, provenance, event, relations, and projection atomically.
