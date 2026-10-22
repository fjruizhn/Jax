# Napkin Runbook

## Curation Rules
- Prioritize recurring guidance; maximum ten entries per category.
- Each entry names a concrete alternative action.

## Execution & Validation
1. **[2026-09-25] Memory tests must never bootstrap production governance.**
   Do instead: use externally supplied `jax_test` credentials and a fresh `jax_memory_test_*` database. Inspect fixture/conftest startup before execution; run the explicit integration driver for real resolver and trigger coverage.
2. **[2026-09-25] Helpers do not operate remote infrastructure.**
   Do instead: let the principal execute SSH, backup, restore, schema provisioning and deployment; reviewers inspect code and evidence without secrets.
3. **[2026-09-25] A successful SQL commit can lose its response.**
   Do instead: inspect durable job and origin markers before retrying; never regenerate paid output or infer rollback from an exception.

## Memory Boundaries
1. **[2026-09-25] Legacy ownership is structured data.**
   Do instead: revalidate the active owner, tenant, referenced rows and project membership inside adoption; quarantine ambiguous or obsolete rows without inventing a destination.
2. **[2026-09-25] Equal event timestamps are not a causal ordering.**
   Do instead: preserve strict event time ordering when one transaction imports and publishes a legacy revision; test replay with inversely ordered UUIDs.
3. **[2026-09-25] Counts and audit logs can expose memory.**
   Do instead: report counts, states and error types; keep content, digests, dumps and credentials in protected files.
