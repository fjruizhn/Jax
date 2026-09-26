# B9 shared memory migrations

The B9 schema is additive inside the physical `jax_memory` shared schema. It
is not executed by application import or worker startup. Apply it only through
the repository's reviewed database migration workflow.

`memory_events` is operational append-only history, not a B4 authority ledger
and not B7 evidence.

## Migration 005 lives only in Python

Migration 005 (project lifecycle -- ARCHIVED/HIDDEN/DISABLED, membership
provenance, idempotent project creation) is applied by
`project_authority_migrations.py::apply_project_authority_migration`, guarded
step-by-step against `information_schema`. There is deliberately **no**
`.sql` copy of it, unlike 003 (`003_project_scope_authority.sql` +
`project_authority_migrations.py`, H2 -- a known, already-accepted piece of
duplication). Adding a second source of truth for 005 would just grow that
same deuda; the guard-per-step Python hook is the only copy, and its bajada
is `revert_project_lifecycle_migration`, run by hand via
`scripts/b9_revertir_005.py --aplicar` (dry-run by default).
