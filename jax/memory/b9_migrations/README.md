# B9 shared memory migrations

The B9 schema is additive inside the physical `jax_memory` shared schema. It
is not executed by application import or worker startup. Apply it only through
the repository's reviewed database migration workflow.

`memory_events` is operational append-only history, not a B4 authority ledger
and not B7 evidence.
