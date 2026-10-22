# Task 3 · Office membership adapter

## RED

- The prior namespace test asserted `office-namespace.1`; Task 3 requires
  `.2` and had no typed, caller-cursor membership mutation surface.
- The MariaDB migration had no Office membership idempotency record or
  `OFFICE` membership provenance.

## GREEN

- Added typed grant/revoke operations with immutable results, fixed role
  mapping, canonical event rows, request digest/replay records and strict
  Office namespace validation.
- Both public membership methods take the caller cursor and never begin,
  commit or roll back a transaction. Authorization runs after tenant/actor/
  target/scope locks and before every idempotency replay lookup.
- Migration creates `jax_namespace_membership_mutation_request`, widens the
  provenance CHECK to include `OFFICE`, and rejects rollback while Office
  scopes or Office mutation requests remain.
- The MariaDB fixture explicitly creates JXP-owned `account_class`; runtime
  queries do not default a missing column or value to INTERNAL.

## Completion matrix

- Real two-connection same-key race: the first transaction holds the tenant
  mutex while the second is already started and waiting; release produces
  exactly one membership, event and request, plus one replay.
- Caller rollback removes membership, event and request together.
- Grant replay after a later revoke and revoke replay after a later regrant
  return their historical results without changing the current membership.
- Current authorization is invoked before every replay; a newly denied actor
  receives no prior result.
- Real requests bind actor, project, target, member kind and execution /
  invitation id.  Valid alternatives with the same key conflict.
- Grant denial covers inactive scope, foreign target tenant and class mismatch;
  revoke permits archived scope plus inactive target cleanup.  Classic
  mutators remain unable to write an Office scope.
- Capability execution persists `office-invitation:<uuid>` / `CAPABILITY`,
  request execution fields and the fixed external VIEWER role.  Unit coverage
  rejects malformed execution and UUID inputs before SQL.
- Migration test exercises apply -> apply -> revert -> apply and proves that
  a durable Office mutation request blocks rollback before destructive DDL.

## Verification

```text
python3 /tmp/cv-office-run-tests.py jax --unit-no-db tests/test_office_namespace.py
12 passed

python3 /tmp/cv-office-run-tests.py jax \
  tests/test_project_authority_mariadb.py::test_office_membership_same_transaction_grant_replay_and_revoke \
  tests/test_project_authority_mariadb.py::test_office_membership_replays_locks_rolls_back_and_keeps_capability_provenance \
  tests/test_project_authority_mariadb.py::test_migracion_apply_apply_revert_apply_y_falla_cerrado_con_historia_office \
  --disable-warnings
3 passed, 19 warnings

python3 /tmp/cv-office-run-tests.py jax \
  tests/test_project_authority_mariadb.py tests/test_office_namespace.py \
  --disable-warnings
66 passed, 19 warnings
```

The real MariaDB test executes `EXPLAIN` for the actual replay lookup and
asserts `PRIMARY` on `(namespace, idempotency_key)`.

## Concurrency and review

The mutation takes the target tenant `FOR UPDATE` first, so same-tenant
same-key calls serialize before their idempotency record lookup; the request
insert is last, after membership and event writes, so a caller rollback
removes all three. The real test starts both connections, holds the first at
the trusted callback after it owns that tenant mutex, and then releases it;
the second returns the exact replay after waiting.

An exploratory fully simultaneous start (both connections attempting the
tenant lock at once) can receive MariaDB 1213, mapped by the public adapter
to `ProjectAuthorityRetryable`. That is a bounded retryable database outcome,
not a duplicate write: callers owning the transaction must retry the same
idempotency key. The deterministic mutex test above proves the successful
serialized path and exact row cardinality. No retry loop was added inside a
caller-owned transaction.

Self-review checked that the core imports no JXP module, exposes no generic
`ProjectRole` input, preserves classic APIs, uses no raw JXP SQL bypass, and
does not add permission caching or blocking I/O.

## Independent root gate
2026-10-20 · Codex: focused membership/namespace/input-validation suite independently passed67tests,19expectedDDLwarnings,in0.84s via own isolatedMariaDB33991 wrapper. No production. ExactSHAreview follows; runtime caller owns retry/rollback and must retry deadlocks with same key.
