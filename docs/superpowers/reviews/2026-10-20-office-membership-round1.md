# Task 3 · fix round 1 · Office membership adapter

## Corrections

- The mutation-request result constraint now accepts every canonical
  `ProjectRole` for revoke history. Grants remain code-fixed to
  `CONTRIBUTOR` or `VIEWER`; a pre-existing `REVIEWER` membership revokes and
  records its actual role without stranding it.
- `NamespaceMembershipMutationAuthorizer` now receives the canonical
  idempotency key as well as the typed execution. The request table has a
  unique `(namespace, invitation_id)` index. Exact-key replay reads its own
  record first; a different key for the same accepted invitation conflicts
  before membership/event side effects. Upgrade detects legacy duplicate
  invitation consumption before creating the index and fails closed without
  rewriting history.
- New membership `created_by` matches the event principal: `user:<id>` for
  interactive work and `office-invitation:<uuid>` for capability acceptance.
  The request row retains the canonical issuer user FK.

## Real MariaDB verification

```text
python3 /tmp/cv-office-run-tests.py jax \
  tests/test_project_authority_mariadb.py tests/test_office_namespace.py \
  --disable-warnings
67 passed, 19 warnings in 0.70s

python3 /tmp/cv-office-run-tests.py jax --unit-no-db \
  tests/test_office_namespace.py --disable-warnings
16 passed in 0.12s
```

The real matrix includes an elevated `REVIEWER` revoke, exact K1 replay,
K2 rejection after a later revoke of the same invitation, capability races
with different keys (one membership/event/request survives), and a simulated
pre-index duplicate upgrade that refuses before the unique index is added.

## Limitation

Two transactions that acquire the tenant mutex at precisely the same instant
can still receive MariaDB 1213, surfaced as `ProjectAuthorityRetryable`; the
caller owns the transaction and retries the same idempotency key. The
controlled two-connection tests prove the serialized path and cardinality.

Root independent68tests passed0.73s,19expectedwarnings, sameisolatedMariaDB33991; includedinputvalidation. No prod. ExactSHAreviewnext.
