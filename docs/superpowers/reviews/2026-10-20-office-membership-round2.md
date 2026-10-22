# Task 3 · fix round 2 · Office membership adapter

## Corrections

- Replaced broad result CHECK v2 with linked CHECK v3:
  `GRANT/INTERNAL → CONTRIBUTOR/ACTIVE`,
  `GRANT/EXTERNAL_CLIENT → VIEWER/ACTIVE`, and
  `REVOKE/null-kind → any canonical role/REVOKED`; all require `changed=1`.
  Existing impossible request rows stop the upgrade before DDL, without
  rewriting audit history.
- Membership idempotency keys and invitation IDs now normalize to lower-case
  canonical UUID strings before the authorization callback, digest, lookup,
  event principal and persistent request row.  The legacy global UUID
  validator remains unchanged for classic API compatibility.
- CHECK v4 closes MariaDB three-valued-logic behavior: grant branches require
  `member_kind IS NOT NULL`, and the upgrade preflight uses
  `COALESCE(..., FALSE)` so legacy NULL-kind grant rows fail closed.
- Optional execution defaults only for literal `None`; falsey untyped values
  (`False`, `0`, empty string and empty list) reject before SQL.

## Real MariaDB verification

```text
python3 /tmp/cv-office-run-tests.py jax \
  tests/test_project_authority_mariadb.py tests/test_office_namespace.py \
  --disable-warnings
67 passed, 19 warnings in 0.73s

python3 /tmp/cv-office-run-tests.py jax --unit-no-db \
  tests/test_office_namespace.py --disable-warnings
20 passed in 0.13s
```

The migration test attempts real FK-valid impossible rows for grant OWNER,
grant REVIEWER/REVOKED, grant NULL-kind and revoke ACTIVE; MariaDB rejects all
four. It also simulates a v3 table with an impossible pre-existing NULL-kind
grant and verifies the v4 upgrade fails closed before restoring its constraint. It separately proves
duplicate invitation upgrades fail closed. The real membership matrix uses
upper-case request and invitation UUIDs and verifies the callback and database
persist the lower-case canonical identity.

Independentroot:72focusedMariaDB/namespace/inputvalidationtests passed0.77s,19expectedDDLwarnings. RootprecheckpointcaughtSQLUNKNOWN-nullgrant hole andfalseyexecutiondefault; correctedwithinround2, meaningfulnegativesincluded. ExactSHAreviewfollows.

## Independent verdict
Sol office_membership_review **APROBADO** exactaeb7278de923ba3417e52e13ec77d01d8bfcd879; noopenBLOCK/MAJOR. Independently72realtests19warnings. CHECKv4NULLsafe/linkedresult, canonicalUUID, strictNoneexecution, capabilitybinding/provenance/elevatedcleanup allclosed. Scope core .2 adapter only; no invitations/accounts/productionenabled by this approval.
