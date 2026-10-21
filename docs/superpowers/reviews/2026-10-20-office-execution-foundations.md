# Task 4A finish report — opaque Office execution foundations

Date: 2026-10-20

## Scope completed

Completed only the disabled 4A contracts, Jacobs schema/store/cupo reservation
and regression tests. No route, executor, provider call, probe policy,
callback, JXP integration, service identity, production database, commit or
push was changed.

Product files in the frozen WIP:

- `jacobs/office_contract.py`
- `jacobs/models.py`
- `jacobs/policy.py`
- `jacobs/cupo.py`
- `jacobs/store.py`
- `tests/test_office_execution_contract.py`
- `tests/test_office_execution_store_db.py`
- `tests/test_store_columna_descarte_acotada.py`

The pre-existing changed binding specification remains owned outside this
completion report.

## RED / GREEN completion

The pre-completion focused suite was green but did not prove the required
acceptance matrix: it accepted a same-name CHECK based only on the substring
`office_accepted_cost_usd`, did not bind every phase to its status and code,
and duplicate recovery selected `client_request_id OR office_stage_id` with
`LIMIT 1`. New negative tests were written for those gaps before the final
implementation was accepted.

The final CHECK uses the full canonical Office grammar. Its expected form is
the exact `information_schema.CHECK_CONSTRAINTS.CHECK_CLAUSE` measurement from
MariaDB 12.3.3 after adding the DDL to the disposable 33991 fixture. The DDL
source and this measured representation are separate because MariaDB renders
`BINARY` as `CAST(... AS CHAR CHARSET binary)`, elides selected parentheses,
and renders `FALSE` as `0`. The comparison normalizes only whitespace and
identifier backticks; it preserves grouping, operators and string literal
case. Tests prove a same-token regrouping and a lower-case `OFFICE_V1` literal
do not compare equal. A real DB test replaces it with a permissive CHECK,
stores invalid historical Office state, and proves that `init_tables()` fails
while restoring the strict CHECK; it does not silently repair invalid history.

## Contract and database evidence

- `office-work.1` is the sole request version and maps to `OFFICE_V1`.
  Request IDs are UUID-normalized; positive IDs and timeout reject strings,
  booleans and floats. Unknown request fields remain forbidden.
- Accepted Office budgets reject float, nonfinite, negative, over-eight-digit
  integer, and over-six-fractional values before calling the shared
  `formatear_usd`. `0`, `0.000001`, and `99999999.999999` are accepted;
  `1`, `1.0`, and `1.000000` bind to one digest. The permanent subtype field
  is checked at startup as exactly `DECIMAL(14,6)` and a real `DECIMAL(14,5)`
  alteration fails closed.
- The real DB test persists each boundary amount, reads the exact canonical
  six-place representation, replays its request, and rejects a changed budget
  under the same client idempotency key.
- The DB CHECK rejects direct SQL for lower-case contract/phase values under a
  case-insensitive database collation, nonterminal `pending`, `FAILED` with no
  code, `RECEIPT_STORED` with a code, `OUTCOME_UNKNOWN` with no code, and an
  Office row carrying the Classic budget. It requires all Office metadata and
  keeps all Office metadata NULL for `CLASSIC`.
- The complete allowed edge table is exercised:
  `RESERVED -> PREFLIGHT_IN_FLIGHT|FAILED|CANCELLED_REVOKED`,
  `PREFLIGHT_IN_FLIGHT -> READY|FAILED|CANCELLED_REVOKED|OUTCOME_UNKNOWN`,
  `READY -> DELIVERY_IN_FLIGHT|FAILED|CANCELLED_REVOKED`, and
  `DELIVERY_IN_FLIGHT -> RECEIPT_STORED|FAILED|CANCELLED_REVOKED|OUTCOME_UNKNOWN`.
  Terminal phases have no generic exit. Status is derived in the store, and
  code presence is derived from the target phase.
- Two simultaneous epoch-CAS attempts have exactly one winner; a stale epoch
  cannot advance the phase. `OUTCOME_UNKNOWN` is terminal and is in the
  no-cupo partition.
- Reservation remains one `INSERT ... SELECT` capacity decision. Duplicate
  handling reads the exact client key first through its held cursor; stage
  collision is checked only after that, so another row cannot be returned by
  an `OR ... LIMIT 1` selection. The pool-size-one duplicate/conflict test
  completes within its three-second bound. The real idempotency lookup has an
  `EXPLAIN` result whose access type is not `ALL`.
- Existing classic rows receive `CLASSIC` by default. The generated classic
  visibility expression explicitly excludes Office rows and only an exact
  known legacy expression is rebuilt; unknown expressions fail closed.

## Commands and results

All database runs used only the root disposable helper fixture at
`127.0.0.1:33991`; no production `3308` connection was used.

Initial focused baseline before the missing acceptance tests:

```bash
python3 /tmp/cv-office-run-tests.py jax -q --disable-warnings tests/test_office_execution_contract.py tests/test_office_execution_store_db.py tests/test_store_columna_descarte_acotada.py
```

```text
31 passed, 20 warnings, 4 subtests passed in 0.18s
```

Focused final validation including the dedicated-pool allowlist guard:

```bash
python3 /tmp/cv-office-run-tests.py jax -q --disable-warnings jacobs/_store_pool_test.py tests/test_office_execution_contract.py tests/test_office_execution_store_db.py tests/test_store_columna_descarte_acotada.py
```

```text
122 passed, 273 warnings, 55 subtests passed in 9.78s
```

Focused canonical-CHECK correction validation:

```bash
python3 /tmp/cv-office-run-tests.py jax -q --disable-warnings tests/test_office_execution_store_db.py tests/test_store_columna_descarte_acotada.py tests/test_office_execution_contract.py
```

```text
44 passed, 45 warnings, 4 subtests passed in 0.27s
```

Required final validation:

```bash
CI=true python3 /tmp/cv-office-run-tests.py jax -q --disable-warnings jacobs/_cupo_io_test.py jacobs/_store_indexes_test.py tests/test_run_epoch_db.py tests/test_office_execution_contract.py tests/test_office_execution_store_db.py tests/test_store_columna_descarte_acotada.py tests/test_project_authority_mariadb.py tests/test_office_namespace.py tests/test_project_authority_input_validation.py
```

```text
159 passed, 228 warnings, 4 subtests passed in 2.16s
```

Also run:

```bash
git diff --check
```

It exited successfully with no output.

## Deliberately still disabled

4A creates no runnable Office API or provider capability. Task 4B still owns
no-tools provider payload enforcement, one-call runtime policy application,
opaque stable errors, catalog transport validation and probe behavior. Task
4C still owns callback materialization/complete/fail bindings, JXP receipt
integration, the dedicated route and executor wiring. Delivery-specific
running-step fencing remains unexposed until 4C provides that contract.

No provider spending, callback, automatic recovery, classic resume/reaper
claim, activation or production deployment follows from this slice.

## Root independent gate

Codex independently inspected product contract/store/cupo/schema. Initialsame-nameCHECKnormalizer removedparentheses andlowercasedliterals; rootrejected beforecheckpoint andfreshTier2 replacedwith exactmeasuredMariaDB12.3.3literal pluswhitespace/backticknormalizationonly. Newtestsdeny same-tokenregrouping/casechanges. Rootfinalrequiredcommand159passed228warnings4subtests2.18s. No rollout or classicconsumer-isolation claim;4B/C remainmandatory.

Historical harness correction: originalworkerclassic_cupo collection failure came from roothelper PYTHONPATH shadowing barecontrato_dispatch withJXPbackend. Rootreproducedimportpath, changedONLY/tmp/cv-office-run-tests.py JAXenv toJAX:las_manos perexistingCI, confirmed15classic_cupocollection then136tests. No productdependency/importrepair needed. Later159includesclassic_cupo.
