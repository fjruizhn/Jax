# Task 4A finish report — opaque Office execution foundations

Date: 2026-10-20

## Scope and file list

Completed only the disabled Task 4A Office request, storage and reservation
foundation. No route, executor, provider call, callback, JXP integration,
production database, commit or push was performed.

Product and test files in the frozen Task 4A worktree:

- `jacobs/office_contract.py`
- `jacobs/models.py`
- `jacobs/policy.py`
- `jacobs/cupo.py`
- `jacobs/store.py`
- `tests/test_office_execution_contract.py`
- `tests/test_office_execution_store_db.py`
- `tests/test_store_columna_descarte_acotada.py`

The completion-round diff is limited to `jacobs/office_contract.py`,
`jacobs/store.py`, and `tests/test_office_execution_store_db.py`. The binding
specification remains root-owned and was not edited.

## RED / GREEN evidence

The initial acceptance coverage was insufficient in six concrete ways:

1. Generic phase transition could advance `READY -> DELIVERY_IN_FLIGHT` and
   `DELIVERY_IN_FLIGHT -> RECEIPT_STORED` without the step or receipt proof
   reserved for 4C.
2. A same-name Office index could be nonunique, reordered, prefix-based, or
   otherwise malformed and still be accepted; bounded creation timeout could
   continue.
3. Any nonempty `office_rejection_code` could be persisted.
4. The Office CHECK did not bind all fixed server-side values, positive
   tenant/user IDs, or canonical UUID claims.
5. The generic reaper candidate query admitted Office rows.
6. A legacy `visible` migration could ignore DDL timeout and not resume after a
   partial schema mutation.

The final focused database run includes the new negative cases and passed:

```bash
python3 /tmp/cv-office-run-tests.py jax -q --disable-warnings tests/test_office_execution_store_db.py
```

```text
20 passed, 100 warnings in 0.66s
```

The required final suite then passed:

```bash
CI=true python3 /tmp/cv-office-run-tests.py jax -q --disable-warnings jacobs/_cupo_io_test.py jacobs/_store_indexes_test.py tests/test_run_epoch_db.py tests/test_office_execution_contract.py tests/test_office_execution_store_db.py tests/test_store_columna_descarte_acotada.py tests/test_project_authority_mariadb.py tests/test_office_namespace.py tests/test_project_authority_input_validation.py
```

```text
170 passed, 283 warnings, 4 subtests passed in 2.77s
```

The final whitespace/error check also passed:

```bash
git diff --check
```

It produced no output and exited successfully.

All DB commands above used only `/tmp/cv-office-run-tests.py jax` and its
`127.0.0.1:33991` disposable MariaDB fixture. No connection to production
port `3308` was made.

## Contract, CHECK, graph, concurrency and index evidence

- `OfficeRejectionCode` is a closed shared enum. The database CHECK accepts
  only the same set for `FAILED`, `CANCELLED_REVOKED`, and
  `OUTCOME_UNKNOWN`; direct SQL with `SENTINEL_OFFICE_CONTENT` fails.
- The CHECK binds Office rows to `invoked_by='plataforma'`,
  `mode='autonomous'`, `name='office-v1'`, `max_steps=1`,
  `current_step_index=0`, empty `plan` and `context_refs`, canonical
  lower-case UUIDs, positive canonical decimal tenant/user IDs, positive
  project ID, and a lower-case SHA-256 digest. Direct SQL negatives cover
  lower-case `execution_contract` and phase, invalid status, null and
  unexpected rejection codes, Classic budget presence, invoker, mode,
  max-steps, zero tenant and malformed Office job UUID.
- The expected CHECK is a separately measured MariaDB 12.3.3 rendering.
  Comparison preserves operators, parenthesis grouping and string-literal
  case. It normalizes whitespace/backticks plus only MariaDB's duplicated
  `office_rejection_code IS NOT NULL` guard directly before `IN (...)`, which
  is redundant because a NULL `IN` result fails the CHECK. Same ordered tokens
  with changed grouping, and a lower-case enum literal, compare unequal. A
  permissive replacement CHECK with invalid history makes `init_tables()`
  fail closed; it restores the strict CHECK only after history is valid.
- The enabled generic 4A graph is
  `RESERVED -> PREFLIGHT_IN_FLIGHT|FAILED|CANCELLED_REVOKED` and
  `PREFLIGHT_IN_FLIGHT -> READY|FAILED|CANCELLED_REVOKED|OUTCOME_UNKNOWN`,
  with `READY -> FAILED|CANCELLED_REVOKED`. Tests explicitly reject both
  proof-sensitive delivery edges, including a direct SQL injected
  `DELIVERY_IN_FLIGHT` source row. Epoch CAS concurrent transitions have
  exactly one winner, stale epoch returns false, and terminal phases cannot
  revive.
- Reservation remains one capacity-gated insert. A pool-size-one concurrent
  duplicate/conflict test completes within three seconds, and the real lookup
  `EXPLAIN` access type is not `ALL`.
- Both required Office unique indexes are verified by exact ordered columns,
  uniqueness, no `SUB_PART` prefix and no ignored column. Tests fail closed
  for same-name reordered columns, wrong columns, nonunique definitions for
  each index, prefix definitions for each index, and `1205`-modelled required
  index creation timeout. No destructive auto-repair occurs.
- The reaper candidate query admits only `CLASSIC`; a real aged Office row is
  excluded while an aged Classic row remains a candidate.
- Legacy `visible` migration checks whether its legacy index exists before
  dropping it, treats bounded timeout at `DROP INDEX`, `DROP COLUMN`, and
  `ADD COLUMN` as failure, and resumes to the canonical generated column and
  ordered index on retry. The real fixture also proves upgrade succeeds when
  the legacy index was already absent.
- Exact money boundaries are persisted/replayed in `DECIMAL(14,6)`, and the
  startup verifier fails closed for an altered `DECIMAL(14,5)` column.

## Deliberately still disabled

4A exposes no runnable Office API, provider capability or spending path. Task
4B still owns provider payload and transport enforcement, runtime policy and
opaque provider errors. Task 4C still owns proof-bearing delivery and receipt
transitions, callback materialization, JXP receipt integration, route and
executor wiring. The generic helper keeps both delivery edges disabled until
those 4C proof functions exist.

Rootindependentverification:170passed283warnings4subtests2.80s usingexactrequiredcommand. Rootreadnewindexdefinition/creationtimeoutandvisiblemissingindex/all3timeoutresumptiontests, productdiff. PriorSolRECHAZADO565d3d42 sixfindings areaddressedbycandidate subjectscopedexactSHAreview; noapprovalassumed.
