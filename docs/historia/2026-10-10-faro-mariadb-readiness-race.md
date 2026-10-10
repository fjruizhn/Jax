# Faro F1.1: readiness race in the MariaDB integration fixture

**Date:** 2026-10-10  
**Type:** HISTORIA / PENDIENTE  
**Source:** GitHub Actions run `38049519208`, job `114205701482`, exact head `3e74c61463d1c7c6727b0fd40134bd774015b3f8`; independent Tier 3 forensic report on the same fixture.

## What happened

The post-merge CI for #391 succeeded, but the exact CI for the floor-retirement grant #390 reproduced the Rule Authority MariaDB failure: 11 tests passed and one errored with `pymysql.err.OperationalError (2013, Lost connection to MySQL server during query)` while `_apply_migration()` executed DDL during fixture setup. The stack shows that the fixture had accepted a connection after checking only that the Unix socket existed. The test harness then removed the container, so that run did not retain its MariaDB logs or `.State` for a definitive diagnosis.

This was the same failure class previously seen in the post-merge CI for #385 (run `38022192036`, while executing `SELECT VERSION()`) and another Rule Authority run (run `38046736780`, while applying a migration). The same suite also passed in an intervening run without changing its fixture. The authority-ledger MariaDB fixture already documents that the MariaDB 12.3.3 entrypoint can expose a temporary server on `port: 0` before the final `port: 3306` server is ready.

## Decision and implementation direction

On Fernando's autonomous GO, Codex decided to repair the Rule Authority fixture before resuming #390/#387. The high-confidence hypothesis is that socket-only readiness admitted a connection to MariaDB's temporary initialization server, which was then shut down while the migration ran. This remains a hypothesis for the failed containers because their logs were discarded.

The repair waits for the entrypoint-complete marker or the final port-3306 server log, then requires `SELECT 1` and `SELECT @@port = 3306` before applying DDL. Startup probes retry only bounded transient connection errors; migration DDL is never retried. Setup failures attach timestamped Docker logs and the non-secret `.State` object before container cleanup. Migration diagnostics identify the statement ordinal without emitting credentials.

## Alternatives rejected

- Retrying the migration after error 2013: DDL may have committed partially, so replaying it could hide a partial schema.
- Retrying CI until it happens to pass: the intervening green did not remove the readiness race.
- Lowering the `authority-rule-storage/mariadb` floor or skipping the fixture: that would conceal the failure rather than establish readiness.

## Pending verification

The implementation is on branch `codex/faro-mariadb-readiness` in an isolated worktree. Its exact CI must show 15 passed in the dedicated MariaDB floor, with no skipped/error cases, before its auditor/preflight/merge. Then #390 must be re-based: both the floor definition hash and the exact diff hash authorized for #387 depend on the new master tree. The #388 owner was notified that their branch touches this fixture and the authority-ledger fixture; their worktree will not be edited by Codex.

## Lesson

An available socket is not proof that a database entrypoint has finished switching from its initialization server to the final server. Preserve enough runtime evidence to distinguish startup handoff from a server crash or resource failure.
