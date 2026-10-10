# Faro Rule Authority MariaDB Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prevent the Rule Authority integration fixture from connecting to MariaDB's temporary initialization server and preserve safe diagnostics if startup or migration fails.

**Architecture:** Reuse the proven readiness contract already present in the authority-ledger MariaDB fixture: wait for entrypoint initialization or the final `port: 3306` server log, then require a successful `SELECT 1` and `@@port = 3306` before applying DDL. Keep retries limited to transient startup probes; never retry migration statements. On setup failure, attach timestamped container logs and non-secret container state to the test failure before cleanup.

**Tech Stack:** Python 3.12/3.14, pytest, PyMySQL 1.2.0, Docker MariaDB 12.3.3.

**Spec:** Forensic Tier 3 report for the repeated `OperationalError 2013` in `tests/policy/test_faro_rule_authority_storage_mariadb.py`; existing readiness reference in `tests/policy/test_authority_ledger_storage_mariadb.py`.

## Global Constraints

- Preserve the exact measured floor in `ci/pisos.json`; increase it only for these regression tests and only from the exact CI count.
- Keep the MariaDB container isolated with `--network none` and one container per test.
- Never retry a migration statement after a lost connection because DDL may have committed partially.
- Never print the root password or full container environment in diagnostics.
- Keep any added regression tests narrowly scoped and remeasure the floor exactly.

## Review Focus

- MariaDB's temporary `port: 0` server must not satisfy readiness; test the parser with realistic temporary and final startup logs.
- A final server that accepts the socket but reports a wrong port must not reach migration; assert the probe requires `@@port = 3306`.
- Transient startup errors may retry only within a bounded deadline; access denied and other permanent errors must propagate immediately.
- Failure diagnostics must include `docker logs --timestamps` and container `.State`, while excluding environment variables and credentials.
- Migration errors must remain fail-fast and identify the statement ordinal without retrying DDL.

---

### Task 1: Harden the Rule Authority test fixture

**Files:**
- Modify: `tests/policy/test_faro_rule_authority_storage_mariadb.py`
- Modify: `ci/pisos.json`
- Modify: `docs/ci/pisos.md`
- Create: `docs/historia/2026-10-10-faro-mariadb-readiness-race.md`
- Create: `docs/superpowers/plans/2026-10-10-faro-rule-authority-mariadb-readiness.md`

**Interfaces:**
- Consumes the MariaDB log readiness pattern already used by the authority-ledger fixture.
- Keeps the existing `db()` and `_apply_migration(connection)` fixture interfaces used by the tests.

- [ ] Add a focused regression proving `port: 0` is rejected and final `port: 3306` / `init process done` is accepted.
- [ ] Run that test and verify it fails against the current fixture before implementation.
- [ ] Add bounded final-server readiness probing, including `SELECT 1` and `SELECT @@port`; do not execute migration until both prove the final server is active.
- [ ] Add safe failure diagnostics (timestamped Docker logs and `.State`) and statement ordinal context; do not include environment values or retry DDL.
- [ ] Run the focused readiness tests and the full Rule Authority MariaDB suite; verify the exact count is 20 passed.
- [ ] Increase the floor and update its history in `ci/pisos.json` and `docs/ci/pisos.md` from the exact CI result; do not lower any floor.
- [ ] Add the incident history with observed evidence, high-confidence root-cause hypothesis, what remains unproven, alternatives rejected, and the next CI verification.
- [ ] Commit the implementation with the required Codex co-author trailer.
