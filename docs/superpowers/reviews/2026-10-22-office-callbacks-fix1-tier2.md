# Office callbacks fix1 Tier 2 implementation record

2026-10-22 · Implements the binding reconciliation ruling for Task 4C2 fix round 1. The runner now uses canonical capability admission, preserves `READY` on pre-claim ambiguity or kill, records exact catalog model and output hash, and performs periodic READY recovery. See `.superpowers/sdd/2026-10-20-office-namespace/task-4c2-fix1-report.md` for the finding-to-test map and commands.

This is an implementation record, not an approval or release verdict. The Office gate remains disabled.

Verification on 2026-10-22 used the private C1 schema runner only:

```
/home/fruiz/jax/.venv/bin/python /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q tests/test_office_runner.py tests/test_office_callbacks.py tests/test_office_submit_db.py tests/test_office_preflight_policy.py tests/test_office_runtime_policy.py tests/test_office_execution_store_db.py
```

Result: `191 passed, 286 warnings in 1.72s`. The selected matrix includes the real `start_reaper_loop` periodic sweep that schedules a READY Office pipeline, `test_real_route_preflight_input_stale_is_unknown_not_cancelled`, and `test_real_route_preflight_execution_error_reconciles_typed_callback`. `git diff --check` was clean.

The reaper EXPLAIN regression uses a self-cleaning private-C1 fixture with 350
terminal Classic rows and three live candidates, followed by `ANALYZE TABLE`
on the private test database. It retains the strict `status`-leading range
assertion for `p`, the `idx_steps_pipeline` ref assertion for `s`, and the
no-filesort/no-temporary checks. The broader verification command was:

```
python3 /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q --disable-warnings tests/test_office_runner.py tests/test_office_callbacks.py tests/test_office_submit_db.py tests/test_office_preflight_policy.py tests/test_office_runtime_policy.py tests/test_office_execution_store_db.py tests/test_dispatch_step_reroute.py tests/test_prevuelo_orquestador.py tests/test_prevuelo_sonda.py tests/test_prevuelo_reglas.py tests/test_jacobs_cancel_reaper_epoca.py tests/test_jacobs_reaper_cas_db.py
```

Result: `330 passed, 316 warnings in 7.71s`; `git diff --check` passed.
