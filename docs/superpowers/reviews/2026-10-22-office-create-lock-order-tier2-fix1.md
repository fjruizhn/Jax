# Office create lock order · Tier 2 fix 1

2026-10-22 · addresses the quality findings in
`2026-10-22-office-create-lock-order-audit-initial.md` for producer SHA
`67be370e8d0ece2f12993acfde92f8482476c3e8`.

The namespaced-create loop now classifies ER_CHECKREAD (1020) only when the
exception is a raw `pymysql`/`aiomysql` driver error. It keeps the retry local
to this create path. After the same five-attempt bound used by the existing
contention paths, raw 1020 raises the existing `ProjectAuthorityRetryable`
with code `reintentar` and a static message; no driver text is exposed. An
ordinary application exception carrying `(1020, ...)` is not retried and
passes through after one mutation attempt.

The module overview now distinguishes classic operations, whose first
statement is the tenant mutex, from namespaced Office creation, whose initial
indexed actor-tenant hint is non-locking and is followed by ascending distinct
tenant locks and the actor recheck.

Focused verification used only the private C1 runner:

```
python3 /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q --disable-warnings tests/test_office_namespace.py tests/test_office_namespace_mariadb.py
# 36 passed, 55 warnings

python3 /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q --disable-warnings tests/test_office_namespace.py tests/test_office_namespace_mariadb.py tests/test_project_authority_mariadb.py tests/test_office_runner.py tests/test_office_callbacks.py tests/test_office_submit_db.py tests/test_office_preflight_policy.py tests/test_office_runtime_policy.py tests/test_office_execution_store_db.py tests/test_dispatch_step_reroute.py tests/test_prevuelo_orquestador.py tests/test_prevuelo_sonda.py tests/test_prevuelo_reglas.py tests/test_jacobs_cancel_reaper_epoca.py tests/test_jacobs_reaper_cas_db.py
# 424 passed, 405 warnings
```

The five-attempt raw-1020 exhaustion case asserts `reintentar`, static text,
and absence of driver text. The application-1020 negative case asserts one
attempt. No public API, protocol, namespace marker, permission source, DDL,
or guard changed.
