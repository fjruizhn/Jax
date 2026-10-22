# Office create lock order · Tier 2 implementation record

2026-10-22 · Base `86858c9b972b5b2f97dace668d7575730495efb2` · implementation
of the bounded producer prerequisite in
`docs/superpowers/specs/2026-10-22-office-create-lock-order.md`.

`create_namespaced_project` now makes a plain `jax_users` PK lookup solely
for the actor tenant hint, locks the distinct hinted/target tenant rows in
numeric order, then locks and rechecks the canonical actor row before calling
the unchanged authorizer. A missing actor denies before authorization or
effects. A changed actor tenant rolls back through a private sentinel and
retries in a new bounded transaction; exhaustion is the existing
`ProjectAuthorityRetryable` with code `reintentar` and static text. The
transaction never takes the newly observed tenant lock late.

No public signature, namespace marker, protocol, permission source, DDL, or
Classic/Office creation semantics changed. MariaDB error 1020 is retried only
on this fresh namespace-create path because the mandated non-locking hint can
otherwise make an idempotency replay observe a concurrent insert as
`ER_CHECKREAD`; 1213/1205 handling remains intact. A deterministic raw-1020
test verifies that this branch discards the failed transaction and authorizes
only in its new transaction.

Focused tests parameterize lower-actor, higher-actor, and same-tenant SQL
traces; they also cover missing actor denial, actor-tenant drift with a fresh
cursor, stable retry exhaustion, raw-1020 fresh retry, and replay
authorization. The disposable MariaDB test uses a real mutation boundary that
observes raw database errors and a barrier after both actor hints. It proves
the PK plan, opposite cross-tenant creates, their bound authorizer calls, and
the absence of raw 1213/1205 even though the implementation could retry them.

Verification, exclusively through the private C1 runner:

```
python3 /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q --disable-warnings tests/test_office_namespace.py tests/test_office_namespace_mariadb.py
# 34 passed, 55 warnings

python3 /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q --disable-warnings tests/test_project_authority_mariadb.py
# 51 passed, 19 warnings

python3 /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q --disable-warnings tests/test_office_namespace.py tests/test_office_namespace_mariadb.py tests/test_project_authority_mariadb.py tests/test_office_runner.py tests/test_office_callbacks.py tests/test_office_submit_db.py tests/test_office_preflight_policy.py tests/test_office_runtime_policy.py tests/test_office_execution_store_db.py tests/test_dispatch_step_reroute.py tests/test_prevuelo_orquestador.py tests/test_prevuelo_sonda.py tests/test_prevuelo_reglas.py tests/test_jacobs_cancel_reaper_epoca.py tests/test_jacobs_reaper_cas_db.py
# 422 passed, 405 warnings (the previous 419 baseline plus three focused cases)
```

`git diff --check` passes. This is an implementation record, not the required
independent Sol audit of the exact producer SHA.
