# Office callbacks fix2 Tier 2 implementation record

2026-10-22 · Implements the strict UTF-8 reconciliation ruling from
`2026-10-22-office-callbacks-audit-fix1.md`. The Office runner initializes a
single `output_sha256` value, maps strict UTF-8 hashing failure to the stable
`OFFICE_PROVIDER_RESPONSE_INVALID` generation failure with
`provider_call_made=true`, and every exception path reuses that value instead
of hashing output again.

`test_surrogate_provider_json_fails_generation_and_reconciles_real_terminal`
obtains an unpaired surrogate through `json.loads` plus the real provider
response parser. With a real C1 store and real fail DTO MockTransport, it
proves that complete is not called; fail sends GENERATION, model-invoked true,
and a null output hash; and FAILED, CANCELLED_REVOKED, and OUTCOME_UNKNOWN
responses persist their corresponding local terminal phase without a receipt
or catalog model. `test_complete_real_transport_errors_are_delivery_ambiguous`
uses real MockTransport 409, 422, 503, and malformed 200 responses to retain
post-attempt delivery ambiguity.

Verification used only the private C1 runner:

```
python3 /home/fruiz/.local/share/codex-workspaces/office-20261020/cv-office-c1-run-tests.py -q --disable-warnings tests/test_office_runner.py tests/test_office_callbacks.py tests/test_office_submit_db.py tests/test_office_preflight_policy.py tests/test_office_runtime_policy.py tests/test_office_execution_store_db.py tests/test_dispatch_step_reroute.py tests/test_prevuelo_orquestador.py tests/test_prevuelo_sonda.py tests/test_prevuelo_reglas.py tests/test_jacobs_cancel_reaper_epoca.py tests/test_jacobs_reaper_cas_db.py
```

Result: `337 passed, 331 warnings in 7.72s`; `git diff --check` passed. The
Office gate remains disabled. This is an implementation record, not an audit
or release verdict.
