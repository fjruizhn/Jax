# Task 4B1 report — direct runtime privacy and no tools

## Correction round1 · 2026-10-21 · Codex

Independent Sol rejected52d54dc1 for untrusted provider metadata entering catalog/usage, hidden JSON exception context, permissive empty response shapes, incorrect timeout/call provenance, Office assemble bypass and incomplete transport/log/cancellation evidence. A cohesive Tier2 wave validates provider response shape and numeric usage, preserves known usage on stable errors, and never forwards provider-controlled model/version observations to durable catalog helpers for Office. Trusted resolved f.model remains authoritative; Classic observation behavior stays unchanged. Error raising leaves the raw exception handler and clones stable errors without their raw traceback; connect failure is known/no-call, read timeout after transport start is OUTCOME_UNKNOWN. Office assemble denies before work.

Parent independently ran focused runtime policy, truncation, Gemini-header and source compatibility files: **60passed in0.57s**. SharedAsyncClient/MockTransport, true cancellation and filter-failure regressions are included. This is fix evidence pending scoped exact-SHA Sol review;4B2/4C and public execution remain disabled.

Date: 2026-10-20

## RED → GREEN

- RED: the initial focused suite exposed that Classic test doubles accept only
  the historical three-argument direct-invoker call shape.
- GREEN: `runtime_policy=None` now calls each Classic invoker with that exact
  historical shape. Office is the only path that receives the private keyword.
- GREEN: 46 focused tests passed under the disposable Office test environment;
  no provider or production endpoint was contacted.

## Source inventory

- `jacobs/office_runtime.py` (new): immutable discriminator policy selection,
  canonical catalog timeout validation, stable `OfficeExecutionError`, opaque
  provider error mapping, and scoped HTTP logger filtering for the six audited
  `httpx`/`httpcore` emitters.
- `jacobs/executor.py`: adds the private `runtime_policy` forwarding only to
  direct HTTP invokers. Office Gemini makes one no-tools request with no
  grounding/retry/sources/redirect resolution. Office OpenAI-compatible and
  Ollama requests are tool-free. Every Office POST uses
  `follow_redirects=False`. Classic remains policy `None`.
- `tests/test_office_runtime_policy.py` (new): MockTransport payload,
  stable-error, discriminator, logging-scope, timeout-context, version-gate,
  and cleanup coverage.

## Tests

Command (using the same disposable environment supplied by
`/tmp/cv-office-run-tests.py`, with this worktree as the code root):

```
/home/fruiz/jax/.venv/bin/python - <<'PY'
import sys, subprocess
sys.path.insert(0, '/tmp')
from cv_office_test_env import environment
root = '/home/fruiz/.local/share/codex-worktrees/jax-office-namespace-20261020'
env = environment(); env['PYTHONPATH'] = root + ':' + root + '/las_manos'
raise SystemExit(subprocess.run([
    '/home/fruiz/jax/.venv/bin/python', '-m', 'pytest',
    'tests/test_office_runtime_policy.py', 'jacobs/_truncado_falla_test.py',
    'tests/test_gemini_key_en_cabecera.py', 'tests/test_hipatia_fuentes.py', '-q'],
    cwd=root, env=env).returncode)
PY
```

Result: `46 passed in 0.54s`.

Additional focused compatibility run:

```
PYTHONPATH=<worktree>:<worktree>/las_manos \
  /home/fruiz/jax/.venv/bin/python -m pytest \
  tests/test_office_runtime_policy.py jacobs/_truncado_falla_test.py \
  tests/test_gemini_key_en_cabecera.py tests/test_hipatia_fuentes.py \
  tests/test_contrato_dispatch_repl_ada.py -q
```

Result: `84 passed in 0.84s` (24 + 60 focused groups).

## Interfaces for 4B2

- `runtime_policy_for_pipeline(pipeline) -> OfficeRuntimePolicy | None`
- `OfficeCapabilityContext(capability)` plus
  `validate_office_capability_timeout(context, timeout)`; it reads the live
  governance capability map and uses `plan._techo_segundos`, including its
  established 300-second fallback.
- `OfficeExecutionError`: `code`, `phase`, `outcome`, `provider_call_made`,
  optional exception class name, and actual token counters for truncation.
- `ensure_office_http_log_privacy()` repairs a removed filter before each
  Office request and fails closed on version or installation failure.

## Remaining scope

- 4B2: preflight/sonda policy propagation, coalescing partition, health detail
  closure, and generation-call limits across probe paths.
- 4C: JXP materialization, receipt and failure callbacks, Office route wiring,
  raw-content avoidance in persistence, and epoch/step delivery fencing.

No Office execution capability, public route, provider configuration, database
schema, or raw Office persistence was enabled by this task.

Root independent gate: 84 passed in 0.68s via /tmp/cv-office-run-tests.py jax --unit-no-db. Exact-SHA scoped review follows; 4B2 and 4C remain disabled.
