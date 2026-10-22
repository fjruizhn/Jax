# Office4B2 preflight/probe policy · OP345

2026-10-21 · Codex. Baseline4B1 independent SolAPROBADO56be8bed. New private optional policy propagates through preflight rules, orchestration, probes and coalescing; ClassicNone preserves historical behavior. Office admits one generation call, checks transport and current capability ceiling before probes, rechecks resolved transport, partitions in-flight probes by error-detail policy and uses sharedHTTP/no redirects/no tools.

Root precheck corrected two initial omissions before checkpoint: raw probe response usage/shape/model metadata still bypassed4B1 validation, and several prepare/generic/outer-timeout branches still used Classic detailed errors. The final candidate uses the same strict provider-response parser before health/usage, retains independently valid known counters, maps all Office error branches to closed stable detail, and propagates cancellation. Office health/usage writer failures log closed codes/classes without provider content. Probe success may intentionally truncate its tiny output; this does not bypass response/usage validation.

Tier2 reported111focused tests5.57s; parent independently reruns the same scope: test_office_preflight_policy.py,test_prevuelo_sonda.py,test_office_runtime_policy.py, using the private isolated harness. Tests cover all three shared-client payloads, malformed response/metadata/usage sentinels, closed health taxonomy, preparation/outer timeout/cancellation, actual coalescing partition and previous4B1 privacy regressions. No DB/provider/production operations. Compilation and diffcheck pass. Exact-SHA scoped review is next;4C execution and public routes remain disabled.

Parent completed that independent run: **111passed in5.62s**, zero failures. This verifies the frozen candidate only, not4C or live provider availability.

## Review-fix round 1 · 2026-10-21

The `a9bdf9f0` review found four remaining boundaries and this round closes all
four without enabling 4C:

1. Office now rejects `hyde` before the subscription cost shortcut, and rejects
   any catalog-resolved subprocess transport before probe work.  Both failures
   carry `PREFLIGHT` phase.
2. A valid 2xx without provider usage is `uso_medido=False`; 5xx and read/outer
   timeouts are likewise unmeasured and conservatively estimate only when a
   request may have reached the provider.  HTTP 4xx is recorded as an opaque
   rejection but has no estimated usage because it is a known non-charge path.
   Valid partial usage from parser failures remains measured.
3. The Office switch must be the exact immutable
   `OFFICE_RUNTIME_POLICY` singleton.  Forged namespaces or replacement
   policies fail closed at the rules, preflight, transport and probe ingress.
4. `require_office_runtime_policy`, transport rejection, and capability-timeout
   rejection now use `OfficeErrorPhase.PREFLIGHT`.

Focused regression coverage includes forged policy expansion, the Hyde shortcut,
missing usage for all three transports, and 4xx/5xx accounting boundaries.
The isolated harness command ran `tests/test_office_preflight_policy.py`,
`tests/test_prevuelo_sonda.py`, `tests/test_prevuelo_orquestador.py`, and
`tests/test_office_runtime_policy.py`: **142 passed in 5.97s**.  Python compile
and `git diff --check` passed; no DB, external provider, production, commit, or
push occurred.

## Pre-check follow-up · 2026-10-21

Usage is now measured only when both provider counters are explicitly present
and strict nonnegative integers; explicit `0` remains measured.  Missing,
`null`, empty, or invalid metadata is unmeasured and reaches the existing
conservative estimate branch.  Parser errors retain safe partial counters but
carry an unmeasured marker when the usage metadata itself was invalid.  HTTP
status errors are unmeasured; 4xx remains non-charge and 5xx estimates.

The Office motor-preparation denial now explicitly has `PREFLIGHT` phase.
Focused isolated coverage adds null/empty/all-invalid and explicit-zero usage
for Gemini/OpenAI/Ollama.  Result: **151 passed in 5.99s**, compilation and
diff check passed.

## Review-fix round 2 · 2026-10-21

Parser failures now retain whether usage was actually complete and measured.
The three probe closures mark parser failures as post-HTTP (`provider_call_made`
true), so valid `13/17` counters from a malformed 2xx response reach the real
usage writer.  A malformed response without usage and a `MockTransport`
`ReadTimeout` are unmeasured and use conservative estimated accounting.

`evaluar_paso()` validates the Office policy before its subprocess shortcut.
Runtime policy/transport helpers now accept an explicit phase context and
default to `GENERATION`; preflight callers pass `PREFLIGHT`.  This avoids
mislabeling direct generation denials while preserving preflight semantics.

Focused full-path tests cover shared AsyncClient → MockTransport → parser →
sondear → usage writer for valid partial usage, absent usage, and ReadTimeout.
Focused isolated suite: **154 passed in 5.99s**; compilation and diff check
passed.

## Final phase-context precheck · 2026-10-21

`validate_office_capability_timeout()` now accepts an explicit phase and
defaults to `GENERATION`; `prevuelo` supplies `PREFLIGHT`.  Its excessive-timeout
coverage verifies both call contexts.  A parser failure after Office probe HTTP
is cloned with `PREFLIGHT`, while retaining its truthful post-call marker.
Focused isolated suite: **155 passed in 5.99s**; compilation and diff check
passed.

## Review-fix round 3 · 2026-10-21

Usage provenance now lives on `OfficeProviderResponse`, rather than the
error-class default.  All three direct Office invokers copy it into truncation
errors; full provider usage remains measured while unknown metadata remains
unmeasured.  Probe estimated accounting now takes the per-counter maximum of
verified partial usage and the conservative estimate, so it never replaces a
real `13` with a lower estimate.

The added tests cover full measured/truncated Gemini/OpenAI/Ollama parser
responses and full probe writer preservation for partial counters.  The prior
155 result was a worktree validation after a parent packaging omission; the
committed candidate then contained 154 tests.  This round's worktree command
passes **158 tests in 6.01s**, plus compilation and diff check.

## Final Classic compatibility precheck · 2026-10-21

Partial-counter preservation is Office-only.  Classic `runtime_policy=None`
continues to write the historical estimated pair exactly; a focused regression
asserts this boundary.  Focused isolated suite: **159 passed in 5.99s**;
compilation and diff check passed.
