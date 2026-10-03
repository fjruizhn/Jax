# AXIOMA 3.0 — GOBERNANZA UNIVERSAL

## F2-E CURRENT EXTERNAL OUTPUT SURFACE MAP

**Discovery snapshot · 2026-10-03 · Codex · JAX `86391a85971b59b460784cbea194cc5ec7e705bf`**

This is a read-only surface inventory, not an authority decision or an
implementation plan. It records routes and producers visible in the current
tree. Tier 3 confirmed a real semantic block: a human decision is required on
F2-C structured projection and `/health` source protocol. No predicate,
authority binding, closed foundation, or runtime behavior is changed by this
document.

`user` below includes an external service or tool consumer. A producer without
one is internal automatically. `NEEDS_F2E_ADAPTER` means a surviving external
surface lacks the shared governed-output/F2-D integration; it does not assert a
new claim or authorize an action.

| ID | Surface / route | Producer → consumer | Dynamic / external | Current governance / F2-D | Classification | Proposed adapter / closed channel |
| --- | --- | --- | --- | --- | --- | --- |
| JAC-001 | `jacobs/routes.py::preflight` `POST /jacobs/preflight` | Jacobs → HTTP client | generated preflight; external | no F2-C/F2-D boundary | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-002 | `plan_only` `POST /jacobs/plan` | Jacobs → HTTP client | plan; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-003 | `create_pipeline` `POST /jacobs/pipeline` | Jacobs → HTTP client | pipeline state; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-004 | `get_pipeline` `GET /jacobs/pipeline/{pipeline_id}` | persisted Jacobs → HTTP client | current pipeline data; external | canonical state exists, no output adapter | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-005 | `cancel_pipeline` `POST .../cancel` | Jacobs → HTTP client | mutation acknowledgement; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-006 | `discard/recover/hide/restore` routes | Jacobs → HTTP client | mutation acknowledgement; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-007 | `resume_pipeline` route | Jacobs → HTTP client | dynamic pipeline result; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-008 | `continue_pipeline` route | Jacobs → HTTP client | dynamic pipeline result; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-009 | `continue_preflight` route | Jacobs → HTTP client | generated preflight; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-010 | `approve_step` route | Jacobs → HTTP client | dynamic acknowledgement; external | none | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-011 | `get_pipeline_results` `GET .../results` | persisted result/source/error/output refs → HTTP client | raw result; external | no governed renderer/lifecycle | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-012 | `GET /pipeline/{pipeline_id}/events` | raw event store → HTTP client | raw events; external | no governed renderer/lifecycle | NEEDS_F2E_ADAPTER | shared typed adapter / `JACOBS` |
| JAC-013–015 | `jacobs/executor.py` provider/motor invocations | provider or Motor → executor | raw typed producer; no direct external consumer | untrusted internal data | INTERNAL_ONLY | none; preserve typed untrusted data |
| JAC-016 | `_run_one_step` private artifact persistence | executor → private persistence | private artifact | no direct consumer shown | INTERNAL_ONLY | none until operator projection |
| JAC-017 | `_persist_step_to_repo` | executor → durable operator-visible artifact | raw Markdown can become visible | no governed output boundary | NEEDS_F2E_ADAPTER | durable-artifact adapter / `JACOBS` |
| JAC-018 | `jacobs/aviso.py::avisar_fin_pipeline` | Jacobs → Telegram | status notice; external operator | no output adapter | NEEDS_F2E_ADAPTER | operator-status adapter / `JACOBS` |
| JAC-019 | Jacobs reaper notification | Jacobs → Telegram | status/error; external operator | no output adapter | NEEDS_F2E_ADAPTER | operator-status adapter / `JACOBS` |
| JAC-020 | facet health notification | Jacobs → Telegram | status; external operator | no output adapter | NEEDS_F2E_ADAPTER | operator-status adapter / `JACOBS` |
| LM-001 | `las_manos/server.py::health` `GET /health` | server state → HTTP client | health status; external | runtime-status evidence exists; no universal output adapter | NEEDS_F2E_ADAPTER | dynamic-status adapter / `LAS_MANOS` |
| LM-002 | `GET /audit/tail` | audit tail → HTTP client | audit data; external | no `AUDIT_EVENT_EXISTS` accreditation | NEEDS_F2E_ADAPTER | audit-data adapter / `LAS_MANOS` |
| LM-003 | `POST /plan` | plan generation → HTTP client | generated output; external | no governed boundary | NEEDS_F2E_ADAPTER | shared typed adapter / `LAS_MANOS` |
| LM-004 | `POST /execute` | legacy handler → HTTP client | static 410 | no dynamic output path | SAFE_STATIC | fixed static response / closed legacy channel |
| MR-001 | `motor_registry/routes.py::governed_dispatch` | Motor Registry → service caller | external service admission/result | authority path; not altered here | NEEDS_F2E_ADAPTER | typed service adapter / `MOTOR_REGISTRY` |
| MR-002 | `dispatch` legacy route | legacy → caller | static 410 | retired path | SAFE_STATIC | fixed static response / closed legacy channel |
| MR-003 | `authorize_facet` | registry → service caller | authorization result; external | authority semantics; no change here | NEEDS_F2E_ADAPTER | typed service adapter / `MOTOR_REGISTRY` |
| MR-004 | `get_job` | JobStore → HTTP client | canonical job data; external | status source exists; no output adapter | NEEDS_F2E_ADAPTER | typed service adapter / `MOTOR_REGISTRY` |
| MR-005 | `cancel_job` | registry → HTTP client | dynamic acknowledgement | no output adapter | NEEDS_F2E_ADAPTER | typed service adapter / `MOTOR_REGISTRY` |
| PROC-001–003 | `procesamiento_routes.py` create/status/cancel | OCR job service → HTTP client | typed dynamic job output; external | no F2-C boundary | NEEDS_F2E_ADAPTER | typed service adapter / `PROCESAMIENTO` |
| CLI-001 | `tools/jacobs_relaunch` | operator command → terminal | operator-visible output | no output adapter | NEEDS_F2E_ADAPTER | operator CLI adapter / `OPERATOR_CLI` |
| CLI-002 | `jaxctl` runtime formatters | authority query → terminal | operator-visible formatted output | query authority is not render authority | NEEDS_F2E_ADAPTER | operator CLI adapter / `OPERATOR_CLI` |
| WRK-001 | `motor_registry/worker.py::_audit_and_notify` | worker → Telegram | failure notification; external | no output adapter | NEEDS_F2E_ADAPTER | operator-status adapter / `MOTOR_REGISTRY` |
| ART-001 | `JobStore.write_result`, `logs/motor_results/*.md` | untrusted job result → internal artifact | no direct external consumer proven | raw untrusted artifact | INTERNAL_ONLY | none until operator projection |
| T16-001 | REPL, `jax --task`, `/api/command`, Command UI | retired mechanisms | retired | T16 closed | RETIRED_BY_T16 | none |
| EJ-001 | Ejecutor proxy SSE | concurrent external output path | external; coordination required | parallel front, no automatic F2-F classification | NEEDS_F2E_ADAPTER | pending owner coordination; channel undecided |

## Frozen constraints for the later plan

- A shared adapter must start with untrusted typed data, resolve authority and
  renderability through the closed F2-A/B/C foundations, then hand a governed
  effective output to F2-D for channel-specific lifecycle handling.
- Channel identifiers must be server-owned, closed and versioned. Candidate
  labels in this inventory are routing placeholders, not final channel IDs.
- Preserve author/origin (`USER`, `ASSISTANT/MODEL`, `TOOL`, `SYSTEM`,
  `AGENT`). History/B9 receives effective governed output only where its
  contract calls for it.
- No new predicate is accredited, and no native agent/engine capability is
  disabled by this mapping.
- The map does not implement Faro or decide that the Ejecutor SSE proxy is
  deferred to F2-F.

## Per-class risk register

| Surface class | Risk if adapted without the pending decision |
| --- | --- |
| JAC-001–012, LM-001–003, MR-001/003–005, PROC-001–003 | Treating raw DTO fields or status as current truth, or losing canonical status protections. |
| JAC-013–017, ART-001 | Promoting provider/Motor/Markdown artifacts from typed untrusted data into claims. |
| JAC-018–020, WRK-001, CLI-001/002 | External operator messages could gain claims or lose `SYSTEM`/`AGENT`/`TOOL` provenance. |
| LM-002 | Audit-tail data could imply the unaccredited predicate `AUDIT_EVENT_EXISTS`. |
| MR-001/003 | Altering the output boundary could accidentally alter execution/admission authority. |
| EJ-001 | Concurrent Ejecutor ownership could be crossed or incorrectly deferred to F2-F. |
| SAFE_STATIC and T16 | Dynamic data could be mislabeled static, or a retired T16 path could be resurrected. |

## Read-verified route and template boundaries

Jacobs routes are `POST /preflight`, `POST /plan`, `POST /pipeline`,
`GET /pipeline/{pipeline_id}`, `POST /pipeline/{pipeline_id}/cancel`, the
four discard/recover/hide/restore actions, resume, continue and its preflight,
approve-step, results and events. Motor Registry covers governed dispatch,
legacy static dispatch, authorize facet, job read and cancel. Processing covers
create/status/cancel work routes. These route families return typed service
data today; their proposed channels are not a claim template or a source
version. Any future structured projection needs its own immutable versioned
template/slot contract, server-owned configuration identity and tests before
it can render a claim-bearing field.

## Validation baseline

Before this mapping, the focused JAX governance suite was reported as **185
passed** across runtime-status, remediation, governed renderer, resolution and
output-lifecycle tests. This document adds no code or test execution.

## Discovery conclusion

The general F2-E discovery is blocked pending a human semantic decision. SR-03
deliberately rejects canonical structured status in `TOOL_DATA`; the existing
claim-reference path produces text only. Reconstituting raw DTOs after render
would bypass the effective governed output. `/health` is also dynamic and
cannot be classified `SAFE_STATIC`: its source-circularity needs an explicit
internal-restricted protocol or external observation-source contract. No JAX
source/runtime behavior is changed by this conclusion.
