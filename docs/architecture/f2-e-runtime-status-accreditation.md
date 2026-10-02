# F2-E runtime-status accreditation contract

Version: `f2-e.runtime-status.2`
Authorization: Fernando, F2-E-SR-001 (2026-10-02)

This amendment adds exactly two closed predicates, `PIPELINE_STATUS` and
`FACET_RUNTIME_STATUS`. It accredits the existing `JOB_STATUS` and
`ENGINE_STATUS` with narrower, explicit sources. No other predicate becomes
accredited. The legacy `policy/governance/validator.py` remains fail-closed for
current-state claims; only F2-B `ResolverRegistry` bindings may resolve these
status claims.

| Predicate | Authoritative observation | Meaning and boundary | Freshness |
|---|---|---|---|
| `JOB_STATUS(job_id,status)` | LAS MANOS Motor Registry `JobStore` locked authoritative snapshot | Status of a Motor Registry job; never a Jacobs pipeline. Ownership must bind the exact tenant and user. Legacy ownerless jobs and unsupported project scopes are unavailable. | 60 seconds from the resolver's locked authoritative read. Transition timestamps remain separate source metadata. |
| `PIPELINE_STATUS(pipeline_id,status)` | Jacobs persisted canonical pipeline row via `jacobs.store.pipeline_get()` | Canonical Jacobs lifecycle status; never the platform UI poller's mapped projection. Tenant and user must match; unsupported project scopes are unavailable. | 60 seconds from the canonical row read; `updated_at` is not treated as observation time. |
| `FACET_RUNTIME_STATUS(name,status)` | JAX Platform server-owned `FacetState` | Only the transient state Platform records (`idle`, `thinking`, `error`, `offline`). It does not establish facet existence, model/provider health, availability, configuration, or external truth. No facet message or user-specific payload is part of the observation. | 15 seconds from the resolver's in-process state read; `last_update` remains transition metadata. |
| `ENGINE_STATUS(name,status)` | Fixed LAS MANOS health probe maintained by JAX Platform | Only the health-check result (`alive`, `down`). It does not establish configuration, model correctness, authorization, or capability availability. No arbitrary URL or engine is resolvable. | 60 seconds from the latest completed probe; probes run every 30 seconds. |

The F2-B `ENGINE_STATUS` source-configuration digest binds the exact effective
health endpoint identity (as a one-way SHA-256), method, path, timeout, poll
interval, and success status code. The URL is not persisted in the binding or
receipt. A source-configuration change therefore requires a matching registry
binding; old evidence cannot be reused under the changed endpoint.

`JOB_STATUS` binds a non-secret identity for the configured canonical JSONL
path, event format, store contract, and append/flush/fsync durability mode.
`PIPELINE_STATUS` binds the server-owned MariaDB host, port, database, table,
and Jacobs store contract, never credentials. A source configuration change
changes the binding and invalidates earlier receipts. Runtime-status API,
resolver, and binding versions are `.2`; `.1` receipts do not validate against
the revised registry.

The JobStore keeps legacy best-effort polling separate from its authoritative
snapshot. Any malformed, truncated, structurally invalid, or undecodable
non-empty event marks loaded history ineligible for current-truth resolution;
the resolver returns unavailable rather than accrediting an earlier event.
Appends flush and `fsync` the event before the in-memory index advances. A new
file's containing directory is also synced. This does not claim atomicity for
multiple records or repair corrupted history.

F2-C governed-domain detection recognizes the exact authorized facet-runtime
template wording in Spanish and English, and the `alive` health status. For
TOOL_DATA and an entire structured NARRATIVE_TEXT JSON payload, it inspects
direct, nested, list-contained, once-encoded, and twice-encoded structures with
limits of 65,536 UTF-8 bytes/chars, two decode layers, depth 32, and 1,024
visited values. Ambiguous/malformed/over-limit structures fail closed;
unrelated JSON such as `{"theme":"dark","status":"selected"}` remains data.
This is structural detection only; arbitrary natural-language paraphrases are
not classified by this rule.

Motor transition timestamps such as `created_at`, `started_at`, and
`finished_at` remain historical transition facts. A fresh resolver read of a
coherent, integrity-valid source snapshot may observe an unchanged stable state
now; it does not rewrite or freshen the transition timestamp. ENGINE_STATUS is
different: its observation time remains the actual health-probe time, so a
resolver read alone cannot refresh an old probe.

Motor and Jacobs sources are exact-response-scope sources. Platform facet and
health observations are installation-global sources, but every receipt remains
bound to the exact response scope and cannot be replayed to another request,
subject, tenant, project, predicate, or arguments. All source identities,
resolver identities, bindings, adapter kinds, and versions are server-owned.
Missing, malformed, unknown, stale, future-dated, mismatched, or out-of-scope
observations fail closed.

F2-C rendering uses explicit versioned templates. In particular, facet wording
states that “JAX Platform currently marks” a runtime state. The wording does
not claim that a model/provider is healthy or available. Tool payload text that
expresses one of these four propositions must use its matching governed claim;
`TOOL_DATA` is not a status-claim bypass.

This contract is a semantic prerequisite for F2-E adapters only. It does not
implement those adapters, accredit any new current predicate beyond the two
named additions, change F2-D transport lifecycle, or alter production
configuration, credentials, or keys.
