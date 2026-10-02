# F2-E runtime-status accreditation contract

Version: `f2-e.runtime-status.1`  
Authorization: Fernando, F2-E-SR-001 (2026-10-02)

This amendment adds exactly two closed predicates, `PIPELINE_STATUS` and
`FACET_RUNTIME_STATUS`. It accredits the existing `JOB_STATUS` and
`ENGINE_STATUS` with narrower, explicit sources. No other predicate becomes
accredited. The legacy `policy/governance/validator.py` remains fail-closed for
current-state claims; only F2-B `ResolverRegistry` bindings may resolve these
status claims.

| Predicate | Authoritative observation | Meaning and boundary | Freshness |
|---|---|---|---|
| `JOB_STATUS(job_id,status)` | LAS MANOS Motor Registry `JobStore` latest event | Status of a Motor Registry job; never a Jacobs pipeline. Ownership must bind the exact tenant and user. Legacy ownerless jobs and unsupported project scopes are unavailable. | 60 seconds from the actual status transition timestamp. |
| `PIPELINE_STATUS(pipeline_id,status)` | Jacobs persisted canonical pipeline row via `jacobs.store.pipeline_get()` | Canonical Jacobs lifecycle status; never the platform UI poller's mapped projection. Tenant and user must match; unsupported project scopes are unavailable. | 60 seconds from canonical `updated_at`. |
| `FACET_RUNTIME_STATUS(name,status)` | JAX Platform server-owned `FacetState` | Only the transient state Platform records (`idle`, `thinking`, `error`, `offline`). It does not establish facet existence, model/provider health, availability, configuration, or external truth. No facet message or user-specific payload is part of the observation. | 15 seconds from `FacetState.last_update`. |
| `ENGINE_STATUS(name,status)` | Fixed LAS MANOS health probe maintained by JAX Platform | Only the health-check result (`alive`, `down`). It does not establish configuration, model correctness, authorization, or capability availability. No arbitrary URL or engine is resolvable. | 60 seconds from the latest completed probe; probes run every 30 seconds. |

The F2-B `ENGINE_STATUS` source-configuration digest binds the exact effective
health endpoint identity (as a one-way SHA-256), method, path, timeout, poll
interval, and success status code. The URL is not persisted in the binding or
receipt. A source-configuration change therefore requires a matching registry
binding; old evidence cannot be reused under the changed endpoint.

F2-C governed-domain detection recognizes the exact authorized facet-runtime
template wording in Spanish and English, and the `alive` health status. The
renderer also rejects `TOOL_DATA` objects with the exact accredited status
argument shapes (including nested payloads); unrelated tool data remains data.

Motor `status_updated_at` changes only when the typed status actually changes.
An idempotent rewrite of the same status does not renew the observation time or
extend the `JOB_STATUS` freshness window.

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
