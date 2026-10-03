# F2-E-SR2 — STEP_STATUS exact-pair contract

**Status:** implementation contract, authorized by Fernando through jax-14 on 2026-10-03. This is limited to the new `STEP_STATUS` source, its governed rendering, exact compatibility and deployment handoff.

## Closed compatibility tuple

The Platform bridge accepts exactly this JAX pair:

```text
GOVERNED_RENDERER_API_VERSION      = f2-c.renderer.3
GOVERNED_DOMAIN_SPEC_VERSION       = f2-c.domain.7
GOVERNED_ENVELOPE_SCHEMA_VERSIONS  = {f2-c.1}
RUNTIME_STATUS_API_VERSION         = f2-e.runtime-status.4
```

All values are exact, server-owned constants. Older, future, partial, wildcard and range matches fail closed. The Platform bridge must verify that the imported JAX governance modules originate under its configured `JAX_REPO_PATH`.

## STEP_STATUS resolver contract

Predicate arguments are exactly `{"step_id": string, "status": string}`. `step_id` is a non-empty canonical Jacobs ID. `status` must be one of the seven `jacobs.models.StepStatus` values:

```text
pending | running | completed | failed | skipped | blocked | blocked_human_gate
```

The canonical read is one server-owned query joining `jacobs_steps` to `jacobs_pipelines` by `pipeline_id`, selected by the `jacobs_steps.step_id` primary key. It returns one immutable `StepStatusSnapshot`:

```text
step_id, status, pipeline_id, tenant_id, user_id,
owner_ack_at, pipeline_status, observed_at
```

`observed_at` is the UTC time after that read completes, not a step transition timestamp. The source requires an acknowledged owner (`owner_ack_at` present), exact tenant and subject match, and a visible canonical parent pipeline. `hidden`, `discarded`, ownerless, orphaned, malformed or unavailable source data cannot accredit a status. Project scope is unsupported; no Platform status alias or caller-selected source participates.

JAX resolver signature:

```python
async JacobsStepStatusResolver.evidence(
    arguments: Mapping[str, object], scope: ResponseScope
) -> RuntimeStatusEvidence
```

Platform bridge signature:

```python
async JacobsStepStatusResolver.evidence(
    arguments: Mapping[str, object], scope
) -> RuntimeStatusEvidence | None
```

The F2-B observation result is exactly `{"step_id": observed_step_id, "status": canonical_status}`. A requested status that differs from the canonical read is `SOURCE_MISMATCH` at registry resolution; it is never rewritten to the requested value. Resolver identity/source are fixed to `JacobsStepStatusResolver` and `jacobs:canonical-step-store`, scoped to the exact authenticated response. Freshness SLA is 60 seconds; a new source read creates a new observation.

## Errors and fail-closed behavior

| Condition | Result |
|---|---|
| Wrong predicate argument keys, unsupported project scope, or non-canonical requested status | `GovernanceContractError` in JAX resolver; Platform bridge returns no evidence |
| Missing/unavailable/corrupt step or source configuration; unacknowledged owner; hidden/discarded pipeline; invalid canonical row | F2-B `UNAVAILABLE` |
| Canonical pipeline owner differs from exact scope tenant/subject | F2-B `WRONG_SCOPE` |
| Requested status differs from observed canonical status | F2-B `SOURCE_MISMATCH` |
| Receipt expired or scope/config/binding identity differs | F2-B rejects it (`STALE`, `WRONG_SCOPE`, or `CONFIGURATION_MISMATCH`) |
| Unsupported F2-C/F2-E version pair | Platform `GovernedChatUnavailable`; server-owned safe fallback, no raw candidate leakage |

The legacy `policy.governance.validator.validate()` path must always return `RESOLVER_NOT_IMPLEMENTED` for `STEP_STATUS`, even if a legacy resolver is injected into `_RESOLVERS`. Only the F2-B `ResolverRegistry` path may accredit it.

## Governed template and policy version

The server-owned Spanish template is:

```text
Jacobs registra actualmente que el paso {step_id} está {status}.
```

Its `TemplateContract` identifies predicate `STEP_STATUS`, runtime contract `f2-e.runtime-status.4`, and locale `es`. The step ID and status are receipt-bound slots; user/model/tool-controlled text is not inserted into this template.

`policy/VERSION` keeps schema `0.1.0`; its `sha256` is the exact-byte hash of the ordered `policy/rules/*.yaml` corpus and `templates_sha256` is the exact-byte hash of `policy/templates/render_templates.yaml`. Recompute both from the final branch contents with the repository scripts. A later master policy change requires the integrating owner to recompute the relevant shared hash on that master; never add deltas to old hashes.

## Exact-pair integration and deployment order

JAX #341 freezes the F2-B resolver and F2-C domain/template contract. Platform SR2 pins the exact JAX PR head and exercises the canonical MariaDB read, F2-B receipt, F2-C renderer and F2-D transport lifecycle in one PR-only job. The artifact records both repositories, PR numbers and full heads.

For deployment, JAX goes first, Platform second, in one coordinated release window. If either side runs alone, the other side's exact compatibility check rejects the pair and returns its server-owned safe fallback; STEP_STATUS remains unavailable until both compatible versions run. No old/new dual acceptance, automatic retry or downgrade is allowed. Deployment remains separate from this implementation and requires Fernando's GO.
