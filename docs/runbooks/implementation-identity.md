# F2-E health source migration — implementation identity

**Scope:** documentation for the authorized F2-E health-source migration.
This file does not change runtime behavior, credentials, routes, policy, tests
or the concurrent JAX14 PR #274 branch.

## Boundaries

The current target is an authenticated, existing-header `GET`/`HEAD`
`/internal/health` source endpoint consumed by the Platform probe. Public
`GET`/`HEAD /health` keeps its exact legacy dynamic behavior during this
transition. The restricted endpoint is an internal source protocol, while all
other dynamic public routes remain external F2-E surfaces.

The Platform probe must use the internal URL and existing authenticated header.
Its dashboard must consume the atomic snapshot, never make a second health
probe. The manual Motor script must validate credentials through the same
helper. The runtime-status source binding/resolver identity is exactly
`resolver.3`; the Platform exact bridge is `exact.3`.

## Ordered migration procedure

1. Record each known consumer and its evidence before restriction. This is an
   inventory of observed consumers, not a claim that no other monitor exists.
2. Add and test the authenticated internal endpoint and migrate every recorded
   consumer to it while preserving its evidence.
3. Confirm the Platform probe uses the internal endpoint/header and updates
   one atomic health observation only.
4. Confirm the dashboard reads that snapshot and the manual Motor script uses
   the common credential validator.
5. Observe and verify the Platform migration against the internal endpoint.
6. Make the later public `/health` contraction an independently deployable
   change: only then may it return fixed `410 health_publico_retirado`.
   This contraction is not part of the current F2-E change.
7. Exercise the exact runtime-status binding/resolver and Platform bridge;
   preserve current freshness semantics and no resolver-read freshening.

## Structured-output compatibility boundary

The active structured composition work is separate: structured renderer
`1`/domain `5`/schema `1`/lifecycle `1` and text renderer `3`/domain `5`/
envelope `1`/lifecycle `2` remain distinct. `runtime_output_composition` is
ongoing shared-core integration, not a final compatibility claim.

## Evidence status

An implementation agent reported **113 JAX passed** and **55 Platform passed,
10 DB skips**. This is an agent-reported HECHO, pending controller verification;
it is not yet a release or CI result. An original normal-DB collection also
reported five unhandled-trigger failures in the clone baseline before health
code/tests; subsequent focused work used `JAX_CI_NO_DB` with fake test
configuration and did not touch production.

## Non-goals

Two current consumer facts were reported by the user: LAS MANOS listens on
loopback and the Platform state/dashboard consumes health; the manual Motor
script remains a current test consumer. They are reported runtime facts, not a
claim that the inventory is globally complete. No production call, deployment, PR,
commit, GitHub action or merge is performed by this runbook. F2-E general
remains implementation-in-progress until the controller completes integration
and independent verification.
