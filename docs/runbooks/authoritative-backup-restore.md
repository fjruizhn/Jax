# Authoritative backup and restore
## Purpose
Restore authoritative schemas safely.
## Scope
`jax_authority`, `jax_decisions`, `jax_execution`, `jax_evidence`.
## Preconditions
Verified backup, recovery authorization, dependency plan.
## Authority impact
Critical integrity operation.
## Safe procedure
Restore approved snapshot in documented dependency order.
## Verification
Run ledger, decision, execution, and evidence checks.
## Fail-closed condition
Unknown provenance or consistency: stop.
## Recovery / escalation
Escalate to authority/database owners.
## Prohibited actions
Do not restore `jax_memory` as authority.
