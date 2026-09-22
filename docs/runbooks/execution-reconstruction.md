# Execution reconstruction
## Purpose
Inspect governed execution state.
## Scope
Block 6 records/events.
## Preconditions
Execution ID and read access.
## Authority impact
Read-only.
## Safe procedure
Use `jaxctl execution <id>` when configured.
## Verification
Check authoritative store result.
## Fail-closed condition
Missing/unverifiable record: stop.
## Recovery / escalation
Escalate.
## Prohibited actions
Do not create, dispatch, or alter execution.
