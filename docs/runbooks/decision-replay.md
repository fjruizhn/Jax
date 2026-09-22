# Decision replay
## Purpose
Read and replay a DecisionRecord.
## Scope
Block 5 decisions.
## Preconditions
Decision ID and read access.
## Authority impact
Read-only.
## Safe procedure
Use `jaxctl decision <id> --replay` when configured.
## Verification
Compare replay result with authoritative record.
## Fail-closed condition
Unverifiable record: stop.
## Recovery / escalation
Escalate integrity failures.
## Prohibited actions
Do not rewrite DecisionRecords.
