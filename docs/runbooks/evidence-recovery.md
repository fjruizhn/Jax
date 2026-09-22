# Evidence recovery
## Purpose
Handle evidence corruption or unavailability.
## Scope
Block 7 evidence store.
## Preconditions
Identify artifact/observation/assertion identity.
## Authority impact
Evidence integrity.
## Safe procedure
Read authoritative evidence and verify bindings.
## Verification
Use B7 authoritative load/query.
## Fail-closed condition
Tamper, missing refs, or unverifiable data: stop.
## Recovery / escalation
Escalate; restore only verified backup.
## Prohibited actions
Do not mint replacement evidence manually.
