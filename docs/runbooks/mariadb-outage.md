# MariaDB outage
## Purpose
Respond to authoritative database outage.
## Scope
Blocks 4-7 schemas.
## Preconditions
Identify affected schemas.
## Authority impact
Potential authority/evidence unavailability.
## Safe procedure
Diagnose reachability and preserve incident evidence.
## Verification
Run read-only diagnostics after recovery.
## Fail-closed condition
Required authoritative source unavailable: stop dependent actions.
## Recovery / escalation
Escalate to database operator.
## Prohibited actions
Do not substitute memory or stale snapshots.
