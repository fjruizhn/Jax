# Worker timeout and cancellation
## Purpose
Handle governed worker timeout/cancellation.
## Scope
Block 6 workers.
## Preconditions
Identify authoritative execution record.
## Authority impact
Execution safety.
## Safe procedure
Use supported governed operational controls only.
## Verification
Inspect authoritative execution state and evidence.
## Fail-closed condition
Missing governed binding: do not dispatch/cancel through an ungoverned path.
## Recovery / escalation
Escalate unsupported cancellation.
## Prohibited actions
No direct worker manipulation that bypasses governance.
