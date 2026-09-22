# Implementation identity
## Purpose
Recover B7 deployment identity availability.
## Scope
Implementation manifest/identity.
## Preconditions
Verified build/deployment source.
## Authority impact
Evidence integrity, not policy authority.
## Safe procedure
Deploy verified identity and manifest together.
## Verification
Run identity/build-manifest verification.
## Fail-closed condition
Drift or missing identity: stop.
## Recovery / escalation
Redeploy verified build or escalate.
## Prohibited actions
Do not hand-edit CLEAN identity claims.
