# Deployment and rollback boundaries
## Purpose
Deploy or recover verified builds.
## Scope
Service deployment and implementation identity.
## Preconditions
Verified build/manifest and approved change.
## Authority impact
Deployment integrity; no policy amendment.
## Safe procedure
Deploy through supported platform procedure and preserve identity bindings.
## Verification
Check startup, identity verification, and diagnostics.
## Fail-closed condition
Manifest drift or unavailable trusted source: stop.
## Recovery / escalation
Redeploy verified prior build or escalate.
## Prohibited actions
Do not reuse stale identity for changed tree.
