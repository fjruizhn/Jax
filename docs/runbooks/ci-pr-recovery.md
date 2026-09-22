# CI and PR recovery
## Purpose
Recover normal feature-branch/PR workflow.
## Scope
Repository changes and CI.
## Preconditions
Clean tree and identified PR/head.
## Authority impact
None.
## Safe procedure
Feature branch, tests, normal commit/push, PR, CI, required audit, human merge, post-merge smoke, cleanup.
## Verification
Confirm exact-head CI and clean repository.
## Fail-closed condition
Changed PR head or failed required CI: stop.
## Recovery / escalation
Fix concrete issue or escalate.
## Prohibited actions
No direct master push, force push, force-with-lease, `--no-verify`, or hook bypass.
