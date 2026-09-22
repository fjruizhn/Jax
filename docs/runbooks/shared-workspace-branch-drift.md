# Shared-workspace branch drift
## Purpose
Recover safely after another process changes branch.
## Scope
Shared Git workspace.
## Preconditions
Stop edits; inspect clean state, branch, head, PR, and remotes.
## Authority impact
None.
## Safe procedure
Fetch, verify PR head, safely switch to expected branch, and use `pull --ff-only` only when no unique local commits exist.
## Verification
Confirm branch/head equals PR head and tree is clean.
## Fail-closed condition
Dirty tree, unpushed unique commits, changed PR, or concurrent branch change: stop.
## Recovery / escalation
Report commit graph to human owner.
## Prohibited actions
No reset, rebase, stash, or force push.
