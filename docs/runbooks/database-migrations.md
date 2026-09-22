# MariaDB migrations
## Purpose
Apply supported B4-B7 migrations.
## Scope
Authoritative schemas.
## Preconditions
Backup and verify target/schema/version.
## Authority impact
Schema changes affect enforcement; no policy mutation.
## Safe procedure
Use tracked migration files in dependency order.
## Verification
Inspect installed schema and run applicable checks.
## Fail-closed condition
Unknown version or partial migration: stop.
## Recovery / escalation
Escalate; do not improvise rollback.
## Prohibited actions
No destructive reset or manual authority edits.
