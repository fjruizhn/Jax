# Trusted external files

| Default | Consumer | Trust domain | Missing behavior / backup |
|---|---|---|---|
| `/etc/jax/authority/trusted-root.json` | B4 root loader | authority | fail closed; critical backup/recovery |
| `/etc/jax/authority/trusted-checkpoint-bootstrap-receipt.json` | B4 checkpoint bootstrap verifier | integrity | fail closed; back up and restore together with the checkpoint log |
| `/var/lib/jax/authority/trusted-checkpoints.log` | B4 checkpoints | integrity | fail closed; back up and restore together with the bootstrap receipt |
| `/etc/jax/execution/trusted-approvers.json` | B6 approver loader | execution authority | fail closed; critical backup/recovery |
| `/etc/jax/build/implementation-identity.json` | B7 identity provider | deployment integrity | fail closed; critical backup/recovery |
| `/etc/jax/PAUSE` | LAS MANOS kill switch | operational safety | inspect through kill-switch runbook |

These paths are sensitive. Exact Unix modes are not asserted by this guide; any hardening recommendation is **RECOMMENDED**, not code-required.

## B4 checkpoint bootstrap and append contract

The trusted checkpoint begins with an explicit **checkpoint-zero bootstrap**. Its
receipt is kept separately at
`/etc/jax/authority/trusted-checkpoint-bootstrap-receipt.json`; the subsequent
append-only checkpoint records are kept at
`/var/lib/jax/authority/trusted-checkpoints.log`.

The receipt represents the approved zero anchor. It is not a replacement for
the log, and the log is not evidence that the zero anchor was approved. A B4
writer may append an authority event only after it has found and verified both
files as its existing anchor. Missing, empty, malformed, or mismatched
bootstrap material or checkpoint log is an integrity failure: stop before an
append or authority decision can proceed.

Backups must preserve the receipt and checkpoint log as one matched recovery
set. The recovery procedure is in
`docs/runbooks/authority-root-recovery.md`. Never create a fresh receipt, an
empty replacement log, or a new zero anchor merely because either trusted file
was lost or is unavailable.

Historical checkpoint logs whose first *log* record is sequence 1 remain
valid. They do not need a synthetic sequence-0 row in the log: checkpoint zero
is represented by the separate bootstrap receipt.
