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

Current-authority verification coordinates with writers through the same
checkpoint sidecar lock. While holding it, the verifier confirms durability
of the checkpoint log and bootstrap receipt, then fsyncs each containing
directory and its ancestors through the filesystem root before reading and
comparing their exact snapshots with the ledger. Directory creation also
persists each new directory entry in its parent. If any durability check fails,
the verifier must not issue a verified current-authority state.

Backups must preserve the receipt and checkpoint log as one matched recovery
set. The recovery procedure is in
`docs/runbooks/authority-root-recovery.md`. Never create a fresh receipt, an
empty replacement log, or a new zero anchor merely because either trusted file
was lost or is unavailable.

Current-authority verification requires the checkpoint log itself to begin at
sequence zero. A legacy log beginning at sequence 1 is not adopted by pairing
it with a deterministic genesis receipt. Until a signed adoption procedure is
implemented and completed, such a deployment must remain unavailable for
current-authority decisions; preserve and escalate its existing artifacts.
