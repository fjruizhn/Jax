# Authority root and checkpoint recovery

## Purpose

Recover B4 trusted-root and trusted-checkpoint availability without inventing
authority or replacing an anchor whose provenance is unknown.

## Scope

External B4 trust files:

- `/etc/jax/authority/trusted-root.json`
- `/etc/jax/authority/trusted-checkpoint-bootstrap-receipt.json`
- `/var/lib/jax/authority/trusted-checkpoints.log`

The bootstrap receipt and checkpoint log are one recovery unit. The receipt is
the explicit checkpoint-zero anchor; the log contains the subsequent
append-only checkpoints.

## Preconditions

1. Stop B4 authority writers and any process that can append to the authority
   ledger or checkpoint log.
2. Identify the incident scope and an authoritative, verified backup.
3. For checkpoint recovery, identify a backup set that contains both the
   bootstrap receipt and checkpoint log from the same recovery point. Do not
   combine either file with a different backup generation.

## Authority impact

Critical authority material.

## Safe procedure

1. Preserve the incident evidence and record the missing, corrupted, or
   unavailable paths before changing the host.
2. Restore only approved, verified material. Restore the bootstrap receipt and
   checkpoint log together from their matched backup set. Restore
   `trusted-root.json` only from its separately verified authority backup when
   it is part of the incident.
3. Verify the restored files before starting writers: the receipt must verify
   as the checkpoint-zero anchor, the log must be canonical and monotonic, and
   the log must match that anchor and the authority ledger.
4. Run the ledger verification using the restored external checkpoint material.
   Reopen writers only after it succeeds.

An existing anchor is a precondition for every append. A missing receipt or
checkpoint log is fail-closed: do not append an event in order to recreate it.

## Verification

Run complete ledger verification against the restored trusted root, bootstrap
receipt, and checkpoint log. Confirm that the verified ledger head matches the
external checkpoint chain before allowing B4 writes.

## Fail-closed condition

Stop if the root, bootstrap receipt, or checkpoint log is missing, empty,
malformed, unverifiable, from an unmatched backup set, or does not match the
ledger. Stop as well if verification cannot establish a checkpoint-zero anchor
or a monotonic checkpoint chain.

## Recovery / escalation

Escalate to the authority owner when no matched, verified backup set is
available. A lost receipt or log is an authority-recovery incident, not an
initialization request.

Historical logs whose first record is sequence 1 remain valid when paired with
their verified separate bootstrap receipt. Do not add an artificial sequence-0
row to such a log during recovery.

## Prohibited actions

- Do not synthesize, overwrite, or re-bootstrap a trusted root.
- Do not rerun checkpoint-zero bootstrap to replace a missing, lost, or
  corrupted receipt or checkpoint log.
- Do not create an empty checkpoint log, edit the log to insert sequence zero,
  truncate it, or append a checkpoint to make recovery appear complete.
- Do not restore the receipt and log from different backup points.
