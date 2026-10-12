# First Block 4 authority bootstrap

## Purpose

Prepare and initialize a genuinely empty Block 4 authority ledger without
mistaking a partial, legacy, or unreadable deployment for first initialization.

## Scope

The Block 4 MariaDB schema (`jax_authority`), its genesis/events/head tables,
and the external trusted-root, bootstrap-receipt, and checkpoint artifacts.

## Preconditions

1. An authority owner has supplied the externally verified Ed25519 public key
   and its `constitutional_key_id`. The source is recorded with the operation.
2. Writers are stopped and cannot append while the inspection and initialization
   occur.
3. A dedicated B4 MariaDB principal is provisioned for the exact observed TCP
   source address. Readers receive SELECT only; writers receive only the B4
   INSERT/UPDATE rights required by the ledger.
4. The expected paths are writable only by the authority service identity and
   no existing authority artifact has been overwritten.
5. The service receives only `JAX_AUTHORITY_DB_HOST`,
   `JAX_AUTHORITY_DB_PORT`, `JAX_AUTHORITY_DB_USER`, and
   `JAX_AUTHORITY_DB_PASSWORD` from root-owned secret configuration (`0640`,
   readable by its configured service group). Do not put the password in argv,
   logs, or a general `JAX_DB_*` setting. Before writers start, connect through
   that exact configuration and record `CURRENT_USER()`; it must be the
   reviewed TCP principal.

## Authority impact

Critical authority material. Staging public values is not authority, a
ratification, or permission to execute an action.

## Safe procedure

1. Inspect MariaDB and external paths with
   `policy.authority_ledger.bootstrap.inspect_authority_bootstrap`. Record the
   returned state and the exact environment/revision. An access denial,
   including MariaDB 1142, is `INSPECTION_UNAVAILABLE`, never an empty ledger.
2. If the state is `CANDIDATE`, apply the reviewed Block 4 migration, then
   inspect again. The second inspection must still be `CANDIDATE` before any
   initialization is considered.
3. Build a public-only candidate with
   `stage_authority_genesis(public_key, constitutional_key_id)`. Independently
   compare its canonical genesis bytes, genesis hash, key fingerprint, and
   trusted-root projection with the supplied public material. Its
   `classification` remains `NOT_AUTHORITY`.
4. Invoke `initialize_staged_authority_ledger` with the exact staged genesis,
   externally approved root, administrator connection factory, and external
   checkpoint store. It inserts only the exact genesis into an empty ledger,
   creates the trusted-root file exclusively (no replacement), then calls the
   existing `initialize_authority_ledger` and `verify_authority_ledger`
   boundaries with the same staged genesis/root.
5. Reinspect. `MATERIAL_PRESENT_UNVERIFIED` is observation only. Reopen
   writers only after `verify_authority_ledger` succeeds with the exact
   approved root, receipt, checkpoint, stored genesis, events, and DB head.

If an interrupted operation leaves a genesis, trusted root, receipt, or
checkpoint artifact, preserve them and inspect their exact relationship. A
matching partial state may only resume after the stored genesis and external
public material exactly match the staged values. Events, a differing genesis,
or a differing external anchor require recovery.

## Verification

Run `verify_authority_ledger` against the stored genesis, all events, trusted
root, bootstrap receipt, and checkpoint log. Confirm its reconstructed
checkpoint equals the current MariaDB head. Confirm the writer account has no
DELETE or UPDATE privilege on genesis/events and the reader has SELECT only.

## Fail-closed condition

Stop for `INSPECTION_UNAVAILABLE`, `RECOVERY`, `LEGACY`, or `INCONSISTENT`.
Stop if the public key source, key id, staged hash, stored genesis, trusted
root, receipt, checkpoint, or MariaDB head does not exactly match. Material
present is never authority until that verification succeeds.

## Recovery / escalation

Use [authority-root-recovery.md](authority-root-recovery.md) for any existing
or partial root/checkpoint material. Escalate a legacy event stream or a
conflicting genesis to the authority owner; first initialization cannot adopt
or repair it.

## Prohibited actions

- Do not derive a public key from MariaDB, create a private key, sign, approve,
  ratify, or fabricate a receipt.
- Do not overwrite, truncate, remove, or recreate trusted-root, receipt, or
  checkpoint artifacts.
- Do not treat a MariaDB permission error as evidence that the schema or
  ledger is empty.
- Do not grant `ALL PRIVILEGES`, wildcard-host accounts, UPDATE/DELETE on
  genesis/events, or use a general `JAX_DB_*` principal for Block 4.
