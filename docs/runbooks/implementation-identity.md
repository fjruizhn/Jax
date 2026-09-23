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

**Exact procedure (jax#260).** `las_manos/server.py::_configure_b7_trusted_runtime`
builds `TrustedImplementationIdentityProvider(evidence_store)` and
`EvidenceLifecycleService` calls `.load()` on it **synchronously during
FastAPI startup** (`@app.on_event("startup")` → `_jacobs_init()`). Without
the two artifacts below in place, LAS MANOS does not come up — startup dies
with `FileNotFoundError` before `/health` ever answers. There is no
degraded/partial mode: fail-closed, per `docs/operations/trusted-files.md`.

1. **Generate on the exact commit being deployed**, from a clean checkout
   of that commit (never from a working tree with local edits):

   ```
   python3 scripts/generar_manifiesto_identidad.py \
       --repo-root /srv/jax-prod/jax \
       --output /etc/jax/build/implementation-identity.json \
       --manifest-output /etc/jax/build/implementation-identity.manifest.json
   ```

   The script refuses (exit 1, writes nothing) if `/srv/jax-prod/jax` has
   uncommitted changes — a production identity can never describe bytes
   that are not in a commit. It writes both output files atomically, mode
   `600`.

2. **Install the identity file** at
   `/etc/jax/build/implementation-identity.json`, owned by the account the
   service runs as (`jaxsvc:jaxsvc`, mode `600` — same account/mode
   discipline as `/etc/jax/.env`, see
   `docs/operations/secret-handling.md`). The generator already writes mode
   `600`; the controller only needs `chown jaxsvc:jaxsvc` after copying it
   into place if it was generated as a different user.

3. **Install the build manifest blob into the evidence store.** The
   identity file only carries `build_manifest_blob_hash` — a *pointer*.
   `verify_build_manifest` resolves it with
   `store.get_evidence_blob(identity.build_manifest_blob_hash)`
   (`policy/enforcement_evidence/implementation_identity.py`), and in
   production `store` is `MariaDBEvidenceStore`
   (`policy/enforcement_evidence/mariadb_store.py`), reading
   `jax_evidence.evidence_blobs`. The manifest bytes at
   `implementation-identity.manifest.json` from step 1 have to be inserted
   there — `MariaDBEvidenceStore.put_evidence_blob(data)` is the exact
   write path (content-addressed by `sha256:` of the bytes, so a duplicate
   insert of the same bytes is a safe no-op). **This script deliberately
   does not do this write**: it needs production MariaDB credentials this
   generator never touches (the production barrier for this change: no DB
   access outside `pytest`, no `/etc/jax/.env` sourced in a dev shell). The
   controller performs this insert as its own deploy step, with the
   deploy's own credentialed access — e.g. loading
   `/etc/jax/.env`, connecting with the resulting `pymysql` connection
   factory, and calling
   `MariaDBEvidenceStore(connection_factory).put_evidence_blob(open(manifest_output,'rb').read())`.
   If this step is skipped, startup does not fail at the identity file —
   it fails one line later, in `verify_build_manifest`, with
   `EvidenceBlobMissingError`.

4. **The `/srv/jax` vs `/srv/jax-prod/jax` mismatch.**
   `_DEPLOYMENT_REPOSITORY_ROOT` in
   `policy/enforcement_evidence/implementation_identity.py` is the fixed
   constant `"/srv/jax"` — and jax#260 left it that way **on purpose**:
   "These are deployment constants, not an API... accepting a path/root
   here would let a caller redefine the bytes whose integrity is being
   claimed." A request can never re-point verification at a different
   root. But production actually lives at `/srv/jax-prod/jax` -- verified live
   on hall9000: `systemctl status jax-las-manos` shows
   `WorkingDirectory=/srv/jax-prod/jax/las_manos`, sourced from the
   systemd drop-in
   `/etc/systemd/system/jax-las-manos.service.d/checkout-de-produccion.conf`,
   whose own comment dates the move to 2026-09-20 ("produccion deja de
   servir desde el checkout de trabajo de un agente").
   Those two facts collide: `verify_build_manifest_bytes` resolves every
   `files` entry as `(Path(repository_root)/name).resolve()` — with
   `repository_root="/srv/jax"`, it will look for
   `/srv/jax/policy/enforcement_evidence/status_engine.py`, not
   `/srv/jax-prod/jax/policy/...`.

   The resolution is a **symlink**, `/srv/jax -> /srv/jax-prod/jax`,
   created and owned by root (`/srv/` itself is root-owned;
   `/srv/jax-prod/jax` is `jaxsvc:jaxsvc`), documented here as an explicit
   **deploy step** — not a code change:

   ```
   sudo ln -s /srv/jax-prod/jax /srv/jax
   ```

   This is the only acceptable fix. Parameterizing
   `_DEPLOYMENT_REPOSITORY_ROOT` to point at `/srv/jax-prod/jax` directly
   would look like a smaller change, but it would reopen exactly the
   authority question jax#260 closed on purpose — a fixed deployment
   constant becoming something a future refactor could thread through as
   a parameter. The symlink keeps the constant untouched and the
   filesystem indirection explicit and auditable (`readlink /srv/jax`).
   Generate the manifest itself against `/srv/jax-prod/jax` (step 1) so its
   `git_commit_sha`/`git_tree_id` describe the real checkout; the symlink
   only has to exist by the time `verify_build_manifest` runs at startup,
   which reads through it.
## Verification
Run identity/build-manifest verification.

Concretely: `systemctl restart jax-las-manos` (or let the next deploy
restart it), then confirm **both**:

```
systemctl is-active jax-las-manos      # expect: active
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7777/health   # expect: 200
```

If either check fails, read `journalctl -u jax-las-manos -n 50` before
retrying anything — `_configure_b7_trusted_runtime` raises with a specific,
distinguishable error for each missing piece
(`FileNotFoundError` for the identity file itself,
`EvidenceBlobMissingError` for a manifest blob never installed in
`jax_evidence.evidence_blobs`, `UntrustedImplementationIdentityError` for a
hash mismatch/drift). Do not restart-loop past a fail-closed startup
without reading which of the three it is.
## Fail-closed condition
Drift or missing identity: stop.

**The manifest is regenerated on every deploy — it is never reused across
commits.** `build_manifest_blob_hash` binds the identity to the exact
source bytes of `_V1_REQUIRED_SOURCE_PATHS` at generation time; the next
commit changes at least one of those files' hashes (that is the entire
point of the manifest: `verify_build_manifest_bytes` recomputes every
listed file's SHA-256 against what's on disk and fails on any mismatch,
`policy/enforcement_evidence/implementation_identity.py`). Deploying new
code with yesterday's `implementation-identity.json` does not silently
serve a stale-but-working identity: it fails closed the same way a missing
file does, because the hashes no longer match. Regenerating (step 1 above)
is the fix — there is no partial/incremental update.
## Recovery / escalation
Redeploy verified build or escalate.
## Prohibited actions
Do not hand-edit CLEAN identity claims.

Do not hand-write `implementation-identity.json` or its manifest blob —
always through `scripts/generar_manifiesto_identidad.py`, so the hashes are
computed by the same algorithm `verify_build_manifest_bytes` checks them
with, never guessed or copy-pasted from a previous deploy.
