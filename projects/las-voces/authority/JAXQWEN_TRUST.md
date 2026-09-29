# `jaxqwen` host trust contract

This contract is separate from the human executor and LAS MANOS credentials.
Its service identity is exactly `jaxqwen`; no `plataforma`, `jacobs`, actor
string or UID is converted into that identity.

## Trust path

1. The governed Ariadna dispatcher records `DISPATCHED` after creating the
   canonical builder worktree. Its explicit `dispatch_jaxqwen(result)` method
   re-reads the durable ACK and sends only that ACK's 64-character idempotency
   key to the fixed host socket. It does not accept mission data, command,
   executable, argv, cwd, environment or model selection.
2. The host checks the Unix peer UID against the dedicated
   `jaxqwen-dispatcher` account. This is a transport gate, not sufficient
   mission authority. The capability independently reads the canonical
   handoff, dispatcher ACK and active lease, and checks task eligibility,
   correlation, project hash, source revision, allowlisted repository,
   configured dispatcher worktree root, branch and worktree identity. It
   holds the existing task-lease registry's shared flock from current-state
   validation through each effect, so lease release cannot race an effect.
3. The host creates a standalone per-mission clone under `/var/lib/jaxqwen`.
   `jaxqwen` cannot write the dispatcher worktree or shared `.git` metadata.
   A durable `MISSION_STARTED` claim has one winner across threads, processes
   and restarts. Only that execution can enter the model/tool loop.
4. The local Qwen model is reached through the existing executor proxy's
   dedicated AF_UNIX listener. The proxy checks `SO_PEERCRED` for the exact
   `jaxqwen` UID and still applies its fixed model, output, C3, C5 and kill
   switch gates. The model receives structured tool schemas and untrusted
   repository content, never the service key or a bearer credential.
5. Before each structured tool effect, the capability requests a short lived
   host broker credential and verifies/consumes it. Tool calls remain limited
   to the fixed capability catalog. No arbitrary shell or command path exists.

## Credential

The host HMAC-SHA256 broker signs canonical JSON with a secret of at least 32
random bytes. The exact signed claims include `service_identity`,
`capability_id`, `mission_id`, `project_id`, `task_id`, `dispatcher_ack_id`,
handoff correlation/idempotency identity, active `lease_id`, canonical
`project_hash`, normalized repository identity, branch, isolated mission
worktree, allowed tools, expiry, nonce, issuer and issuer version. The issuer
and verifier use a host composition-bound current-state callback; requests
cannot supply either callback or verifier.

The secret is provisioned by the host operator at
`/etc/jax/secrets/jaxqwen-trust.key`, root-owned mode `0600`, and delivered to
the service only with systemd `LoadCredential`. It is not committed, placed in
the environment, included in model prompts, or logged. The host configuration
and trust/replay ledgers are also outside Git. A missing or malformed replay
ledger prevents startup or authentication; it is never recreated over prior
state.

To rotate the HMAC key, an operator stops the unit, atomically replaces the
root-owned `0600` key with a newly generated 32-byte value, and starts the unit
with the existing replay ledger intact. Any credential signed by the old key
then fails verification; an in-flight mission must be cancelled or reconciled
before it is restarted. `SERVICE_REVOKED` is terminal in the existing ledger;
re-enabling that identity requires a separate Human Authority decision and a
new reviewed provisioning generation, never an edit by Qwen.

Credentials expire after at most 300 seconds and are consumed once under a
cross-process lock. A nonce cannot be issued twice. Issued, accepted,
rejected, replay, expiry and revocation records are fsynced without secret or
MAC material. Restart preserves replay and revocation state. Mission and
service revocation block later effects; root can also cancel an active mission
and close its model transport. The model has no control over these operations.

## Provisioning and start gates

The checked-in `config/sysusers.d/jaxqwen.conf`, `config/tmpfiles.d/jaxqwen.conf`
and `config/systemd/jaxqwen.service` are installation templates. Do not deploy
them as part of this PR. A later explicitly authorized host rollout must:

1. Install the sysusers and tmpfiles definitions; provision the dispatcher
   worktree root at `/var/lib/jaxqwen-dispatch/worktrees` with read-only access
   for group `jaxqwen-dispatch`; configure the governed dispatcher to create
   its canonical task worktrees there. Configure host paths to the actual
   canonical checkout, producer control directory, dispatcher state, and that
   worktree root. The host rejects any ACK worktree outside that exact root.
2. Create `/etc/jax/secrets/jaxqwen-trust.key` with 32 random bytes, owner
   `root:root`, mode `0600`. Install `/etc/jax/jaxqwen.json` as root-owned,
   non-writable by the service, with exact schema from `scripts/jaxqwen_host.py`;
   keep `dispatch_enabled` false. Its `dispatcher_uid` and `dispatch_gid` must
   be numeric IDs resolved from the dedicated sysusers entries;
   `model_socket_uid` and `model_socket_gid` must identify the existing proxy
   service and dedicated `jaxqwen-proxy` group.
3. Configure the existing executor proxy with the paired
   `JAX_PROXY_CARRIL_JAXQWEN_SOCKET=/run/jaxqwen-proxy/jaxqwen-model.sock`,
   `JAX_PROXY_CARRIL_JAXQWEN_UID=<numeric jaxqwen UID>` and
   `JAX_PROXY_CARRIL_JAXQWEN_GID=<numeric jaxqwen-proxy GID>`. Add
   `SupplementaryGroups=jaxqwen-proxy` to that proxy service. Store the
   environment file as `root:<proxy-service-group>` mode `0640`. All three
   values are required together; a partial configuration fails startup.
4. Verify the dispatcher can read only its approved producer state and can
   write its builder worktree and ACK state; verify `jaxqwen` has read-only
   access to canonical source, producer state, ACK state and dispatcher source
   worktrees, and write access only to `/var/lib/jaxqwen`. Grant read/search
   access to the task-leases lock and ledger so the service can hold the shared
   lock; grant no write access to that control state or Git metadata. Install
   the unit,
   inspect `systemd-analyze security`, then start (do not enable) proxy and
   host only under the separate human deploy gate. The service default remains
   `dispatch_enabled=false`.

No live secret, unit, socket, Qwen process or mission was provisioned during
development.

## First live dispatch

After merge and the separately approved deployment above, leave
`dispatch_enabled=false`. A human reviews the exact `DISPATCHED` ACK and
confirms task, canonical project hash, repository, branch, worktree, active
lease, acceptance criteria and the approved named tool set. The human then
authorizes that exact first mission and the operator changes only the
host-owned `dispatch_enabled` setting for the approved dispatch window, then
restarts the service. The governed dispatcher invokes its fixed
`dispatch_jaxqwen()` method only from `DISPATCHED` or a replay `NOOP` backed by
the same durable `DISPATCHED` ACK. Its explicit CLI form is
`scripts/ariadna_dispatch.py --root <checkout> --canonical-root <canonical-checkout> --handoff-state-dir <producer-control-dir> --worktree-root /var/lib/jaxqwen-dispatch/worktrees --idempotency-key <reviewed-ack-key> --consume --request-jaxqwen --jaxqwen-socket /run/jaxqwen/dispatch.sock --jaxqwen-service-uid <numeric-jaxqwen-uid>`.
The credential and mission remain bounded to that ACK and lease. Any timeout
or ambiguous transport result requires checking the durable mission ledger
before another attempt; no retry starts a second execution.

This PR does not merge, deploy, enable first dispatch, consume a live LV-001
handoff or start Qwen.
