# Ariadna authority contract — LV-003

Human Authority has approved Ariadna (`ariadna-project-manager`) as an active
governed PM runtime. This grants no human authority, capability, merge,
deploy, or production-mutation power. The deterministic evaluator returns only
`ALLOW`, `DENY`, or `HUMAN_REQUIRED`; no LLM decides authorization. Merge,
deploy, production mutation, capability grant, and runtime execution are always
`DENY`, including if a human could separately authorize the real-world action.

## DONE evidence and trust boundary

`DONE` needs a task-bound `JAX_TEST_EVIDENCE_MANIFEST`, a `commit:<sha>` or
PR record resolving to the exact same SHA, and one task-bound acceptance record
for that exact SHA. The test manifest follows the existing policy CI evidence
shape: schema/version, task, GitHub Actions provider, repository, commit,
workflow/run/job identity, CI environment, test list/counts, and raw artifact
hash. The acceptance record names task, human acceptance actor, human authority
source/record, commit, explicit criteria/results, evidence refs, and decision.

Hashes bind artifacts but do not prove who ran CI or approved acceptance. The
local evaluator therefore requires controlled CI and human-authority verifiers
configured only when the privileged integration constructs an
`AuthorityEngine`. They are not request fields and a request cannot select,
replace, or inject either verifier. A local JUnit file, JSON
manifest, filename, git message, commit author, actor string, or self-computed
hash is never sufficient. If that independent service is unavailable, the
result is `HUMAN_REQUIRED`; a rejected/malformed/missing record is `DENY`.
This is the explicit offline trust boundary, not invented cryptographic trust.

For one authorization, every direct local evidence record and the complete
supported transitive closure (currently a PR record's
`execution_manifest`) is opened exactly once into an immutable in-memory
snapshot. Structural validation and controlled CI/human verification consume
only those snapshots; they never reopen an evidence path. The normalized
evidence digest includes each snapshot's repository-relative reference and
byte digest, including a transitively referenced PR execution manifest. Thus
the intent/commit binding is for the exact bytes evaluated, rather than just
the names of mutable files.

The implementation commit must resolve and be an ancestor of the repository
context under evaluation. Test manifest, commit/PR evidence, and acceptance
must name precisely one identical SHA. Commit-message wording is informational
only. A PR record names its provider, repository, numeric id, exact head SHA,
and a controlled execution-manifest reference. Offline PR numbers cannot
independently prove a head; without that bound controlled record completion is
not allowed.

`BLOCKED` requires a task-bound, open blocker with a non-empty blocker fact.
All other allowed status transitions still validate canonical task identity and
lifecycle. Unknown actions outside the deterministic scope are
`HUMAN_REQUIRED`, never executed automatically.

## Recoverable state/audit protocol

`transition()` obtains an exclusive lock and fails closed if prior journal
intents are unfinished. It creates its audit fields internally; callers cannot
supply duplicate event fields. For a transition id it durably appends and
fsyncs `TRANSITION_INTENT`, performs a byte-for-byte compare-and-swap guarded
atomic state write plus directory fsync, then durably appends and fsyncs
`TRANSITION_COMMITTED`. Both records bind transition id, task, from/to status,
project hash before, actor, and decision. Its `evidence_binding_hash` is a
digest of the normalized, byte-digested evidence snapshot that produced
authorization (test-manifest, acceptance-record, PR if present, and commit
bindings), never merely evidence path names. The committed event repeats that
exact binding and its component digests; commit also binds the resulting
project hash. An intent without a matching bound commit is
explicitly `interrupted` and requires human reconciliation. It is never
silently repaired or discarded.

Per-file atomic replacement is not represented as multi-file atomicity. The
journal makes every persistence-boundary failure detectable. The activity log
is append-only; past events are neither rewritten nor caller-authored.

## Handoffs

Handoffs use the existing `sync/message-envelope.schema.json`, not a third
protocol. `authority_context.handoff` must provide `owner`, `branch_worktree`,
`scope`, `acceptance_criteria`, `commit_pr`, `test_evidence`, `blockers`, and
`next_action`, alongside the MessageEnvelope fields.

## Governed builder handoff consumer

`ariadna_dispatcher.py` is a separate privileged composition boundary, not an
Ariadna capability and not a builder runner. It accepts only an exact
append-only outbox record that is structurally valid, has one matching
runtime-built `ALLOW`/`EFFECT_EMIT_HANDOFF` audit row, validates against the
current canonical task and a live matching lease, and is still allowed by a
fresh AuthorityEngine evaluation. It derives the branch and worktree from the
canonical task id; it never accepts a command, executable, branch, or
filesystem path from a handoff as an execution instruction.

The dispatcher serializes replay with a local flock and appends durable ACKs
`RECEIVED`, `ACCEPTED`, `DISPATCHED`, or `REJECTED`. `ACCEPTED` precedes the
only effect: fixed-argument, hook-disabled Git creation of a clean isolated
development worktree. `DISPATCHED` means **WORKTREE_ONLY** and explicitly does
not mean a builder process started. Replays of `DISPATCHED` or `REJECTED` are
no-ops; a released, stale, malformed, conflicting, or ambiguous handoff is
rejected without deleting history or forcing a worktree.

The producer's canonical checkout/control state and the builder checkout are
explicit independent roots. Their canonical project bytes must match; ACKs
remain under the builder checkout's Git common directory rather than the
Ariadna producer control directory. There is currently no authenticated
Ariadna-to-Qwen execution adapter. A later adapter must establish a service
identity, repository allowlist, task/lease/hash binding, and mission
idempotency; until then no dispatcher code may spawn Qwen or another builder.
The one-shot consumer also requires a deliberately provisioned dispatcher
identity that can read the producer control lock and write only the approved
development worktree/ACK paths. It must not be run as root to bridge those
ownership domains; absent that narrow host identity or an equivalent
privilege-dropping broker, consumption fails closed.

## LV-004 hosted runtime

ariadna_runtime.py is a local, host-invoked run_once control loop, not an
activation, network endpoint, or execution capability. `ariadna_host.py` is
the privileged local composition root: it constructs fixed verifier adapters,
the engine, scheduler, kill-switch checks, and health state; proposal data
cannot replace any of them. Its repository systemd unit is an uninstalled
template, not a production deployment claim. Host readiness means only that
its local lease is held; ACTIVE_GOVERNED is a canonical human-authorized
lifecycle, not a production deployment claim. It accepts data-only proposals, rebinds them to the
current project hash, and passes authorized transitions through this contract's
locked expected-hash/CAS journal path. It cannot provide or replace verifiers,
edit this contract, execute commands, or make runtime_execution ALLOW.

The host uses nonblocking local flock ownership. This is intentionally not
distributed consensus: a competing local process fails closed, stale metadata
is reported after an OS-released crash lock, and no lock is silently stolen.
Task leases are append-only records and deny concurrent identical task,
worktree, or ancestor/descendant writable-scope ownership. A host stop request
prevents a new effectful tick; explicit shutdown releases only its local
control lease.

An unreleased task lease from a dead or unknown process is surfaced as
`RECONCILIATION_REQUIRED`, never silently stolen. The GitHub Actions adapter
requires a configured repository and exact numeric run/job plus commit binding;
unavailable API evidence is `HUMAN_REQUIRED`. The Human Authority adapter
requires a strict append-only task-acceptance record, not an actor string.

Each terminal cycle is idempotently appended with host-built instance, cycle,
project hash, task/action, verdict, evidence refs, lease, and result fields.
Planner data cannot provide an audit verdict. Repeated unchanged proposals do
not append duplicate handoffs, escalations, or cycle records. DENY and
HUMAN_REQUIRED never execute; time/retry alone cannot change either verdict.
An unresolved LV-003 transition intent places a new host in
RECONCILIATION_REQUIRED, emits no automatic repair, and permits no work.

The host-owned deterministic proposal source reads only canonical task bytes.
It considers only `READY` tasks whose declared dependencies are all `DONE`,
which have no blocker fact and no conflicting local ownership. It ranks parsed
priority (`P0` before `P1`) and task id deterministically, then creates a
MessageEnvelope handoff for the canonical owner. The planner cannot construct
verifiers, mutate project state, run commands, or dispatch a builder; the
durable handoff outbox is the current integration boundary. Until canonical
writable-scope metadata exists, an assignment conservatively leases the whole
repository boundary, so the host never guesses that concurrent writers are
safe. Repeated unchanged observations reuse the semantic handoff key and do
not append another assignment; a changed canonical project hash permits a new
consideration.
