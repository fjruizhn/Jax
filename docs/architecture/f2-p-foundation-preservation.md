# F2-P — Foundation Preservation for Long-Term Learning (v1)

## Purpose and boundary

This is a directional contract for governed future learning. It preserves
provenance and safety boundaries; it does not create training, promotion,
provider access, or a model-audit UI.

Memory records historical context. Storage, extraction, synthesis,
deduplication, or an audit label never make it current truth, authority,
evidence, training authorization, or a production claim. Even human-verified
knowledge is not automatically training material; dataset inclusion and every
later learning stage require separate governed authorization.

## Preserved contracts

- Extracted items retain exact locked source message IDs, turn numbers and
  author roles in provenance. The extraction transformation is `b9-worker-v3`.
  This proves source identity and attribution only: the extractor chooses its
  cited turns, and a valid `source_turns` set does not deterministically prove
  exhaustive semantic support, `Lifecycle.VERIFIED`, `SOURCE_AUDITED`, or
  current truth.
- Only an authenticated `USER`, with the existing resolved authorization, can
  create `Lifecycle.VERIFIED`. Models, agents and services cannot verify.
- Deduplication, merge, extraction and synthesis never verify. Derived content
  retains exact source revisions and inherits the worst source provenance.
  A corrected, revoked, tombstoned or purged source makes its synthesis
  unavailable for normal reading until a new governed derivation is made.
- Tenant, user, project and visibility boundaries apply before discovery,
  grouping, selection or mutation. A tenant experience never feeds another.
- A future model audit is orthogonal to lifecycle. Its allowed statuses are
  `NOT_AUDITED`, `SOURCE_AUDITED`/`MODEL_AUDITED`, `AUDIT_INCONCLUSIVE`,
  `AUDIT_REJECTED`, and `AUDIT_REVOKED`. None implies `VERIFIED`, current
  observation, authority, evidence, or training authorization.

## Future one-button audit

The intended future operation is: Fernando selects memories and chooses
**Auditar** once; the server freezes scope, revision IDs, source set and
digests; an independent eligible model returns a structured verdict; server
policy validates it and applies only the pre-authorized operation; a durable
report retains producer/auditor identities, prompt and schema versions,
snapshot, coverage, limitations, verdict and applied operation. The model
never writes MariaDB, expands scope or invokes arbitrary mutation.

Producer and auditor must be independent where required. Eligibility depends
on tenant, project, data classification and provider authorization; scopes
that prohibit external providers require a local independent auditor. An
inconclusive audit makes no mutation. A model-audited merge creates a new
revision/event/relationship, preserves originals and cannot improve
provenance, erase contradiction or cross scope.

## Deferred work

No F2-P code creates datasets, training runs, artifacts, candidates,
evaluations, promotions, adapters, model weights, deployment manifests, or
feedback/knowledge types. Future sealed datasets, candidates, evaluations and
promotions must be immutable/content-addressed; training completion does not
promote, promotion evaluation is independent, and a deployment must identify
immutable base, adapter and runtime contracts while retaining its prior
deployment reference. Provider catalogs are mutable and are not promoted-model
identity; every promoted version must be explainable by its evidence. Production
interaction never modifies model weights or adapters, and a future binding
change requires a historical record. Attachment content digests (C2) remain a
human privacy decision.
