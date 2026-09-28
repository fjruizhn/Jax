# AXIOMA 3.0 — F2-B authoritative resolution layer

F2-B adds the server-owned accreditation contract behind future governed
current-state claims.  It does not render, emit, migrate an egress, or make
any existing response path governed.

## Predicate status

| Predicate | Existing implementation | F2-B accreditation status |
| --- | --- | --- |
| `CAPABILITY_AVAILABLE` | `policy.governance.validator._resolve_capability_available` | adapter-capable; enabled only with a human-owned binding and configuration digest |
| `FILE_EXISTS` | `policy.governance.validator._resolve_file_exists` | adapter-capable; enabled only with an allowlisted filesystem binding and configuration digest |
| `B9_DESIGNATED_CURRENT_SOURCE` | `jax.memory.b9_resolvers.DesignatedSourceResolver` | adapter-capable only when upstream source, observation time, and scope evidence are independently preserved; otherwise unavailable |
| `ENGINE_STATUS` | none | disabled / unaccredited |
| `FACET_EXISTS` | none | disabled / unaccredited |
| `CONFIG_VALUE` | none | disabled / unaccredited |
| `AUDIT_EVENT_EXISTS` | none | disabled / unaccredited |
| `JOB_STATUS` | none | disabled / unaccredited |
| `MEMORY_ENTRY_EXISTS` | none | disabled / unaccredited |

`PredicateAuthorityBinding` pins predicate/version, source identity and
owner, environment, explicit scope rules, freshness SLA, conflict policy,
resolver identity/version, and source configuration digest where meaningful.
Registration of a callable is never accreditation.  The sealed registry stores
only an immutable `TrustedAdapterRegistration` specification (`adapter_kind`,
identity/version, designated source, and canonical configuration).  A
server-owned dispatcher consumes explicit observed inputs; neither a callable,
closure, nor resolver-returned strings can self-attest authority metadata.

`ResolverRegistry` seals an immutable approved graph (binding, trusted adapter
identity, argument contract, and configuration identity) before calculating
its snapshot digest. The registry object itself is sealed: its graph,
snapshot, and authenticator cannot be replaced in place. Observation data is
untrusted and cannot select source, resolver identity, version, or
configuration digest.
`GovernedResolutionReceipt` is minted only by `ResolverRegistry`; it includes
an absolute upstream-derived `observed_at` and `not_after`. Only
`ResolverRegistry.verify_receipt(receipt, expected_scope, validation_time)`
may decide currentness. Reuse is exact-scope in F2-B, including request and
trace identity, and fails after binding, registry, source configuration, or
resolver changes. B7 evidence, B8 authority, and B9 memory remain references;
none can mint current truth.

Receipt content identity is not treated as issuer authority. Each receipt also
carries an HMAC authentication tag from a server-owned `ReceiptAuthenticator`;
the key is never serialized or provided by a resolver. F2-B supplies only the
abstraction and ephemeral test keys. Production key provisioning remains a
future composition-root concern and no production receipt path is enabled.

The currently supported conflict policies are typed registry policy:
`SINGLE_SOURCE_REQUIRED`, `ALL_SOURCES_AGREE`, and
`PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK`. Current adapters use the strict
single-source form; conflict and malformed/partial upstream observations stay
non-current.

Human decisions remain required before any unresolved predicate gains an
approved source binding.  F2-B creates no universal current-state claim path;
F2-C/D will own presentation and transport-time revalidation.

## Required adversarial coverage map

`tests/test_governed_resolution.py` includes final adversarial regressions for
the sealed registry, explicit B9 upstream timing/source data, authenticated
receipt tampering and exact-scope replay, and multi-source agreement. The
previous adapter-specific integration examples were deliberately replaced by
the trusted dispatcher contract; host wiring is not authorized in F2-B.
