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
Registration of a callable is never accreditation.

`ResolverRegistry` seals an immutable approved graph (binding, trusted adapter
identity, argument contract, and configuration identity) before calculating
its snapshot digest. A callable returns untrusted observation data only; it
cannot select source, resolver identity, version, or configuration digest.
`GovernedResolutionReceipt` is minted only by `ResolverRegistry`; it includes
an absolute upstream-derived `observed_at` and `not_after`. Only
`ResolverRegistry.verify_receipt(receipt, expected_scope, validation_time)`
may decide currentness. Reuse is exact-scope in F2-B, including request and
trace identity, and fails after binding, registry, source configuration, or
resolver changes. B7 evidence, B8 authority, and B9 memory remain references;
none can mint current truth.

The currently supported conflict policies are typed registry policy:
`SINGLE_SOURCE_REQUIRED`, `ALL_SOURCES_AGREE`, and
`PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK`. Current adapters use the strict
single-source form; conflict and malformed/partial upstream observations stay
non-current.

Human decisions remain required before any unresolved predicate gains an
approved source binding.  F2-B creates no universal current-state claim path;
F2-C/D will own presentation and transport-time revalidation.

## Required adversarial coverage map

`tests/test_governed_resolution.py` maps requirements 1–30 as follows:

| Requirements | Test coverage |
| --- | --- |
| 1, 15, 16 | `test_f2b_registered_resolver_without_binding_is_not_current`, `test_f2b_conflict_unknown_and_disabled_never_become_current` |
| 2, 11–13 | `test_f2b_accreditation_mismatches_fail_closed` |
| 3–7 | `test_f2b_complete_scope_rejects_environment_tenant_project_subject_audience_and_actor_substitution` |
| 8–10 | `test_f2b_stale_expired_and_not_after_are_deterministic` |
| 14 | `test_f2b_conflict_unknown_and_disabled_never_become_current` |
| 17–18 | `test_f2b_receipt_is_server_minted_and_receipt_identity_changes_with_binding` |
| 19–22 | `test_f2b_no_b7_b8_b9_reference_can_mint_current_receipt`, `test_f2b_reference_validation_preserves_historical_and_fails_closed` |
| 23–24 | `test_f2b_reference_validation_preserves_historical_and_fails_closed` |
| 25 | `test_f2b_25_capability_adapter_distinguishes_ops_catalog_and_conflict` |
| 26 | `test_f2b_26_file_exists_adapter_mints_valid_receipt_with_mocked_source` |
| 27 | `test_f2b_27_b9_adapter_preserves_designated_source_result_with_mock` |
| 28–29 | `test_f2b_registry_digest_is_deterministic_and_import_has_no_runtime_side_effects`, `test_f2b_receipt_is_server_minted_and_receipt_identity_changes_with_binding` |
| 30 | `test_f2b_registry_digest_is_deterministic_and_import_has_no_runtime_side_effects` |
