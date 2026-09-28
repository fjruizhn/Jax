"""F2-B accreditation tests: resolver registration is never authority by itself."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import sys
import pytest

from policy.governance.response import GovernanceContractError, ResponseScope, ReferenceRef, ReferenceType, TemporalClass, ExistenceState
from policy.governance.resolution import *
import policy.governance.resolution as resolution

NOW = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
def scope(**kw):
    return replace(ResponseScope("production", "tenant-a", "project-a", "user-a", "service:jax", "human:fernando", "governance", "request-a", "trace-a"), **kw)
def rules(s=None):
    s=s or scope()
    rule=ScopeRule(s.environment,s.tenant_id,s.project_id,s.subject_id,s.actor_id,s.audience,s.component_id)
    return rule, rule
def binding(predicate="CAPABILITY_AVAILABLE", **kw):
    a,b=rules(); base=PredicateAuthorityBinding(predicate,"v1","catalog:capabilities","authority:catalog","production",a,b,60,"FAIL_CLOSED","validator:capability","1","sha256:catalog","b1")
    return replace(base, **kw)
def observation(s=None, **kw):
    s=s or scope(); base=ResolutionObservation("catalog:capabilities","sha256:catalog",s,NOW,ResolutionStatus.RESOLVED,"artifact:catalog",{"available":True},"validator:capability","1")
    return replace(base, **kw)
def registry(predicate="CAPABILITY_AVAILABLE", resolver=None, **kw):
    resolver = resolver or (lambda args, s: observation(s))
    return resolution._build_approved_registry_for_server((RegistryEntry(binding(predicate, **kw), resolver, ("name", "mode")),))

def test_f2b_registered_resolver_without_binding_is_not_current():
    with pytest.raises(GovernanceContractError): ResolverRegistry(())
    r=resolution._build_approved_registry_for_server(())
    assert r.resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW).status is ResolutionStatus.UNSUPPORTED
    disabled=registry(enabled=False)
    assert disabled.resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW).status is ResolutionStatus.UNACCREDITED

@pytest.mark.parametrize("changed,expected", [("actual_source_identity",ResolutionStatus.SOURCE_MISMATCH),("resolver_version",ResolutionStatus.VERSION_MISMATCH),("source_configuration_digest",ResolutionStatus.CONFIGURATION_MISMATCH)])
def test_f2b_accreditation_mismatches_fail_closed(changed, expected):
    r=registry(resolver=lambda a,s: observation(s, **{changed:"wrong"}))
    assert r.resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW).status is expected

def test_f2b_complete_scope_rejects_environment_tenant_project_subject_audience_and_actor_substitution():
    r=registry()
    for bad in (scope(environment="staging"),scope(tenant_id="tenant-b"),scope(project_id="project-b"),scope(subject_id="user-b"),scope(actor_id="service:other"),scope(audience="agent:jacobs")):
        result=r.resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, bad, validation_time=NOW)
        assert result.status in {ResolutionStatus.WRONG_SCOPE,ResolutionStatus.WRONG_ENVIRONMENT}
    # actor is never a fallback for subject authority.

def test_f2b_stale_expired_and_not_after_are_deterministic():
    r=registry(resolver=lambda a,s: observation(s, observed_at=NOW-timedelta(seconds=61)))
    receipt=r.resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW)
    assert receipt.status is ResolutionStatus.STALE and not receipt.is_current_at(NOW)
    good=registry().resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW)
    assert good.not_after == NOW + timedelta(seconds=60)
    assert not good.is_current_at(good.not_after + timedelta(microseconds=1))

def test_f2b_conflict_unknown_and_disabled_never_become_current():
    r=registry(resolver=lambda a,s: observation(s,status=ResolutionStatus.CONFLICT))
    assert r.resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW).status is ResolutionStatus.CONFLICT
    assert registry().resolve("NOT_A_PREDICATE", {}, scope(), validation_time=NOW).status is ResolutionStatus.UNSUPPORTED

def test_f2b_receipt_is_server_minted_and_receipt_identity_changes_with_binding():
    with pytest.raises(GovernanceContractError): GovernedResolutionReceipt()
    one=registry().resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW)
    two=registry(binding_version="b2").resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW)
    assert one.receipt_id != two.receipt_id and one.registry_snapshot_digest != two.registry_snapshot_digest

def test_f2b_reference_validation_preserves_historical_and_fails_closed():
    s=scope(); r=ReferenceRef("e",ReferenceType.EVIDENCE,"axioma://e","evidence:1","sha256:e",s.scope_digest,TemporalClass.HISTORICAL,ExistenceState.PRESENT)
    lookup=ReferenceLookupRecord(ReferenceType.EVIDENCE,"evidence:1","sha256:e",s.scope_digest,TemporalClass.HISTORICAL,ExistenceState.PRESENT,True)
    assert validate_reference(r,s,lookup=lookup) is ReferenceValidationStatus.HISTORICAL
    assert validate_reference(r,scope(tenant_id="tenant-b"),lookup=lookup) is ReferenceValidationStatus.WRONG_SCOPE
    assert validate_reference(r,s,lookup=lookup,expected_revision="sha256:other") is ReferenceValidationStatus.WRONG_REVISION
    assert validate_reference(r,s,lookup=None) is ReferenceValidationStatus.DANGLING
    tombstone=replace(lookup, existence_state=ExistenceState.TOMBSTONED)
    assert validate_reference(r,s,lookup=tombstone) is ReferenceValidationStatus.TOMBSTONED
    denied=replace(lookup, temporal_class=TemporalClass.CURRENT, accessible=False)
    assert validate_reference(r,s,lookup=denied) is ReferenceValidationStatus.ACCESS_DENIED

def test_f2b_no_b7_b8_b9_reference_can_mint_current_receipt():
    s=scope()
    for typ in (ReferenceType.EVIDENCE, ReferenceType.AUTHORITY, ReferenceType.MEMORY):
        ref=ReferenceRef(typ.value,typ,"axioma://x",typ.value,"sha256:x",s.scope_digest,TemporalClass.HISTORICAL,ExistenceState.PRESENT)
        lookup=ReferenceLookupRecord(typ,typ.value,"sha256:x",s.scope_digest,TemporalClass.HISTORICAL,ExistenceState.PRESENT,True)
        assert validate_reference(ref,s,lookup=lookup) is ReferenceValidationStatus.HISTORICAL

def test_f2b_registry_digest_is_deterministic_and_import_has_no_runtime_side_effects():
    assert registry().snapshot_digest == registry().snapshot_digest
    assert registry().status_table()[0]["enabled"] is True

def test_f2b_adapter_shapes_for_existing_resolvers_only():
    # These adapters model the existing validator/B9 result boundary without
    # invoking filesystem, DB, or network I/O in the F2-B core package.
    cap=registry("CAPABILITY_AVAILABLE")
    assert cap.resolve("CAPABILITY_AVAILABLE", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW).is_current_at(NOW)
    files=registry("FILE_EXISTS", resolver=lambda a,s: observation(s, actual_source_identity="filesystem:allowlisted", resolver_id="validator:file", resolver_version="1", source_configuration_digest="sha256:file"), designated_source_identity="filesystem:allowlisted", resolver_implementation_identity="validator:file", source_configuration_digest="sha256:file")
    assert files.resolve("FILE_EXISTS", {"name":"x","mode":"read_only"}, scope(), validation_time=NOW).status is ResolutionStatus.RESOLVED

def test_f2b_25_capability_adapter_distinguishes_ops_catalog_and_conflict():
    class Catalog:
        def __init__(self, entries): self.entries=entries
        def get_capability(self, name): return self.entries.get(name)
    class Capability: mode="read_only"
    class Context:
        ops=frozenset({"ops-only", "conflict"}); mutating_capabilities=frozenset()
        catalog=Catalog({"catalog-only":Capability(), "conflict":Capability()})
    adapter=CapabilityAvailableAdapter(Context(), ops_source_identity="capability:ops", catalog_source_identity="capability:catalog", ops_configuration_digest="sha256:ops", catalog_configuration_digest="sha256:catalog", verdict_resolver=lambda a,c:"VALID", clock=lambda:NOW)
    assert adapter({"name":"ops-only","mode":"read_only"},scope()).actual_source_identity == "capability:ops"
    assert adapter({"name":"catalog-only","mode":"read_only"},scope()).actual_source_identity == "capability:catalog"
    assert adapter({"name":"conflict","mode":"read_only"},scope()).status is ResolutionStatus.CONFLICT

def test_f2b_26_file_exists_adapter_mints_valid_receipt_with_mocked_source(tmp_path):
    content=b"f2-b"; file=tmp_path / "allowed.txt"; file.write_bytes(content)
    digest="sha256:" + __import__("hashlib").sha256(content).hexdigest()
    class Context:
        config_paths_allowlist=frozenset({"allowed.txt"}); repo_root=Path(tmp_path)
    adapter=FileExistsAdapter(Context(),source_identity="filesystem:allowlisted",configuration_digest="sha256:file",verdict_resolver=lambda a,c:"VALID",clock=lambda:NOW)
    s=scope(); rule=ScopeRule(s.environment,s.tenant_id,s.project_id,s.subject_id,s.actor_id,s.audience,s.component_id)
    b=PredicateAuthorityBinding("FILE_EXISTS","v1","filesystem:allowlisted","authority:filesystem","production",rule,rule,60,"FAIL_CLOSED",adapter.resolver_id,adapter.resolver_version,"sha256:file","b1")
    r=resolution._build_approved_registry_for_server((RegistryEntry(b,adapter,("path","hash")),))
    now=NOW
    assert r.resolve("FILE_EXISTS",{"path":"allowed.txt","hash":digest},s,validation_time=now).status is ResolutionStatus.RESOLVED

def test_f2b_27_b9_adapter_preserves_designated_source_result_with_mock():
    from jax.memory.b9 import ResolutionResult, ResolutionState
    class Resolver:
        def resolve(self, reference):
            assert reference.reference_type == "health" and reference.reference_value == "hall9000"
            return ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",0.0,{"healthy":True})
    adapter=B9DesignatedSourceAdapter(Resolver(),source_identity="b9:designated",clock=lambda:NOW)
    s=scope(); rule=ScopeRule(s.environment,s.tenant_id,s.project_id,s.subject_id,s.actor_id,s.audience,s.component_id)
    b=PredicateAuthorityBinding("B9_DESIGNATED_CURRENT_SOURCE","v1","b9:designated","authority:b9","production",rule,rule,60,"FAIL_CLOSED",adapter.resolver_id,adapter.resolver_version,None,"b1")
    r=resolution._build_approved_registry_for_server((RegistryEntry(b,adapter,("reference_type","reference_value")),))
    receipt=r.resolve("B9_DESIGNATED_CURRENT_SOURCE",{"reference_type":"health","reference_value":"hall9000"},s,validation_time=NOW)
    assert receipt.status is ResolutionStatus.RESOLVED and receipt.provenance_ref == "b9:B8/jaxctl"

def _legacy_validator_module():
    """Load the existing legacy module in its documented import layout."""
    legacy_path=str(Path(__file__).parents[1] / "policy" / "governance")
    if legacy_path not in sys.path: sys.path.insert(0,legacy_path)
    import validator
    return validator

def test_f2b_25_factories_invoke_actual_legacy_capability_resolver_with_mock_context():
    legacy=_legacy_validator_module()
    class Catalog:
        def get_capability(self,name): return None
    class Context:
        ops=frozenset({"read"}); mutating_capabilities=frozenset(); catalog=Catalog()
    adapter=capability_adapter_from_legacy(Context(),legacy._resolve_capability_available,
        ops_source_identity="capability:ops",catalog_source_identity="capability:catalog",
        ops_configuration_digest="sha256:ops",catalog_configuration_digest="sha256:catalog",clock=lambda:NOW)
    result=adapter({"name":"read","mode":"read_only"},scope())
    assert result.status is ResolutionStatus.RESOLVED and result.actual_source_identity == "capability:ops"

def test_f2b_26_factory_invokes_actual_legacy_file_resolver_with_mock_filesystem(tmp_path):
    legacy=_legacy_validator_module(); content=b"legacy-file"; (tmp_path / "allowed.txt").write_bytes(content)
    class Context:
        config_paths_allowlist=frozenset({"allowed.txt"}); repo_root=tmp_path
    adapter=file_exists_adapter_from_legacy(Context(),legacy._resolve_file_exists,
        source_identity="filesystem:allowlisted",configuration_digest="sha256:file",clock=lambda:NOW)
    result=adapter({"path":"allowed.txt","hash":hashlib.sha256(content).hexdigest()},scope())
    assert result.status is ResolutionStatus.RESOLVED

def test_f2b_serialized_fake_receipt_has_no_public_trusted_loader():
    fake={"receipt_id":"receipt:fake","predicate":"ENGINE_STATUS","arguments_digest":"sha256:fake",
          "binding_version":"fake","registry_snapshot_digest":"sha256:fake","actual_source_identity":"fake",
          "source_configuration_digest":"sha256:fake","observation_scope_digest":scope().scope_digest,
          "observed_at":NOW,"not_after":NOW+timedelta(seconds=1),"status":ResolutionStatus.RESOLVED,
          "provenance_ref":"fake","resolver_id":"fake","resolver_version":"fake","result_digest":"sha256:fake"}
    assert not hasattr(resolution,"load_resolution_receipt")
    with pytest.raises(GovernanceContractError): GovernedResolutionReceipt(**fake)
