"""F2-B resolver accreditation and F2B-A01..A04 adversarial regressions."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
import pytest

from policy.governance.response import GovernanceContractError, ResponseScope
from policy.governance.resolution import *
import policy.governance.resolution as resolution

NOW=datetime(2026,9,28,12,tzinfo=timezone.utc)
def scope(**kw):return replace(ResponseScope("production","tenant-a","project-a","user-a","service:jax","human:fernando","governance","request-a","trace-a"),**kw)
def rules(s=None):
    s=s or scope();r=ScopeRule(s.environment,s.tenant_id,s.project_id,s.subject_id,s.actor_id,s.audience,s.component_id);return r,r
def binding(**kw):
    a,b=rules();return replace(PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","catalog:capabilities","authority:catalog","production",a,b,60,ConflictPolicy.SINGLE_SOURCE_REQUIRED,"adapter:capability","1","sha256:catalog","b1"),**kw)
def observation(**kw):return replace(ResolutionObservation(ResolutionStatus.RESOLVED,NOW,"artifact:catalog",{"available":True}),**kw)
def entry(**kw):
    b=binding(**{k:v for k,v in kw.items() if k in {"enabled","binding_version","resolver_version","source_configuration_digest","designated_source_identity","conflict_policy"}})
    resolver=kw.get("resolver",lambda a,s:observation())
    adapter=TrustedAdapterRegistration(resolver,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,b.source_configuration_digest)
    return RegistryEntry(b,adapter,("name","mode"))
def registry(*entries):return resolution._build_approved_registry_for_server(tuple(entries or (entry(),)))
ARGS={"name":"x","mode":"read_only"}

def test_f2b_registered_is_not_authoritative_without_accredited_entry():
    r=resolution._build_approved_registry_for_server(())
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW).status is ResolutionStatus.UNSUPPORTED
    assert registry(entry(enabled=False)).resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW).status is ResolutionStatus.UNACCREDITED
    with pytest.raises(GovernanceContractError):ResolverRegistry()

def test_f2b_a01_registry_graph_is_immutable_and_snapshot_is_exact():
    r=registry();old=r.snapshot_digest
    assert isinstance(r._entries,MappingProxyType)
    with pytest.raises(TypeError):r._entries["ENGINE_STATUS"]=entry()
    with pytest.raises(TypeError):del r._entries["CAPABILITY_AVAILABLE"]
    with pytest.raises((AttributeError,GovernanceContractError)):r._entries["CAPABILITY_AVAILABLE"].binding.binding_version="evil"
    with pytest.raises(TypeError):r._snapshot_projection["x"]="evil"
    assert r.snapshot_digest==old
    assert r.resolve("ENGINE_STATUS",{},scope(),validation_time=NOW).status is ResolutionStatus.UNSUPPORTED
    assert registry().snapshot_digest==old
    assert registry(entry(binding_version="b2")).snapshot_digest!=old

def test_f2b_a02_b9_preserves_upstream_time_source_scope_or_fails_noncurrent():
    from jax.memory.b9 import ResolutionResult,ResolutionState
    upstream=NOW-timedelta(seconds=20)
    class Source:
        def __init__(self,result):self.result=result
        def resolve(self,ref):return self.result
    s=scope();b=binding(predicate="B9_DESIGNATED_CURRENT_SOURCE",designated_source_identity="B8/jaxctl",resolver_implementation_identity=B9DesignatedSourceAdapter.resolver_id,resolver_version=B9DesignatedSourceAdapter.resolver_version,source_configuration_digest=None)
    good=B9DesignatedSourceAdapter(Source(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",upstream.timestamp(),{"healthy":True})),designated_source_identity="B8/jaxctl",scope_evidence=lambda result,actual:actual==s)
    r=registry(RegistryEntry(b,TrustedAdapterRegistration(good,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,None),("reference_type","reference_value")))
    receipt=r.resolve("B9_DESIGNATED_CURRENT_SOURCE",{"reference_type":"health","reference_value":"hall"},s,validation_time=NOW)
    assert receipt.status is ResolutionStatus.RESOLVED and receipt.observed_at==upstream
    old=replace(receipt, observed_at=NOW) if False else None
    expired=Source(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",(NOW-timedelta(seconds=61)).timestamp(),{}))
    bad=B9DesignatedSourceAdapter(expired,designated_source_identity="B8/jaxctl",scope_evidence=lambda *_:True)
    rr=registry(RegistryEntry(b,TrustedAdapterRegistration(bad,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,None),("reference_type","reference_value")))
    assert rr.resolve("B9_DESIGNATED_CURRENT_SOURCE",{"reference_type":"health","reference_value":"hall"},s,validation_time=NOW).status is ResolutionStatus.STALE
    historical=B9DesignatedSourceAdapter(Source(ResolutionResult(ResolutionState.RESOLVED_HISTORICAL,"B8/jaxctl",NOW.timestamp(),{})),designated_source_identity="B8/jaxctl",scope_evidence=lambda *_:True)
    rh=registry(RegistryEntry(b,TrustedAdapterRegistration(historical,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,None),("reference_type","reference_value")))
    assert rh.resolve("B9_DESIGNATED_CURRENT_SOURCE",{"reference_type":"health","reference_value":"hall"},s,validation_time=NOW).status is ResolutionStatus.UNAVAILABLE

def test_f2b_a03_only_registry_revalidates_receipt_and_blocks_replay():
    r=registry();receipt=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW)
    assert r.verify_receipt(receipt,scope(),validation_time=NOW)
    for changed in ("environment","tenant_id","project_id","subject_id","audience","component_id","request_id","trace_id"):
        value="staging" if changed=="environment" else "other"
        assert not r.verify_receipt(receipt,scope(**{changed:value}),validation_time=NOW)
    assert not r.verify_receipt(receipt,scope(),validation_time=NOW+timedelta(seconds=61))
    assert not registry(entry(binding_version="b2")).verify_receipt(receipt,scope(),validation_time=NOW)
    assert not registry(entry(resolver_version="2")).verify_receipt(receipt,scope(),validation_time=NOW)
    assert not registry(entry(source_configuration_digest="sha256:new")).verify_receipt(receipt,scope(),validation_time=NOW)
    with pytest.raises(GovernanceContractError):receipt.is_current_at(NOW)

def test_f2b_a04_observation_cannot_self_attest_identity_and_conflict_is_executable():
    # Legacy attacker-controlled identity strings no longer exist in observation.
    with pytest.raises(TypeError):ResolutionObservation(ResolutionStatus.RESOLVED,NOW,"x",{},actual_source_identity="catalog:capabilities")
    r=registry(entry(resolver=lambda a,s:ResolutionObservation(ResolutionStatus.CONFLICT,NOW,"x",{})))
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW).status is ResolutionStatus.CONFLICT
    with pytest.raises(GovernanceContractError):binding(conflict_policy="FAIL_CLOSED")
    preferred=registry(entry(conflict_policy=ConflictPolicy.PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK,
                             resolver=lambda a,s:ResolutionObservation(ResolutionStatus.CONFLICT,NOW,"x",{})))
    # No fallback adapter is accredited, so explicit fallback policy is safely unavailable.
    assert preferred.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW).status is ResolutionStatus.UNAVAILABLE

def test_fake_receipts_and_invalid_adapter_binding_cannot_mint_trust():
    with pytest.raises(GovernanceContractError):GovernedResolutionReceipt()
    b=binding()
    with pytest.raises(GovernanceContractError):RegistryEntry(b,TrustedAdapterRegistration(lambda a,s:observation(),"evil","1",b.designated_source_identity,b.source_configuration_digest),("name","mode"))

def test_scope_and_freshness_fail_closed():
    r=registry()
    for bad in (scope(environment="staging"),scope(tenant_id="b"),scope(project_id="b"),scope(subject_id="b"),scope(actor_id="b"),scope(audience="b")):
        assert r.resolve("CAPABILITY_AVAILABLE",ARGS,bad,validation_time=NOW).status in {ResolutionStatus.WRONG_SCOPE,ResolutionStatus.WRONG_ENVIRONMENT}
    stale=registry(entry(resolver=lambda a,s:observation(observed_at=NOW-timedelta(seconds=61))))
    assert stale.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW).status is ResolutionStatus.STALE

def test_reference_validation_remains_historical_and_fail_closed():
    from policy.governance.response import ReferenceRef,ReferenceType,TemporalClass,ExistenceState
    s=scope();ref=ReferenceRef("m",ReferenceType.MEMORY,"x","m","sha256:x",s.scope_digest,TemporalClass.HISTORICAL,ExistenceState.PRESENT)
    lookup=ReferenceLookupRecord(ReferenceType.MEMORY,"m","sha256:x",s.scope_digest,TemporalClass.HISTORICAL,ExistenceState.PRESENT,True)
    assert validate_reference(ref,s,lookup=lookup) is ReferenceValidationStatus.HISTORICAL
    assert validate_reference(ref,scope(tenant_id="b"),lookup=lookup) is ReferenceValidationStatus.WRONG_SCOPE

def test_disabled_predicates_cannot_be_current():
    r=registry()
    for predicate in ("ENGINE_STATUS","FACET_EXISTS","CONFIG_VALUE","AUDIT_EVENT_EXISTS","JOB_STATUS","MEMORY_ENTRY_EXISTS"):
        assert r.resolve(predicate,{},scope(),validation_time=NOW).status is ResolutionStatus.UNSUPPORTED


def test_f2ba01_registry_rejects_replace_and_nested_adapter_mutation():
    r=registry(); item=r._entries["CAPABILITY_AVAILABLE"]
    with pytest.raises(TypeError): r._entries["CAPABILITY_AVAILABLE"]=item
    with pytest.raises((AttributeError,GovernanceContractError)): item.adapter.resolver_id="changed"


def test_f2ba01_registry_digest_changes_for_material_source_binding_change():
    assert registry().snapshot_digest != registry(entry(designated_source_identity="catalog:other")).snapshot_digest


def test_f2ba02_b9_wrong_source_and_missing_scope_evidence_are_noncurrent():
    from jax.memory.b9 import ResolutionResult,ResolutionState
    class Source:
        def resolve(self,ref):return ResolutionResult(ResolutionState.RESOLVED_CURRENT,"wrong",NOW.timestamp(),{})
    adapter=B9DesignatedSourceAdapter(Source(),designated_source_identity="B8/jaxctl",scope_evidence=lambda *_:True)
    assert adapter({"reference_type":"health","reference_value":"x"},scope()).status is ResolutionStatus.SOURCE_MISMATCH
    assert B9DesignatedSourceAdapter(Source(),designated_source_identity="B8/jaxctl")({"reference_type":"health","reference_value":"x"},scope()).status is ResolutionStatus.UNAVAILABLE


def test_f2ba02_b9_wrapper_never_stamps_current_time():
    from jax.memory.b9 import ResolutionResult,ResolutionState
    old=NOW-timedelta(seconds=11)
    class Source:
        def resolve(self,ref):return ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",old.timestamp(),{})
    out=B9DesignatedSourceAdapter(Source(),designated_source_identity="B8/jaxctl",scope_evidence=lambda *_:True)({"reference_type":"health","reference_value":"x"},scope())
    assert out.observed_at==old


def test_f2ba03_registry_snapshot_change_invalidates_existing_receipt():
    r=registry(); receipt=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW)
    changed=registry(entry(binding_version="changed"))
    assert not changed.verify_receipt(receipt,scope(),validation_time=NOW)


def test_f2ba03_receipt_id_detects_tampering():
    r=registry(); receipt=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW)
    forged=object.__new__(GovernedResolutionReceipt)
    for field in receipt.__annotations__: object.__setattr__(forged,field,getattr(receipt,field))
    object.__setattr__(forged,"result_digest","sha256:forged")
    assert not r.verify_receipt(forged,scope(),validation_time=NOW)


def test_f2ba04_untrusted_result_custom_object_and_bad_status_fail_closed():
    with pytest.raises(GovernanceContractError): ResolutionObservation(ResolutionStatus.RESOLVED,NOW,"x",{"bad":object()})
    r=registry(entry(resolver=lambda a,s:"not observation"))
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW).status is ResolutionStatus.UNAVAILABLE


def test_f2ba04_strict_conflict_is_never_current_even_when_fresh():
    r=registry(entry(resolver=lambda a,s:ResolutionObservation(ResolutionStatus.CONFLICT,NOW,"two sources",{})))
    receipt=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW)
    assert receipt.status is ResolutionStatus.CONFLICT and not r.verify_receipt(receipt,scope(),validation_time=NOW)


def test_import_has_no_runtime_resolution_side_effects():
    # The module exposes types and explicit construction only; it has no default registry.
    assert not hasattr(resolution,"DEFAULT_REGISTRY")


def test_capability_adapter_derives_only_its_registered_source_kind():
    class Catalog:
        def get_capability(self,name):return object() if name=="catalog" else None
    class Context:
        ops=frozenset({"ops"});catalog=Catalog()
    adapter=CapabilityAvailableAdapter(Context(),source_kind="ops",verdict_resolver=lambda *_:"VALID",clock=lambda:NOW)
    assert adapter({"name":"ops","mode":"read_only"},scope()).status is ResolutionStatus.RESOLVED
    assert adapter({"name":"catalog","mode":"read_only"},scope()).status is ResolutionStatus.UNAVAILABLE
