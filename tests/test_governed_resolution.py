"""F2-B adversarial regressions, including final A01--A04 remediation."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
import hashlib, hmac, json
import pytest
from policy.governance.response import GovernanceContractError, ResponseScope
from policy.governance.resolution import *
import policy.governance.resolution as mod
from jax.memory.b9 import MemoryEnvelope, ResolutionResult, ResolutionState

NOW=datetime(2026,9,28,12,tzinfo=timezone.utc)
KEY=b"f2-b-test-secret-material-longer-than-thirty-two-bytes"
ARGS={"name":"x","mode":"read_only"}
def scope(**kw):return replace(ResponseScope("production","tenant-a","project-a","user-a","service:jax","human:fernando","governance","request-a","trace-a"),**kw)
def rules():
    s=scope();r=ScopeRule(s.environment,s.tenant_id,s.project_id,s.subject_id,s.actor_id,s.audience,s.component_id);return r,r
def binding(**kw):
    a,b=rules();return replace(PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","catalog:capabilities","authority:catalog","production",a,b,60,ConflictPolicy.SINGLE_SOURCE_REQUIRED,"adapter:capability","1","sha256:catalog","b1"),**kw)
def entry(**kw):
    b=binding(**{k:v for k,v in kw.items() if k in {"enabled","binding_version","resolver_version","source_configuration_digest","designated_source_identity","conflict_policy","designated_source_identities"}})
    kind=kw.get("kind",AdapterKind.CAPABILITY_AVAILABLE)
    return RegistryEntry(b,TrustedAdapterRegistration(kind,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,b.source_configuration_digest,{"root":"approved"}),("name","mode"))
def registry(*entries,key=KEY):return mod._build_approved_registry_for_server(tuple(entries or (entry(),)),authenticator=ReceiptAuthenticator.for_testing(key))
def obs(status=ResolutionStatus.RESOLVED,at=NOW,result=None):return ResolutionObservation(status,at,"artifact:catalog",result if result is not None else {"available":True})
def server_input(values):return ServerAdapterInput._mint(mod._ADAPTER_INPUT_TOKEN,tuple(values))
def inp(b,*values):return server_input(values or ((b.designated_source_identity,obs()),))

def test_f2ba01_registry_object_and_graph_are_immutable():
    r=registry(); old=r.snapshot_digest; e=r._entries["CAPABILITY_AVAILABLE"]
    for name,value in (("_entries",MappingProxyType({})),("_snapshot_projection",MappingProxyType({})),("_snapshot_digest","sha256:evil"),("_authenticator",ReceiptAuthenticator.for_testing(KEY))):
        with pytest.raises(GovernanceContractError):setattr(r,name,value)
    with pytest.raises(GovernanceContractError):del r._entries
    with pytest.raises(TypeError):r._entries["ENGINE_STATUS"]=e
    with pytest.raises((TypeError,AttributeError,GovernanceContractError)):e.adapter.configuration["root"]="evil"
    assert r.snapshot_digest==old
    assert r.resolve("ENGINE_STATUS",{},scope(),validation_time=NOW).status is ResolutionStatus.UNSUPPORTED
    assert registry(entry(binding_version="b2")).snapshot_digest!=old

def test_f2ba02_b9_evidence_is_explicit_preserved_and_not_freshened():
    b=binding(predicate="B9_DESIGNATED_CURRENT_SOURCE",designated_source_identity="B8/jaxctl",designated_source_identities=("B8/jaxctl",),resolver_implementation_identity="jax.memory.b9_resolvers:DesignatedSourceResolver",resolver_version="existing-v1",source_configuration_digest=None)
    e=RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.B9_DESIGNATED_CURRENT_SOURCE,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,None,{}),("reference_type","reference_value"));r=registry(e)
    old=NOW-timedelta(seconds=20);args={"reference_type":"health","reference_value":"hall"}
    upstream=ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",old.timestamp(),{"value":"x"})
    good=mod._b9_evidence_from_server(upstream,scope(),"b9:immutable-upstream")
    rec=r.resolve(b.predicate,args,scope(),validation_time=NOW,b9_evidence=good)
    assert rec.status is ResolutionStatus.RESOLVED and rec.observed_at==old
    wrong=mod._b9_evidence_from_server(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"wrong",old.timestamp(),{}),scope(),"b9:wrong")
    assert r.resolve(b.predicate,args,scope(),validation_time=NOW,b9_evidence=wrong).status is ResolutionStatus.SOURCE_MISMATCH
    assert r.resolve(b.predicate,args,scope(),validation_time=NOW).status is ResolutionStatus.UNAVAILABLE
    unavailable=mod._b9_evidence_from_server(ResolutionResult(ResolutionState.SOURCE_UNAVAILABLE,"B8/jaxctl",old.timestamp(),{}),scope(),"b9:unavailable")
    assert r.resolve(b.predicate,args,scope(),validation_time=NOW,b9_evidence=unavailable).status is ResolutionStatus.UNAVAILABLE
    stale=mod._b9_evidence_from_server(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",(NOW-timedelta(seconds=61)).timestamp(),{}),scope(),"b9:stale")
    assert r.resolve(b.predicate,args,scope(),validation_time=NOW,b9_evidence=stale).status is ResolutionStatus.STALE
    assert r.resolve(b.predicate,args,scope(),validation_time=NOW,server_input=server_input((("B8/jaxctl",obs()),))).status is ResolutionStatus.UNAVAILABLE
    with pytest.raises(GovernanceContractError):mod._b9_evidence_from_server({"state":"RESOLVED_CURRENT"},scope(),"fake")

def test_b9_h1_evidence_value_is_canonical_frozen_and_digest_bound():
    original={"nested":{"items":["x",{"v":1}]}}
    upstream=ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",NOW.timestamp(),original)
    evidence=mod._b9_evidence_from_server(upstream,scope(),"b9:frozen")
    retained=evidence.result.value
    original["nested"]["items"][1]["v"]=2
    original["nested"]["items"].append("later")
    assert retained["nested"]["items"]==("x",MappingProxyType({"v":1}))
    with pytest.raises(TypeError):retained["nested"]["items"][1]["v"]=3
    same=mod._b9_evidence_from_server(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",NOW.timestamp(),{"nested":{"items":["x",{"v":1}]}}),scope(),"b9:same")
    changed=mod._b9_evidence_from_server(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",NOW.timestamp(),{"nested":{"items":["x",{"v":2}]}}),scope(),"b9:changed")
    assert mod._digest(evidence.result.value)==mod._digest(same.result.value)
    assert mod._digest(evidence.result.value)!=mod._digest(changed.result.value)

@pytest.mark.parametrize("value",["text",42,["list"],object(),{1:"bad-key"}])
def test_b9_h1_rejects_unsupported_or_noncanonical_current_values(value):
    with pytest.raises(GovernanceContractError):
        mod._b9_evidence_from_server(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",NOW.timestamp(),value),scope(),"b9:invalid")

def test_b9_h2_binding_and_runtime_are_strictly_single_source():
    a,c=rules()
    common=dict(predicate="B9_DESIGNATED_CURRENT_SOURCE",designated_source_identity="B8/jaxctl",resolver_implementation_identity="jax.memory.b9_resolvers:DesignatedSourceResolver",resolver_version="existing-v1",source_configuration_digest=None)
    with pytest.raises(GovernanceContractError):
        RegistryEntry(
            binding(**common,conflict_policy=ConflictPolicy.ALL_SOURCES_AGREE,designated_source_identities=("B8/jaxctl","B8/other")),
            TrustedAdapterRegistration(AdapterKind.B9_DESIGNATED_CURRENT_SOURCE,"jax.memory.b9_resolvers:DesignatedSourceResolver","existing-v1","B8/jaxctl",None,{}),
            ("reference_type","reference_value"),
        )
    with pytest.raises(GovernanceContractError):
        RegistryEntry(
            binding(**common,designated_source_identities=("B8/jaxctl","B8/other")),
            TrustedAdapterRegistration(AdapterKind.B9_DESIGNATED_CURRENT_SOURCE,"jax.memory.b9_resolvers:DesignatedSourceResolver","existing-v1","B8/jaxctl",None,{}),
            ("reference_type","reference_value"),
        )
    with pytest.raises(GovernanceContractError):
        binding(**common,designated_source_identities=())
    b=binding(**common,designated_source_identities=("B8/jaxctl",))
    r=registry(RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.B9_DESIGNATED_CURRENT_SOURCE,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,None,{}),("reference_type","reference_value")))
    evidence=mod._b9_evidence_from_server(ResolutionResult(ResolutionState.RESOLVED_CURRENT,"B8/jaxctl",NOW.timestamp(),{"value":"x"}),scope(),"b9:single")
    assert r.resolve(b.predicate,{"reference_type":"health","reference_value":"hall"},scope(),validation_time=NOW,b9_evidence=evidence).status is ResolutionStatus.RESOLVED
    assert r.resolve(b.predicate,{"reference_type":"health","reference_value":"hall"},scope(),validation_time=NOW,b9_evidence=evidence,server_input=server_input((("B8/jaxctl",obs()),))).status is ResolutionStatus.SOURCE_MISMATCH

def test_f2ba03_receipt_is_authenticated_and_replay_checked():
    b=binding();r=registry();rec=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=inp(b))
    assert r.verify_receipt(rec,scope(),validation_time=NOW)
    forged=object.__new__(GovernedResolutionReceipt)
    for n in rec.__annotations__:object.__setattr__(forged,n,getattr(rec,n))
    object.__setattr__(forged,"result_digest","sha256:forged")
    assert not r.verify_receipt(forged,scope(),validation_time=NOW)
    for n,v in (("environment","staging"),("tenant_id","other"),("project_id","other"),("subject_id","other"),("audience","other"),("request_id","other"),("trace_id","other")):
        assert not r.verify_receipt(rec,scope(**{n:v}),validation_time=NOW)
    assert not r.verify_receipt(rec,scope(),validation_time=NOW+timedelta(seconds=61))
    assert not registry(key=b"another-test-secret-material-longer-than-32-bytes").verify_receipt(rec,scope(),validation_time=NOW)
    assert not registry(entry(binding_version="b2")).verify_receipt(rec,scope(),validation_time=NOW)
    with pytest.raises(GovernanceContractError):rec.is_current_at(NOW)

def test_f2ba04_exact_multisource_conflict_contract():
    a,b_rule=rules();b=PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","catalog:a","authority:catalog","production",a,b_rule,60,ConflictPolicy.ALL_SOURCES_AGREE,"adapter:capability","1",None,"b1",designated_source_identities=("catalog:a","catalog:b"))
    e=RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE,"adapter:capability","1","catalog:a",None,{}),("name","mode"));r=registry(e)
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("catalog:a",obs()),))).status is ResolutionStatus.UNAVAILABLE
    agree=server_input((("catalog:a",obs(result={"v":1})),("catalog:b",obs(result={"v":1}))))
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=agree).status is ResolutionStatus.RESOLVED
    conflict=server_input((("catalog:a",obs(result={"v":1})),("catalog:b",obs(result={"v":2}))))
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=conflict).status is ResolutionStatus.CONFLICT
    with pytest.raises(GovernanceContractError):PredicateAuthorityBinding("x","v","a","o","production",a,b_rule,1,ConflictPolicy.ALL_SOURCES_AGREE,"i","v",None,"b")

def test_source_set_rejects_duplicates_after_f2a_nfc_normalization():
    a,c=rules()
    with pytest.raises(GovernanceContractError,match="duplicate canonical"):
        PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","source:\u00e9","owner","production",a,c,60,ConflictPolicy.ALL_SOURCES_AGREE,"adapter:capability","1",None,"b1",designated_source_identities=("source:\u00e9","source:e\u0301"))
    with pytest.raises(GovernanceContractError,match="duplicate canonical"):
        PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","source:a","owner","production",a,c,60,ConflictPolicy.ALL_SOURCES_AGREE,"adapter:capability","1",None,"b1",designated_source_identities=("source:a","source:a"))

def test_source_set_is_canonical_for_runtime_matching_and_snapshot_identity():
    a,c=rules()
    left="source:\u00e9";right="source:b"
    b=PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1",left,"owner","production",a,c,60,ConflictPolicy.ALL_SOURCES_AGREE,"adapter:capability","1",None,"b1",designated_source_identities=(right,left))
    equivalent=PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","source:e\u0301","owner","production",a,c,60,ConflictPolicy.ALL_SOURCES_AGREE,"adapter:capability","1",None,"b1",designated_source_identities=("source:e\u0301",right))
    assert b.designated_source_identities==(right,left)
    assert b.digest==equivalent.digest
    r=registry(RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE,"adapter:capability","1",left,None,{}),("name","mode")))
    # An NFD spelling is one canonical participant, not a second source.
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("source:e\u0301",obs(result={"v":1})),))).status is ResolutionStatus.UNAVAILABLE
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("source:e\u0301",obs(result={"v":1})),(right,obs(result={"v":1}))))).status is ResolutionStatus.RESOLVED
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input(((left,obs(result={"v":1})),(right,obs(result={"v":2}))))).status is ResolutionStatus.CONFLICT
    with pytest.raises(GovernanceContractError,match="duplicate"):
        server_input(((left,obs()),("source:e\u0301",obs())))

def test_single_source_uses_canonical_source_identity_without_regression():
    a,c=rules();b=PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","source:\u00e9","owner","production",a,c,60,ConflictPolicy.SINGLE_SOURCE_REQUIRED,"adapter:capability","1",None,"b1",designated_source_identities=("source:e\u0301",))
    r=registry(RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE,"adapter:capability","1","source:\u00e9",None,{}),("name","mode")))
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("source:e\u0301",obs()),))).status is ResolutionStatus.RESOLVED

def test_no_callable_accreditation_or_observation_self_attestation():
    with pytest.raises(GovernanceContractError):TrustedAdapterRegistration(lambda:None,"x","1","source")
    with pytest.raises(TypeError):ResolutionObservation(ResolutionStatus.RESOLVED,NOW,"x",{},actual_source_identity="source")

def test_scope_and_disabled_predicates_fail_closed():
    b=binding();r=registry()
    for n,v in (("environment","staging"),("tenant_id","b"),("project_id","b"),("subject_id","b"),("actor_id","b"),("audience","b")):
        assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(**{n:v}),validation_time=NOW,server_input=inp(b)).status in {ResolutionStatus.WRONG_SCOPE,ResolutionStatus.WRONG_ENVIRONMENT}
    for p in ("ENGINE_STATUS","FACET_EXISTS","CONFIG_VALUE","AUDIT_EVENT_EXISTS","JOB_STATUS","MEMORY_ENTRY_EXISTS"):
        assert r.resolve(p,{},scope(),validation_time=NOW).status is ResolutionStatus.UNSUPPORTED

def test_import_has_no_default_registry_or_runtime_side_effects():assert not hasattr(mod,"DEFAULT_REGISTRY")

def test_registry_snapshot_is_deterministic_for_same_entries():assert registry().snapshot_digest==registry().snapshot_digest
def test_material_source_change_requires_new_snapshot():assert registry().snapshot_digest!=registry(entry(designated_source_identity="catalog:other",designated_source_identities=("catalog:other",))).snapshot_digest
def test_all_sources_agree_unavailable_member_is_noncurrent():
    a,c=rules();b=PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","a","owner","production",a,c,60,ConflictPolicy.ALL_SOURCES_AGREE,"i","1",None,"b",designated_source_identities=("a","b"));r=registry(RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE,"i","1","a",None,{}),("name","mode")))
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("a",obs(result={"x":1})),("b",obs(ResolutionStatus.UNAVAILABLE,result={}))))).status is ResolutionStatus.UNAVAILABLE
def test_all_sources_agree_stale_member_is_noncurrent_and_expiry_is_earliest():
    a,c=rules();b=PredicateAuthorityBinding("CAPABILITY_AVAILABLE","v1","a","owner","production",a,c,60,ConflictPolicy.ALL_SOURCES_AGREE,"i","1",None,"b",designated_source_identities=("a","b"));r=registry(RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE,"i","1","a",None,{}),("name","mode")))
    stale=obs(at=NOW-timedelta(seconds=61),result={"x":1})
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("a",obs(result={"x":1})),("b",stale)))).status is ResolutionStatus.STALE
    early=obs(result={"x":1});late=ResolutionObservation(ResolutionStatus.RESOLVED,NOW,"b",{"x":1},NOW+timedelta(seconds=10))
    rec=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("a",early),("b",late))))
    assert rec.status is ResolutionStatus.RESOLVED and rec.not_after<=NOW+timedelta(seconds=10)
def test_unexpected_source_is_rejected():
    b=binding();assert registry().resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input((("other",obs()),))).status is ResolutionStatus.SOURCE_MISMATCH
def test_receipt_tag_tampering_is_rejected():
    b=binding();r=registry();x=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=inp(b));f=object.__new__(GovernedResolutionReceipt)
    for n in x.__annotations__:object.__setattr__(f,n,getattr(x,n))
    object.__setattr__(f,"authentication_tag","0"*64);assert not r.verify_receipt(f,scope(),validation_time=NOW)
def test_receipt_domain_and_key_id_are_authenticated():
    auth=ReceiptAuthenticator.for_testing(KEY);body={"x":"y"}
    assert auth.sign(body)==hmac.new(KEY,mod._RECEIPT_DOMAIN+json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode(),hashlib.sha256).hexdigest()
    b=binding();r=registry();rec=r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=inp(b));f=object.__new__(GovernedResolutionReceipt)
    for n in rec.__annotations__:object.__setattr__(f,n,getattr(rec,n))
    object.__setattr__(f,"issuer_key_id","unknown-key")
    assert not r.verify_receipt(f,scope(),validation_time=NOW)
def test_receipt_rejects_config_and_resolver_change():
    b=binding();x=registry().resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=inp(b))
    assert not registry(entry(source_configuration_digest="sha256:new")).verify_receipt(x,scope(),validation_time=NOW)
    assert not registry(entry(resolver_version="2")).verify_receipt(x,scope(),validation_time=NOW)
def test_receipt_constructor_and_registry_constructor_are_not_public():
    with pytest.raises(GovernanceContractError):GovernedResolutionReceipt()
    with pytest.raises(GovernanceContractError):ResolverRegistry()
    with pytest.raises(GovernanceContractError):ServerAdapterInput(())
def test_single_source_requires_the_exact_named_source():
    b=binding();r=registry();assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=server_input(((b.designated_source_identity,obs()),("extra",obs())))).status is ResolutionStatus.SOURCE_MISMATCH
def test_preferred_fallback_is_disabled_until_a_dispatcher_exists():
    b=binding(conflict_policy=ConflictPolicy.PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK,designated_source_identities=("catalog:capabilities","catalog:fallback"));r=registry(RegistryEntry(b,TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE,b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,b.source_configuration_digest,{}),("name","mode")))
    assert r.resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=inp(b)).status is ResolutionStatus.UNAVAILABLE
def test_adapter_configuration_is_frozen_and_canonical():
    a=TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE,"i","1","s",None,{"nested":{"x":1}})
    with pytest.raises(TypeError):a.configuration["nested"]["x"]=2
def test_status_table_exposes_only_sealed_entries():
    assert registry().status_table()[0]["predicate"]=="CAPABILITY_AVAILABLE"
def test_observed_at_future_is_not_current():
    b=binding();assert registry().resolve("CAPABILITY_AVAILABLE",ARGS,scope(),validation_time=NOW,server_input=inp(b,(b.designated_source_identity,obs(at=NOW+timedelta(seconds=1))))).status is ResolutionStatus.STALE
