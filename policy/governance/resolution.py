"""F2-B accredited, side-effect-free resolution contracts.

There is deliberately no default registry or production key here.  The host
composition root must supply its sealed registry and server-owned key.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib, hmac, json, secrets, math
from types import MappingProxyType
from typing import Any, Mapping
from .response import ExistenceState, GovernanceContractError, ReferenceRef, ReferenceType, ResponseScope, TemporalClass, _freeze, _plain, _text
from jax.memory.b9 import ResolutionResult as B9ResolutionResult, ResolutionState as B9ResolutionState

_RECEIPT_DOMAIN = b"AXIOMA:F2B:RESOLUTION_RECEIPT:v1\x00"

class ResolutionStatus(str,Enum):
    RESOLVED="RESOLVED"; UNAVAILABLE="UNAVAILABLE"; STALE="STALE"; CONFLICT="CONFLICT"; WRONG_SCOPE="WRONG_SCOPE"; WRONG_ENVIRONMENT="WRONG_ENVIRONMENT"; SOURCE_MISMATCH="SOURCE_MISMATCH"; UNSUPPORTED="UNSUPPORTED"; UNACCREDITED="UNACCREDITED"; CONFIGURATION_MISMATCH="CONFIGURATION_MISMATCH"; VERSION_MISMATCH="VERSION_MISMATCH"; AUTHENTICATION_FAILED="AUTHENTICATION_FAILED"
class ReferenceValidationStatus(str,Enum):
    VALID="VALID"; DANGLING="DANGLING"; TOMBSTONED="TOMBSTONED"; WRONG_TYPE="WRONG_TYPE"; WRONG_SCOPE="WRONG_SCOPE"; WRONG_REVISION="WRONG_REVISION"; HISTORICAL="HISTORICAL"; ACCESS_DENIED="ACCESS_DENIED"
class ConflictPolicy(str,Enum):
    SINGLE_SOURCE_REQUIRED="SINGLE_SOURCE_REQUIRED"; ALL_SOURCES_AGREE="ALL_SOURCES_AGREE"; PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK="PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK"
class AdapterKind(str,Enum):
    CAPABILITY_AVAILABLE="CAPABILITY_AVAILABLE"; FILE_EXISTS="FILE_EXISTS"; B9_DESIGNATED_CURRENT_SOURCE="B9_DESIGNATED_CURRENT_SOURCE"
def _digest(v:Any)->str:return "sha256:"+hashlib.sha256(json.dumps(_plain(_freeze(v)),sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def _time(v:datetime,n:str)->datetime:
    if not isinstance(v,datetime) or v.tzinfo is None:raise GovernanceContractError(f"{n} must be timezone-aware datetime")
    return v.astimezone(timezone.utc)
def _enum(v:Any,t:type[Enum],n:str):
    if not isinstance(v,t):raise GovernanceContractError(f"{n} must be {t.__name__}")

@dataclass(frozen=True)
class ScopeRule:
    environment:str; tenant_id:str|None; project_id:str|None; subject_id:str|None; actor_id:str|None; audience:str|None; component_id:str|None
    def __post_init__(self):
        object.__setattr__(self,"environment",_text(self.environment,"environment"))
        for n in ("tenant_id","project_id","subject_id","actor_id","audience","component_id"):
            if getattr(self,n) is not None:object.__setattr__(self,n,_text(getattr(self,n),n))
    def matches(self,s:ResponseScope):
        if s.environment!=self.environment:return ResolutionStatus.WRONG_ENVIRONMENT
        return next((ResolutionStatus.WRONG_SCOPE for n in ("tenant_id","project_id","subject_id","actor_id","audience","component_id") if getattr(s,n)!=getattr(self,n)),None)
    def projection(self):return {n:getattr(self,n) for n in ("environment","tenant_id","project_id","subject_id","actor_id","audience","component_id")}

@dataclass(frozen=True)
class PredicateAuthorityBinding:
    predicate:str; predicate_version:str; designated_source_identity:str; source_owner_authority_ref:str; environment:str; tenant_project_scope_rule:ScopeRule; subject_audience_scope_rule:ScopeRule; freshness_sla_seconds:int; conflict_policy:ConflictPolicy; resolver_implementation_identity:str; resolver_version:str; source_configuration_digest:str|None; binding_version:str; enabled:bool=True; designated_source_identities:tuple[str,...]|None=None
    def __post_init__(self):
        for n in ("predicate","predicate_version","designated_source_identity","source_owner_authority_ref","environment","resolver_implementation_identity","resolver_version","binding_version"):object.__setattr__(self,n,_text(getattr(self,n),n))
        _enum(self.conflict_policy,ConflictPolicy,"conflict_policy")
        if not isinstance(self.tenant_project_scope_rule,ScopeRule) or not isinstance(self.subject_audience_scope_rule,ScopeRule):raise GovernanceContractError("binding scope rules must be ScopeRule")
        if self.environment!=self.tenant_project_scope_rule.environment or self.environment!=self.subject_audience_scope_rule.environment:raise GovernanceContractError("binding environment must match scope rules")
        if not isinstance(self.freshness_sla_seconds,int) or self.freshness_sla_seconds<=0:raise GovernanceContractError("freshness_sla_seconds must be positive int")
        if self.source_configuration_digest is not None:object.__setattr__(self,"source_configuration_digest",_text(self.source_configuration_digest,"source_configuration_digest"))
        # Omission means the legacy single designated source.  An explicit
        # empty tuple is an invalid accredited source set, never a silent
        # fallback to the primary identity.
        sources=(self.designated_source_identity,) if self.designated_source_identities is None else self.designated_source_identities
        if not isinstance(sources,tuple) or not sources or len(set(sources))!=len(sources):raise GovernanceContractError("designated source set must be nonempty unique tuple")
        sources=tuple(_text(x,"designated_source_identity") for x in sources)
        if self.designated_source_identity not in sources:raise GovernanceContractError("primary source absent from source set")
        if self.conflict_policy is ConflictPolicy.ALL_SOURCES_AGREE and len(sources)<2:raise GovernanceContractError("ALL_SOURCES_AGREE requires at least two sources")
        if self.conflict_policy is ConflictPolicy.PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK and len(sources)<2:raise GovernanceContractError("fallback policy requires named fallback")
        object.__setattr__(self,"designated_source_identities",sources)
    def projection(self):return {"predicate":self.predicate,"predicate_version":self.predicate_version,"designated_source_identity":self.designated_source_identity,"designated_source_identities":list(self.designated_source_identities),"source_owner_authority_ref":self.source_owner_authority_ref,"environment":self.environment,"tenant_project_scope_rule":self.tenant_project_scope_rule.projection(),"subject_audience_scope_rule":self.subject_audience_scope_rule.projection(),"freshness_sla_seconds":self.freshness_sla_seconds,"conflict_policy":self.conflict_policy.value,"resolver_implementation_identity":self.resolver_implementation_identity,"resolver_version":self.resolver_version,"source_configuration_digest":self.source_configuration_digest,"binding_version":self.binding_version,"enabled":self.enabled}
    @property
    def digest(self):return _digest(self.projection())

@dataclass(frozen=True)
class ResolutionObservation:
    """Untrusted observed data; it cannot self-attest source/adapter identity."""
    status:ResolutionStatus; observed_at:datetime; provenance_ref:str; result:Mapping[str,Any]=field(default_factory=dict); upstream_not_after:datetime|None=None
    def __post_init__(self):
        _enum(self.status,ResolutionStatus,"resolution status");object.__setattr__(self,"observed_at",_time(self.observed_at,"observed_at"));object.__setattr__(self,"provenance_ref",_text(self.provenance_ref,"provenance_ref"))
        if self.upstream_not_after is not None:
            object.__setattr__(self,"upstream_not_after",_time(self.upstream_not_after,"upstream_not_after"))
            if self.upstream_not_after<self.observed_at:raise GovernanceContractError("upstream_not_after precedes observed_at")
        if not isinstance(self.result,Mapping):raise GovernanceContractError("result must be mapping")
        object.__setattr__(self,"result",_freeze(self.result,"result"))

@dataclass(frozen=True)
class TrustedAdapterRegistration:
    """Immutable server-owned specification. It never accepts a callable."""
    adapter_kind:AdapterKind; resolver_id:str; resolver_version:str; source_identity:str; source_configuration_digest:str|None=None; configuration:Mapping[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        _enum(self.adapter_kind,AdapterKind,"adapter_kind")
        for n in ("resolver_id","resolver_version","source_identity"):object.__setattr__(self,n,_text(getattr(self,n),n))
        if self.source_configuration_digest is not None:object.__setattr__(self,"source_configuration_digest",_text(self.source_configuration_digest,"source_configuration_digest"))
        if not isinstance(self.configuration,Mapping):raise GovernanceContractError("adapter configuration must be mapping")
        object.__setattr__(self,"configuration",_freeze(self.configuration,"adapter configuration"))
    def projection(self):return {"adapter_kind":self.adapter_kind.value,"resolver_id":self.resolver_id,"resolver_version":self.resolver_version,"source_identity":self.source_identity,"source_configuration_digest":self.source_configuration_digest,"configuration":_plain(self.configuration)}

@dataclass(frozen=True)
class RegistryEntry:
    binding:PredicateAuthorityBinding; adapter:TrustedAdapterRegistration; argument_keys:tuple[str,...]; template_contract_ref:str|None=None
    def __post_init__(self):
        if not isinstance(self.binding,PredicateAuthorityBinding) or not isinstance(self.adapter,TrustedAdapterRegistration):raise GovernanceContractError("registry entry requires typed binding and adapter")
        if not isinstance(self.argument_keys,tuple) or not all(isinstance(x,str) and x for x in self.argument_keys) or len(set(self.argument_keys))!=len(self.argument_keys):raise GovernanceContractError("argument_keys must be unique strings tuple")
        if self.template_contract_ref is not None:object.__setattr__(self,"template_contract_ref",_text(self.template_contract_ref,"template_contract_ref"))
        b,a=self.binding,self.adapter
        if (a.resolver_id,a.resolver_version,a.source_identity,a.source_configuration_digest)!=(b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,b.source_configuration_digest):raise GovernanceContractError("adapter must exactly match approved binding")
        expected={AdapterKind.CAPABILITY_AVAILABLE:"CAPABILITY_AVAILABLE",AdapterKind.FILE_EXISTS:"FILE_EXISTS",AdapterKind.B9_DESIGNATED_CURRENT_SOURCE:"B9_DESIGNATED_CURRENT_SOURCE"}[a.adapter_kind]
        if b.predicate!=expected:raise GovernanceContractError("adapter kind/predicate mismatch")
        # B9's existing designated-current-source resolver has exactly one
        # upstream source.  It is not a multi-source reconciliation adapter;
        # accepting an agreement policy here would make its one-result
        # dispatch path bypass the accredited source-set contract.
        if a.adapter_kind is AdapterKind.B9_DESIGNATED_CURRENT_SOURCE and (
            b.conflict_policy is not ConflictPolicy.SINGLE_SOURCE_REQUIRED
            or b.designated_source_identities != (b.designated_source_identity,)
        ):
            raise GovernanceContractError(
                "B9 designated current source requires exactly one SINGLE_SOURCE_REQUIRED source"
            )
    def projection(self):return {"binding":self.binding.projection(),"adapter":self.adapter.projection(),"argument_keys":list(self.argument_keys),"template_contract_ref":self.template_contract_ref}

class ReceiptAuthenticator:
    __slots__=("_key","key_id","_sealed")
    def __init__(self,key:bytes,*,key_id:str="test-ephemeral"):
        if not isinstance(key,(bytes,bytearray)) or len(key)<32:raise GovernanceContractError("receipt authentication key must contain at least 256 bits")
        object.__setattr__(self,"_key",bytes(key));object.__setattr__(self,"key_id",_text(key_id,"key_id"));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):
        if getattr(self,"_sealed",False):raise GovernanceContractError("receipt authenticator is immutable")
        object.__setattr__(self,n,v)
    @classmethod
    def for_testing(cls,key:bytes|None=None):return cls(key or secrets.token_bytes(32))
    def _authenticated_bytes(self,b:Mapping[str,Any])->bytes:
        """Canonical, domain-separated bytes; this format is receipt-only."""
        return _RECEIPT_DOMAIN+json.dumps(_plain(_freeze(b,"receipt body")),sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")
    def sign(self,b:Mapping[str,Any]):return hmac.new(self._key,self._authenticated_bytes(b),hashlib.sha256).hexdigest()
    def verify(self,b:Mapping[str,Any],tag:str,key_id:str):return key_id==self.key_id and isinstance(tag,str) and hmac.compare_digest(self.sign(b),tag)

_RECEIPT_TOKEN=object()
@dataclass(frozen=True,init=False)
class GovernedResolutionReceipt:
    receipt_id:str; predicate:str; arguments_digest:str; binding_version:str; registry_snapshot_digest:str; actual_source_identity:str; source_configuration_digest:str|None; observation_scope_digest:str; observed_at:datetime; not_after:datetime; status:ResolutionStatus; provenance_ref:str; resolver_id:str; resolver_version:str; result_digest:str; issuer_key_id:str; authentication_tag:str
    def __init__(self,*a,**kw):raise GovernanceContractError("receipts are server-minted")
    @classmethod
    def _mint(cls,t,auth:ReceiptAuthenticator,**v):
        if t is not _RECEIPT_TOKEN or not isinstance(auth,ReceiptAuthenticator):raise GovernanceContractError("receipt minting requires server authenticator")
        _enum(v["status"],ResolutionStatus,"receipt status");v["observed_at"]=_time(v["observed_at"],"observed_at");v["not_after"]=_time(v["not_after"],"not_after")
        if v["not_after"]<v["observed_at"]:raise GovernanceContractError("not_after precedes observed_at")
        for n in ("predicate","arguments_digest","binding_version","registry_snapshot_digest","actual_source_identity","observation_scope_digest","provenance_ref","resolver_id","resolver_version","result_digest"):v[n]=_text(v[n],n)
        if v.get("source_configuration_digest") is not None:v["source_configuration_digest"]=_text(v["source_configuration_digest"],"source_configuration_digest")
        v["issuer_key_id"]=auth.key_id
        body={k:(v[k].isoformat() if k in {"observed_at","not_after"} else _plain(v[k])) for k in ("predicate","arguments_digest","binding_version","registry_snapshot_digest","actual_source_identity","source_configuration_digest","observation_scope_digest","observed_at","not_after","status","provenance_ref","resolver_id","resolver_version","result_digest","issuer_key_id")}
        v["authentication_tag"]=auth.sign(body);v["receipt_id"]="receipt:"+_digest({**body,"authentication_tag":v["authentication_tag"]})[7:]
        x=object.__new__(cls)
        for n,value in v.items():object.__setattr__(x,n,value)
        return x
    def authenticated_body(self):return {k:(getattr(self,k).isoformat() if k in {"observed_at","not_after"} else _plain(getattr(self,k))) for k in ("predicate","arguments_digest","binding_version","registry_snapshot_digest","actual_source_identity","source_configuration_digest","observation_scope_digest","observed_at","not_after","status","provenance_ref","resolver_id","resolver_version","result_digest","issuer_key_id")}
    def is_temporally_fresh_at(self,n):n=_time(n,"validation_time");return self.status is ResolutionStatus.RESOLVED and self.observed_at<=n<=self.not_after
    def is_current_at(self,n):raise GovernanceContractError("use ResolverRegistry.verify_receipt")

_ADAPTER_INPUT_TOKEN=object()
@dataclass(frozen=True,init=False)
class ServerAdapterInput:
    """Internal execution input minted by the server-owned adapter boundary.

    It deliberately has no public constructor: a client/model cannot convert a
    convenient `RESOLVED` dictionary into a trusted adapter invocation.
    """
    observations:tuple[tuple[str,ResolutionObservation],...]
    def __init__(self,*a,**kw):raise GovernanceContractError("adapter input is server-minted")
    @classmethod
    def _mint(cls,t,observations):
        if t is not _ADAPTER_INPUT_TOKEN:raise GovernanceContractError("adapter input requires server pathway")
        if not isinstance(observations,tuple) or not observations:raise GovernanceContractError("observations must be nonempty tuple")
        seen=set()
        for source,o in observations:
            _text(source,"observation source")
            if source in seen or not isinstance(o,ResolutionObservation):raise GovernanceContractError("duplicate or invalid observation")
            seen.add(source)
        x=object.__new__(cls);object.__setattr__(x,"observations",tuple(observations));return x


@dataclass(frozen=True, init=False)
class B9ResolutionEvidence:
    """Narrow server boundary over an actual B9 ``ResolutionResult``.

    It intentionally cannot be built from a dict, a MemoryEnvelope, or a
    lookalike object. Scope/provenance are supplied by the trusted B9 adapter
    composition, while source/time/value remain those returned by B9.
    """
    result: B9ResolutionResult
    observation_scope: ResponseScope
    provenance_ref: str
    def __init__(self,*a,**kw):raise GovernanceContractError("B9 evidence requires server B9 pathway")
    @classmethod
    def _from_b9_result(cls,t,result,observation_scope,provenance_ref):
        if t is not _B9_EVIDENCE_TOKEN or not isinstance(result,B9ResolutionResult) or not isinstance(observation_scope,ResponseScope):
            raise GovernanceContractError("typed B9 ResolutionResult and scope required")
        if not isinstance(result.state,B9ResolutionState):
            raise GovernanceContractError("B9 resolution state must be typed")
        source=_text(result.source,"B9 source")
        if not isinstance(result.observed_at,(int,float)) or isinstance(result.observed_at,bool) or not math.isfinite(result.observed_at):
            raise GovernanceContractError("B9 observed_at must be numeric")
        # ``ResolutionResult`` is frozen but its optional Mapping value is
        # caller-owned.  Snapshot its exact supported contract here so a
        # later caller mutation cannot change an authenticated receipt.
        # Current B9 results are mapping-valued by the B9 contract; never
        # collapse an unsupported value into an empty mapping.
        if result.value is not None and not isinstance(result.value,Mapping):
            raise GovernanceContractError("B9 resolution value must be a mapping or None")
        if result.state is B9ResolutionState.RESOLVED_CURRENT and not isinstance(result.value,Mapping):
            raise GovernanceContractError("current B9 resolution requires mapping value")
        value=None if result.value is None else _freeze(result.value,"B9 resolution value")
        snapshot=B9ResolutionResult(result.state,source,result.observed_at,value)
        x=object.__new__(cls);object.__setattr__(x,"result",snapshot);object.__setattr__(x,"observation_scope",observation_scope);object.__setattr__(x,"provenance_ref",_text(provenance_ref,"b9 provenance_ref"));return x


_B9_EVIDENCE_TOKEN=object()
def _b9_evidence_from_server(result, observation_scope, provenance_ref):
    """Internal composition-only conversion; external serialized data is rejected."""
    return B9ResolutionEvidence._from_b9_result(_B9_EVIDENCE_TOKEN,result,observation_scope,provenance_ref)

class ResolverRegistry:
    __slots__=("_entries","_snapshot_projection","_snapshot_digest","_authenticator","_sealed")
    def __init__(self,*a,**kw):raise GovernanceContractError("registries require server-approved sealing")
    def __setattr__(self,n,v):
        if getattr(self,"_sealed",False):raise GovernanceContractError("sealed registry is immutable")
        object.__setattr__(self,n,v)
    def __delattr__(self,n):raise GovernanceContractError("sealed registry is immutable")
    @classmethod
    def _from_approved_entries(cls,t,entries,auth):
        if t is not _REGISTRY_TOKEN or not isinstance(auth,ReceiptAuthenticator):raise GovernanceContractError("registry construction requires server authenticator")
        if not isinstance(entries,tuple) or not all(isinstance(e,RegistryEntry) for e in entries):raise GovernanceContractError("entries must be tuple RegistryEntry")
        if len({e.binding.predicate for e in entries})!=len(entries):raise GovernanceContractError("duplicate registry predicate")
        x=object.__new__(cls);graph=MappingProxyType({e.binding.predicate:e for e in sorted(entries,key=lambda e:e.binding.predicate)})
        object.__setattr__(x,"_entries",graph);object.__setattr__(x,"_snapshot_projection",_freeze({p:e.projection() for p,e in graph.items()},"approved_registry"));object.__setattr__(x,"_snapshot_digest",_digest(x._snapshot_projection));object.__setattr__(x,"_authenticator",auth);object.__setattr__(x,"_sealed",True);return x
    @property
    def snapshot_digest(self):return self._snapshot_digest
    def _entry(self,p):
        if _digest(self._snapshot_projection)!=self._snapshot_digest:raise GovernanceContractError("registry snapshot integrity failure")
        return self._entries.get(p)
    def status_table(self):return tuple(MappingProxyType({"predicate":p,"enabled":e.binding.enabled,"source_owner":e.binding.source_owner_authority_ref,"freshness_sla_seconds":e.binding.freshness_sla_seconds,"human_decision_required":not e.binding.enabled}) for p,e in self._entries.items())
    def resolve(self,predicate,args,scope,*,validation_time,server_input:ServerAdapterInput|None=None,b9_evidence:B9ResolutionEvidence|None=None):
        predicate=_text(predicate,"predicate");now=_time(validation_time,"validation_time")
        if not isinstance(args,Mapping) or not isinstance(scope,ResponseScope):raise GovernanceContractError("arguments and scope must be typed")
        e=self._entry(predicate)
        if e is None:return self._failure(predicate,args,scope,now,ResolutionStatus.UNSUPPORTED)
        b,a=e.binding,e.adapter
        if not b.enabled:return self._failure(predicate,args,scope,now,ResolutionStatus.UNACCREDITED,b,a)
        if set(args)!=set(e.argument_keys):return self._failure(predicate,args,scope,now,ResolutionStatus.UNAVAILABLE,b,a)
        mismatch=b.tenant_project_scope_rule.matches(scope) or b.subject_audience_scope_rule.matches(scope)
        if mismatch:return self._failure(predicate,args,scope,now,mismatch,b,a)
        status,o=self._dispatch(e,server_input,b9_evidence,scope,now)
        if status is ResolutionStatus.RESOLVED and (o.observed_at>now or now>o.observed_at+timedelta(seconds=b.freshness_sla_seconds) or(o.upstream_not_after and now>o.upstream_not_after)):status=ResolutionStatus.STALE
        not_after=min(o.upstream_not_after or o.observed_at+timedelta(seconds=b.freshness_sla_seconds),o.observed_at+timedelta(seconds=b.freshness_sla_seconds))
        return self._mint(predicate,args,scope,b,a,o,status,not_after)
    def _fresh_observation(self,o,b,now):
        if o.status is not ResolutionStatus.RESOLVED:return o.status
        if o.observed_at>now or now>o.observed_at+timedelta(seconds=b.freshness_sla_seconds) or (o.upstream_not_after and now>o.upstream_not_after):return ResolutionStatus.STALE
        return ResolutionStatus.RESOLVED
    def _dispatch(self,e,inp,b9_evidence,scope,now):
        b=e.binding
        if e.adapter.adapter_kind is AdapterKind.B9_DESIGNATED_CURRENT_SOURCE:
            # B9 is never adapted from generic observations.  Its evidence has
            # to be the real typed B9 result plus trusted scope provenance.
            if b9_evidence is None:
                return ResolutionStatus.UNAVAILABLE,ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,"server:b9-missing-evidence",{})
            # The typed evidence is one, and only one, upstream B9
            # observation.  Generic adapter observations are not an
            # additional source channel for B9.
            if inp is not None:
                return ResolutionStatus.SOURCE_MISMATCH,ResolutionObservation(ResolutionStatus.SOURCE_MISMATCH,now,"server:b9-unexpected-observations",{})
            if not isinstance(b9_evidence,B9ResolutionEvidence) or b9_evidence.observation_scope.scope_digest!=scope.scope_digest:
                return ResolutionStatus.WRONG_SCOPE,ResolutionObservation(ResolutionStatus.WRONG_SCOPE,now,"server:b9-scope",{})
            result=b9_evidence.result
            if result.state is not B9ResolutionState.RESOLVED_CURRENT:return ResolutionStatus.UNAVAILABLE,ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,b9_evidence.provenance_ref,{})
            if result.source!=b.designated_source_identity:return ResolutionStatus.SOURCE_MISMATCH,ResolutionObservation(ResolutionStatus.SOURCE_MISMATCH,now,b9_evidence.provenance_ref,{})
            observed=_time(datetime.fromtimestamp(result.observed_at,timezone.utc),"b9 observed_at")
            # B9ResolutionEvidence has already frozen and validated this
            # mapping.  Keep the exact value; a malformed value is never
            # normalized into a successful empty result.
            if not isinstance(result.value,Mapping):
                return ResolutionStatus.UNAVAILABLE,ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,b9_evidence.provenance_ref,{})
            value=result.value
            o=ResolutionObservation(ResolutionStatus.RESOLVED,observed,b9_evidence.provenance_ref,value)
            return self._fresh_observation(o,b,now),o
        if not isinstance(inp,ServerAdapterInput):return ResolutionStatus.UNAVAILABLE,ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,"server:missing-input",{})
        got=dict(inp.observations)
        if set(got)-set(b.designated_source_identities):return ResolutionStatus.SOURCE_MISMATCH,ResolutionObservation(ResolutionStatus.SOURCE_MISMATCH,now,"server:unexpected-source",{})
        if b.conflict_policy is ConflictPolicy.SINGLE_SOURCE_REQUIRED:
            if set(got)!={b.designated_source_identity}:return ResolutionStatus.SOURCE_MISMATCH,ResolutionObservation(ResolutionStatus.SOURCE_MISMATCH,now,"server:source-set",{})
            o=got[b.designated_source_identity];return self._fresh_observation(o,b,now),o
        if b.conflict_policy is ConflictPolicy.ALL_SOURCES_AGREE:
            if set(got)!=set(b.designated_source_identities):return ResolutionStatus.UNAVAILABLE,ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,"server:incomplete-source-set",{})
            values=list(got.values())
            states=[self._fresh_observation(o,b,now) for o in values]
            if any(x is ResolutionStatus.STALE for x in states):return ResolutionStatus.STALE,ResolutionObservation(ResolutionStatus.STALE,now,"server:source-stale",{})
            if any(x is not ResolutionStatus.RESOLVED for x in states):return ResolutionStatus.UNAVAILABLE,ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,"server:source-unavailable",{})
            if len({_digest(o.result) for o in values})!=1:return ResolutionStatus.CONFLICT,ResolutionObservation(ResolutionStatus.CONFLICT,now,"server:source-conflict",{})
            # Receipt lifetime is bounded by every independently valid source.
            ends=[o.upstream_not_after or o.observed_at+timedelta(seconds=b.freshness_sla_seconds) for o in values]
            selected=min(zip(ends,values),key=lambda pair:pair[0])[1]
            return ResolutionStatus.RESOLVED,ResolutionObservation(ResolutionStatus.RESOLVED,selected.observed_at,selected.provenance_ref,selected.result,min(ends))
        return ResolutionStatus.UNAVAILABLE,ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,"server:fallback-not-implemented",{})
    def _failure(self,predicate,args,scope,now,status,b=None,a=None):return self._mint(predicate,args,scope,b,a,ResolutionObservation(status,now,"none",{}),status,now)
    def _mint(self,predicate,args,scope,b,a,o,status,not_after):
        v={"predicate":predicate,"arguments_digest":_digest(args),"binding_version":b.binding_version if b else "none","registry_snapshot_digest":self.snapshot_digest,"actual_source_identity":a.source_identity if a else "unaccredited","source_configuration_digest":a.source_configuration_digest if a else None,"observation_scope_digest":scope.scope_digest,"observed_at":o.observed_at,"not_after":not_after,"status":status,"provenance_ref":o.provenance_ref,"resolver_id":a.resolver_id if a else "none","resolver_version":a.resolver_version if a else "none","result_digest":_digest(o.result)}
        return GovernedResolutionReceipt._mint(_RECEIPT_TOKEN,self._authenticator,**v)
    def verify_receipt(self,r,expected_scope,*,validation_time):
        if not isinstance(r,GovernedResolutionReceipt) or not isinstance(expected_scope,ResponseScope):return False
        try:now=_time(validation_time,"validation_time");e=self._entry(r.predicate)
        except GovernanceContractError:return False
        if e is None or not e.binding.enabled or not self._authenticator.verify(r.authenticated_body(),r.authentication_tag,r.issuer_key_id):return False
        b,a=e.binding,e.adapter;expected_id="receipt:"+_digest({**r.authenticated_body(),"authentication_tag":r.authentication_tag})[7:]
        if r.receipt_id!=expected_id or r.status is not ResolutionStatus.RESOLVED or r.registry_snapshot_digest!=self.snapshot_digest:return False
        if (r.binding_version,r.resolver_id,r.resolver_version,r.actual_source_identity,r.source_configuration_digest)!=(b.binding_version,a.resolver_id,a.resolver_version,a.source_identity,a.source_configuration_digest):return False
        return r.observation_scope_digest==expected_scope.scope_digest and r.is_temporally_fresh_at(now)

_REGISTRY_TOKEN=object()
def _build_approved_registry_for_server(entries,*,authenticator):return ResolverRegistry._from_approved_entries(_REGISTRY_TOKEN,entries,authenticator)

@dataclass(frozen=True)
class ReferenceLookupRecord:
    ref_type:ReferenceType; immutable_identity:str; revision_or_digest:str; scope_digest:str; temporal_class:TemporalClass; existence_state:ExistenceState; accessible:bool
    def __post_init__(self):
        _enum(self.ref_type,ReferenceType,"ref_type");_enum(self.temporal_class,TemporalClass,"temporal_class");_enum(self.existence_state,ExistenceState,"existence_state")
        for n in ("immutable_identity","revision_or_digest","scope_digest"):object.__setattr__(self,n,_text(getattr(self,n),n))
def validate_reference(ref,scope,*,lookup,expected_type=None,expected_revision=None):
    if not isinstance(ref,ReferenceRef) or not isinstance(scope,ResponseScope):raise GovernanceContractError("typed reference and scope required")
    if lookup is None or lookup.existence_state is ExistenceState.UNKNOWN:return ReferenceValidationStatus.DANGLING
    if lookup.existence_state is ExistenceState.TOMBSTONED:return ReferenceValidationStatus.TOMBSTONED
    if not lookup.accessible:return ReferenceValidationStatus.ACCESS_DENIED
    if (expected_type is not None and ref.ref_type is not expected_type) or ref.ref_type is not lookup.ref_type:return ReferenceValidationStatus.WRONG_TYPE
    if (expected_revision is not None and ref.revision_or_digest!=expected_revision) or ref.immutable_identity!=lookup.immutable_identity or ref.revision_or_digest!=lookup.revision_or_digest:return ReferenceValidationStatus.WRONG_REVISION
    if ref.scope_digest!=scope.scope_digest or lookup.scope_digest!=scope.scope_digest:return ReferenceValidationStatus.WRONG_SCOPE
    return ReferenceValidationStatus.HISTORICAL if ref.temporal_class is not lookup.temporal_class or ref.temporal_class is TemporalClass.HISTORICAL else ReferenceValidationStatus.VALID
