"""F2-B accredited, side-effect-free current-source resolution contracts."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib, json
from types import MappingProxyType
from typing import Any, Callable, Mapping
from .response import ExistenceState, GovernanceContractError, ReferenceRef, ReferenceType, ResponseScope, TemporalClass, _freeze, _plain, _text

class ResolutionStatus(str, Enum):
    RESOLVED="RESOLVED"; UNAVAILABLE="UNAVAILABLE"; STALE="STALE"; CONFLICT="CONFLICT"; WRONG_SCOPE="WRONG_SCOPE"; WRONG_ENVIRONMENT="WRONG_ENVIRONMENT"; SOURCE_MISMATCH="SOURCE_MISMATCH"; UNSUPPORTED="UNSUPPORTED"; UNACCREDITED="UNACCREDITED"; CONFIGURATION_MISMATCH="CONFIGURATION_MISMATCH"; VERSION_MISMATCH="VERSION_MISMATCH"
class ReferenceValidationStatus(str, Enum):
    VALID="VALID"; DANGLING="DANGLING"; TOMBSTONED="TOMBSTONED"; WRONG_TYPE="WRONG_TYPE"; WRONG_SCOPE="WRONG_SCOPE"; WRONG_REVISION="WRONG_REVISION"; HISTORICAL="HISTORICAL"; ACCESS_DENIED="ACCESS_DENIED"
class ConflictPolicy(str, Enum):
    SINGLE_SOURCE_REQUIRED="SINGLE_SOURCE_REQUIRED"; ALL_SOURCES_AGREE="ALL_SOURCES_AGREE"; PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK="PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK"
def _digest(value: Any)->str:
    return "sha256:"+hashlib.sha256(json.dumps(_plain(_freeze(value)),sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def _time(value:datetime,name:str)->datetime:
    if not isinstance(value,datetime) or value.tzinfo is None: raise GovernanceContractError(f"{name} must be timezone-aware datetime")
    return value.astimezone(timezone.utc)
def _enum(value:Any,typ:type[Enum],name:str)->None:
    if not isinstance(value,typ): raise GovernanceContractError(f"{name} must be {typ.__name__}")

@dataclass(frozen=True)
class ScopeRule:
    environment:str; tenant_id:str|None; project_id:str|None; subject_id:str|None; actor_id:str|None; audience:str|None; component_id:str|None
    def __post_init__(self):
        object.__setattr__(self,"environment",_text(self.environment,"environment"))
        for n in ("tenant_id","project_id","subject_id","actor_id","audience","component_id"):
            if getattr(self,n) is not None: object.__setattr__(self,n,_text(getattr(self,n),n))
    def matches(self,scope:ResponseScope)->ResolutionStatus|None:
        if scope.environment!=self.environment:return ResolutionStatus.WRONG_ENVIRONMENT
        return next((ResolutionStatus.WRONG_SCOPE for n in ("tenant_id","project_id","subject_id","actor_id","audience","component_id") if getattr(scope,n)!=getattr(self,n)),None)
    def projection(self): return {n:getattr(self,n) for n in ("environment","tenant_id","project_id","subject_id","actor_id","audience","component_id")}

@dataclass(frozen=True)
class PredicateAuthorityBinding:
    predicate:str; predicate_version:str; designated_source_identity:str; source_owner_authority_ref:str; environment:str; tenant_project_scope_rule:ScopeRule; subject_audience_scope_rule:ScopeRule; freshness_sla_seconds:int; conflict_policy:ConflictPolicy; resolver_implementation_identity:str; resolver_version:str; source_configuration_digest:str|None; binding_version:str; enabled:bool=True
    def __post_init__(self):
        for n in ("predicate","predicate_version","designated_source_identity","source_owner_authority_ref","environment","resolver_implementation_identity","resolver_version","binding_version"):object.__setattr__(self,n,_text(getattr(self,n),n))
        _enum(self.conflict_policy,ConflictPolicy,"conflict_policy")
        if not isinstance(self.tenant_project_scope_rule,ScopeRule) or not isinstance(self.subject_audience_scope_rule,ScopeRule):raise GovernanceContractError("binding scope rules must be ScopeRule")
        if self.environment!=self.tenant_project_scope_rule.environment or self.environment!=self.subject_audience_scope_rule.environment:raise GovernanceContractError("binding environment must match all scope rules")
        if not isinstance(self.freshness_sla_seconds,int) or self.freshness_sla_seconds<=0:raise GovernanceContractError("freshness_sla_seconds must be positive int")
        if self.source_configuration_digest is not None:object.__setattr__(self,"source_configuration_digest",_text(self.source_configuration_digest,"source_configuration_digest"))
    def projection(self):return {"predicate":self.predicate,"predicate_version":self.predicate_version,"designated_source_identity":self.designated_source_identity,"source_owner_authority_ref":self.source_owner_authority_ref,"environment":self.environment,"tenant_project_scope_rule":self.tenant_project_scope_rule.projection(),"subject_audience_scope_rule":self.subject_audience_scope_rule.projection(),"freshness_sla_seconds":self.freshness_sla_seconds,"conflict_policy":self.conflict_policy.value,"resolver_implementation_identity":self.resolver_implementation_identity,"resolver_version":self.resolver_version,"source_configuration_digest":self.source_configuration_digest,"binding_version":self.binding_version,"enabled":self.enabled}
    @property
    def digest(self):return _digest(self.projection())

@dataclass(frozen=True)
class ResolutionObservation:
    """Untrusted observation data: it cannot self-attest source/adapter identity."""
    status:ResolutionStatus; observed_at:datetime; provenance_ref:str; result:Mapping[str,Any]=field(default_factory=dict); upstream_not_after:datetime|None=None
    def __post_init__(self):
        _enum(self.status,ResolutionStatus,"resolution status");object.__setattr__(self,"observed_at",_time(self.observed_at,"observed_at"));object.__setattr__(self,"provenance_ref",_text(self.provenance_ref,"provenance_ref"))
        if self.upstream_not_after is not None:
            object.__setattr__(self,"upstream_not_after",_time(self.upstream_not_after,"upstream_not_after"))
            if self.upstream_not_after<self.observed_at:raise GovernanceContractError("upstream_not_after precedes observed_at")
        if not isinstance(self.result,Mapping):raise GovernanceContractError("result must be mapping")
        object.__setattr__(self,"result",_freeze(self.result,"result"))
Resolver=Callable[[Mapping[str,Any],ResponseScope],ResolutionObservation]

@dataclass(frozen=True)
class TrustedAdapterRegistration:
    """Reviewed identity for a callable.  Observation payload never sets these fields."""
    resolver:Resolver|None; resolver_id:str; resolver_version:str; source_identity:str; source_configuration_digest:str|None=None
    def __post_init__(self):
        if self.resolver is not None and not callable(self.resolver):raise GovernanceContractError("resolver must be callable")
        for n in ("resolver_id","resolver_version","source_identity"):object.__setattr__(self,n,_text(getattr(self,n),n))
        if self.source_configuration_digest is not None:object.__setattr__(self,"source_configuration_digest",_text(self.source_configuration_digest,"source_configuration_digest"))
    def projection(self):return {"resolver_id":self.resolver_id,"resolver_version":self.resolver_version,"source_identity":self.source_identity,"source_configuration_digest":self.source_configuration_digest}

_RECEIPT_TOKEN=object()
@dataclass(frozen=True,init=False)
class GovernedResolutionReceipt:
    receipt_id:str; predicate:str; arguments_digest:str; binding_version:str; registry_snapshot_digest:str; actual_source_identity:str; source_configuration_digest:str|None; observation_scope_digest:str; observed_at:datetime; not_after:datetime; status:ResolutionStatus; provenance_ref:str; resolver_id:str; resolver_version:str; result_digest:str
    def __init__(self,*args,**kwargs):raise GovernanceContractError("receipts are server-minted by ResolverRegistry")
    @classmethod
    def _mint(cls,token,**v):
        if token is not _RECEIPT_TOKEN:raise GovernanceContractError("receipt requires server minting pathway")
        _enum(v["status"],ResolutionStatus,"receipt status");v["observed_at"]=_time(v["observed_at"],"observed_at");v["not_after"]=_time(v["not_after"],"not_after")
        if v["not_after"]<v["observed_at"]:raise GovernanceContractError("not_after precedes observed_at")
        for n in ("receipt_id","predicate","arguments_digest","binding_version","registry_snapshot_digest","actual_source_identity","observation_scope_digest","provenance_ref","resolver_id","resolver_version","result_digest"):v[n]=_text(v[n],n)
        if v["source_configuration_digest"] is not None:v["source_configuration_digest"]=_text(v["source_configuration_digest"],"source_configuration_digest")
        x=object.__new__(cls)
        for n,value in v.items():object.__setattr__(x,n,value)
        return x
    def projection(self,*,include_identity=True):return {n:(getattr(self,n).isoformat() if n in {"observed_at","not_after"} else _plain(getattr(self,n))) for n in self.__annotations__ if include_identity or n!="receipt_id"}
    def is_temporally_fresh_at(self,now):
        now=_time(now,"validation_time");return self.status is ResolutionStatus.RESOLVED and self.observed_at<=now<=self.not_after
    def is_current_at(self,now):raise GovernanceContractError("use ResolverRegistry.verify_receipt for trusted currentness")

@dataclass(frozen=True)
class RegistryEntry:
    binding:PredicateAuthorityBinding; adapter:TrustedAdapterRegistration; argument_keys:tuple[str,...]; template_contract_ref:str|None=None
    def __post_init__(self):
        if not isinstance(self.binding,PredicateAuthorityBinding) or not isinstance(self.adapter,TrustedAdapterRegistration):raise GovernanceContractError("registry entry requires typed binding and adapter")
        if not isinstance(self.argument_keys,tuple) or not all(isinstance(x,str) and x for x in self.argument_keys) or len(set(self.argument_keys))!=len(self.argument_keys):raise GovernanceContractError("argument_keys must be unique nonempty strings tuple")
        if self.template_contract_ref is not None:object.__setattr__(self,"template_contract_ref",_text(self.template_contract_ref,"template_contract_ref"))
        b,a=self.binding,self.adapter
        if (a.resolver_id,a.resolver_version,a.source_identity,a.source_configuration_digest)!=(b.resolver_implementation_identity,b.resolver_version,b.designated_source_identity,b.source_configuration_digest):raise GovernanceContractError("approved adapter must exactly match binding")
    def projection(self):return {"binding":self.binding.projection(),"adapter":self.adapter.projection(),"argument_keys":list(self.argument_keys),"template_contract_ref":self.template_contract_ref}

class ResolverRegistry:
    """Sealed immutable graph. Unknown predicates cannot be dynamically injected."""
    def __init__(self,*args,**kwargs):raise GovernanceContractError("registries require the server-owned approved construction pathway")
    @classmethod
    def _from_approved_entries(cls,token,entries):
        if token is not _REGISTRY_TOKEN:raise GovernanceContractError("registry entries require server approval")
        if not isinstance(entries,tuple) or not all(isinstance(x,RegistryEntry) for x in entries):raise GovernanceContractError("registry entries must be tuple[RegistryEntry,...]")
        if len({x.binding.predicate for x in entries})!=len(entries):raise GovernanceContractError("duplicate registry predicate")
        x=object.__new__(cls);graph=MappingProxyType({e.binding.predicate:e for e in sorted(entries,key=lambda i:i.binding.predicate)})
        object.__setattr__(x,"_entries",graph);object.__setattr__(x,"_snapshot_projection",_freeze({p:e.projection() for p,e in graph.items()},"approved_registry"));object.__setattr__(x,"_snapshot_digest",_digest(x._snapshot_projection));return x
    @property
    def snapshot_digest(self):return self._snapshot_digest
    def _entry(self,predicate):
        if _digest(self._snapshot_projection)!=self._snapshot_digest:raise GovernanceContractError("registry snapshot integrity failure")
        return self._entries.get(predicate)
    def status_table(self):return tuple(MappingProxyType({"predicate":p,"enabled":e.binding.enabled and e.adapter.resolver is not None,"source_owner":e.binding.source_owner_authority_ref,"freshness_sla_seconds":e.binding.freshness_sla_seconds,"human_decision_required":not(e.binding.enabled and e.adapter.resolver is not None)}) for p,e in self._entries.items())
    def resolve(self,predicate,arguments,scope,*,validation_time):
        predicate=_text(predicate,"predicate");now=_time(validation_time,"validation_time")
        if not isinstance(arguments,Mapping) or not isinstance(scope,ResponseScope):raise GovernanceContractError("arguments and scope must be typed")
        e=self._entry(predicate)
        if e is None:return self._failure(predicate,arguments,scope,now,ResolutionStatus.UNSUPPORTED)
        b,a=e.binding,e.adapter
        if not b.enabled or a.resolver is None:return self._failure(predicate,arguments,scope,now,ResolutionStatus.UNACCREDITED,b,a)
        if set(arguments)!=set(e.argument_keys):return self._failure(predicate,arguments,scope,now,ResolutionStatus.UNAVAILABLE,b,a)
        mismatch=b.tenant_project_scope_rule.matches(scope) or b.subject_audience_scope_rule.matches(scope)
        if mismatch:return self._failure(predicate,arguments,scope,now,mismatch,b,a)
        try:o=a.resolver(_freeze(arguments,"arguments"),scope)
        except Exception: # fail-soft: availability failure is UNAVAILABLE, never current truth.
            return self._failure(predicate,arguments,scope,now,ResolutionStatus.UNAVAILABLE,b,a)
        if not isinstance(o,ResolutionObservation):return self._failure(predicate,arguments,scope,now,ResolutionStatus.UNAVAILABLE,b,a)
        # No current adapter registers a fallback source.  Therefore a
        # preferred-source binding has an explicit, deterministic unavailable
        # outcome on disagreement rather than silently choosing a candidate.
        status=(ResolutionStatus.UNAVAILABLE if o.status is ResolutionStatus.CONFLICT and b.conflict_policy is ConflictPolicy.PREFERRED_SOURCE_WITH_EXPLICIT_FALLBACK else o.status)
        if status is ResolutionStatus.RESOLVED:
            status=ResolutionStatus.STALE if o.observed_at>now or now>o.observed_at+timedelta(seconds=b.freshness_sla_seconds) or(o.upstream_not_after is not None and now>o.upstream_not_after) else ResolutionStatus.RESOLVED
        upstream=o.upstream_not_after or o.observed_at+timedelta(seconds=b.freshness_sla_seconds);return self._mint(predicate,arguments,scope,b,a,o,status,min(upstream,o.observed_at+timedelta(seconds=b.freshness_sla_seconds)))
    def _failure(self,predicate,args,scope,now,status,b=None,a=None):return self._mint(predicate,args,scope,b,a,ResolutionObservation(status,now,"none",{}),status,now)
    def _mint(self,predicate,args,scope,b,a,o,status,not_after):
        v={"predicate":predicate,"arguments_digest":_digest(args),"binding_version":b.binding_version if b else "none","registry_snapshot_digest":self.snapshot_digest,"actual_source_identity":a.source_identity if a else "unaccredited","source_configuration_digest":a.source_configuration_digest if a else None,"observation_scope_digest":scope.scope_digest,"observed_at":o.observed_at,"not_after":not_after,"status":status,"provenance_ref":o.provenance_ref,"resolver_id":a.resolver_id if a else "none","resolver_version":a.resolver_version if a else "none","result_digest":_digest(o.result)}
        seed=dict(v);seed["observed_at"]=v["observed_at"].isoformat();seed["not_after"]=v["not_after"].isoformat();return GovernedResolutionReceipt._mint(_RECEIPT_TOKEN,receipt_id="receipt:"+_digest(seed)[7:],**v)
    def verify_receipt(self,receipt,expected_scope,*,validation_time):
        """Only this server-owned registry may decide a receipt remains current."""
        if not isinstance(receipt,GovernedResolutionReceipt) or not isinstance(expected_scope,ResponseScope):return False
        now=_time(validation_time,"validation_time");e=self._entry(receipt.predicate)
        if e is None or not e.binding.enabled or e.adapter.resolver is None:return False
        b,a=e.binding,e.adapter
        if receipt.status is not ResolutionStatus.RESOLVED or receipt.registry_snapshot_digest!=self.snapshot_digest:return False
        if (receipt.binding_version,receipt.resolver_id,receipt.resolver_version,receipt.actual_source_identity,receipt.source_configuration_digest)!=(b.binding_version,a.resolver_id,a.resolver_version,a.source_identity,a.source_configuration_digest):return False
        # Exact scope reuse is deliberately required: includes request/trace.
        if receipt.observation_scope_digest!=expected_scope.scope_digest or not receipt.is_temporally_fresh_at(now):return False
        return receipt.receipt_id=="receipt:"+_digest(receipt.projection(include_identity=False))[7:]

_REGISTRY_TOKEN=object()
def _build_approved_registry_for_server(entries):return ResolverRegistry._from_approved_entries(_REGISTRY_TOKEN,entries)

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

class CapabilityAvailableAdapter:
    resolver_id="policy.governance.validator:CAPABILITY_AVAILABLE";resolver_version="existing-v1"
    def __init__(self,validation_context,*,source_kind,verdict_resolver=None,clock=None,**_):
        if source_kind not in {"ops","catalog"}:raise GovernanceContractError("capability source_kind must be ops or catalog")
        self._context=validation_context;self._source_kind=source_kind;self._verdict=verdict_resolver;self._clock=clock or(lambda:datetime.now(timezone.utc))
    def __call__(self,args,scope):
        verdict=self._verdict(args,self._context) if self._verdict else "UNAVAILABLE";name=args.get("name");ops=name in self._context.ops;catalog=self._context.catalog.get_capability(name) is not None
        if ops and catalog:status=ResolutionStatus.CONFLICT
        elif self._source_kind=="ops":status=ResolutionStatus.RESOLVED if ops and verdict=="VALID" else ResolutionStatus.UNAVAILABLE
        else:status=ResolutionStatus.RESOLVED if catalog and verdict=="VALID" else ResolutionStatus.UNAVAILABLE
        return ResolutionObservation(status,self._clock(),"governance:capability-validator",dict(args))
@dataclass(frozen=True)
class _LegacyClaimInput:predicate:str;args:Mapping[str,Any]
def capability_adapter_from_legacy(context,legacy_resolver,**kwargs):
    if not callable(legacy_resolver):raise GovernanceContractError("legacy capability resolver must be callable")
    return CapabilityAvailableAdapter(context,verdict_resolver=lambda args,c:getattr(legacy_resolver(_LegacyClaimInput("CAPABILITY_AVAILABLE",dict(args)),c),"status","UNAVAILABLE"),**kwargs)
class FileExistsAdapter:
    resolver_id="policy.governance.validator:FILE_EXISTS";resolver_version="existing-v1"
    def __init__(self,context,*,verdict_resolver=None,clock=None,**_):self._context=context;self._verdict=verdict_resolver;self._clock=clock or(lambda:datetime.now(timezone.utc))
    def __call__(self,args,scope):return ResolutionObservation(ResolutionStatus.RESOLVED if self._verdict and self._verdict(args,self._context)=="VALID" else ResolutionStatus.UNAVAILABLE,self._clock(),"governance:file-validator",dict(args))
def file_exists_adapter_from_legacy(context,legacy_resolver,**kwargs):
    if not callable(legacy_resolver):raise GovernanceContractError("legacy file resolver must be callable")
    return FileExistsAdapter(context,verdict_resolver=lambda args,c:getattr(legacy_resolver(_LegacyClaimInput("FILE_EXISTS",dict(args)),c),"status","UNAVAILABLE"),**kwargs)
class B9DesignatedSourceAdapter:
    resolver_id="jax.memory.b9_resolvers:DesignatedSourceResolver";resolver_version="existing-v1"
    def __init__(self,resolver,*,designated_source_identity,scope_evidence=None):self._resolver=resolver;self._source=_text(designated_source_identity,"designated_source_identity");self._scope_evidence=scope_evidence
    def __call__(self,args,scope):
        from jax.memory.b9 import MemoryReference,ResolutionState
        now=datetime.now(timezone.utc);kind,value=args.get("reference_type"),args.get("reference_value")
        if not isinstance(kind,str) or not isinstance(value,str) or self._scope_evidence is None:return ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,"b9:none",{})
        result=self._resolver.resolve(MemoryReference(kind,value))
        try:observed=datetime.fromtimestamp(result.observed_at,tz=timezone.utc)
        except (TypeError,ValueError,OSError): # fail-soft: malformed upstream time is unavailable, never current truth.
            return ResolutionObservation(ResolutionStatus.UNAVAILABLE,now,"b9:none",{})
        if result.state is not ResolutionState.RESOLVED_CURRENT:return ResolutionObservation(ResolutionStatus.UNAVAILABLE,observed,"b9:"+str(result.source),{"state":result.state.value})
        if result.source!=self._source:return ResolutionObservation(ResolutionStatus.SOURCE_MISMATCH,observed,"b9:"+result.source,{"state":result.state.value})
        if not self._scope_evidence(result,scope):return ResolutionObservation(ResolutionStatus.WRONG_SCOPE,observed,"b9:"+result.source,{"state":result.state.value})
        limit=None
        if isinstance(result.value,Mapping) and isinstance(result.value.get("not_after"),(int,float)):limit=datetime.fromtimestamp(result.value["not_after"],tz=timezone.utc)
        return ResolutionObservation(ResolutionStatus.RESOLVED,observed,"b9:"+result.source,dict(result.value or {}),limit)
