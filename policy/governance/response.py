"""F2-A immutable, structural response-governance contracts; no I/O or resolver semantics."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib, json, math, unicodedata
from types import MappingProxyType
from typing import Any, Mapping

class GovernanceContractError(ValueError): pass
class SourceClass(str, Enum):
    CURRENT_SOURCE="CURRENT_SOURCE"; EVIDENCE="EVIDENCE"; TOOL_RESULT="TOOL_RESULT"; MEMORY="MEMORY"; USER_INPUT="USER_INPUT"; MODEL_KNOWLEDGE="MODEL_KNOWLEDGE"; INFERENCE="INFERENCE"; UNKNOWN="UNKNOWN"
class EpistemicStatus(str, Enum):
    CURRENT_OBSERVATION="CURRENT_OBSERVATION"; EVIDENCE_BOUND="EVIDENCE_BOUND"; USER_ASSERTED="USER_ASSERTED"; MEMORY_DERIVED="MEMORY_DERIVED"; MODEL_KNOWLEDGE="MODEL_KNOWLEDGE"; INFERRED="INFERRED"; UNVERIFIED="UNVERIFIED"; UNAVAILABLE="UNAVAILABLE"
class AuthorityOrigin(str, Enum): HUMAN="HUMAN"; POLICY="POLICY"; SERVICE="SERVICE"; DELEGATED="DELEGATED"; SYSTEM="SYSTEM"
class ClaimDisposition(str, Enum): ASSERTABLE="ASSERTABLE"; WITHHELD="WITHHELD"; UNAVAILABLE="UNAVAILABLE"
class ContractState(str, Enum): VALID="VALID"; DEGRADED_STRUCTURED="DEGRADED_STRUCTURED"; BLOCKED_SYSTEM_CLAIM="BLOCKED_SYSTEM_CLAIM"; UNAVAILABLE="UNAVAILABLE"
class ReferenceType(str, Enum):
    AUTHORITY="AUTHORITY"; EVIDENCE="EVIDENCE"; MEMORY="MEMORY"; TOOL_RESULT="TOOL_RESULT"; CURRENT_SOURCE="CURRENT_SOURCE"; USER_ASSERTION="USER_ASSERTION"; MODEL_KNOWLEDGE="MODEL_KNOWLEDGE"; INFERENCE="INFERENCE"; RESOLUTION_RECEIPT="RESOLUTION_RECEIPT"; ARTIFACT="ARTIFACT"
class TemporalClass(str, Enum): CURRENT="CURRENT"; HISTORICAL="HISTORICAL"; ATEMPORAL="ATEMPORAL"
class ExistenceState(str, Enum): PRESENT="PRESENT"; TOMBSTONED="TOMBSTONED"; UNKNOWN="UNKNOWN"
class ContentBlockKind(str, Enum): NARRATIVE_TEXT="NARRATIVE_TEXT"; CLAIM_REF_BLOCK="CLAIM_REF_BLOCK"; ATTRIBUTED_QUOTE="ATTRIBUTED_QUOTE"; TOOL_DATA="TOOL_DATA"; SAFE_STATIC_NOTICE="SAFE_STATIC_NOTICE"; ERROR_NOTICE="ERROR_NOTICE"
_TRUSTED=frozenset({"authority","authority_refs","authority_origin","source","source_class","verification","verified","epistemic_status","trust","trusted_label","current_observation","citation","citations","memory_label","claim_id","scope_digest","governance_receipt","current"})

def _text(v: str,n: str)->str:
    if not isinstance(v,str) or not v.strip(): raise GovernanceContractError(f"{n} must be a non-empty string")
    return unicodedata.normalize("NFC",v)
def _enum(v:Any,t:type[Enum],n:str)->None:
    if not isinstance(v,t): raise GovernanceContractError(f"{n} must be {t.__name__}")
def _ids(v:tuple[str,...],n:str)->tuple[str,...]:
    if not isinstance(v,tuple): raise GovernanceContractError(f"{n} must be a tuple")
    r=tuple(_text(x,n) for x in v)
    if len(r)!=len(set(r)): raise GovernanceContractError(f"{n} contains duplicate identifiers")
    return tuple(sorted(r))
def _freeze(v:Any,path="value")->Any:
    if v is None or isinstance(v,(bool,int)): return v
    if isinstance(v,float):
        if not math.isfinite(v): raise GovernanceContractError(f"{path} may not contain NaN or infinity")
        raise GovernanceContractError(f"{path} may not contain floats")
    if isinstance(v,str): return unicodedata.normalize("NFC",v)
    if isinstance(v,Mapping):
        r={}
        for k,x in v.items():
            if not isinstance(k,str): raise GovernanceContractError(f"{path} mapping keys must be strings")
            k=unicodedata.normalize("NFC",k)
            if k in r: raise GovernanceContractError(f"{path} has duplicate keys after NFC normalization")
            r[k]=_freeze(x,f"{path}.{k}")
        return MappingProxyType(dict(sorted(r.items())))
    if isinstance(v,(list,tuple)): return tuple(_freeze(x,f"{path}[{i}]") for i,x in enumerate(v))
    raise GovernanceContractError(f"{path} must contain only canonical JSON values")
def _plain(v:Any)->Any:
    if isinstance(v,Enum): return v.value
    if isinstance(v,Mapping): return {k:_plain(x) for k,x in v.items()}
    if isinstance(v,tuple): return [_plain(x) for x in v]
    if hasattr(v,"semantic_projection"): return _plain(v.semantic_projection())
    return v
def _trusted(v:Any)->bool:
    if isinstance(v,Mapping): return any(k.lower() in _TRUSTED or _trusted(x) for k,x in v.items())
    return isinstance(v,tuple) and any(_trusted(x) for x in v)
def _bytes(v:Any)->bytes: return json.dumps(v,ensure_ascii=True,allow_nan=False,sort_keys=True,separators=(",",":")).encode()

@dataclass(frozen=True)
class ResponseScope:
    environment:str; tenant_id:str; project_id:str|None; subject_id:str|None; actor_id:str|None; audience:str; component_id:str; request_id:str; trace_id:str
    def __post_init__(self):
        for n in ("environment","tenant_id","audience","component_id","request_id","trace_id"): object.__setattr__(self,n,_text(getattr(self,n),n))
        for n in ("project_id","subject_id","actor_id"):
            if getattr(self,n) is not None: object.__setattr__(self,n,_text(getattr(self,n),n))
    def semantic_projection(self): return {n:getattr(self,n) for n in ("environment","tenant_id","project_id","subject_id","actor_id","audience","component_id","request_id","trace_id")}
    @property
    def scope_digest(self): return "sha256:"+hashlib.sha256(_bytes(self.semantic_projection())).hexdigest()

@dataclass(frozen=True)
class ReferenceRef:
    ref_id:str; ref_type:ReferenceType; canonical_locator:str; immutable_identity:str; revision_or_digest:str; scope_digest:str; temporal_class:TemporalClass; existence_state:ExistenceState; asserter_id:str|None=None
    def __post_init__(self):
        _enum(self.ref_type,ReferenceType,"ref_type"); _enum(self.temporal_class,TemporalClass,"temporal_class"); _enum(self.existence_state,ExistenceState,"existence_state")
        for n in ("ref_id","canonical_locator","immutable_identity","revision_or_digest","scope_digest"): object.__setattr__(self,n,_text(getattr(self,n),n))
        if self.existence_state is not ExistenceState.PRESENT: raise GovernanceContractError("a response reference must be structurally present")
        if self.asserter_id is not None: object.__setattr__(self,"asserter_id",_text(self.asserter_id,"asserter_id"))
        if self.ref_type is ReferenceType.USER_ASSERTION and self.asserter_id is None: raise GovernanceContractError("USER_ASSERTION requires asserter_id")
    def semantic_projection(self): return {n:_plain(getattr(self,n)) for n in ("ref_id","ref_type","canonical_locator","immutable_identity","revision_or_digest","scope_digest","temporal_class","existence_state","asserter_id")}

@dataclass(frozen=True)
class TemplateContract:
    template_id:str; template_version:str; locale:str
    def __post_init__(self):
        for n in ("template_id","template_version","locale"): object.__setattr__(self,n,_text(getattr(self,n),n))
    def semantic_projection(self): return {"template_id":self.template_id,"template_version":self.template_version,"locale":self.locale}

@dataclass(frozen=True)
class ClaimRecord:
    claim_id:str; predicate:str; typed_arguments:Mapping[str,Any]; claim_scope:ResponseScope; source_class:SourceClass; epistemic_status:EpistemicStatus; authority_refs:tuple[str,...]=(); basis_refs:tuple[str,...]=(); resolution_receipt_ref:str|None=None; disposition:ClaimDisposition=ClaimDisposition.WITHHELD; reason_codes:tuple[str,...]=(); template_contract:TemplateContract|None=None; owner_response_id:str|None=None
    def __post_init__(self):
        _text(self.claim_id,"claim_id"); _text(self.predicate,"predicate")
        if not isinstance(self.claim_scope,ResponseScope): raise GovernanceContractError("claim_scope must be ResponseScope")
        _enum(self.source_class,SourceClass,"source_class"); _enum(self.epistemic_status,EpistemicStatus,"epistemic_status"); _enum(self.disposition,ClaimDisposition,"disposition")
        if self.template_contract is not None and not isinstance(self.template_contract,TemplateContract): raise GovernanceContractError("template_contract must be TemplateContract")
        if not isinstance(self.typed_arguments,Mapping): raise GovernanceContractError("typed_arguments must be a mapping")
        object.__setattr__(self,"typed_arguments",_freeze(self.typed_arguments,"typed_arguments")); object.__setattr__(self,"authority_refs",_ids(self.authority_refs,"authority_refs")); object.__setattr__(self,"basis_refs",_ids(self.basis_refs,"basis_refs")); object.__setattr__(self,"reason_codes",_ids(self.reason_codes,"reason_codes"))
        if self.resolution_receipt_ref is not None: object.__setattr__(self,"resolution_receipt_ref",_text(self.resolution_receipt_ref,"resolution_receipt_ref"))
        if self.owner_response_id is not None: object.__setattr__(self,"owner_response_id",_text(self.owner_response_id,"owner_response_id"))
        expected={EpistemicStatus.CURRENT_OBSERVATION:SourceClass.CURRENT_SOURCE,EpistemicStatus.EVIDENCE_BOUND:SourceClass.EVIDENCE,EpistemicStatus.USER_ASSERTED:SourceClass.USER_INPUT,EpistemicStatus.MEMORY_DERIVED:SourceClass.MEMORY,EpistemicStatus.MODEL_KNOWLEDGE:SourceClass.MODEL_KNOWLEDGE,EpistemicStatus.INFERRED:SourceClass.INFERENCE}.get(self.epistemic_status)
        if expected is not None and self.source_class is not expected: raise GovernanceContractError("source_class and epistemic_status are not a permitted pairing")
        if self.epistemic_status is EpistemicStatus.CURRENT_OBSERVATION and (not self.resolution_receipt_ref or self.disposition is not ClaimDisposition.ASSERTABLE): raise GovernanceContractError("CURRENT_OBSERVATION requires resolution_receipt_ref and ASSERTABLE disposition")
        if self.epistemic_status is EpistemicStatus.USER_ASSERTED and not self.basis_refs: raise GovernanceContractError("USER_ASSERTED requires an attributed source reference")
        if self.disposition is ClaimDisposition.ASSERTABLE and self.template_contract is None: raise GovernanceContractError("ASSERTABLE claims require a template_contract")
    def semantic_projection(self): return {n:_plain(getattr(self,n)) for n in ("claim_id","predicate","typed_arguments","claim_scope","source_class","epistemic_status","authority_refs","basis_refs","resolution_receipt_ref","disposition","reason_codes","template_contract","owner_response_id")}

@dataclass(frozen=True)
class ContentBlock:
    kind:ContentBlockKind; payload:Any=None; claim_refs:tuple[str,...]=(); attribution_ref:str|None=None; speaker:str|None=None; notice_id:str|None=None
    def __post_init__(self):
        _enum(self.kind,ContentBlockKind,"kind"); object.__setattr__(self,"claim_refs",_ids(self.claim_refs,"claim_refs"))
        for n in ("attribution_ref","speaker","notice_id"):
            if getattr(self,n) is not None: object.__setattr__(self,n,_text(getattr(self,n),n))
        if self.kind is ContentBlockKind.NARRATIVE_TEXT:
            if not isinstance(self.payload,str) or self.claim_refs or self.attribution_ref or self.speaker or self.notice_id: raise GovernanceContractError("NARRATIVE_TEXT cannot own claims, attribution, or notices")
            object.__setattr__(self,"payload",_freeze(self.payload,"narrative_text"))
        elif self.kind is ContentBlockKind.CLAIM_REF_BLOCK:
            if self.payload is not None or not self.claim_refs or self.attribution_ref or self.speaker or self.notice_id: raise GovernanceContractError("CLAIM_REF_BLOCK contains only claim identities")
        elif self.kind is ContentBlockKind.ATTRIBUTED_QUOTE:
            if not isinstance(self.payload,str) or len(self.claim_refs)!=1 or not self.attribution_ref or not self.speaker or self.notice_id: raise GovernanceContractError("ATTRIBUTED_QUOTE requires one claim, attribution, and speaker")
            object.__setattr__(self,"payload",_freeze(self.payload,"attributed_quote"))
        elif self.kind in {ContentBlockKind.SAFE_STATIC_NOTICE,ContentBlockKind.ERROR_NOTICE}:
            if self.claim_refs or self.attribution_ref or self.speaker or not self.notice_id or not isinstance(self.payload,Mapping): raise GovernanceContractError("notice blocks require server-owned notice_id and safe parameters")
            object.__setattr__(self,"payload",_freeze(self.payload,"notice_params"))
        elif self.kind is ContentBlockKind.TOOL_DATA:
            if self.claim_refs or self.attribution_ref or self.speaker or self.notice_id: raise GovernanceContractError("TOOL_DATA cannot own claims, attribution, or notices")
            object.__setattr__(self,"payload",_freeze(self.payload,"tool_data"))
        if self.kind in {ContentBlockKind.TOOL_DATA,ContentBlockKind.SAFE_STATIC_NOTICE,ContentBlockKind.ERROR_NOTICE} and _trusted(self.payload): raise GovernanceContractError("untrusted payload may not define trusted metadata")
    def semantic_projection(self): return {n:_plain(getattr(self,n)) for n in ("kind","payload","claim_refs","attribution_ref","speaker","notice_id")}

@dataclass(frozen=True)
class GovernanceReceipt:
    policy_version:str; vocabulary_version:str; registry_snapshot_digest:str; validator_version:str; renderer_plan_version:str
    def __post_init__(self):
        for n in ("policy_version","vocabulary_version","registry_snapshot_digest","validator_version","renderer_plan_version"): object.__setattr__(self,n,_text(getattr(self,n),n))
    def semantic_projection(self): return {"policy_version":self.policy_version,"vocabulary_version":self.vocabulary_version,"registry_snapshot_digest":self.registry_snapshot_digest,"validator_version":self.validator_version,"renderer_plan_version":self.renderer_plan_version}

@dataclass(frozen=True)
class GovernedResponseCandidate:
    schema_version:str; response_id:str; request_id:str; trace_id:str; response_scope:ResponseScope; producer:str; issuance_authority_refs:tuple[str,...]; content_blocks:tuple[ContentBlock,...]; claims:tuple[ClaimRecord,...]; references:tuple[ReferenceRef,...]
    def __post_init__(self):
        for n in ("schema_version","response_id","request_id","trace_id","producer"): object.__setattr__(self,n,_text(getattr(self,n),n))
        if not isinstance(self.response_scope,ResponseScope): raise GovernanceContractError("response_scope must be ResponseScope")
        if self.producer!=self.response_scope.component_id: raise GovernanceContractError("producer must match response scope component_id")
        if self.request_id!=self.response_scope.request_id or self.trace_id!=self.response_scope.trace_id: raise GovernanceContractError("envelope identity must match response scope")
        if not isinstance(self.content_blocks,tuple) or not isinstance(self.claims,tuple) or not isinstance(self.references,tuple): raise GovernanceContractError("candidate graph collections must be tuples")
        if not all(isinstance(x,ContentBlock) for x in self.content_blocks) or not all(isinstance(x,ClaimRecord) for x in self.claims) or not all(isinstance(x,ReferenceRef) for x in self.references): raise GovernanceContractError("candidate graph contains invalid type")
        object.__setattr__(self,"issuance_authority_refs",_ids(self.issuance_authority_refs,"issuance_authority_refs")); self._validate_graph()
    def _validate_graph(self):
        if len({x.claim_id for x in self.claims})!=len(self.claims): raise GovernanceContractError("duplicate claim_id")
        refs={x.ref_id:x for x in self.references}
        if len(refs)!=len(self.references): raise GovernanceContractError("duplicate ref_id")
        pin={}
        for x in self.references:
            if x.scope_digest!=self.response_scope.scope_digest: raise GovernanceContractError("all envelope references must match response scope")
            k=(x.immutable_identity,x.revision_or_digest); old=pin.get(k)
            if old and (old.ref_type!=x.ref_type or old.canonical_locator!=x.canonical_locator or old.scope_digest!=x.scope_digest or old.temporal_class!=x.temporal_class or old.existence_state!=x.existence_state): raise GovernanceContractError("pinned reference identity may not be reclassified")
            pin[k]=x
        for i in self.issuance_authority_refs:
            if i not in refs or refs[i].ref_type is not ReferenceType.AUTHORITY: raise GovernanceContractError("issuance authority must reference AUTHORITY")
        bound=[]
        for c in self.claims:
            if c.claim_scope.scope_digest!=self.response_scope.scope_digest: raise GovernanceContractError("claim scope must equal response scope in F2-A")
            if c.owner_response_id is not None and c.owner_response_id!=self.response_id: raise GovernanceContractError("claim is already owned by another response")
            if any(i not in refs or refs[i].ref_type is not ReferenceType.AUTHORITY for i in c.authority_refs): raise GovernanceContractError("claim authority must reference AUTHORITY")
            if any(i not in refs for i in c.basis_refs): raise GovernanceContractError("claim basis reference is missing")
            if c.resolution_receipt_ref is not None and (c.resolution_receipt_ref not in refs or refs[c.resolution_receipt_ref].ref_type is not ReferenceType.RESOLUTION_RECEIPT): raise GovernanceContractError("resolution receipt must reference RESOLUTION_RECEIPT")
            if c.epistemic_status is EpistemicStatus.CURRENT_OBSERVATION and refs[c.resolution_receipt_ref].temporal_class is not TemporalClass.CURRENT: raise GovernanceContractError("CURRENT_OBSERVATION requires CURRENT resolution receipt")
            if c.epistemic_status is EpistemicStatus.USER_ASSERTED and not any(refs[i].ref_type is ReferenceType.USER_ASSERTION for i in c.basis_refs): raise GovernanceContractError("USER_ASSERTED requires USER_ASSERTION basis")
            if c.epistemic_status is EpistemicStatus.MEMORY_DERIVED and not any(refs[i].ref_type is ReferenceType.MEMORY for i in c.basis_refs): raise GovernanceContractError("MEMORY_DERIVED requires MEMORY basis")
            if c.epistemic_status is EpistemicStatus.EVIDENCE_BOUND and not any(refs[i].ref_type is ReferenceType.EVIDENCE for i in c.basis_refs): raise GovernanceContractError("EVIDENCE_BOUND requires EVIDENCE basis")
            bound.append(replace(c,owner_response_id=self.response_id))
        object.__setattr__(self,"claims",tuple(bound)); by={x.claim_id:x for x in self.claims}
        for b in self.content_blocks:
            if b.kind is ContentBlockKind.CLAIM_REF_BLOCK and any(i not in by or by[i].disposition is not ClaimDisposition.ASSERTABLE for i in b.claim_refs): raise GovernanceContractError("CLAIM_REF_BLOCK may reference only ASSERTABLE claims")
            if b.kind is ContentBlockKind.ATTRIBUTED_QUOTE:
                c=by.get(b.claim_refs[0]); r=refs.get(b.attribution_ref or "")
                if c is None or c.epistemic_status is not EpistemicStatus.USER_ASSERTED or r is None or r.ref_type is not ReferenceType.USER_ASSERTION or b.attribution_ref not in c.basis_refs or b.speaker != r.asserter_id: raise GovernanceContractError("ATTRIBUTED_QUOTE must bind exact USER_ASSERTED claim and attribution")
    def semantic_projection(self): return {"schema_version":self.schema_version,"response_id":self.response_id,"request_id":self.request_id,"trace_id":self.trace_id,"response_scope":self.response_scope.semantic_projection(),"producer":self.producer,"issuance_authority_refs":list(self.issuance_authority_refs),"content_blocks":[x.semantic_projection() for x in self.content_blocks],"claims":[x.semantic_projection() for x in sorted(self.claims,key=lambda x:x.claim_id)],"references":[x.semantic_projection() for x in sorted(self.references,key=lambda x:x.ref_id)]}

_SEAL_TOKEN=object()
@dataclass(frozen=True,init=False)
class GovernedResponseEnvelope:
    candidate:GovernedResponseCandidate; contract_state:ContractState; governance_receipt:GovernanceReceipt; envelope_digest:str
    def __init__(self,*a,**k): raise GovernanceContractError("use seal_candidate to create GovernedResponseEnvelope")
    @classmethod
    def _seal(cls,t,c,s,r):
        if t is not _SEAL_TOKEN: raise GovernanceContractError("sealed envelope requires server sealing pathway")
        if not isinstance(c,GovernedResponseCandidate): raise GovernanceContractError("seal_candidate requires GovernedResponseCandidate")
        _enum(s,ContractState,"contract_state")
        if not isinstance(r,GovernanceReceipt): raise GovernanceContractError("governance_receipt must be GovernanceReceipt")
        _state(c,s); x=object.__new__(cls); object.__setattr__(x,"candidate",c); object.__setattr__(x,"contract_state",s); object.__setattr__(x,"governance_receipt",r); object.__setattr__(x,"envelope_digest",x.compute_digest()); return x
    def __getattr__(self,n):
        if n in {"schema_version","response_id","request_id","trace_id","response_scope","producer","issuance_authority_refs","content_blocks","claims","references"}: return getattr(self.candidate,n)
        raise AttributeError(n)
    def canonical_projection(self):
        x=self.candidate.semantic_projection(); x.update({"contract_state":self.contract_state.value,"governance_receipt":self.governance_receipt.semantic_projection()}); return x
    def to_canonical_bytes(self): return _bytes(self.canonical_projection())
    def compute_digest(self): return "sha256:"+hashlib.sha256(self.to_canonical_bytes()).hexdigest()
def _state(c,s):
    cur={x.claim_id for x in c.claims if x.disposition is ClaimDisposition.ASSERTABLE and x.epistemic_status is EpistemicStatus.CURRENT_OBSERVATION}; shown={i for b in c.content_blocks if b.kind is ContentBlockKind.CLAIM_REF_BLOCK for i in b.claim_refs}
    if s in {ContractState.BLOCKED_SYSTEM_CLAIM,ContractState.UNAVAILABLE} and cur&shown: raise GovernanceContractError(f"{s.value} may not present ASSERTABLE CURRENT_OBSERVATION")
    if s is ContractState.DEGRADED_STRUCTURED and (cur or any(b.kind in {ContentBlockKind.NARRATIVE_TEXT,ContentBlockKind.TOOL_DATA,ContentBlockKind.CLAIM_REF_BLOCK} for b in c.content_blocks)): raise GovernanceContractError("DEGRADED_STRUCTURED may contain only safe notices or attributed quotes")
def _seal_candidate_for_server(candidate:GovernedResponseCandidate,*,contract_state:ContractState,
                               governance_receipt:GovernanceReceipt)->GovernedResponseEnvelope:
    """Module-private server sealing pathway.

    This is intentionally not part of the response-contract public API.  F2-A
    has no public operation by which caller-supplied receipt strings can turn a
    candidate into a trusted/validated envelope.  Future server integration
    owns the call site and its non-user-provided sealing authority.
    """
    return GovernedResponseEnvelope._seal(_SEAL_TOKEN,candidate,contract_state,governance_receipt)

def candidate_from_canonical_projection(v:Mapping[str,Any])->GovernedResponseCandidate:
    if not isinstance(v,Mapping): raise GovernanceContractError("candidate projection must be a mapping")
    s=ResponseScope(**v["response_scope"]); rs=tuple(ReferenceRef(ref_id=x["ref_id"],ref_type=ReferenceType(x["ref_type"]),canonical_locator=x["canonical_locator"],immutable_identity=x["immutable_identity"],revision_or_digest=x["revision_or_digest"],scope_digest=x["scope_digest"],temporal_class=TemporalClass(x["temporal_class"]),existence_state=ExistenceState(x["existence_state"]),asserter_id=x.get("asserter_id")) for x in v["references"])
    cs=tuple(ClaimRecord(claim_id=x["claim_id"],predicate=x["predicate"],typed_arguments=x["typed_arguments"],claim_scope=ResponseScope(**x["claim_scope"]),source_class=SourceClass(x["source_class"]),epistemic_status=EpistemicStatus(x["epistemic_status"]),authority_refs=tuple(x["authority_refs"]),basis_refs=tuple(x["basis_refs"]),resolution_receipt_ref=x["resolution_receipt_ref"],disposition=ClaimDisposition(x["disposition"]),reason_codes=tuple(x["reason_codes"]),template_contract=TemplateContract(**x["template_contract"]) if x["template_contract"] else None,owner_response_id=x.get("owner_response_id")) for x in v["claims"])
    bs=tuple(ContentBlock(kind=ContentBlockKind(x["kind"]),payload=x["payload"],claim_refs=tuple(x["claim_refs"]),attribution_ref=x["attribution_ref"],speaker=x.get("speaker"),notice_id=x.get("notice_id")) for x in v["content_blocks"])
    return GovernedResponseCandidate(v["schema_version"],v["response_id"],v["request_id"],v["trace_id"],s,v["producer"],tuple(v["issuance_authority_refs"]),bs,cs,rs)
def load_sealed_envelope(v:Mapping[str,Any],*,_verification_token:object|None=None)->GovernedResponseEnvelope:
    if _verification_token is not _SEAL_TOKEN: raise GovernanceContractError("sealed deserialization requires trusted verification pathway")
    c=candidate_from_canonical_projection(v); r=v.get("governance_receipt")
    if not isinstance(r,Mapping): raise GovernanceContractError("sealed projection requires governance_receipt")
    x=_seal_candidate_for_server(c,contract_state=ContractState(v.get("contract_state")),governance_receipt=GovernanceReceipt(**r))
    if not isinstance(v.get("envelope_digest"),str) or v["envelope_digest"]!=x.envelope_digest: raise GovernanceContractError("sealed projection digest verification failed")
    return x
