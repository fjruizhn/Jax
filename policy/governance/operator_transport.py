"""Closed F2-D text wire encoders for operator-facing transports."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from .output_lifecycle import GovernedTransportUnit, OutputLifecycleError, revalidate_for_transport
from .governed_renderer import RenderedText

_CHANNEL_VERSION={"OPERATOR_CLI_TEXT_V1":"cli-utf8-v1","EXTERNAL_ALERT_V1":"telegram-json-v1","DURABLE_JOB_RESULT_V1":"durable-text-v1"}
OPERATOR_WIRE_VERSIONS=frozenset(_CHANNEL_VERSION.values())
@dataclass(frozen=True)
class OperatorTransportMetadata:
    channel_id:str; wire_version:str; destination_configuration_identity:str
    chat_id:str|None=None
    def __post_init__(self):
        if _CHANNEL_VERSION.get(self.channel_id)!=self.wire_version: raise OutputLifecycleError("unsupported operator transport pairing")
        if not self.destination_configuration_identity: raise OutputLifecycleError("destination identity required")
        if self.wire_version=="telegram-json-v1" and not self.chat_id: raise OutputLifecycleError("server chat id required")
        if self.wire_version!="telegram-json-v1" and self.chat_id is not None: raise OutputLifecycleError("chat id unsupported")
    @property
    def destination_digest(self): return "sha256:"+hashlib.sha256(self.destination_configuration_identity.encode()).hexdigest()
    def projection(self): return {"channel_id":self.channel_id,"wire_version":self.wire_version,"destination_digest":self.destination_digest,"chat_id_digest":None if self.chat_id is None else "sha256:"+hashlib.sha256(self.chat_id.encode()).hexdigest()}
_TOKEN=object()
@dataclass(frozen=True,init=False)
class OperatorWireUnit:
    unit:GovernedTransportUnit; metadata:OperatorTransportMetadata; wire_bytes:bytes; wire_digest:str; bound_digest:str
    def __init__(self,*a,**k): raise OutputLifecycleError("operator wire requires server minting")
    @classmethod
    def _mint(cls,t,**k):
        if t is not _TOKEN: raise OutputLifecycleError("operator wire requires server minting")
        x=object.__new__(cls)
        for n,v in k.items(): object.__setattr__(x,n,v)
        return x
    def durable_projection(self): return {"parent_projection_digest":self.unit.effective_projection_digest,"metadata":self.metadata.projection(),"wire_digest":self.wire_digest,"bound_digest":self.bound_digest}
def mint_operator_wire(unit:GovernedTransportUnit, rendered:RenderedText, metadata:OperatorTransportMetadata, *, now):
    actual=revalidate_for_transport(unit,now)
    if actual!=rendered: raise OutputLifecycleError("operator text changed before wire")
    if metadata.wire_version=="telegram-json-v1": wire=json.dumps({"chat_id":metadata.chat_id,"text":rendered.text},ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    elif metadata.wire_version=="cli-utf8-v1": wire=(rendered.text+"\n").encode()
    else: wire=rendered.text.encode()
    digest="sha256:"+hashlib.sha256(wire).hexdigest()
    bound="sha256:"+hashlib.sha256(json.dumps({"parent":unit.effective_projection_digest,"metadata":metadata.projection(),"wire":digest},sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return OperatorWireUnit._mint(_TOKEN,unit=unit,metadata=metadata,wire_bytes=wire,wire_digest=digest,bound_digest=bound)
def revalidate_operator_wire(value:OperatorWireUnit, *, now):
    if not isinstance(value,OperatorWireUnit): raise OutputLifecycleError("operator wire unit required")
    actual=mint_operator_wire(value.unit,value.unit.rendered,value.metadata,now=now)
    if actual.wire_bytes!=value.wire_bytes or actual.wire_digest!=value.wire_digest or actual.bound_digest!=value.bound_digest: raise OutputLifecycleError("operator wire changed before transport")
    return actual
