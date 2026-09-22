"""The packaged, non-normative V1 control registry."""
from __future__ import annotations
from dataclasses import dataclass
import weakref
from .ids import domain_hash
from .errors import UnknownControlError, UnsupportedControlVersionError, UntrustedControlDefinitionError
from .models import ClaimLevel, EvidenceSubjectType
_trusted: dict[int, weakref.ReferenceType] = {}
@dataclass(frozen=True)
class ControlDefinition:
 control_id:str; control_version:int; mechanism:str; supported_subject_types:tuple[EvidenceSubjectType,...]; allowed_reason_codes:tuple[str,...]; supported_claim_levels:tuple[ClaimLevel,...]; governing_rule_id:str|None=None; freshness_hours:int=24; normative_effect:str="NONE"; schema_version:str="1.0"; kind:str="JAX_CONTROL_DEFINITION"
 def __post_init__(self):
  if self.normative_effect!="NONE" or not self.control_id or self.control_version!=1: raise ValueError("ControlDefinition inválida")
 @property
 def control_definition_hash(self): return domain_hash("JAX-CONTROL-DEFINITION/1",self.projection())
 def projection(self): return {"schema_version":self.schema_version,"kind":self.kind,"control_id":self.control_id,"control_version":self.control_version,"governing_rule_id":self.governing_rule_id,"mechanism":self.mechanism,"supported_subject_types":[x.value for x in self.supported_subject_types],"required_profiles":{"written":True,"tested":True,"runtime":ClaimLevel.ENFORCED in self.supported_claim_levels},"freshness":{"hours":self.freshness_hours},"allowed_reason_codes":list(self.allowed_reason_codes),"normative_effect":"NONE"}
def _make(cid, mech, subjects, levels=(ClaimLevel.WRITTEN,ClaimLevel.TESTED,ClaimLevel.ENFORCED), rule=None):
 return ControlDefinition(cid,1,mech,subjects,("SATISFIED","DENIED","FAILED","ERROR"),levels,rule)
_controls = {x.control_id:x for x in (
 _make("CTL.B5.DECISION_PROVENANCE","DecisionRecord trusted lifecycle",(EvidenceSubjectType.DECISION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("CTL.B6.AUTHORIZATION_PROVENANCE","ExecutionAuthorization trusted lifecycle",(EvidenceSubjectType.AUTHORIZATION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("CTL.B6.ONE_DECISION_ONE_EXECUTION","MariaDB unique decision constraint",(EvidenceSubjectType.EXECUTION,EvidenceSubjectType.DATABASE_SCHEMA)),
 _make("CTL.B6.HUMAN_APPROVAL_BINDING","operational approval binding",(EvidenceSubjectType.EXECUTION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("CTL.B6.GOVERNED_DISPATCH","governed Motor boundary",(EvidenceSubjectType.EXECUTION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("CTL.B6.SANDBOX_ONLY","sandbox request validation",(EvidenceSubjectType.EXECUTION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("CTL.B6.KILL_SWITCH","kill switch gate",(EvidenceSubjectType.EXECUTION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("CTL.B6.AUTHORIZATION_EXPIRY","authorization expiry gate",(EvidenceSubjectType.AUTHORIZATION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("CTL.B6.TIMEOUT_CEILING","authorization timeout ceiling",(EvidenceSubjectType.AUTHORIZATION,EvidenceSubjectType.OPERATION_ATTEMPT)),
 _make("RULE.P10","P10 static fail-open scan",(EvidenceSubjectType.POLICY_RULE,), (ClaimLevel.WRITTEN,ClaimLevel.TESTED),"P10"),
 _make("RULE.P11","P11 live policy comparison",(EvidenceSubjectType.POLICY_RULE,), (ClaimLevel.WRITTEN,ClaimLevel.TESTED),"P11"),)}
def load_control_definition(control_id:str, control_version:int=1)->ControlDefinition:
 if control_id not in _controls: raise UnknownControlError(control_id)
 value=_controls[control_id]
 if value.control_version!=control_version: raise UnsupportedControlVersionError(control_id)
 _trusted[id(value)]=weakref.ref(value); return value
def is_trusted_control_definition(value: object)->bool:
 r=_trusted.get(id(value)); return r is not None and r() is value
def require_trusted_definition(value:ControlDefinition)->None:
 if not is_trusted_control_definition(value): raise UntrustedControlDefinitionError("definición no cargada del registro")
