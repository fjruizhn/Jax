"""F2-A structural and adversarial contract tests."""
from dataclasses import replace
from types import MappingProxyType
import math
import pytest
import policy.governance.response as response
from policy.governance.response import *

def scope(**kw):
    return replace(ResponseScope("production","tenant-a","project-a","user:fernando","service:jax","human:fernando","web-chat","request-1","trace-1"),**kw)
def ref(i,t,s,*,identity=None,revision="sha256:"+"a"*64,**kw):
    temporal=kw.get("temporal",TemporalClass.CURRENT if t is ReferenceType.RESOLUTION_RECEIPT else TemporalClass.HISTORICAL)
    return ReferenceRef(i,t,kw.get("locator",f"axioma://{i}"),identity or f"immutable:{i}",revision,s.scope_digest,temporal,ExistenceState.PRESENT,kw.get("asserter_id"))
def template(): return TemplateContract("claim-current","1","es-HN")
def claim(s,**kw):
    return replace(ClaimRecord("claim-1","ENGINE_STATUS",{"engine":"jax"},s,SourceClass.CURRENT_SOURCE,EpistemicStatus.CURRENT_OBSERVATION,resolution_receipt_ref="receipt-1",disposition=ClaimDisposition.ASSERTABLE,template_contract=template()),**kw)
def receipt(): return GovernanceReceipt("policy-v1","vocab-f2-a","sha256:"+"b"*64,"f2-a","renderer-plan-f2-a")
def candidate(s=None,**kw):
    s=s or scope(); r=ref("receipt-1",ReferenceType.RESOLUTION_RECEIPT,s)
    base=GovernedResponseCandidate("f2-a.1","response-1",s.request_id,s.trace_id,s,"web-chat",(),(ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK,claim_refs=("claim-1",)),),(claim(s),),(r,))
    return replace(base,**kw)
def envelope(s=None,**kw):
    state=kw.pop("contract_state",ContractState.VALID); return response._seal_candidate_for_server(candidate(s,**kw),contract_state=state,governance_receipt=receipt())

def test_vocabularies_are_distinct_and_runtime_typed():
    assert AuthorityOrigin.HUMAN.value == "HUMAN"
    with pytest.raises(GovernanceContractError): claim(scope(),source_class="CURRENT_SOURCE")
    with pytest.raises(GovernanceContractError): ContentBlock("NARRATIVE_TEXT","x")

@pytest.mark.parametrize("status,source",[(EpistemicStatus.MEMORY_DERIVED,SourceClass.CURRENT_SOURCE),(EpistemicStatus.EVIDENCE_BOUND,SourceClass.CURRENT_SOURCE),(EpistemicStatus.USER_ASSERTED,SourceClass.CURRENT_SOURCE),(EpistemicStatus.MODEL_KNOWLEDGE,SourceClass.CURRENT_SOURCE),(EpistemicStatus.INFERRED,SourceClass.CURRENT_SOURCE)])
def test_nonconflation_sources_cannot_be_current(status,source):
    with pytest.raises(GovernanceContractError): claim(scope(),epistemic_status=status,source_class=source,resolution_receipt_ref=None,disposition=ClaimDisposition.WITHHELD,template_contract=None)

def test_issuance_authority_is_not_claim_authority():
    s=scope(); a=ref("authority-1",ReferenceType.AUTHORITY,s)
    result=envelope(s,issuance_authority_refs=("authority-1",),references=(ref("receipt-1",ReferenceType.RESOLUTION_RECEIPT,s),a))
    assert result.claims[0].authority_refs == ()

def test_content_block_escapes_fail_closed():
    with pytest.raises(GovernanceContractError): ContentBlock(ContentBlockKind.NARRATIVE_TEXT,"x",claim_refs=("claim-1",))
    with pytest.raises(GovernanceContractError): ContentBlock(ContentBlockKind.TOOL_DATA,{"source_class":"CURRENT_SOURCE"})
    with pytest.raises(GovernanceContractError): ContentBlock(ContentBlockKind.SAFE_STATIC_NOTICE,"dynamic")
    with pytest.raises(GovernanceContractError): envelope(content_blocks=(ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK,claim_refs=("missing",)),))

def test_f2a07_user_assertion_requires_exact_quote_binding():
    s=scope(); u=ref("user-a",ReferenceType.USER_ASSERTION,s,asserter_id="Fernando")
    c=ClaimRecord("user-claim","USER_REPORT",{"text":"down"},s,SourceClass.USER_INPUT,EpistemicStatus.USER_ASSERTED,basis_refs=("user-a",))
    b=ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE,"Hall9000 is down",claim_refs=("user-claim",),attribution_ref="user-a",speaker="Fernando")
    result=response._seal_candidate_for_server(GovernedResponseCandidate("f2-a.1","response-u",s.request_id,s.trace_id,s,"web-chat",(),(b,),(c,),(u,)),contract_state=ContractState.VALID,governance_receipt=receipt())
    assert result.claims[0].owner_response_id == "response-u"
    with pytest.raises(GovernanceContractError): ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE,"x",claim_refs=("user-claim",),attribution_ref="user-a",speaker="")

def test_scope_is_complete_and_digest_sensitive():
    s=scope(); assert s.scope_digest != scope(environment="staging").scope_digest
    assert s.scope_digest != scope(project_id=None).scope_digest
    assert s.scope_digest != scope(audience="agent:jacobs").scope_digest
    with pytest.raises(GovernanceContractError): candidate(s,producer="provider:untrusted")

def test_reference_contract_and_f2a04_reclassification():
    s=scope()
    with pytest.raises(GovernanceContractError): ReferenceRef("x",ReferenceType.ARTIFACT,"","id","sha:x",s.scope_digest,TemporalClass.HISTORICAL,ExistenceState.PRESENT)
    e=ref("e",ReferenceType.EVIDENCE,s,identity="same"); m=ref("m",ReferenceType.MEMORY,s,identity="same")
    with pytest.raises(GovernanceContractError): candidate(s,references=(ref("receipt-1",ReferenceType.RESOLUTION_RECEIPT,s),e,m))
    bad=ref("bad",ReferenceType.EVIDENCE,scope(environment="staging"))
    with pytest.raises(GovernanceContractError): candidate(s,references=(ref("receipt-1",ReferenceType.RESOLUTION_RECEIPT,s),bad))

def test_f2a01_freezes_all_caller_owned_payloads():
    s=scope(); args={"nested":{"items":["a"]}}; tool={"nested":[{"x":1}]}
    c=claim(s,typed_arguments=args); block=ContentBlock(ContentBlockKind.TOOL_DATA,tool)
    args["nested"]["items"].append("b"); tool["nested"][0]["x"]=2
    assert c.typed_arguments["nested"]["items"] == ("a",); assert block.payload["nested"][0]["x"] == 1
    with pytest.raises(TypeError): c.typed_arguments["x"]="y"
    with pytest.raises(TypeError): block.payload["x"]=1
    with pytest.raises(GovernanceContractError):
        GovernedResponseCandidate("f","r",s.request_id,s.trace_id,s,"web-chat",(),[],(),())

@pytest.mark.parametrize("bad",[{1:"a"},{"x":{1:"a"}},{"x":float("nan")},{"x":float("inf")},{"x":object()}])
def test_f2a05_canonical_values_fail_closed(bad):
    with pytest.raises(GovernanceContractError): ClaimRecord("x","P",bad,scope(),SourceClass.UNKNOWN,EpistemicStatus.UNAVAILABLE)

def test_f2a05_unicode_nfc_and_roundtrip_stability():
    a=ClaimRecord("x","P",{"e\u0301":"v"},scope(),SourceClass.UNKNOWN,EpistemicStatus.UNAVAILABLE)
    assert list(a.typed_arguments) == ["é"]

def test_f2a06_unordered_sets_sorted_content_order_preserved():
    s=scope(); r=ref("receipt-1",ReferenceType.RESOLUTION_RECEIPT,s); c1=claim(s); c2=replace(c1,claim_id="claim-2")
    b1=ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK,claim_refs=("claim-1",)); b2=ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK,claim_refs=("claim-2",))
    x=GovernedResponseCandidate("f","r",s.request_id,s.trace_id,s,"web-chat",(),(b1,b2),(c1,c2),(r,))
    y=GovernedResponseCandidate("f","r",s.request_id,s.trace_id,s,"web-chat",(),(b1,b2),(c2,c1),(r,))
    assert response._seal_candidate_for_server(x,contract_state=ContractState.VALID,governance_receipt=receipt()).envelope_digest == response._seal_candidate_for_server(y,contract_state=ContractState.VALID,governance_receipt=receipt()).envelope_digest
    z=replace(x,content_blocks=(b2,b1)); assert response._seal_candidate_for_server(x,contract_state=ContractState.VALID,governance_receipt=receipt()).envelope_digest != response._seal_candidate_for_server(z,contract_state=ContractState.VALID,governance_receipt=receipt()).envelope_digest

@pytest.mark.parametrize("state",[ContractState.BLOCKED_SYSTEM_CLAIM,ContractState.UNAVAILABLE,ContractState.DEGRADED_STRUCTURED])
def test_f2a03_contract_state_matrix_blocks_current_presentation(state):
    with pytest.raises(GovernanceContractError): envelope(contract_state=state)

def test_f2a02_candidate_and_sealed_trust_are_distinct():
    c=candidate(); assert not hasattr(c,"envelope_digest")
    assert not hasattr(response,"seal_candidate")
    with pytest.raises(GovernanceContractError): GovernedResponseEnvelope()
    sealed=envelope(); assert sealed.contract_state is ContractState.VALID
    projection=sealed.canonical_projection() | {"envelope_digest":sealed.envelope_digest}
    with pytest.raises(GovernanceContractError): load_sealed_envelope(projection)
    projection["governance_receipt"]["policy_version"]="fake"
    with pytest.raises(GovernanceContractError): load_sealed_envelope(projection)

def test_current_observation_requires_current_receipt_and_quote_speaker_is_bound():
    s=scope()
    stale=ref("receipt-1",ReferenceType.RESOLUTION_RECEIPT,s,temporal=TemporalClass.HISTORICAL)
    with pytest.raises(GovernanceContractError):
        candidate(s,references=(stale,))
    u=ref("u",ReferenceType.USER_ASSERTION,s,asserter_id="Fernando")
    c=ClaimRecord("uc","USER_REPORT",{},s,SourceClass.USER_INPUT,EpistemicStatus.USER_ASSERTED,basis_refs=("u",))
    b=ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE,"x",claim_refs=("uc",),attribution_ref="u",speaker="Mallory")
    with pytest.raises(GovernanceContractError):
        GovernedResponseCandidate("f","r",s.request_id,s.trace_id,s,"web-chat",(),(b,),(c,),(u,))

def test_digest_covers_security_relevant_fields_and_is_immutable():
    first=envelope(); assert first.compute_digest()==first.envelope_digest
    assert first.envelope_digest != envelope(scope(tenant_id="tenant-b",request_id="r2",trace_id="t2"),request_id="r2",trace_id="t2").envelope_digest
    with pytest.raises(Exception): first.contract_state=ContractState.UNAVAILABLE
    with pytest.raises(Exception): first.candidate.claims=()

def test_b7_b8_b9_remain_reference_only():
    s=scope(); e=ref("b7",ReferenceType.EVIDENCE,s); a=ref("b8",ReferenceType.AUTHORITY,s); m=ref("b9",ReferenceType.MEMORY,s)
    c=ClaimRecord("memory","MEMORY_FACT",{},s,SourceClass.MEMORY,EpistemicStatus.MEMORY_DERIVED,basis_refs=("b9",))
    x=GovernedResponseCandidate("f","r",s.request_id,s.trace_id,s,"web-chat",("b8",),(ContentBlock(ContentBlockKind.NARRATIVE_TEXT,"historical"),),(c,),(e,a,m))
    assert response._seal_candidate_for_server(x,contract_state=ContractState.VALID,governance_receipt=receipt()).claims[0].epistemic_status is EpistemicStatus.MEMORY_DERIVED
