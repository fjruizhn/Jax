from dataclasses import replace

import pytest

from policy.governance.response import (
    AuthorityOrigin, ClaimDisposition, ClaimRecord, ContentBlock,
    ContentBlockKind, ContractState, EpistemicStatus, ExistenceState,
    GovernanceContractError, GovernanceReceipt, GovernedResponseEnvelope,
    ReferenceRef, ReferenceType, ResponseScope, SourceClass, TemplateContract,
    TemporalClass,
)


def scope(**changes):
    value = ResponseScope(
        environment="production", tenant_id="tenant-a", project_id="project-a",
        subject_id="user:fernando", actor_id="service:jax", audience="human:fernando",
        component_id="web-chat", request_id="request-1", trace_id="trace-1",
    )
    return replace(value, **changes)


def ref(ref_id, ref_type, response_scope, *, revision="sha256:" + "a" * 64):
    return ReferenceRef(
        ref_id=ref_id, ref_type=ref_type, canonical_locator=f"axioma://{ref_id}",
        immutable_identity=f"immutable:{ref_id}", revision_or_digest=revision,
        scope_digest=response_scope.scope_digest, temporal_class=TemporalClass.HISTORICAL,
        existence_state=ExistenceState.PRESENT,
    )


def template():
    return TemplateContract("claim-current", "1", "es-HN")


def claim(response_scope, **changes):
    value = ClaimRecord(
        claim_id="claim-1", predicate="ENGINE_STATUS", typed_arguments={"engine": "jax"},
        claim_scope=response_scope, source_class=SourceClass.CURRENT_SOURCE,
        epistemic_status=EpistemicStatus.CURRENT_OBSERVATION,
        resolution_receipt_ref="receipt-1", disposition=ClaimDisposition.ASSERTABLE,
        template_contract=template(),
    )
    return replace(value, **changes)


def envelope(response_scope=None, **changes):
    response_scope = response_scope or scope()
    receipt = ref("receipt-1", ReferenceType.RESOLUTION_RECEIPT, response_scope)
    value = GovernedResponseEnvelope(
        schema_version="f2-a.1", response_id="response-1", request_id=response_scope.request_id,
        trace_id=response_scope.trace_id, response_scope=response_scope, producer="web-chat",
        issuance_authority_refs=(),
        content_blocks=(ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK, claim_refs=("claim-1",)),),
        claims=(claim(response_scope),), references=(receipt,), contract_state=ContractState.VALID,
        governance_receipt=GovernanceReceipt("policy-v1", "sha256:" + "b" * 64, "f2-a"),
    )
    changes.setdefault("envelope_digest", None)
    return replace(value, **changes)


def test_vocabularies_are_distinct_dimensions():
    assert AuthorityOrigin.HUMAN.value == "HUMAN"
    assert SourceClass.MEMORY.value != EpistemicStatus.MEMORY_DERIVED.value
    with pytest.raises(GovernanceContractError, match="permitted pairing"):
        claim(scope(), source_class=SourceClass.MEMORY)


def test_issuance_authority_does_not_become_claim_authority():
    s = scope(); authority = ref("authority-1", ReferenceType.AUTHORITY, s)
    result = envelope(s, issuance_authority_refs=("authority-1",), references=(ref("receipt-1", ReferenceType.RESOLUTION_RECEIPT, s), authority))
    assert result.claims[0].authority_refs == ()


def test_narrative_cannot_own_claims_or_attribution():
    with pytest.raises(GovernanceContractError):
        ContentBlock(ContentBlockKind.NARRATIVE_TEXT, "engine healthy", claim_refs=("claim-1",))


def test_claim_block_cannot_reference_non_assertable_claim():
    s = scope()
    withheld = ClaimRecord("claim-1", "ENGINE_STATUS", {}, s, SourceClass.UNKNOWN,
                           EpistemicStatus.UNAVAILABLE, disposition=ClaimDisposition.WITHHELD)
    with pytest.raises(GovernanceContractError, match="ASSERTABLE"):
        envelope(s, claims=(withheld,))


@pytest.mark.parametrize(("status", "source"), [
    (EpistemicStatus.MEMORY_DERIVED, SourceClass.CURRENT_SOURCE),
    (EpistemicStatus.EVIDENCE_BOUND, SourceClass.CURRENT_SOURCE),
    (EpistemicStatus.USER_ASSERTED, SourceClass.CURRENT_SOURCE),
    (EpistemicStatus.MODEL_KNOWLEDGE, SourceClass.CURRENT_SOURCE),
    (EpistemicStatus.INFERRED, SourceClass.CURRENT_SOURCE),
])
def test_non_current_sources_cannot_become_current(status, source):
    with pytest.raises(GovernanceContractError):
        claim(scope(), epistemic_status=status, source_class=source,
              resolution_receipt_ref=None, disposition=ClaimDisposition.WITHHELD,
              template_contract=None)


def test_current_observation_requires_receipt_and_current_source():
    with pytest.raises(GovernanceContractError, match="resolution_receipt"):
        claim(scope(), resolution_receipt_ref=None)
    with pytest.raises(GovernanceContractError, match="permitted pairing"):
        claim(scope(), source_class=SourceClass.TOOL_RESULT)


@pytest.mark.parametrize("basis_kind", [ReferenceType.EVIDENCE, ReferenceType.AUTHORITY, ReferenceType.MEMORY])
def test_b7_b8_b9_reference_alone_cannot_create_current_observation(basis_kind):
    s = scope()
    basis = ref("basis-1", basis_kind, s)
    with pytest.raises(GovernanceContractError, match="resolution_receipt"):
        ClaimRecord("claim-current", "ENGINE_STATUS", {}, s, SourceClass.CURRENT_SOURCE,
                    EpistemicStatus.CURRENT_OBSERVATION, basis_refs=(basis.ref_id,),
                    disposition=ClaimDisposition.ASSERTABLE, template_contract=template())


def test_user_assertion_requires_structural_attribution():
    s = scope(); user = ref("user-1", ReferenceType.USER_ASSERTION, s)
    user_claim = ClaimRecord("claim-user", "USER_REPORT", {"text": "down"}, s,
                             SourceClass.USER_INPUT, EpistemicStatus.USER_ASSERTED,
                             basis_refs=("user-1",))
    result = GovernedResponseEnvelope(
        "f2-a.1", "response-user", s.request_id, s.trace_id, s, "web-chat", (),
        (ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE, "Hall9000 is down", attribution_ref="user-1"),),
        (user_claim,), (user,), ContractState.VALID,
        GovernanceReceipt("policy-v1", "sha256:" + "b" * 64, "f2-a"),
    )
    assert result.claims[0].epistemic_status is EpistemicStatus.USER_ASSERTED


def test_actor_subject_and_audience_are_distinct_and_scope_digest_is_sensitive():
    s = scope()
    assert s.actor_id != s.subject_id != s.audience
    assert s.scope_digest != scope(environment="staging").scope_digest
    assert s.scope_digest != scope(audience="agent:jacobs").scope_digest


def test_cross_environment_or_claim_scope_mismatch_fails_closed():
    s = scope()
    with pytest.raises(GovernanceContractError, match="claim scope"):
        envelope(s, claims=(claim(scope(environment="staging")),))


def test_referenced_basis_scope_mismatch_fails_closed():
    s = scope()
    staging_basis = ref("b7-evidence", ReferenceType.EVIDENCE, scope(environment="staging"))
    evidence_claim = ClaimRecord(
        "claim-evidence", "HISTORICAL_CONTROL", {}, s, SourceClass.EVIDENCE,
        EpistemicStatus.EVIDENCE_BOUND, basis_refs=("b7-evidence",),
    )
    with pytest.raises(GovernanceContractError, match="basis reference scope mismatch"):
        GovernedResponseEnvelope(
            "f2-a.1", "response-scope", s.request_id, s.trace_id, s, "web-chat", (),
            (ContentBlock(ContentBlockKind.NARRATIVE_TEXT, "Historical context only."),),
            (evidence_claim,), (staging_basis,), ContractState.VALID,
            GovernanceReceipt("policy-v1", "sha256:" + "b" * 64, "f2-a"),
        )


def test_reference_requires_identity_revision_and_presence():
    s = scope()
    with pytest.raises(GovernanceContractError):
        ref("evidence-1", ReferenceType.EVIDENCE, s, revision=None)
    with pytest.raises(GovernanceContractError):
        ReferenceRef("x", ReferenceType.ARTIFACT, "axioma://x", "identity:x", None,
                     s.scope_digest, TemporalClass.HISTORICAL, ExistenceState.TOMBSTONED)


def test_payload_cannot_define_trusted_metadata():
    with pytest.raises(GovernanceContractError, match="trusted metadata"):
        ContentBlock(ContentBlockKind.TOOL_DATA, {"source_class": "CURRENT_SOURCE"})
    with pytest.raises(GovernanceContractError, match="trusted metadata"):
        ContentBlock(ContentBlockKind.NARRATIVE_TEXT, {"verified": True})


def test_envelope_digest_is_deterministic_and_semantic_changes_change_it():
    first, second = envelope(), envelope()
    assert first.envelope_digest == second.envelope_digest
    changed = envelope(scope(tenant_id="tenant-b", request_id="request-2", trace_id="trace-2"),
                       request_id="request-2", trace_id="trace-2")
    assert first.envelope_digest != changed.envelope_digest


def test_canonical_projection_roundtrip_is_stable():
    result = envelope()
    projection = result.canonical_projection()
    restored = GovernedResponseEnvelope.from_canonical_projection(projection)
    assert restored.canonical_projection() == projection
    assert restored.envelope_digest == result.envelope_digest
    assert restored.to_canonical_bytes() == result.to_canonical_bytes()
    assert projection["response_scope"]["tenant_id"] == "tenant-a"
    assert projection["claims"][0]["epistemic_status"] == "CURRENT_OBSERVATION"


def test_invalid_digest_and_invalid_claim_reference_fail_closed():
    with pytest.raises(GovernanceContractError, match="envelope_digest"):
        envelope(envelope_digest="sha256:" + "0" * 64)
    with pytest.raises(GovernanceContractError):
        envelope(content_blocks=(ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK, claim_refs=("unknown",)),))


def test_b7_b8_b9_refs_remain_references_without_reclassification():
    s = scope()
    evidence = ref("b7-evidence", ReferenceType.EVIDENCE, s)
    authority = ref("b8-authority", ReferenceType.AUTHORITY, s)
    memory = ref("b9-memory", ReferenceType.MEMORY, s)
    memory_claim = ClaimRecord("memory-claim", "MEMORY_FACT", {}, s, SourceClass.MEMORY,
                               EpistemicStatus.MEMORY_DERIVED, basis_refs=("b9-memory",))
    result = GovernedResponseEnvelope(
        "f2-a.1", "response-refs", s.request_id, s.trace_id, s, "repl", ("b8-authority",),
        (ContentBlock(ContentBlockKind.NARRATIVE_TEXT, "Historical context only."),),
        (memory_claim,), (evidence, authority, memory), ContractState.VALID,
        GovernanceReceipt("policy-v1", "sha256:" + "b" * 64, "f2-a"),
    )
    assert result.claims[0].source_class is SourceClass.MEMORY
    assert result.claims[0].epistemic_status is EpistemicStatus.MEMORY_DERIVED
