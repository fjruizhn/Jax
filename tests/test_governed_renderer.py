"""F2-C core renderer contracts; all dependencies are explicit test composition."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import pytest

import policy.governance.response as response
import policy.governance.resolution as resolution
from policy.governance.governed_renderer import (
    GovernedDomainRegistry, GovernedRenderError, GovernedRenderer, RenderContext,
    WebChatGovernanceAdapter,
)
from policy.governance.response import *
from policy.governance.resolution import *
from policy.governance.governed_domain import GovernedDomainSpecification

NOW = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
KEY = b"f2-c-test-secret-material-longer-than-thirty-two-bytes"

def scope(**kw):
    return replace(ResponseScope("production", "tenant-a", "project-a", "user-a", "service:jax", "human:fernando", "web-chat", "request-a", "trace-a"), **kw)

def receipt():
    return GovernanceReceipt("policy-v1", "vocab-v1", "sha256:" + "a" * 64, "f2-c", "renderer-plan-f2-c")

def ref(ref_id, typ, s, *, temporal=None, asserter=None):
    return ReferenceRef(ref_id, typ, "axioma://" + ref_id, "immutable:" + ref_id, "sha256:" + ref_id, s.scope_digest, temporal or (TemporalClass.CURRENT if typ is ReferenceType.RESOLUTION_RECEIPT else TemporalClass.HISTORICAL), ExistenceState.PRESENT, asserter)

def registry_and_receipt(s):
    rule = ScopeRule(s.environment, s.tenant_id, s.project_id, s.subject_id, s.actor_id, s.audience, s.component_id)
    binding = PredicateAuthorityBinding("CAPABILITY_AVAILABLE", "v1", "catalog:capabilities", "authority:catalog", s.environment, rule, rule, 60, ConflictPolicy.SINGLE_SOURCE_REQUIRED, "adapter:capability", "1", "sha256:catalog", "b1")
    entry = RegistryEntry(binding, TrustedAdapterRegistration(AdapterKind.CAPABILITY_AVAILABLE, "adapter:capability", "1", "catalog:capabilities", "sha256:catalog", {}), ("name", "mode"), "capability@1:en")
    registry = resolution._build_approved_registry_for_server((entry,), authenticator=ReceiptAuthenticator.for_testing(KEY))
    observed = ResolutionObservation(ResolutionStatus.RESOLVED, NOW, "catalog:proof", {"available": True})
    inp = ServerAdapterInput._mint(resolution._ADAPTER_INPUT_TOKEN, (("catalog:capabilities", observed),))
    receipt_value = registry.resolve("CAPABILITY_AVAILABLE", {"name": "x", "mode": "read"}, s, validation_time=NOW, server_input=inp)
    return registry, receipt_value

def sealed_current(s=None, *, state=ContractState.VALID):
    s = s or scope(); registry, resolution_receipt = registry_and_receipt(s)
    claim = ClaimRecord("claim-1", "CAPABILITY_AVAILABLE", {"name": "x", "mode": "read"}, s, SourceClass.CURRENT_SOURCE, EpistemicStatus.CURRENT_OBSERVATION, resolution_receipt_ref="receipt-1", disposition=ClaimDisposition.ASSERTABLE, template_contract=TemplateContract("capability", "1", "en"))
    receipt_ref = replace(ref("receipt-1", ReferenceType.RESOLUTION_RECEIPT, s), revision_or_digest=resolution_receipt.receipt_id)
    candidate = GovernedResponseCandidate("f2-c.1", "response-1", s.request_id, s.trace_id, s, "web-chat", (), (ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK, claim_refs=("claim-1",)),), (claim,), (receipt_ref,))
    env = response._seal_candidate_for_server(candidate, contract_state=state, governance_receipt=receipt())
    context = RenderContext(registry, {"receipt-1": resolution_receipt}, {("capability", "1", "en"): "Capability {name} is available."}, {}, GovernedDomainRegistry(), lambda _ref, _scope: True, lambda: NOW)
    return env, context

def test_renderer_accepts_only_sealed_envelope_and_revalidates_current_receipt():
    env, ctx = sealed_current()
    rendered = GovernedRenderer().render_text(env, ctx, chunk_size=8)
    assert rendered.text == "Capability x is available." and not rendered.chunks
    with pytest.raises(GovernedRenderError):
        GovernedRenderer().render_text(env.candidate, ctx)
    stale = RenderContext(ctx.registry, ctx.receipts, ctx.templates, {}, ctx.domain_registry, lambda _ref, _scope: True, lambda: NOW + timedelta(seconds=61))
    assert GovernedRenderer().render_text(env, stale).text == GovernedRenderer.unavailable_text
    # Structural references alone are insufficient: F2-B dereference/access
    # validation must be supplied by trusted composition before display.
    missing_reference_validation = RenderContext(ctx.registry, ctx.receipts, ctx.templates, {}, ctx.domain_registry, None, lambda: NOW)
    assert GovernedRenderer().render_text(env, missing_reference_validation).text == GovernedRenderer.unavailable_text

def test_narrative_registered_governed_term_fails_to_safe_unavailable_without_leaking_prose():
    s = scope(); adapter = WebChatGovernanceAdapter(s, receipt())
    env = adapter.seal_non_governed_candidate(response_id="response-n", candidate_text="The capability is available now.")
    ctx = RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), None, lambda: NOW)
    result = GovernedRenderer().render_text(env, ctx)
    assert result.text == GovernedRenderer.unavailable_text

def test_untrusted_markup_controls_and_tool_data_do_not_become_trusted_presentation():
    s = scope(); candidate = GovernedResponseCandidate("f2-c.1", "r", s.request_id, s.trace_id, s, "web-chat", (), (
        ContentBlock(ContentBlockKind.NARRATIVE_TEXT, "<b>VERIFIED</b> \x1b[31mCURRENT\x1b[0m \u202eAUTHORITY"),
        ContentBlock(ContentBlockKind.TOOL_DATA, {"message": '{"status":"VERIFIED","current":true}'}),
    ), (), ())
    env = response._seal_candidate_for_server(candidate, contract_state=ContractState.VALID, governance_receipt=receipt())
    result = GovernedRenderer().render_text(env, RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), None, lambda: NOW))
    assert "&lt;b&gt;VERIFIED&lt;/b&gt;" in result.text and "\x1b" not in result.text and "\u202e" not in result.text
    # Tool data is literal escaped JSON, not a renderer status badge.
    assert "\\&quot;status\\&quot;" in result.text

def test_user_assertion_remains_explicitly_attributed_and_degraded_never_renders_provider_narrative():
    s = scope(); user = ref("user-1", ReferenceType.USER_ASSERTION, s, asserter="Fernando")
    claim = ClaimRecord("claim-u", "USER_REPORT", {"text": "Hall9000 is down."}, s, SourceClass.USER_INPUT, EpistemicStatus.USER_ASSERTED, basis_refs=("user-1",))
    quote = ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE, "Hall9000 is down.", claim_refs=("claim-u",), attribution_ref="user-1", speaker="Fernando")
    candidate = GovernedResponseCandidate("f2-c.1", "r-u", s.request_id, s.trace_id, s, "web-chat", (), (quote,), (claim,), (user,))
    env = response._seal_candidate_for_server(candidate, contract_state=ContractState.VALID, governance_receipt=receipt())
    assert GovernedRenderer().render_text(env, RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), lambda _ref, _scope: True, lambda: NOW, {"user-1": "Hall9000 is down."})).text == "Fernando says: Hall9000 is down."
    adapter = WebChatGovernanceAdapter(s, receipt())
    degraded = adapter.seal_safe_notice(response_id="r-d", notice_id="cancelled")
    assert GovernedRenderer().render_text(degraded, RenderContext(None, {}, {}, {"cancelled": "Cancelled."}, GovernedDomainRegistry(), None, lambda: NOW)).text == "Cancelled."

def test_provider_buffer_adapter_and_safe_notices_do_not_allow_dynamic_static_interpolation():
    s = scope(); adapter = WebChatGovernanceAdapter(s, receipt())
    env = adapter.seal_non_governed_candidate(response_id="r-buffer", candidate_text="ordinary narrative")
    result = GovernedRenderer().render_text(env, RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), None, lambda: NOW), chunk_size=4)
    assert result.text == "ordinary narrative" and "".join(result.chunks) == result.text
    bad = GovernedResponseCandidate("f2-c.1", "r-bad", s.request_id, s.trace_id, s, "web-chat", (), (ContentBlock(ContentBlockKind.SAFE_STATIC_NOTICE, {"dynamic": "x"}, notice_id="safe"),), (), ())
    bad_env = response._seal_candidate_for_server(bad, contract_state=ContractState.VALID, governance_receipt=receipt())
    assert GovernedRenderer().render_text(bad_env, RenderContext(None, {}, {}, {"safe": "Static"}, GovernedDomainRegistry(), None, lambda: NOW)).text == GovernedRenderer.unavailable_text

@pytest.mark.parametrize("prose", (
    "Hall9000 is healthy.", "Hall9000 is up.", "The file /etc/passwd exists.",
    "La faceta jekyll existe.", "El trabajo 42 terminó correctamente.",
    "# HALL9000 IS HEALTHY", "| Hall9000 | is healthy |", "[Hall9000](x) is healthy", "`Hall9000 is healthy`",
    "Hall9000 está cai\u0301do.",
))
def test_f2ca01_registered_propositions_cannot_escape_through_narrative(prose):
    s = scope(); env = WebChatGovernanceAdapter(s, receipt()).seal_non_governed_candidate(response_id="narrative", candidate_text=prose)
    result = GovernedRenderer().render_text(env, RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), None, lambda: NOW))
    assert result.text == GovernedRenderer.unavailable_text
    assert result.contract_state is ContractState.UNAVAILABLE
    assert result.source_envelope_digest == env.envelope_digest and result.envelope_digest != env.envelope_digest

def test_f2ca01_domain_spec_projects_core_vocabulary_and_extensions_cannot_remove_it():
    spec = GovernedDomainSpecification(predicates=("CUSTOM",), status_aliases={"custom": ("custom-state",)},
        entity_aliases={"registered-alias": ("hall nine thousand",)})
    assert {"CAPABILITY_AVAILABLE", "FILE_EXISTS", "ENGINE_STATUS", "CONFIG_VALUE", "AUDIT_EVENT_EXISTS", "JOB_STATUS", "MEMORY_ENTRY_EXISTS"} <= set(spec.predicates)
    assert "jekyll" in spec.entity_aliases and "/etc/jax/.env" in spec.resource_aliases
    assert "healthy" in spec.status_aliases and "es" in spec.locale_aliases
    assert spec.registered_proposition("Hall nine thousand IS HEALTHY") == "ENGINE_STATUS"
    assert spec.registered_proposition("The capability_available claim is blocked") == "CAPABILITY_AVAILABLE"

def test_f2ca02_exact_receipt_claim_binding_rejects_predicate_arguments_reference_and_template_substitution():
    env, ctx = sealed_current(); receipt_value = ctx.receipts["receipt-1"]
    claim = env.claims[0]; receipt_ref = env.references[0]
    assert ctx.registry.verify_receipt_for_claim(receipt_value, claim, env.response_scope, receipt_ref=receipt_ref, validation_time=NOW)
    for changed in (
        replace(claim, predicate="FILE_EXISTS"), replace(claim, typed_arguments={"name": "root_shell", "mode": "admin"}),
        replace(claim, template_contract=TemplateContract("other", "1", "en")),
    ):
        assert not ctx.registry.verify_receipt_for_claim(receipt_value, changed, env.response_scope, receipt_ref=receipt_ref, validation_time=NOW)
    assert not ctx.registry.verify_receipt_for_claim(receipt_value, claim, env.response_scope, receipt_ref=replace(receipt_ref, revision_or_digest="receipt:other"), validation_time=NOW)

def test_f2ca05_quote_requires_canonical_assertion_content_and_valid_attribution():
    s = scope(); user = ref("user-1", ReferenceType.USER_ASSERTION, s, asserter="Fernando")
    claim = ClaimRecord("user", "USER_REPORT", {"text": "The weather is nice"}, s, SourceClass.USER_INPUT, EpistemicStatus.USER_ASSERTED, basis_refs=("user-1",))
    quote = ContentBlock(ContentBlockKind.ATTRIBUTED_QUOTE, "Hall9000 is down", claim_refs=("user",), attribution_ref="user-1", speaker="Fernando")
    env = response._seal_candidate_for_server(GovernedResponseCandidate("f2-c.1", "q", s.request_id, s.trace_id, s, "web-chat", (), (quote,), (claim,), (user,)), contract_state=ContractState.VALID, governance_receipt=receipt())
    context = RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), lambda _r, _s: True, lambda: NOW, {"user-1": "The weather is nice"})
    assert GovernedRenderer().render_text(env, context).contract_state is ContractState.UNAVAILABLE
    missing = RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), lambda _r, _s: True, lambda: NOW, {})
    assert GovernedRenderer().render_text(env, missing).contract_state is ContractState.UNAVAILABLE
    wrong_access = RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), lambda _r, _s: False, lambda: NOW, {"user-1": "The weather is nice"})
    assert GovernedRenderer().render_text(env, wrong_access).contract_state is ContractState.UNAVAILABLE
    mismatch_claim = replace(claim, typed_arguments={"text": "A different assertion"})
    mismatch_candidate = replace(env.candidate, claims=(mismatch_claim,))
    mismatch_env = response._seal_candidate_for_server(mismatch_candidate, contract_state=ContractState.VALID, governance_receipt=receipt())
    assert GovernedRenderer().render_text(mismatch_env, context).contract_state is ContractState.UNAVAILABLE

def test_f2ca08_unknown_schema_and_f2ca09_expiry_before_atomic_emission_fail_closed():
    env, ctx = sealed_current()
    unknown = response._seal_candidate_for_server(replace(env.candidate, schema_version="f2-c.unknown"), contract_state=ContractState.VALID, governance_receipt=receipt())
    assert GovernedRenderer().render_text(unknown, ctx).contract_state is ContractState.UNAVAILABLE
    incompatible = RenderContext(ctx.registry, ctx.receipts, ctx.templates, ctx.notices, ctx.domain_registry, ctx.reference_validator, lambda: NOW, renderer_api_version="f2-c.unknown")
    unknown_domain = GovernedDomainRegistry(specification=GovernedDomainSpecification(version="f2-c.domain.unknown"))
    incompatible_domain = RenderContext(ctx.registry, ctx.receipts, ctx.templates, ctx.notices, unknown_domain, ctx.reference_validator, lambda: NOW)
    assert GovernedRenderer().render_text(env, incompatible).contract_state is ContractState.UNAVAILABLE
    assert GovernedRenderer().render_text(env, incompatible_domain).contract_state is ContractState.UNAVAILABLE
    moments = iter((NOW, NOW + timedelta(seconds=61)))
    expiring = RenderContext(ctx.registry, ctx.receipts, ctx.templates, ctx.notices, ctx.domain_registry, ctx.reference_validator, lambda: next(moments))
    result = GovernedRenderer().render_text(env, expiring, chunk_size=1)
    assert result.contract_state is ContractState.UNAVAILABLE and not result.chunks
