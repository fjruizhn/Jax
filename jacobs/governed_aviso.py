"""Trusted F2-A/B/C composition for the Jacobs completion notice channel."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
import secrets
import uuid
from pathlib import Path

from policy.governance.governed_domain import GOVERNED_RENDERER_API_VERSION
from policy.governance.governed_renderer import GovernedDomainRegistry, RenderContext
from policy.governance.loaders import load_templates
from policy.governance.resolution import (
    AdapterKind,
    ConflictPolicy,
    PredicateAuthorityBinding,
    ReceiptAuthenticator,
    ReferenceLookupRecord,
    RegistryEntry,
    ResolutionStatus,
    ScopeRule,
    SourceScopeClass,
    TrustedAdapterRegistration,
    _build_approved_registry_for_server,
)
from policy.governance.response import (
    ClaimDisposition,
    ClaimRecord,
    ContentBlock,
    ContentBlockKind,
    ContractState,
    EpistemicStatus,
    GovernanceContractError,
    GovernanceReceipt,
    GovernedResponseCandidate,
    ReferenceRef,
    ReferenceType,
    ResponseScope,
    SourceClass,
    TemplateContract,
    TemporalClass,
    ExistenceState,
    _seal_candidate_for_server,
)
from policy.governance.runtime_status import (
    JacobsPipelineStatusResolver,
    _jacobs_source_configuration,
    runtime_status_source_configuration_digest,
)


_PREDICATE = "PIPELINE_STATUS"
_BINDING_VERSION = "f2-e.runtime-status.3"
_RESOLVER_ID = "policy.governance.runtime_status:JacobsPipelineStatusResolver"
_RESOLVER_VERSION = "f2-e.runtime-status-resolver.3"
_TEMPLATE_ID = "PIPELINE_STATUS"
_LOCALE = "es"
_AUTHENTICATOR = ReceiptAuthenticator(
    secrets.token_bytes(32), key_id="jacobs-notice-process:" + secrets.token_hex(8),
)


def _registry(scope: ResponseScope):
    source_config = _jacobs_source_configuration(require_config=True)
    config_digest = runtime_status_source_configuration_digest(_PREDICATE, source_config)
    rule = ScopeRule(scope.environment, scope.tenant_id, scope.project_id,
        scope.subject_id, scope.actor_id, scope.audience, scope.component_id)
    binding = PredicateAuthorityBinding(
        _PREDICATE, _BINDING_VERSION, "jacobs:canonical-store", "authority:jacobs",
        scope.environment, rule, rule, 60, ConflictPolicy.SINGLE_SOURCE_REQUIRED,
        _RESOLVER_ID, _RESOLVER_VERSION, config_digest, _BINDING_VERSION,
        source_scope_class=SourceScopeClass.EXACT_RESPONSE_SCOPE,
    )
    adapter = TrustedAdapterRegistration(
        AdapterKind.JACOBS_PIPELINE_STATUS, _RESOLVER_ID, _RESOLVER_VERSION,
        "jacobs:canonical-store", config_digest,
    )
    entry = RegistryEntry(binding, adapter, ("pipeline_id", "status"),
        f"{_PREDICATE}@{_BINDING_VERSION}:{_LOCALE}")
    return _build_approved_registry_for_server((entry,), authenticator=_AUTHENTICATOR)


def _governance_receipt(registry) -> GovernanceReceipt:
    policy_dir = Path(__file__).resolve().parents[1] / "policy"
    version_path = policy_dir / "VERSION"
    version_text = version_path.read_text(encoding="utf-8")
    policy_hash = next((line.split(":", 1)[1].strip() for line in version_text.splitlines()
        if line.startswith("sha256:")), None)
    if not policy_hash:
        raise GovernanceContractError("policy version identity unavailable")
    vocabulary_hash = hashlib.sha256(
        (policy_dir / "vocabulary" / "closed_vocabulary.yaml").read_bytes()
    ).hexdigest()
    validator_hash = hashlib.sha256(
        (policy_dir / "governance" / "validator.py").read_bytes()
    ).hexdigest()
    return GovernanceReceipt(
        "sha256:" + policy_hash,
        "sha256:" + vocabulary_hash,
        registry.snapshot_digest,
        "sha256:" + validator_hash,
        GOVERNED_RENDERER_API_VERSION,
    )


async def compose_pipeline_notice(*, pipeline_id: str, requested_status: str):
    """Build a sealed claim from the canonical Jacobs source, never the caller's status."""
    from jacobs import store

    identity = await store.pipeline_get(pipeline_id)
    if identity is None or not identity.tenant_id or not identity.user_id:
        raise GovernanceContractError("Jacobs pipeline identity is unavailable")
    environment = os.environ.get("JAX_GOVERNANCE_ENVIRONMENT", "production")
    request_id, trace_id, response_id = (str(uuid.uuid4()) for _ in range(3))
    scope = ResponseScope(
        environment, identity.tenant_id, None, identity.user_id, "service:jacobs",
        "operator", "jacobs-aviso", request_id, trace_id,
    )
    arguments = {"pipeline_id": pipeline_id, "status": requested_status}
    registry = _registry(scope)
    evidence = await JacobsPipelineStatusResolver().evidence(arguments, scope)
    resolved = registry.resolve(_PREDICATE, arguments, scope,
        validation_time=datetime.now(timezone.utc), runtime_status_evidence=evidence)
    receipt_ref_id = "pipeline-status-receipt:" + response_id
    receipt_ref = ReferenceRef(
        receipt_ref_id, ReferenceType.RESOLUTION_RECEIPT,
        "axioma://runtime-resolution/PIPELINE_STATUS/" + response_id,
        "f2-b-resolution:" + resolved.receipt_id, resolved.receipt_id,
        scope.scope_digest, TemporalClass.CURRENT, ExistenceState.PRESENT,
    )
    claim = ClaimRecord(
        "pipeline-status:" + response_id, _PREDICATE, arguments, scope,
        SourceClass.CURRENT_SOURCE, EpistemicStatus.CURRENT_OBSERVATION,
        resolution_receipt_ref=receipt_ref_id,
        disposition=ClaimDisposition.ASSERTABLE,
        template_contract=TemplateContract(_TEMPLATE_ID, _BINDING_VERSION, _LOCALE),
    )
    candidate = GovernedResponseCandidate(
        "f2-c.1", response_id, request_id, trace_id, scope, "jacobs-aviso", (),
        (ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK, claim_refs=(claim.claim_id,)),),
        (claim,), (receipt_ref,),
    )
    envelope = _seal_candidate_for_server(
        candidate, contract_state=ContractState.VALID,
        governance_receipt=_governance_receipt(registry),
    )
    receipt_lookup = ReferenceLookupRecord(
        receipt_ref.ref_id, receipt_ref.ref_type, receipt_ref.canonical_locator,
        receipt_ref.immutable_identity, receipt_ref.revision_or_digest,
        receipt_ref.scope_digest, receipt_ref.temporal_class,
        receipt_ref.existence_state, True,
    )
    templates = load_templates()
    template = templates.get(_TEMPLATE_ID)
    if template is None or template.status != "definida" or not template.template:
        raise GovernanceContractError("PIPELINE_STATUS render template unavailable")
    context = RenderContext(
        registry=registry,
        receipts={receipt_ref_id: resolved},
        templates={(_TEMPLATE_ID, _BINDING_VERSION, _LOCALE): template.template},
        domain_registry=GovernedDomainRegistry(),
        reference_validator=lambda ref, actual_scope: (
            ref == receipt_ref and actual_scope.scope_digest == scope.scope_digest
        ),
        now=lambda: datetime.now(timezone.utc),
        receipt_reference_resolver=lambda ref, actual_scope: (
            receipt_lookup if ref == receipt_ref and actual_scope.scope_digest == scope.scope_digest else None
        ),
    )
    return envelope, context
