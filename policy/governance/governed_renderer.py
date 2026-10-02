"""F2-C side-effect-free final rendering boundary for Web Chat.

This module deliberately does not perform transport, provider dispatch, or
durable output lifecycle work.  It turns a *sealed* F2-A envelope into a
channel-safe text projection after rechecking F2-B receipts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html
import json
import re
import unicodedata
from types import MappingProxyType
from typing import Callable, Mapping

from .response import (
    ClaimDisposition, ContentBlockKind, ContractState, EpistemicStatus,
    GovernedResponseCandidate, GovernedResponseEnvelope, GovernanceContractError,
    GovernanceReceipt, ResponseScope, ContentBlock, _freeze, _plain,
    _seal_candidate_for_server, _text,
)
from .resolution import GovernedResolutionReceipt, ReferenceLookupRecord, ResolverRegistry
from .governed_domain import (GOVERNED_DOMAIN_SPEC_VERSION, GOVERNED_ENVELOPE_SCHEMA_VERSIONS,
    GOVERNED_RENDERER_API_VERSION, GovernedDomainSpecification)


class GovernedRenderError(GovernanceContractError):
    """The candidate/envelope is not permitted to become visible."""


@dataclass(frozen=True)
class GovernedDomainRegistry:
    """Compatibility wrapper around the canonical JAX domain specification."""
    phrases: Mapping[str, str] = field(default_factory=dict)
    specification: GovernedDomainSpecification = field(default_factory=GovernedDomainSpecification)

    def __post_init__(self) -> None:
        if not isinstance(self.phrases, Mapping) or not isinstance(self.specification, GovernedDomainSpecification):
            raise GovernanceContractError("governed phrases must be a mapping")
        if self.phrases:
            raise GovernanceContractError("legacy phrase registries are unsupported; supply canonical GovernedDomainSpecification")
        object.__setattr__(self, "phrases", MappingProxyType({}))

    def hit(self, text: str) -> str | None:
        hit = self.specification.registered_proposition(text)
        if hit is not None:
            return hit
        return None


@dataclass(frozen=True)
class RenderedText:
    """Structural projection for Web Chat; ``text`` contains escaped payload."""
    text: str
    response_id: str
    envelope_digest: str
    contract_state: ContractState
    claim_ids: tuple[str, ...] = ()
    chunks: tuple[str, ...] = ()
    source_envelope_digest: str = ""
    renderer_api_version: str = GOVERNED_RENDERER_API_VERSION
    domain_spec_version: str = GOVERNED_DOMAIN_SPEC_VERSION


@dataclass(frozen=True)
class RenderContext:
    """Server-owned dependencies; no user/provider payload supplies these."""
    registry: ResolverRegistry | None
    receipts: Mapping[str, GovernedResolutionReceipt] = field(default_factory=dict)
    templates: Mapping[tuple[str, str, str], str] = field(default_factory=dict)
    notices: Mapping[str, str] = field(default_factory=dict)
    domain_registry: GovernedDomainRegistry = field(default_factory=GovernedDomainRegistry)
    # The host supplies F2-B dereference/access validation.  It is optional
    # only for envelopes without claims; a claim with basis/receipt references
    # fails closed if no server validator is composed.
    reference_validator: Callable[[object, ResponseScope], bool] | None = None
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    # Canonical immutable USER_ASSERTED text, fetched by trusted composition
    # after reference access/revision validation.  Provider text is never it.
    user_assertion_content: Mapping[str, str] = field(default_factory=dict)
    renderer_api_version: str = GOVERNED_RENDERER_API_VERSION
    receipt_reference_resolver: Callable[[object, ResponseScope], ReferenceLookupRecord | None] | None = None

    def __post_init__(self) -> None:
        if self.registry is not None and not isinstance(self.registry, ResolverRegistry):
            raise GovernanceContractError("render registry must be ResolverRegistry or None")
        if not isinstance(self.receipts, Mapping) or not all(isinstance(k, str) and isinstance(v, GovernedResolutionReceipt) for k, v in self.receipts.items()):
            raise GovernanceContractError("render receipts must be typed mapping")
        for name, values in (("templates", self.templates), ("notices", self.notices)):
            if not isinstance(values, Mapping) or not all(isinstance(v, str) for v in values.values()):
                raise GovernanceContractError(f"{name} must be string mapping")
        if not isinstance(self.domain_registry, GovernedDomainRegistry) or not callable(self.now):
            raise GovernanceContractError("invalid server render context")
        if not isinstance(self.renderer_api_version, str):
            raise GovernanceContractError("renderer API version must be string")
        if not isinstance(self.user_assertion_content, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in self.user_assertion_content.items()):
            raise GovernanceContractError("user assertion content must be string mapping")
        if self.reference_validator is not None and not callable(self.reference_validator):
            raise GovernanceContractError("reference_validator must be server callable or None")
        if self.receipt_reference_resolver is not None and not callable(self.receipt_reference_resolver):
            raise GovernanceContractError("receipt_reference_resolver must be server callable or None")
        object.__setattr__(self, "receipts", MappingProxyType(dict(self.receipts)))
        object.__setattr__(self, "templates", MappingProxyType(dict(self.templates)))
        object.__setattr__(self, "notices", MappingProxyType(dict(self.notices)))
        object.__setattr__(self, "user_assertion_content", MappingProxyType(dict(self.user_assertion_content)))


def _safe_payload(text: str) -> str:
    """Literal text projection: strip terminal/control spoofing and escape markup."""
    if not isinstance(text, str):
        raise GovernedRenderError("text payload must be string")
    # Bidirectional/zero-width controls and ANSI ESC cannot establish renderer
    # metadata.  Remaining markup is literal HTML, not executable Markdown.
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = "".join(c for c in text if unicodedata.category(c) not in {"Cf", "Cc"} or c in "\n\t")
    return html.escape(text, quote=True)


def _format(template: str, arguments: Mapping[str, object]) -> str:
    try:
        # Arguments are typed/frozen F2-A values.  They are still escaped as
        # untrusted values; the template alone is server-owned wording.
        values = {k: _safe_payload(str(v)) for k, v in arguments.items()}
        return template.format_map(values)
    except (KeyError, ValueError) as exc:
        raise GovernedRenderError("server template contract cannot render claim") from exc


class GovernedRenderer:
    """The only F2-C Web Chat text renderer.

    It accepts neither candidates nor raw provider strings.  A blocked
    governed proposition becomes a server-owned unavailable notice rather
    than being rendered as provider prose.
    """
    unavailable_text = "I could not verify the current state."
    cancelled_text = "The response was cancelled before it could be verified."

    def render_text(self, envelope: GovernedResponseEnvelope, context: RenderContext, *, chunk_size: int = 0) -> RenderedText:
        if not isinstance(envelope, GovernedResponseEnvelope):
            raise GovernedRenderError("renderer accepts only sealed GovernedResponseEnvelope")
        if envelope.compute_digest() != envelope.envelope_digest:
            raise GovernedRenderError("sealed envelope digest mismatch")
        if not isinstance(context, RenderContext):
            raise GovernedRenderError("renderer requires server RenderContext")
        if (envelope.candidate.schema_version not in GOVERNED_ENVELOPE_SCHEMA_VERSIONS
                or context.renderer_api_version != GOVERNED_RENDERER_API_VERSION
                or context.domain_registry.specification.version != GOVERNED_DOMAIN_SPEC_VERSION):
            return self._safe(envelope, self.unavailable_text)
        now = context.now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise GovernedRenderError("server renderer clock must be timezone-aware")
        if envelope.contract_state in {ContractState.BLOCKED_SYSTEM_CLAIM, ContractState.UNAVAILABLE}:
            return self._safe(envelope, self.unavailable_text)
        claims = {claim.claim_id: claim for claim in envelope.claims}
        fragments: list[str] = []
        shown: list[str] = []
        for block in envelope.content_blocks:
            if block.kind is ContentBlockKind.NARRATIVE_TEXT:
                hit = context.domain_registry.hit(block.payload)
                if hit is not None:
                    return self._safe(envelope, self.unavailable_text)
                fragments.append(_safe_payload(block.payload))
            elif block.kind is ContentBlockKind.CLAIM_REF_BLOCK:
                for claim_id in block.claim_refs:
                    claim = claims.get(claim_id)
                    if claim is None or claim.disposition is not ClaimDisposition.ASSERTABLE:
                        return self._safe(envelope, self.unavailable_text)
                    try:
                        self._validate_claim(claim, envelope.response_scope, context, now, {r.ref_id: r for r in envelope.references})
                        fragments.append(self._render_claim(claim, context))
                    except GovernedRenderError:
                        # A structurally separable governed proposition is
                        # replaced by a server-owned unavailable form; no
                        # provider text becomes a fallback current assertion.
                        return self._safe(envelope, self.unavailable_text)
                    shown.append(claim_id)
            elif block.kind is ContentBlockKind.ATTRIBUTED_QUOTE:
                claim = claims.get(block.claim_refs[0])
                if not self._validate_quote(claim, block, envelope.response_scope, context, {r.ref_id: r for r in envelope.references}):
                    return self._safe(envelope, self.unavailable_text)
                fragments.append(f"{_safe_payload(block.speaker or '')} says: {_safe_payload(block.payload)}")
                shown.append(claim.claim_id)
            elif block.kind is ContentBlockKind.TOOL_DATA:
                if context.domain_registry.specification.runtime_status_tool_data_predicate(_plain(block.payload)) is not None:
                    return self._safe(envelope, self.unavailable_text)
                serialized_tool_data = json.dumps(_plain(block.payload), ensure_ascii=False, sort_keys=True)
                if context.domain_registry.hit(serialized_tool_data) is not None:
                    return self._safe(envelope, self.unavailable_text)
                fragments.append(_safe_payload(serialized_tool_data))
            elif block.kind in {ContentBlockKind.SAFE_STATIC_NOTICE, ContentBlockKind.ERROR_NOTICE}:
                notice = context.notices.get(block.notice_id or "")
                if notice is None:
                    return self._safe(envelope, self.unavailable_text)
                # Notice parameters are never a template source; keeping them
                # out prevents dynamic SAFE_STATIC_TEXT interpolation.
                if block.payload:
                    return self._safe(envelope, self.unavailable_text)
                fragments.append(notice)
            else:  # defensive for future enum additions
                return self._safe(envelope, self.unavailable_text)
        text = "\n".join(fragments)
        # Current observations must cross the display boundary atomically.  A
        # second validation occurs immediately before returning the unit.
        if any(claims[x].epistemic_status is EpistemicStatus.CURRENT_OBSERVATION for x in shown):
            final_now = context.now()
            try:
                for claim_id in shown:
                    claim = claims[claim_id]
                    if claim.epistemic_status is EpistemicStatus.CURRENT_OBSERVATION:
                        self._validate_claim(claim, envelope.response_scope, context, final_now, {r.ref_id: r for r in envelope.references})
            except GovernedRenderError:
                return self._safe(envelope, self.unavailable_text)
            chunks = ()
        else:
            chunks = self._chunks(text, chunk_size)
        return self._effective(text, envelope, envelope.contract_state, tuple(shown), chunks)

    def _validate_claim(self, claim, scope: ResponseScope, context: RenderContext, now: datetime, references: Mapping[str, object]) -> None:
        if claim.claim_scope.scope_digest != scope.scope_digest:
            raise GovernedRenderError("claim scope mismatch")
        required_refs = tuple(claim.basis_refs) + ((claim.resolution_receipt_ref,) if claim.resolution_receipt_ref else ())
        if required_refs:
            if context.reference_validator is None:
                raise GovernedRenderError("claim references require server dereference validation")
            for ref_id in required_refs:
                ref = references.get(ref_id)
                if ref is None or not context.reference_validator(ref, scope):
                    raise GovernedRenderError("claim reference is not valid for rendering")
        if claim.epistemic_status is EpistemicStatus.CURRENT_OBSERVATION:
            receipt = context.receipts.get(claim.resolution_receipt_ref or "")
            receipt_ref = references.get(claim.resolution_receipt_ref or "")
            trusted_reference = (context.receipt_reference_resolver(receipt_ref, scope)
                if context.receipt_reference_resolver is not None and receipt_ref is not None else None)
            if (context.registry is None or receipt is None or receipt_ref is None or trusted_reference is None
                    or not context.registry.verify_receipt_for_claim(receipt, claim, scope,
                        receipt_ref=receipt_ref, reference_lookup=trusted_reference, validation_time=now)):
                raise GovernedRenderError("current claim receipt is invalid at render time")
        elif claim.epistemic_status in {EpistemicStatus.MEMORY_DERIVED, EpistemicStatus.MODEL_KNOWLEDGE, EpistemicStatus.INFERRED}:
            # These statuses can be rendered only from non-system typed claims.
            # A registered exact governed predicate is never narrative fallback.
            if claim.predicate in context.domain_registry.specification.predicates:
                raise GovernedRenderError("non-current source cannot assert governed predicate")

    def _render_claim(self, claim, context: RenderContext) -> str:
        contract = claim.template_contract
        if contract is None:
            raise GovernedRenderError("assertable claim needs template")
        key = (contract.template_id, contract.template_version, contract.locale)
        template = context.templates.get(key)
        if template is None:
            raise GovernedRenderError("claim template is not server-owned")
        return _format(template, claim.typed_arguments)

    def _validate_quote(self, claim, block, scope, context, references) -> bool:
        if claim is None or claim.epistemic_status is not EpistemicStatus.USER_ASSERTED:
            return False
        ref = references.get(block.attribution_ref or "")
        if ref is None or block.attribution_ref not in claim.basis_refs or ref.asserter_id != block.speaker:
            return False
        if context.reference_validator is None or not context.reference_validator(ref, scope):
            return False
        canonical = context.user_assertion_content.get(ref.ref_id)
        if canonical is None:
            return False
        canonical = _text(canonical, "canonical assertion")
        claimed = claim.typed_arguments.get("text")
        return (isinstance(claimed, str)
            and _text(claimed, "claimed assertion") == canonical
            and canonical == _text(block.payload, "quote payload"))

    def _effective(self, text: str, envelope: GovernedResponseEnvelope, state: ContractState, claim_ids=(), chunks=()) -> RenderedText:
        digest = "sha256:" + hashlib.sha256((state.value + "\x00" + text).encode("utf-8")).hexdigest()
        return RenderedText(text, envelope.response_id, digest, state, tuple(claim_ids), tuple(chunks), envelope.envelope_digest)

    def _safe(self, envelope: GovernedResponseEnvelope, text: str) -> RenderedText:
        return self._effective(text, envelope, ContractState.UNAVAILABLE)

    @staticmethod
    def _chunks(text: str, chunk_size: int) -> tuple[str, ...]:
        if not isinstance(chunk_size, int) or chunk_size < 0:
            raise GovernedRenderError("chunk_size must be nonnegative int")
        if not chunk_size:
            return ()
        return tuple(text[i:i + chunk_size] for i in range(0, len(text), chunk_size))


@dataclass(frozen=True)
class WebChatGovernanceAdapter:
    """Stable server-side bridge: buffers candidate text then seals a response.

    This is not provider streaming.  Callers may pass only complete candidate
    text; rendering remains a separate sealed-envelope operation.
    """
    scope: ResponseScope
    governance_receipt: GovernanceReceipt
    schema_version: str = "f2-c.1"
    producer: str = "web-chat"

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResponseScope) or not isinstance(self.governance_receipt, GovernanceReceipt):
            raise GovernanceContractError("WebChatGovernanceAdapter requires typed scope and receipt")
        if self.producer != self.scope.component_id:
            raise GovernanceContractError("Web Chat producer must match scope component")

    def seal_non_governed_candidate(self, *, response_id: str, candidate_text: str) -> GovernedResponseEnvelope:
        candidate = GovernedResponseCandidate(
            self.schema_version, response_id, self.scope.request_id, self.scope.trace_id,
            self.scope, self.producer, (),
            (ContentBlock(ContentBlockKind.NARRATIVE_TEXT, candidate_text),), (), (),
        )
        return _seal_candidate_for_server(candidate, contract_state=ContractState.VALID, governance_receipt=self.governance_receipt)

    def seal_safe_notice(self, *, response_id: str, notice_id: str, contract_state: ContractState = ContractState.DEGRADED_STRUCTURED) -> GovernedResponseEnvelope:
        candidate = GovernedResponseCandidate(
            self.schema_version, response_id, self.scope.request_id, self.scope.trace_id,
            self.scope, self.producer, (),
            (ContentBlock(ContentBlockKind.ERROR_NOTICE, {}, notice_id=notice_id),), (), (),
        )
        return _seal_candidate_for_server(candidate, contract_state=contract_state, governance_receipt=self.governance_receipt)
