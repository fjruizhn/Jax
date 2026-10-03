"""F2-D core-owned governed-output lifecycle contract.

This module is deliberately persistence and transport agnostic.  It mints a
server-only transport unit only from the *effective* F2-C rendering, and gives
the persistence/transport composition a small, fail-closed state machine.  A
database outbox owns durability and attempt lineage; it must never infer
authority from caller supplied lifecycle fields.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Mapping

from .governed_domain import GOVERNED_DOMAIN_SPEC_VERSION, GOVERNED_ENVELOPE_SCHEMA_VERSIONS, GOVERNED_RENDERER_API_VERSION
from .governed_renderer import GovernedRenderError, GovernedRenderer, RenderContext, RenderedText
from .response import ContractState, EpistemicStatus, GovernedResponseEnvelope, GovernanceContractError, _plain
from .structured_output import (
    GovernedStructuredRenderer, RenderedStructuredOutput, StructuredLayout,
    StructuredLayoutRegistry, STRUCTURED_OUTPUT_SCHEMA_VERSION,
    STRUCTURED_RENDERER_API_VERSION,
)


OUTPUT_LIFECYCLE_API_VERSION = "f2-d.lifecycle.2"
STRUCTURED_OUTPUT_LIFECYCLE_API_VERSION = "f2-d.structured.1"
STRUCTURED_WIRE_ENCODING_VERSION = "f2-d.structured-wire.1"


class OutputLifecycleError(GovernanceContractError):
    """A lifecycle or transport-authorization invariant was violated."""


class OutputLifecycleState(str, Enum):
    OUTPUT_PREPARED = "OUTPUT_PREPARED"
    TRANSPORT_COMMITTING = "TRANSPORT_COMMITTING"
    OUTPUT_COMMITTED_TO_TRANSPORT = "OUTPUT_COMMITTED_TO_TRANSPORT"
    DELIVERY_ACKNOWLEDGED = "DELIVERY_ACKNOWLEDGED"
    FAILED_BEFORE_COMMIT = "FAILED_BEFORE_COMMIT"
    CANCELLED_BEFORE_COMMIT = "CANCELLED_BEFORE_COMMIT"
    TRANSPORT_OUTCOME_UNKNOWN = "TRANSPORT_OUTCOME_UNKNOWN"


_ALLOWED = {
    OutputLifecycleState.OUTPUT_PREPARED: frozenset({
        OutputLifecycleState.TRANSPORT_COMMITTING,
        OutputLifecycleState.FAILED_BEFORE_COMMIT,
        OutputLifecycleState.CANCELLED_BEFORE_COMMIT,
    }),
    OutputLifecycleState.TRANSPORT_COMMITTING: frozenset({
        OutputLifecycleState.OUTPUT_COMMITTED_TO_TRANSPORT,
        OutputLifecycleState.TRANSPORT_OUTCOME_UNKNOWN,
    }),
}


def validate_lifecycle_version(version: str) -> None:
    if version != OUTPUT_LIFECYCLE_API_VERSION:
        raise OutputLifecycleError("unsupported F2-D lifecycle version")


def validate_structured_lifecycle_version(version: str) -> None:
    if version != STRUCTURED_OUTPUT_LIFECYCLE_API_VERSION:
        raise OutputLifecycleError("unsupported structured F2-D lifecycle version")


def validate_lifecycle_transition(
    current: OutputLifecycleState,
    target: OutputLifecycleState,
    *,
    same_identity: bool = False,
    before_send: bool = False,
) -> None:
    """Validate a single durable state transition.

    Repeating a state is allowed only for an idempotent operation on the exact
    same outbox/attempt identity.  ``TRANSPORT_COMMITTING`` may be cancelled
    only when the server-owned adapter proves no ASGI send has occurred yet.
    This integration has no authenticated client acknowledgement protocol, so
    ``DELIVERY_ACKNOWLEDGED`` is intentionally unreachable.
    """
    if not isinstance(current, OutputLifecycleState) or not isinstance(target, OutputLifecycleState):
        raise OutputLifecycleError("lifecycle states must be typed")
    if current is target:
        if same_identity:
            return
        raise OutputLifecycleError("repeated lifecycle transition requires exact identity")
    if target is OutputLifecycleState.DELIVERY_ACKNOWLEDGED:
        raise OutputLifecycleError("delivery acknowledgement has no authenticated protocol")
    if (current is OutputLifecycleState.TRANSPORT_COMMITTING
            and target is OutputLifecycleState.CANCELLED_BEFORE_COMMIT):
        if before_send:
            return
        raise OutputLifecycleError("transport commitment may be cancelled only before first ASGI send")
    if target not in _ALLOWED.get(current, frozenset()):
        raise OutputLifecycleError(f"invalid non-monotonic lifecycle transition: {current.value} -> {target.value}")


def record_delivery_acknowledgement(*_args: object, **_kwargs: object) -> None:
    """Reject synthetic acknowledgements until a real authenticated protocol exists."""
    raise OutputLifecycleError("delivery acknowledgement is unavailable for Web Chat HTTP")


def _time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OutputLifecycleError("transport validation time must be timezone-aware")
    return value


def _effective_projection(rendered: RenderedText) -> Mapping[str, object]:
    if not isinstance(rendered, RenderedText):
        raise OutputLifecycleError("transport unit requires RenderedText")
    return {
        "response_id": rendered.response_id,
        "text": rendered.text,
        "effective_output_digest": rendered.envelope_digest,
        "effective_contract_state": rendered.contract_state.value,
        "claim_ids": list(rendered.claim_ids),
        # Chunk boundaries are an implementation detail of the F2-C
        # post-seal generator.  The F2-D transport authorization binds the
        # complete semantic text and metadata; current claims have no chunks.
        "original_envelope_digest": rendered.source_envelope_digest,
        "renderer_api_version": rendered.renderer_api_version,
        "domain_spec_version": rendered.domain_spec_version,
    }


def _digest(value: Mapping[str, object]) -> str:
    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


_UNIT_TOKEN = object()


@dataclass(frozen=True, init=False)
class GovernedTransportUnit:
    """Opaque in-process authority to prepare one exact effective output.

    It is not serializable request input and its constructor is unavailable to
    callers.  Durable outbox storage persists the explicit projection/digests,
    while this object gives the server composition the trusted context needed
    to revalidate current claims at the next boundary.
    """
    envelope: GovernedResponseEnvelope
    rendered: RenderedText
    context: RenderContext
    transport_kind: str
    idempotency_key: str
    lifecycle_version: str
    effective_projection_digest: str
    original_envelope_digest: str
    response_id: str
    request_id: str
    trace_id: str
    tenant_id: str
    project_id: str
    subject_id: str
    audience: str
    scope_digest: str
    schema_version: str
    governance_reference_ids: tuple[str, ...]
    contains_current_claim: bool
    current_not_after: datetime | None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise OutputLifecycleError("GovernedTransportUnit requires server minting")

    @classmethod
    def _mint(cls, token: object, **values: object) -> "GovernedTransportUnit":
        if token is not _UNIT_TOKEN:
            raise OutputLifecycleError("GovernedTransportUnit requires server minting")
        unit = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(unit, key, value)
        return unit

    def durable_projection(self) -> Mapping[str, object]:
        """Only non-secret immutable facts required by a durable outbox row."""
        return {
            "lifecycle_version": self.lifecycle_version,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "subject_id": self.subject_id,
            "audience": self.audience,
            "scope_digest": self.scope_digest,
            "schema_version": self.schema_version,
            "governance_reference_ids": list(self.governance_reference_ids),
            "transport_kind": self.transport_kind,
            "idempotency_key": self.idempotency_key,
            "effective_output_digest": self.rendered.envelope_digest,
            "effective_projection_digest": self.effective_projection_digest,
            "effective_contract_state": self.rendered.contract_state.value,
            "original_envelope_digest": self.original_envelope_digest,
            "renderer_api_version": self.rendered.renderer_api_version,
            "domain_spec_version": self.rendered.domain_spec_version,
            "claim_ids": list(self.rendered.claim_ids),
            "contains_current_claim": self.contains_current_claim,
            "current_not_after": self.current_not_after.isoformat() if self.current_not_after else None,
        }


def _compatible(envelope: GovernedResponseEnvelope, rendered: RenderedText, context: RenderContext) -> None:
    if not isinstance(envelope, GovernedResponseEnvelope) or not isinstance(context, RenderContext):
        raise OutputLifecycleError("transport unit requires sealed envelope and trusted render context")
    if envelope.compute_digest() != envelope.envelope_digest:
        raise OutputLifecycleError("sealed envelope digest mismatch")
    if (envelope.candidate.schema_version not in GOVERNED_ENVELOPE_SCHEMA_VERSIONS
            or context.renderer_api_version != GOVERNED_RENDERER_API_VERSION
            or rendered.renderer_api_version != GOVERNED_RENDERER_API_VERSION
            or context.domain_registry.specification.version != GOVERNED_DOMAIN_SPEC_VERSION
            or rendered.domain_spec_version != GOVERNED_DOMAIN_SPEC_VERSION):
        raise OutputLifecycleError("incompatible governed output versions")
    if rendered.response_id != envelope.response_id or rendered.source_envelope_digest != envelope.envelope_digest:
        raise OutputLifecycleError("rendered output is not bound to sealed envelope")


def mint_governed_transport_unit(
    envelope: GovernedResponseEnvelope,
    rendered: RenderedText,
    context: RenderContext,
    *,
    transport_kind: str,
    idempotency_key: str,
    now: datetime | None = None,
) -> GovernedTransportUnit:
    """Mint a server-owned unit after proving the supplied F2-C projection.

    The renderer is invoked again at this durable boundary.  Thus a caller
    cannot supply a stale, foreign, or altered effective projection and current
    claims must still satisfy the F2-B receipt checks immediately before
    preparation.
    """
    if not isinstance(transport_kind, str) or not transport_kind:
        raise OutputLifecycleError("transport kind is required")
    if not isinstance(idempotency_key, str) or not idempotency_key:
        raise OutputLifecycleError("idempotency key is required")
    _compatible(envelope, rendered, context)
    validation_time = _time(now if now is not None else context.now())
    revalidation_context = replace(context, now=lambda: validation_time)
    actual = GovernedRenderer().render_text(envelope, revalidation_context)
    if _effective_projection(actual) != _effective_projection(rendered):
        raise OutputLifecycleError("effective output changed or is not the trusted renderer projection")
    claims = {claim.claim_id: claim for claim in envelope.claims}
    contains_current = any(
        claims[claim_id].epistemic_status is EpistemicStatus.CURRENT_OBSERVATION
        for claim_id in rendered.claim_ids
        if claim_id in claims
    )
    current_expiries = [
        context.receipts[claims[claim_id].resolution_receipt_ref].not_after
        for claim_id in rendered.claim_ids
        if claim_id in claims
        and claims[claim_id].epistemic_status is EpistemicStatus.CURRENT_OBSERVATION
        and claims[claim_id].resolution_receipt_ref in context.receipts
    ]
    if contains_current and not current_expiries:
        raise OutputLifecycleError("current claim has no authenticated receipt expiry")
    return GovernedTransportUnit._mint(
        _UNIT_TOKEN,
        envelope=envelope,
        rendered=rendered,
        context=context,
        transport_kind=transport_kind,
        idempotency_key=idempotency_key,
        lifecycle_version=OUTPUT_LIFECYCLE_API_VERSION,
        effective_projection_digest=_digest(_effective_projection(rendered)),
        original_envelope_digest=envelope.envelope_digest,
        response_id=envelope.response_id,
        request_id=envelope.request_id,
        trace_id=envelope.trace_id,
        tenant_id=envelope.response_scope.tenant_id,
        project_id=envelope.response_scope.project_id,
        subject_id=envelope.response_scope.subject_id,
        audience=envelope.response_scope.audience,
        scope_digest=envelope.response_scope.scope_digest,
        schema_version=envelope.schema_version,
        governance_reference_ids=tuple(sorted({
            *(reference.ref_id for reference in envelope.references),
            *envelope.issuance_authority_refs,
        })),
        contains_current_claim=contains_current,
        current_not_after=min(current_expiries) if current_expiries else None,
    )


def revalidate_for_transport(unit: GovernedTransportUnit, now: datetime) -> RenderedText:
    """Re-render at commit time and return the exact authorized projection.

    Current observations are rechecked through the F2-C/F2-B renderer.  Any
    stale receipt or semantic change fails closed; a platform must prepare a
    new governed unavailable/re-resolved response rather than mutate this one.
    """
    if not isinstance(unit, GovernedTransportUnit):
        raise OutputLifecycleError("transport authorization requires GovernedTransportUnit")
    validate_lifecycle_version(unit.lifecycle_version)
    validation_time = _time(now)
    _compatible(unit.envelope, unit.rendered, unit.context)
    actual = GovernedRenderer().render_text(unit.envelope, replace(unit.context, now=lambda: validation_time))
    if (_effective_projection(actual) != _effective_projection(unit.rendered)
            or _digest(_effective_projection(actual)) != unit.effective_projection_digest):
        raise OutputLifecycleError("prepared output is stale or differs at transport commitment")
    return actual


@dataclass(frozen=True)
class StructuredTransportMetadata:
    """Server route-contract facts that must travel with exact JSON bytes."""
    channel_id: str
    media_type: str
    outcome: str
    http_status: int | None = None
    frame_type: str | None = None
    event_type: str | None = None
    wire_encoding: str = "http-json"

    def __post_init__(self) -> None:
        for name in ("channel_id", "media_type", "outcome", "wire_encoding"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise OutputLifecycleError(f"structured transport {name} is required")
        if self.http_status is not None and (not isinstance(self.http_status, int) or not 100 <= self.http_status <= 599):
            raise OutputLifecycleError("structured HTTP status must be valid")
        for name in ("frame_type", "event_type"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise OutputLifecycleError(f"structured transport {name} must be nonempty string")
        if self.wire_encoding not in {"http-json", "websocket-text", "sse-data"}:
            raise OutputLifecycleError("unknown structured wire encoding")
        if self.wire_encoding == "websocket-text" and self.frame_type != "text":
            raise OutputLifecycleError("websocket structured output requires text frame")
        if self.wire_encoding == "sse-data" and self.frame_type is not None:
            raise OutputLifecycleError("SSE structured output cannot select websocket frame")
        if self.event_type is not None and ("\n" in self.event_type or "\r" in self.event_type or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.event_type)):
            raise OutputLifecycleError("structured event type is unsafe")

    def projection(self) -> Mapping[str, object]:
        return {"channel_id": self.channel_id, "media_type": self.media_type,
            "outcome": self.outcome, "http_status": self.http_status,
            "frame_type": self.frame_type, "event_type": self.event_type,
            "wire_encoding": self.wire_encoding,
            "wire_encoding_version": STRUCTURED_WIRE_ENCODING_VERSION}

    @property
    def digest(self) -> str:
        return _digest(self.projection())


@dataclass(frozen=True, init=False)
class StructuredGovernedTransportUnit:
    """Opaque F2-D authority for one exact structured body and metadata tuple."""
    envelope: GovernedResponseEnvelope
    rendered: RenderedStructuredOutput
    context: RenderContext
    layout: StructuredLayout
    layout_registry: StructuredLayoutRegistry
    metadata: StructuredTransportMetadata
    idempotency_key: str
    lifecycle_version: str
    canonical_bytes_digest: str
    wire_bytes: bytes
    wire_payload_digest: str
    effective_projection_digest: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise OutputLifecycleError("StructuredGovernedTransportUnit requires server minting")

    @classmethod
    def _mint(cls, token: object, **values: object) -> "StructuredGovernedTransportUnit":
        if token is not _UNIT_TOKEN:
            raise OutputLifecycleError("StructuredGovernedTransportUnit requires server minting")
        unit = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(unit, key, value)
        return unit

    def durable_projection(self) -> Mapping[str, object]:
        return {"lifecycle_version": self.lifecycle_version,
            "response_id": self.rendered.response_id,
            "source_envelope_digest": self.rendered.source_envelope_digest,
            "effective_output_digest": self.rendered.output_digest,
            "canonical_bytes_digest": self.canonical_bytes_digest,
            "wire_payload_digest": self.wire_payload_digest,
            "effective_projection_digest": self.effective_projection_digest,
            "layout_id": self.rendered.layout_id, "layout_digest": self.rendered.layout_digest,
            "channel_id": self.rendered.channel_id,
            "transport_metadata": self.metadata.projection(),
            "transport_metadata_digest": self.metadata.digest,
            "renderer_api_version": self.rendered.renderer_api_version,
            "domain_spec_version": self.rendered.domain_spec_version,
            "structured_schema_version": self.rendered.structured_schema_version,
            "claim_ids": list(self.rendered.claim_ids),
            "protected_subtrees_digest": self.rendered.protected_subtrees_digest,
            "origin_manifest_digest": self.rendered.origin_manifest_digest,
            "idempotency_key": self.idempotency_key}


def _structured_projection(rendered: RenderedStructuredOutput,
                           metadata: StructuredTransportMetadata) -> Mapping[str, object]:
    return {"response_id": rendered.response_id, "source_envelope_digest": rendered.source_envelope_digest,
        "effective_output_digest": rendered.output_digest,
        "canonical_bytes_digest": "sha256:" + hashlib.sha256(rendered.canonical_bytes).hexdigest(),
        "layout_id": rendered.layout_id, "layout_digest": rendered.layout_digest,
        "channel_id": rendered.channel_id, "metadata_digest": metadata.digest,
        "claim_ids": list(rendered.claim_ids),
        "protected_subtrees_digest": rendered.protected_subtrees_digest,
        "origin_manifest_digest": rendered.origin_manifest_digest,
        "renderer_api_version": rendered.renderer_api_version,
        "domain_spec_version": rendered.domain_spec_version,
        "structured_schema_version": rendered.structured_schema_version}


def _structured_wire_bytes(rendered: RenderedStructuredOutput, metadata: StructuredTransportMetadata) -> bytes:
    """Only server metadata may choose one closed framing contract."""
    payload = rendered.canonical_bytes
    if metadata.wire_encoding in {"http-json", "websocket-text"}:
        return payload
    event = b"" if metadata.event_type is None else b"event:" + metadata.event_type.encode("ascii") + b"\n"
    return event + b"data:" + payload + b"\n\n"


def mint_structured_governed_transport_unit(
    envelope: GovernedResponseEnvelope, rendered: RenderedStructuredOutput,
    context: RenderContext, layout: StructuredLayout, layout_registry: StructuredLayoutRegistry,
    *, metadata: StructuredTransportMetadata, idempotency_key: str,
    now: datetime | None = None,
) -> StructuredGovernedTransportUnit:
    if not isinstance(rendered, RenderedStructuredOutput) or not isinstance(metadata, StructuredTransportMetadata):
        raise OutputLifecycleError("structured transport requires typed output and metadata")
    if not isinstance(idempotency_key, str) or not idempotency_key:
        raise OutputLifecycleError("structured idempotency key is required")
    if (rendered.renderer_api_version != STRUCTURED_RENDERER_API_VERSION
            or rendered.structured_schema_version != STRUCTURED_OUTPUT_SCHEMA_VERSION
            or rendered.channel_id != metadata.channel_id
            or rendered.source_envelope_digest != envelope.envelope_digest):
        raise OutputLifecycleError("incompatible structured output tuple")
    validation_time = _time(now if now is not None else context.now())
    actual = GovernedStructuredRenderer().render(envelope, replace(context, now=lambda: validation_time), layout, layout_registry)
    if (_structured_projection(actual, metadata) != _structured_projection(rendered, metadata)
            or actual.canonical_bytes != rendered.canonical_bytes):
        raise OutputLifecycleError("structured output differs from trusted renderer projection")
    projection = _structured_projection(rendered, metadata)
    wire = _structured_wire_bytes(rendered, metadata)
    return StructuredGovernedTransportUnit._mint(_UNIT_TOKEN, envelope=envelope, rendered=rendered,
        context=context, layout=layout, layout_registry=layout_registry, metadata=metadata,
        idempotency_key=idempotency_key, lifecycle_version=STRUCTURED_OUTPUT_LIFECYCLE_API_VERSION,
        canonical_bytes_digest="sha256:" + hashlib.sha256(rendered.canonical_bytes).hexdigest(),
        wire_bytes=wire, wire_payload_digest="sha256:" + hashlib.sha256(wire).hexdigest(),
        effective_projection_digest=_digest(projection))


def revalidate_structured_for_transport(unit: StructuredGovernedTransportUnit, now: datetime) -> RenderedStructuredOutput:
    if not isinstance(unit, StructuredGovernedTransportUnit):
        raise OutputLifecycleError("structured transport requires server minted unit")
    validate_structured_lifecycle_version(unit.lifecycle_version)
    validation_time = _time(now)
    actual = GovernedStructuredRenderer().render(unit.envelope, replace(unit.context, now=lambda: validation_time),
        unit.layout, unit.layout_registry)
    wire = _structured_wire_bytes(actual, unit.metadata)
    if (actual.canonical_bytes != unit.rendered.canonical_bytes
            or _structured_projection(actual, unit.metadata) != _structured_projection(unit.rendered, unit.metadata)
            or _digest(_structured_projection(actual, unit.metadata)) != unit.effective_projection_digest
            or wire != unit.wire_bytes
            or "sha256:" + hashlib.sha256(wire).hexdigest() != unit.wire_payload_digest):
        raise OutputLifecycleError("structured output changed before transport commitment")
    return actual
