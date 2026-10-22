"""Additive F2-D exact-byte lifecycle binding for structured projections."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import Mapping

from .governed_renderer import RenderContext
from .output_lifecycle import (
    OutputLifecycleError,
    OutputLifecycleState,
    validate_lifecycle_transition,
)
from .response import GovernanceContractError, GovernedResponseEnvelope
from .structured_projection import (
    GovernedStructuredProjection,
    GovernedStructuredRenderer,
    PIPELINE_LIST_SCHEMA_VERSION,
    StructuredProjectionError,
)


STRUCTURED_BYTES_LIFECYCLE_API_VERSION = "f2-d.structured-bytes.1"
_UNIT_TOKEN = object()


class StructuredOutputLifecycleError(GovernanceContractError):
    """The structured bytes or lifecycle cannot be trusted at transport."""


class StructuredOutputChannelId(str, Enum):
    PIPELINE_LIST_HTTP_JSON_V1 = "pipeline.list.http-json.v1"


@dataclass(frozen=True)
class StructuredOutputChannel:
    channel_id: StructuredOutputChannelId
    projection_schema_version: str
    transport_kind: str
    destination: str
    supports_authenticated_ack: bool


_PIPELINE_LIST_CHANNEL = StructuredOutputChannel(
    StructuredOutputChannelId.PIPELINE_LIST_HTTP_JSON_V1,
    PIPELINE_LIST_SCHEMA_VERSION,
    "http-json-response",
    "authenticated_user",
    False,
)
STRUCTURED_OUTPUT_CHANNELS: Mapping[StructuredOutputChannelId, StructuredOutputChannel] = MappingProxyType(
    {_PIPELINE_LIST_CHANNEL.channel_id: _PIPELINE_LIST_CHANNEL})


@dataclass(frozen=True, init=False)
class StructuredTransportUnit:
    envelope: GovernedResponseEnvelope
    context: RenderContext
    projection: GovernedStructuredProjection
    channel: StructuredOutputChannel
    canonical_bytes: bytes
    bytes_digest: str
    idempotency_key: str
    api_version: str
    response_id: str
    tenant_id: str
    subject_id: str | None
    scope_digest: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuredOutputLifecycleError("structured transport unit requires server minting")

    @classmethod
    def _mint(cls, token: object, **values: object) -> "StructuredTransportUnit":
        if token is not _UNIT_TOKEN:
            raise StructuredOutputLifecycleError("structured transport unit requires server minting")
        unit = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(unit, key, value)
        return unit


@dataclass(frozen=True)
class PreparedStructuredOutput:
    channel_id: StructuredOutputChannelId
    transport_unit: StructuredTransportUnit
    canonical_bytes: bytes
    state: OutputLifecycleState = OutputLifecycleState.OUTPUT_PREPARED


def _channel(channel_id: StructuredOutputChannelId) -> StructuredOutputChannel:
    if not isinstance(channel_id, StructuredOutputChannelId):
        raise StructuredOutputLifecycleError("caller supplied an unknown structured output channel")
    result = STRUCTURED_OUTPUT_CHANNELS.get(channel_id)
    if result is None:
        raise StructuredOutputLifecycleError("structured output channel is not registered")
    return result


def mint_structured_transport_unit(
    envelope: GovernedResponseEnvelope,
    projection: GovernedStructuredProjection,
    context: RenderContext,
    *,
    channel_id: StructuredOutputChannelId = StructuredOutputChannelId.PIPELINE_LIST_HTTP_JSON_V1,
    idempotency_key: str,
    now: datetime,
) -> StructuredTransportUnit:
    channel = _channel(channel_id)
    if not isinstance(envelope, GovernedResponseEnvelope) or not isinstance(context, RenderContext):
        raise StructuredOutputLifecycleError("F2-D requires a sealed envelope and server render context")
    if not isinstance(projection, GovernedStructuredProjection):
        raise StructuredOutputLifecycleError("F2-D requires an F2-C structured projection")
    if projection.schema_version != channel.projection_schema_version:
        raise StructuredOutputLifecycleError("projection schema does not match closed channel")
    if not isinstance(idempotency_key, str) or not idempotency_key:
        raise StructuredOutputLifecycleError("server idempotency key is required")
    if envelope.compute_digest() != envelope.envelope_digest:
        raise StructuredOutputLifecycleError("sealed envelope digest mismatch")
    if projection.envelope_digest != envelope.envelope_digest:
        raise StructuredOutputLifecycleError("F2-C projection belongs to a different envelope")
    if not isinstance(projection.canonical_bytes, bytes):
        raise StructuredOutputLifecycleError("canonical output must be immutable bytes")
    expected_digest = "sha256:" + hashlib.sha256(projection.canonical_bytes).hexdigest()
    if projection.bytes_digest != expected_digest:
        raise StructuredOutputLifecycleError("F2-C canonical byte digest mismatch")
    try:
        GovernedStructuredRenderer().revalidate_claims_for_transport(
            envelope, context, projection, now)
    except (StructuredProjectionError, GovernanceContractError, TypeError, ValueError) as exc:
        raise StructuredOutputLifecycleError("F2-B claim validation failed at preparation") from exc
    return StructuredTransportUnit._mint(_UNIT_TOKEN,
        envelope=envelope,
        context=context,
        projection=projection,
        channel=channel,
        canonical_bytes=projection.canonical_bytes,
        bytes_digest=expected_digest,
        idempotency_key=idempotency_key,
        api_version=STRUCTURED_BYTES_LIFECYCLE_API_VERSION,
        response_id=envelope.response_id,
        tenant_id=envelope.response_scope.tenant_id,
        subject_id=envelope.response_scope.subject_id,
        scope_digest=envelope.response_scope.scope_digest,
    )


def prepare_structured_output(
    envelope: GovernedResponseEnvelope,
    projection: GovernedStructuredProjection,
    context: RenderContext,
    *,
    channel_id: StructuredOutputChannelId = StructuredOutputChannelId.PIPELINE_LIST_HTTP_JSON_V1,
    now: datetime,
) -> PreparedStructuredOutput:
    channel = _channel(channel_id)
    unit = mint_structured_transport_unit(envelope, projection, context,
        channel_id=channel_id, idempotency_key=f"{channel_id.value}:{envelope.response_id}", now=now)
    return PreparedStructuredOutput(channel_id, unit, projection.canonical_bytes)


def revalidate_structured_for_transport(
    unit: StructuredTransportUnit,
    now: datetime,
) -> bytes:
    if not isinstance(unit, StructuredTransportUnit):
        raise StructuredOutputLifecycleError("transport authorization requires a server-minted unit")
    if unit.api_version != STRUCTURED_BYTES_LIFECYCLE_API_VERSION:
        raise StructuredOutputLifecycleError("unsupported F2-D structured bytes version")
    if _channel(unit.channel.channel_id) != unit.channel:
        raise StructuredOutputLifecycleError("transport channel contract changed")
    if unit.envelope.compute_digest() != unit.envelope.envelope_digest:
        raise StructuredOutputLifecycleError("sealed envelope digest mismatch")
    actual_digest = "sha256:" + hashlib.sha256(unit.canonical_bytes).hexdigest()
    if (actual_digest != unit.bytes_digest
            or unit.canonical_bytes != unit.projection.canonical_bytes
            or unit.projection.bytes_digest != actual_digest):
        raise StructuredOutputLifecycleError("prepared canonical bytes changed")
    try:
        GovernedStructuredRenderer().revalidate_claims_for_transport(
            unit.envelope, unit.context, unit.projection, now)
    except (StructuredProjectionError, GovernanceContractError, TypeError, ValueError) as exc:
        raise StructuredOutputLifecycleError("F2-B claim validation failed at transport") from exc
    return unit.canonical_bytes


def validate_structured_lifecycle_transition(
    current: OutputLifecycleState,
    target: OutputLifecycleState,
    *,
    same_identity: bool = False,
    before_send: bool = False,
) -> None:
    """Apply the established F2-D transition rules to the byte-bound unit."""
    try:
        validate_lifecycle_transition(current, target, same_identity=same_identity,
            before_send=before_send)
    except OutputLifecycleError as exc:
        raise StructuredOutputLifecycleError("structured output lifecycle transition rejected") from exc


def record_structured_delivery_acknowledgement(*_args: object, **_kwargs: object) -> None:
    raise StructuredOutputLifecycleError(
        "structured HTTP channel has no authenticated delivery acknowledgement protocol")
