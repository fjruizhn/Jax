"""Closed server-owned external-output channel registry for F2-E.

This module deliberately owns channel and transport selection. Producers submit
sealed F2-A material; they do not choose a resolver, receipt, renderer policy,
or F2-D state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .governed_renderer import RenderContext
from .output_lifecycle import (
    StructuredGovernedTransportUnit, StructuredTransportMetadata,
    mint_structured_governed_transport_unit,
)
from .response import GovernedResponseEnvelope, GovernanceContractError, ResponseScope, _text
from .structured_output import (
    GovernedStructuredRenderer, OutputOrigin, StructuredLayout,
    StructuredLayoutRegistry,
)


EXTERNAL_OUTPUT_CHANNEL_REGISTRY_VERSION = "f2-e.channels.1"


class ExternalOutputChannelId(str, Enum):
    WEB_CHAT_HTTP_JSON_V1 = "WEB_CHAT_HTTP_JSON_V1"
    PLATFORM_HTTP_JSON_V1 = "PLATFORM_HTTP_JSON_V1"
    JACOBS_SERVICE_HTTP_JSON_V1 = "JACOBS_SERVICE_HTTP_JSON_V1"
    LAS_MANOS_SERVICE_HTTP_JSON_V1 = "LAS_MANOS_SERVICE_HTTP_JSON_V1"
    PLATFORM_WEBSOCKET_JSON_V1 = "PLATFORM_WEBSOCKET_JSON_V1"
    PLATFORM_SSE_JSON_V1 = "PLATFORM_SSE_JSON_V1"
    DURABLE_JOB_RESULT_V1 = "DURABLE_JOB_RESULT_V1"
    OPERATOR_CLI_TEXT_V1 = "OPERATOR_CLI_TEXT_V1"
    EXTERNAL_ALERT_V1 = "EXTERNAL_ALERT_V1"


@dataclass(frozen=True)
class ChannelDefinition:
    channel_id: ExternalOutputChannelId
    destination: str
    medium: str
    transport_kind: str
    requires_f2d: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.channel_id, ExternalOutputChannelId):
            raise GovernanceContractError("channel identity must be closed enum")
        for name in ("destination", "medium", "transport_kind"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.requires_f2d, bool):
            raise GovernanceContractError("channel lifecycle classification must be bool")


_CHANNELS = (
    ChannelDefinition(ExternalOutputChannelId.WEB_CHAT_HTTP_JSON_V1, "human", "http", "web-chat-http-json"),
    ChannelDefinition(ExternalOutputChannelId.PLATFORM_HTTP_JSON_V1, "http-client", "http", "platform-http-json"),
    ChannelDefinition(ExternalOutputChannelId.JACOBS_SERVICE_HTTP_JSON_V1, "service", "http", "jacobs-service-http-json"),
    ChannelDefinition(ExternalOutputChannelId.LAS_MANOS_SERVICE_HTTP_JSON_V1, "service", "http", "las-manos-service-http-json"),
    ChannelDefinition(ExternalOutputChannelId.PLATFORM_WEBSOCKET_JSON_V1, "human", "websocket", "platform-websocket-json"),
    ChannelDefinition(ExternalOutputChannelId.PLATFORM_SSE_JSON_V1, "human", "sse", "platform-sse-json"),
    ChannelDefinition(ExternalOutputChannelId.DURABLE_JOB_RESULT_V1, "human", "durable-job-result", "durable-job-result"),
    ChannelDefinition(ExternalOutputChannelId.OPERATOR_CLI_TEXT_V1, "operator", "cli", "operator-cli-text"),
    ChannelDefinition(ExternalOutputChannelId.EXTERNAL_ALERT_V1, "external-service", "alert", "external-alert"),
)
CHANNELS: Mapping[ExternalOutputChannelId, ChannelDefinition] = MappingProxyType({item.channel_id: item for item in _CHANNELS})


@dataclass(frozen=True)
class GovernedExternalOutputSubmission:
    """Server composition input after F2-A sealing, never wire/request input."""
    envelope: GovernedResponseEnvelope
    scope: ResponseScope
    origin: OutputOrigin
    channel_id: ExternalOutputChannelId
    output_reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, GovernedResponseEnvelope) or not isinstance(self.scope, ResponseScope):
            raise GovernanceContractError("external submission requires sealed envelope and scope")
        if self.envelope.response_scope.scope_digest != self.scope.scope_digest:
            raise GovernanceContractError("external submission scope must equal sealed envelope")
        if not isinstance(self.origin, OutputOrigin) or not isinstance(self.channel_id, ExternalOutputChannelId):
            raise GovernanceContractError("external submission origin/channel must be typed")
        if self.output_reference is not None:
            object.__setattr__(self, "output_reference", _text(self.output_reference, "output reference"))


@dataclass(frozen=True)
class FutureFaroFinalOutputHandoff:
    """Stable data-only seam. It imports no Faro runtime or provisional API."""
    output_reference: str
    scope: ResponseScope
    origin: OutputOrigin
    channel_id: ExternalOutputChannelId

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_reference", _text(self.output_reference, "output reference"))
        if not isinstance(self.scope, ResponseScope) or not isinstance(self.origin, OutputOrigin) or not isinstance(self.channel_id, ExternalOutputChannelId):
            raise GovernanceContractError("future handoff needs typed scope, origin, and closed channel")


@dataclass(frozen=True)
class GovernedExternalOutputAdapter:
    """Server-owned composition for structured F2-C plus F2-D preparation."""
    context: RenderContext
    layout_registry: StructuredLayoutRegistry
    channel: ChannelDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.context, RenderContext) or not isinstance(self.layout_registry, StructuredLayoutRegistry) or not isinstance(self.channel, ChannelDefinition):
            raise GovernanceContractError("external adapter requires trusted server composition")
        if CHANNELS.get(self.channel.channel_id) != self.channel:
            raise GovernanceContractError("external adapter requires canonical channel definition")

    def prepare_structured(self, submission: GovernedExternalOutputSubmission, *,
                           layout: StructuredLayout, metadata: StructuredTransportMetadata,
                           idempotency_key: str, now=None) -> StructuredGovernedTransportUnit:
        if not isinstance(submission, GovernedExternalOutputSubmission):
            raise GovernanceContractError("external adapter requires typed submission")
        if (submission.channel_id is not self.channel.channel_id
                or layout.channel_id != self.channel.channel_id.value
                or metadata.channel_id != self.channel.channel_id.value
                or not self.channel.requires_f2d):
            raise GovernanceContractError("server channel/layout/transport contract mismatch")
        rendered = GovernedStructuredRenderer().render(submission.envelope, self.context, layout, self.layout_registry)
        return mint_structured_governed_transport_unit(submission.envelope, rendered, self.context,
            layout, self.layout_registry, metadata=metadata, idempotency_key=idempotency_key, now=now)
