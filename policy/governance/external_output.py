"""Server-owned boundary for the first F2-E external output channel.

Producers provide only a sealed F2-A envelope and trusted server context.
Channel, destination, renderer, F2-D transport kind, commitment state, and
transport implementation remain fixed in this module. Telegram has no
authenticated delivery acknowledgement protocol, so that channel rejects ACKs.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .governed_renderer import GovernedRenderError, GovernedRenderer, RenderContext, RenderedText
from .output_lifecycle import (
    GovernedTransportUnit,
    OutputLifecycleError,
    OutputLifecycleState,
    mint_governed_transport_unit,
    record_delivery_acknowledgement,
    revalidate_for_transport,
    validate_lifecycle_transition,
)
from .response import GovernanceContractError, GovernedResponseEnvelope


EXTERNAL_OUTPUT_CHANNEL_REGISTRY_VERSION = "f2-e.channels.1"


class ExternalOutputChannelId(str, Enum):
    JACOBS_PIPELINE_NOTICE_TEXT_V1 = "jacobs.pipeline.notice.text.v1"


class OutputOrigin(str, Enum):
    USER = "USER"
    ASSISTANT_MODEL = "ASSISTANT/MODEL"
    TOOL = "TOOL"
    SYSTEM = "SYSTEM"
    AGENT = "AGENT"


class ProjectionKind(str, Enum):
    TEXT = "TEXT"
    STRUCTURED_JSON = "STRUCTURED_JSON"  # reserved for a separately versioned future API


@dataclass(frozen=True)
class ExternalOutputChannel:
    channel_id: ExternalOutputChannelId
    projection_kind: ProjectionKind
    transport_kind: str
    destination: str
    origin: OutputOrigin
    supports_authenticated_ack: bool


_JACOBS_TELEGRAM = ExternalOutputChannel(
    ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
    ProjectionKind.TEXT,
    "telegram-bot-api",
    "telegram.operator_chat",
    OutputOrigin.SYSTEM,
    False,
)
CHANNELS: Mapping[ExternalOutputChannelId, ExternalOutputChannel] = MappingProxyType({
    _JACOBS_TELEGRAM.channel_id: _JACOBS_TELEGRAM,
})


class ExternalOutputError(GovernanceContractError):
    """The channel or its effective governed output could not be established."""


@dataclass(frozen=True)
class PreparedExternalOutput:
    channel_id: ExternalOutputChannelId
    origin: OutputOrigin
    transport_unit: GovernedTransportUnit
    rendered: RenderedText
    canonical_bytes: bytes

    @property
    def text(self) -> str:
        return self.canonical_bytes.decode("utf-8")


@dataclass(frozen=True)
class ExternalOutputCommit:
    channel_id: ExternalOutputChannelId
    state: OutputLifecycleState
    transport_receipt: str | None


@dataclass(frozen=True)
class GovernedExternalOutputAdapter:
    """F2-C → F2-D adapter for a closed, server-owned transport channel."""

    channel: ExternalOutputChannel

    def __post_init__(self) -> None:
        if not isinstance(self.channel, ExternalOutputChannel):
            raise ExternalOutputError("external channel must be server registered")
        if CHANNELS.get(self.channel.channel_id) != self.channel:
            raise ExternalOutputError("external channel is not in the closed server registry")
        if self.channel.projection_kind is not ProjectionKind.TEXT:
            raise ExternalOutputError("structured external projection is reserved for a future version")

    @classmethod
    def for_channel(cls, channel_id: ExternalOutputChannelId) -> "GovernedExternalOutputAdapter":
        if not isinstance(channel_id, ExternalOutputChannelId):
            raise ExternalOutputError("caller supplied an unknown external-output channel")
        definition = CHANNELS.get(channel_id)
        if definition is None:
            raise ExternalOutputError("external-output channel is not registered")
        return cls(definition)

    def prepare_text(
        self,
        envelope: GovernedResponseEnvelope,
        context: RenderContext,
        *,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> PreparedExternalOutput:
        if not isinstance(envelope, GovernedResponseEnvelope) or not isinstance(context, RenderContext):
            raise ExternalOutputError("preparation requires a sealed envelope and trusted render context")
        try:
            rendered = GovernedRenderer().render_text(envelope, context)
            unit = mint_governed_transport_unit(
                envelope,
                rendered,
                context,
                transport_kind=self.channel.transport_kind,
                idempotency_key=idempotency_key,
                now=now,
            )
        except (GovernedRenderError, OutputLifecycleError, GovernanceContractError, TypeError, ValueError) as exc:
            raise ExternalOutputError("governed output preparation failed closed") from exc
        if rendered.contract_state.value == "UNAVAILABLE" and rendered.text != GovernedRenderer.unavailable_text:
            raise ExternalOutputError("unavailable rendering is not the server fallback")
        raw = rendered.text.encode("utf-8")
        if raw.decode("utf-8") != rendered.text:
            raise ExternalOutputError("rendered transport bytes are not canonical UTF-8 text")
        return PreparedExternalOutput(self.channel.channel_id, self.channel.origin, unit, rendered, raw)

    async def commit(
        self,
        prepared: PreparedExternalOutput,
        *,
        now: datetime,
    ) -> ExternalOutputCommit:
        """Commit the exact prepared text to the channel's fixed Telegram sender."""
        if not isinstance(prepared, PreparedExternalOutput) or prepared.channel_id is not self.channel.channel_id:
            raise ExternalOutputError("prepared output does not belong to this channel")
        revalidated = revalidate_for_transport(prepared.transport_unit, now)
        revalidated_bytes = revalidated.text.encode("utf-8")
        if prepared.canonical_bytes != revalidated_bytes:
            raise ExternalOutputError("prepared bytes differ from the F2-D revalidated rendering")
        validate_lifecycle_transition(
            OutputLifecycleState.OUTPUT_PREPARED,
            OutputLifecycleState.TRANSPORT_COMMITTING,
        )
        # Fixed server-owned transport; callers cannot supply a sender or destination.
        from jacobs.reaper import send_telegram_alert

        try:
            result = await send_telegram_alert(revalidated.text)
        except asyncio.CancelledError:
            raise
        except (OSError, TimeoutError, RuntimeError) as exc:
            # A transport exception after entering COMMITTING has an unknown outcome.
            validate_lifecycle_transition(
                OutputLifecycleState.TRANSPORT_COMMITTING,
                OutputLifecycleState.TRANSPORT_OUTCOME_UNKNOWN,
            )
            raise ExternalOutputError("transport outcome is unknown; no delivery claim is made") from exc
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            validate_lifecycle_transition(
                OutputLifecycleState.TRANSPORT_COMMITTING,
                OutputLifecycleState.TRANSPORT_OUTCOME_UNKNOWN,
            )
            return ExternalOutputCommit(self.channel.channel_id,
                OutputLifecycleState.TRANSPORT_OUTCOME_UNKNOWN, None)
        validate_lifecycle_transition(
            OutputLifecycleState.TRANSPORT_COMMITTING,
            OutputLifecycleState.OUTPUT_COMMITTED_TO_TRANSPORT,
        )
        receipt = result.get("message_id")
        return ExternalOutputCommit(self.channel.channel_id,
            OutputLifecycleState.OUTPUT_COMMITTED_TO_TRANSPORT,
            str(receipt) if isinstance(receipt, (str, int)) and not isinstance(receipt, bool) else None)

    def acknowledge(self, _commit: ExternalOutputCommit) -> None:
        if not self.channel.supports_authenticated_ack:
            raise OutputLifecycleError("channel has no authenticated delivery acknowledgement protocol")
        record_delivery_acknowledgement(_commit)
