import asyncio
import dataclasses
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from policy.governance.external_output import (
    ExternalOutputChannelId,
    ExternalOutputError,
    GovernedExternalOutputAdapter,
    OutputOrigin,
)
from policy.governance.governed_renderer import RenderContext, WebChatGovernanceAdapter
from policy.governance.output_lifecycle import OutputLifecycleError, OutputLifecycleState
from policy.governance.response import GovernanceReceipt, ResponseScope


NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)


def _scope():
    return ResponseScope(
        "test", "tenant-a", None, "operator-a", "jacobs", "operator",
        "jacobs-aviso", "request-a", "trace-a",
    )


def _context():
    return RenderContext(None, now=lambda: NOW)


def _envelope(text="Aviso estático del servidor."):
    scope = _scope()
    receipt = GovernanceReceipt("policy-test", "vocabulary-test", "sha256:" + "a" * 64,
                                "validator-test", "renderer-test")
    return WebChatGovernanceAdapter(scope, receipt, producer="jacobs-aviso").seal_non_governed_candidate(
        response_id="response-a", candidate_text=text)


def _prepared():
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
    )
    prepared = adapter.prepare_text(
        _envelope(), _context(), idempotency_key="pipeline:p-7:completed", now=NOW,
    )
    return adapter, prepared


def test_channel_registry_is_closed_versioned_and_declares_destination_and_ack_policy():
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
    )
    assert adapter.channel.channel_id.value.endswith(".v1")
    assert adapter.channel.destination == "telegram.operator_chat"
    assert adapter.channel.transport_kind == "telegram-bot-api"
    assert adapter.channel.origin is OutputOrigin.SYSTEM
    assert adapter.channel.supports_authenticated_ack is False
    with pytest.raises(ValueError):
        ExternalOutputChannelId("caller-invented-channel")


def test_prepare_binds_immutable_governed_text_and_origin_to_f2d_transport_unit():
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
    )
    prepared = adapter.prepare_text(
        _envelope("Aviso estático del servidor."), _context(),
        idempotency_key="pipeline:p-7:completed",
        now=NOW,
    )
    assert prepared.text == "Aviso estático del servidor."
    assert prepared.canonical_bytes == "Aviso estático del servidor.".encode("utf-8")
    assert prepared.origin is OutputOrigin.SYSTEM
    assert prepared.transport_unit.transport_kind == "telegram-bot-api"
    with pytest.raises((AttributeError, TypeError)):
        prepared.text = "mutated"
    with pytest.raises((AttributeError, TypeError)):
        prepared.canonical_bytes = b"mutated"


def test_commit_records_f2d_commit_but_acknowledgement_fails_closed(monkeypatch):
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
    )
    prepared = adapter.prepare_text(
        _envelope(), _context(),
        idempotency_key="pipeline:p-7:completed", now=NOW,
    )
    sent = []

    async def telegram_send(message):
        sent.append(message)
        return {"ok": True, "message_id": 17}

    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", AsyncMock(side_effect=telegram_send))
    result = asyncio.run(adapter.commit(prepared, now=NOW))
    assert sent == [prepared.text]
    assert result.state is OutputLifecycleState.OUTPUT_COMMITTED_TO_TRANSPORT
    with pytest.raises(OutputLifecycleError, match="acknowledgement"):
        adapter.acknowledge(result)


def test_commit_rejects_bytes_changed_after_f2d_preparation(monkeypatch):
    adapter, prepared = _prepared()
    changed = dataclasses.replace(prepared, canonical_bytes=b"forged after render")
    telegram = AsyncMock(return_value={"ok": True, "message_id": 2})
    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", telegram)

    with pytest.raises(ExternalOutputError, match="bytes differ"):
        asyncio.run(adapter.commit(changed, now=NOW))
    telegram.assert_not_awaited()


def test_renderer_failure_never_sends_raw_candidate(monkeypatch):
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
    )
    envelope = _envelope("Pipeline p-7 is running.")
    from policy.governance.governed_renderer import GovernedRenderer, GovernedRenderError
    monkeypatch.setattr(GovernedRenderer, "render_text", lambda *_a, **_k: (_ for _ in ()).throw(GovernedRenderError("render broke")))
    with pytest.raises(ExternalOutputError):
        adapter.prepare_text(envelope, _context(),
                             idempotency_key="p-7", now=NOW)


def test_canonical_pipeline_status_in_raw_narrative_is_only_safe_unavailable():
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
    )
    prepared = adapter.prepare_text(
        _envelope("Pipeline p-7 is completed."), _context(),
        idempotency_key="pipeline:p-7:completed",
        now=NOW,
    )
    assert prepared.text == "I could not verify the current state."
    assert "p-7" not in prepared.text and "completed" not in prepared.text
