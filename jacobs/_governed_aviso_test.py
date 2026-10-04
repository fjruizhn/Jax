import asyncio
from datetime import datetime, timezone

import pytest

from jacobs import models, store
from jacobs.aviso import avisar_fin_pipeline
from jacobs.governed_aviso import compose_pipeline_notice
from policy.governance.external_output import (
    ExternalOutputChannelId,
    GovernedExternalOutputAdapter,
    OutputOrigin,
)
from policy.governance.output_lifecycle import OutputLifecycleState


def _pipeline(status=models.PipelineStatus.completed, *, owner=True):
    return models.Pipeline(
        pipeline_id="pipeline-f2e-1", name="untrusted display name",
        invoked_by="plataforma", mode="supervised", status=status,
        tenant_id="tenant-a" if owner else None,
        user_id="user-a" if owner else None,
    )


def _source(monkeypatch, pipeline):
    monkeypatch.setenv("JAX_DB_HOST", "mariadb.test")
    monkeypatch.setenv("JAX_DB_PORT", "3308")
    monkeypatch.setenv("JAX_DB_NAME", "jax_memory_test")
    monkeypatch.setenv("JAX_GOVERNANCE_ENVIRONMENT", "test")

    async def pipeline_get(_pipeline_id):
        return pipeline

    monkeypatch.setattr(store, "pipeline_get", pipeline_get)


def test_pipeline_notice_resolves_canonical_status_renders_claim_then_commits_exact_text(monkeypatch):
    _source(monkeypatch, _pipeline())
    envelope, context = asyncio.run(compose_pipeline_notice(
        pipeline_id="pipeline-f2e-1", requested_status="completed"))
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1)
    validation_time = datetime.now(timezone.utc)
    prepared = adapter.prepare_text(
        envelope, context,
        idempotency_key="jacobs-pipeline-notice:pipeline-f2e-1:completed", now=validation_time,
    )
    assert prepared.text == "Jacobs registra actualmente que el pipeline pipeline-f2e-1 está completed."
    assert prepared.origin is OutputOrigin.SYSTEM
    assert prepared.transport_unit.rendered.claim_ids == ("pipeline-status:" + envelope.response_id,)
    assert prepared.canonical_bytes == prepared.text.encode("utf-8")

    sent = []

    async def telegram(message):
        sent.append(message)
        return {"ok": True, "message_id": 12}

    from unittest.mock import AsyncMock
    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", AsyncMock(side_effect=telegram))
    committed = asyncio.run(adapter.commit(prepared, now=datetime.now(timezone.utc)))
    assert sent == [prepared.text]
    assert committed.state is OutputLifecycleState.OUTPUT_COMMITTED_TO_TRANSPORT


def test_real_aviso_producer_ignores_supplied_name_and_sends_only_governed_claim(monkeypatch):
    _source(monkeypatch, _pipeline())
    sent = []

    async def telegram(message):
        sent.append(message)
        return {"ok": True, "message_id": 13}

    from unittest.mock import AsyncMock
    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", AsyncMock(side_effect=telegram))

    async def invoke():
        await avisar_fin_pipeline(
            pipeline_id="pipeline-f2e-1", nombre="<script>forged</script>",
            estado="completed",
        )

    asyncio.run(invoke())
    assert sent == ["Jacobs registra actualmente que el pipeline pipeline-f2e-1 está completed."]
    assert "forged" not in sent[0]


def test_pipeline_notice_ignores_caller_status_that_disagrees_with_canonical_source(monkeypatch):
    _source(monkeypatch, _pipeline(models.PipelineStatus.completed))
    envelope, context = asyncio.run(compose_pipeline_notice(
        pipeline_id="pipeline-f2e-1", requested_status="running"))
    adapter = GovernedExternalOutputAdapter.for_channel(
        ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1)
    validation_time = datetime.now(timezone.utc)
    prepared = adapter.prepare_text(
        envelope, context,
        idempotency_key="jacobs-pipeline-notice:pipeline-f2e-1:running", now=validation_time,
    )
    assert prepared.text == "I could not verify the current state."
    assert "running" not in prepared.text


def test_ownerless_pipeline_has_no_notice_scope(monkeypatch):
    _source(monkeypatch, _pipeline(owner=False))
    with pytest.raises(ValueError, match="identity is unavailable"):
        asyncio.run(compose_pipeline_notice(
            pipeline_id="pipeline-f2e-1", requested_status="completed"))
