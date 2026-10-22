"""The callback boundary accepts only immutable, text-opaque envelopes."""
from __future__ import annotations

import hashlib
import uuid

import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from unittest.mock import AsyncMock, patch

from jacobs.office_contract import (
    OFFICE_WORK_CONTRACT_VERSION,
    OfficeCompleteRequest,
    OfficeMaterializeRequest,
    OfficeMaterializedResponse,
    deterministic_correlation_id,
    deterministic_step_id,
    office_utf8_sha256,
)


def _binding(**changes):
    pipeline_id = str(uuid.uuid4())
    values = {
        "contract_version": OFFICE_WORK_CONTRACT_VERSION,
        "pipeline_id": pipeline_id,
        "run_epoch": 0,
        "step_id": deterministic_step_id(pipeline_id),
        "correlation_id": deterministic_correlation_id(pipeline_id),
        "request_sha256": "a" * 64,
    }
    values.update(changes)
    return values


def test_complete_request_preserves_provider_text_and_checks_its_exact_utf8_digest():
    """Catches a JSON parse/trim/normalization changing provider bytes."""
    raw = '{\n  "café": [1, 2] \n}'
    request = OfficeCompleteRequest(**_binding(
        output_json=raw,
        output_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        modelo_real="gemini-2.5-pro",
    ))
    assert request.output_json == raw
    assert request.output_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    with pytest.raises(ValidationError):
        OfficeCompleteRequest(**_binding(
            output_json='{"café":[1,2]}', output_sha256=request.output_sha256,
            modelo_real="gemini-2.5-pro",
        ))


@pytest.mark.parametrize("raw", ["not-json\n", "{\"a\":1}", "{ \"a\": 1 }", "é", "é\n"])
def test_provider_text_hash_distinguishes_exact_unicode_whitespace_and_non_json(raw):
    """Catches treating output as a domain JSON document before JXP does."""
    assert office_utf8_sha256(raw) == hashlib.sha256(raw.encode("utf-8", errors="strict")).hexdigest()


def test_callback_envelopes_reject_coercion_extra_keys_and_noncanonical_ids():
    """Catches permissive callback parsing that would weaken the persisted binding."""
    with pytest.raises(ValidationError):
        OfficeMaterializeRequest(**_binding(run_epoch=True, phase="PREFLIGHT"))
    with pytest.raises(ValidationError):
        OfficeMaterializeRequest(**_binding(pipeline_id=str(uuid.uuid4()).upper(), phase="PREFLIGHT"))
    with pytest.raises(ValidationError):
        OfficeMaterializeRequest(**_binding(phase="PREFLIGHT", unexpected="x"))


def test_materialized_prompt_must_match_exact_utf8_digest_without_leaking_text():
    """Catches accepting a materialization whose stored digest is unrelated to its prompt."""
    raw = "instrucción\n"
    response = OfficeMaterializedResponse(
        outcome="MATERIALIZED", prompt=raw, materialization_sha256=office_utf8_sha256(raw),
        binding_version=1, contract_version=OFFICE_WORK_CONTRACT_VERSION,
    )
    assert response.prompt == raw
    with pytest.raises(ValidationError) as caught:
        OfficeMaterializedResponse(
            outcome="MATERIALIZED", prompt="other", materialization_sha256=response.materialization_sha256,
            binding_version=1, contract_version=OFFICE_WORK_CONTRACT_VERSION,
        )
    assert raw not in str(caught.value)


def test_classic_events_returns_404_before_the_events_reader_for_office_id():
    """Catches the former 200 empty list disclosure shape for an Office pipeline."""
    from jacobs import routes
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=None)), \
         patch.object(routes.store, "events_by_pipeline", AsyncMock()) as events:
        with pytest.raises(HTTPException) as caught:
            import asyncio; asyncio.run(routes.get_events(str(uuid.uuid4())))
    assert caught.value.status_code == 404
    events.assert_not_awaited()
