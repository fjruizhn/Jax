from decimal import Decimal
import asyncio
import traceback
import uuid

import pytest
from pydantic import ValidationError

from jacobs import store
from jacobs.office_contract import (
    OFFICE_RUNTIME_POLICY, OFFICE_WORK_CONTRACT_VERSION, OfficePhase, OfficeRunRequest, OfficeRuntimePolicy,
    OfficeCompleteRequest, OfficeMaterializedResponse, OfficeReceipt,
    deterministic_correlation_id, deterministic_step_id, office_request_digest, office_utf8_sha256,
)


def _claims(**overrides):
    values = {
        "contract_version": OFFICE_WORK_CONTRACT_VERSION,
        "office_job_id": uuid.uuid4(), "office_stage_id": uuid.uuid4(),
        "client_request_id": uuid.uuid4(), "project_id": 7, "tenant_id": 9,
        "usage_user_id": 11, "facet": "hipatia", "capability": "design",
        "timeout_seconds": 60, "accepted_cost_usd": Decimal("1.250000"),
    }
    values.update(overrides)
    values["request_sha256"] = office_request_digest(values)
    return values


def test_strict_opaque_request_binds_all_claims_and_has_stable_ids():
    request = OfficeRunRequest(**_claims())
    assert request.request_sha256 == office_request_digest(request)
    pipeline_id = str(uuid.uuid4())
    assert deterministic_step_id(pipeline_id) == deterministic_step_id(pipeline_id)
    assert deterministic_correlation_id(pipeline_id) == deterministic_correlation_id(pipeline_id)
    assert OFFICE_RUNTIME_POLICY.generation_calls == 1
    assert OFFICE_RUNTIME_POLICY.tool_mode == "NONE"
    with pytest.raises(ValueError):
        OfficeRuntimePolicy(generation_calls=2)
    with pytest.raises(ValueError):
        OfficeRuntimePolicy(generation_calls=True)
    with pytest.raises(ValueError):
        OfficeRuntimePolicy(allowed_transports={"http_gemini"})


def test_uuid_strings_have_the_same_canonical_digest_as_uuid_values():
    claims = _claims()
    uuid_keys = {"office_job_id", "office_stage_id", "client_request_id"}
    string_claims = {key: (str(value).upper() if key in uuid_keys else value) for key, value in claims.items()}
    string_claims["request_sha256"] = office_request_digest(string_claims)
    parsed = OfficeRunRequest(**string_claims)
    assert parsed.request_sha256 == office_request_digest(parsed)


@pytest.mark.parametrize("amount", ["0", "0.000001", "99999999.999999"])
def test_exact_office_budget_boundaries_are_canonical_and_digest_stable(amount):
    base = _claims(accepted_cost_usd=amount)
    request = OfficeRunRequest(**base)
    assert request.request_sha256 == office_request_digest(request)
    identity = _claims()
    digests = []
    for equivalent in ("1", "1.0", "1.000000"):
        claims = {**identity, "accepted_cost_usd": equivalent}
        claims["request_sha256"] = office_request_digest(claims)
        digests.append(OfficeRunRequest(**claims).request_sha256)
    assert len(set(digests)) == 1


@pytest.mark.parametrize("field,value", [
    ("prompt", "secret"), ("provider_url", "https://bad.invalid"),
    ("accepted_cost_usd", Decimal("1.250001")),
])
def test_opaque_request_rejects_raw_or_digest_mismatched_fields(field, value):
    claims = _claims()
    claims[field] = value
    with pytest.raises(ValidationError):
        OfficeRunRequest(**claims)


@pytest.mark.parametrize("field,value", [
    ("project_id", "7"), ("tenant_id", True), ("timeout_seconds", 1.5),
    ("contract_version", 1), ("facet", 7), ("capability", False),
    ("accepted_cost_usd", 1.0), ("accepted_cost_usd", "NaN"),
    ("accepted_cost_usd", "Infinity"), ("accepted_cost_usd", "0.0000001"),
    ("accepted_cost_usd", "999999999.999999"), ("accepted_cost_usd", "100000000"),
])
def test_strict_json_claim_types_and_exact_money(field, value):
    claims = _claims()
    claims[field] = value
    with pytest.raises(ValidationError):
        OfficeRunRequest(**claims)


def test_unknown_office_rejection_code_is_opaque_before_any_database_work(monkeypatch, caplog):
    sentinel = "SENTINEL_UNTRUSTED_OFFICE_REJECTION"
    connection_called = False

    def unexpected_connection():
        nonlocal connection_called
        connection_called = True
        raise AssertionError("rejection validation must precede database work")

    monkeypatch.setattr(store, "conexion", unexpected_connection)
    caplog.set_level("DEBUG")
    for untrusted in (sentinel, 7, []):
        with pytest.raises(ValueError) as caught:
            asyncio.run(store.office_transition_phase(
                "unused", 0, OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.FAILED,
                rejection_code=untrusted,
            ))
        assert str(caught.value) == "generic Office phase transitions are disabled"
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
        assert sentinel not in rendered
    assert not connection_called
    assert sentinel not in caplog.text


def test_callback_dtos_are_strict_and_preserve_opaque_provider_bytes():
    pipeline_id = str(uuid.uuid4())
    envelope = {
        "contract_version": OFFICE_WORK_CONTRACT_VERSION, "pipeline_id": pipeline_id,
        "run_epoch": 0, "step_id": deterministic_step_id(pipeline_id),
        "correlation_id": deterministic_correlation_id(pipeline_id), "request_sha256": "a" * 64,
    }
    raw = '{"x":"\\u00e9"}\n'
    request = OfficeCompleteRequest(**envelope, output_json=raw,
        output_sha256=office_utf8_sha256(raw), modelo_real="catalog/model-v1")
    assert request.output_json == raw
    with pytest.raises(ValidationError):
        OfficeCompleteRequest(**envelope, output_json=raw, output_sha256=office_utf8_sha256(raw), modelo_real="catalog/model", extra="no")
    with pytest.raises(ValidationError):
        OfficeCompleteRequest(**envelope, output_json=raw, output_sha256="a" * 64, modelo_real="catalog/model")
    with pytest.raises(ValidationError):
        OfficeMaterializedResponse(outcome="MATERIALIZED", prompt="SENTINEL_PRIVATE_PROMPT", materialization_sha256="a" * 64, binding_version=True, contract_version=OFFICE_WORK_CONTRACT_VERSION)
    with pytest.raises(ValidationError) as caught:
        OfficeReceipt(receipt_id=str(uuid.uuid4()).upper(), effect_kind="NOTE_CREATED", effect_id="note:1", output_sha256="a" * 64, office_stage_version=1)
    assert "NOTE_CREATED" not in str(caught.value)
