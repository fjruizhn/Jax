from __future__ import annotations

from datetime import datetime, timezone

import pytest

from policy.rule_authority.models import (
    RuleDecision,
    RuleDecisionStatus,
    RuleEvaluationRequest,
    RuleLimits,
)
from policy.rule_authority.store import InMemoryRuleDecisionStore
from policy.authority_ledger.errors import AuthorityEventValidationError, AuthorityStateError
from jax.faro.catalogo_topes import cargar_catalogo_bytes

CATALOG = cargar_catalogo_bytes(b'''{"version":1,"decision":"test fixture","clases":{"monto_dinero":["hnl","usd"],"actos_externos":["mensajes","correos","publicaciones","compras","pagos"],"frecuencia":["por_hora","por_dia"],"duracion":["segundos"],"tokens_costo":["tokens","usd"]}}''')

FORBIDDEN_LIMIT_FIELDS = ("connections", "concurrency", "workers", "threads", "processes", "agents")


def request(**overrides):
    values = {
        "request_id": "0199f8a1-8c00-7000-8000-000000000001",
        "rule_id": "rule-one",
        "subject": "actor:fernando",
        "capability": "mail.send",
        "objective": "notify-client",
        "resource_id": "message:invoice-42",
        "arguments": {"recipient": "client@example.test", "body": "Invoice ready"},
        "catalogo": CATALOG,
        "quantity": 1,
        "quantity_unit": "mensajes",
        "amount": None,
    }
    values.update(overrides)
    return RuleEvaluationRequest(**values)


def test_request_hash_is_derived_from_closed_canonical_projection():
    original = request()
    reordered = request(arguments={"body": "Invoice ready", "recipient": "client@example.test"})
    changed = request(arguments={"recipient": "other@example.test", "body": "Invoice ready"})

    assert original.request_hash == reordered.request_hash
    assert original.request_hash != changed.request_hash
    assert original.arguments["recipient"] == "client@example.test"
    with pytest.raises(TypeError):
        original.arguments["recipient"] = "tampered@example.test"
    with pytest.raises((AttributeError, TypeError)):
        original.request_hash = "sha256:" + "0" * 64
    with pytest.raises(AuthorityEventValidationError):
        request(request_hash="sha256:" + "0" * 64)
    assert request(arguments={"nested": {}}).request_hash != request(arguments={"nested": []}).request_hash


def test_request_arguments_are_normalized_to_nfc_before_hashing():
    composed = request(arguments={"texto": "café"})
    decomposed = request(arguments={"texto": "cafe\u0301"})
    assert composed.arguments["texto"] == "café"
    assert decomposed.arguments["texto"] == "café"
    assert composed.request_hash == decomposed.request_hash
    with pytest.raises(AuthorityEventValidationError, match="claves duplicadas tras NFC"):
        request(arguments={"é": 1, "e\u0301": 2})


def test_request_hash_includes_quantity_unit():
    assert request(quantity_unit="mensajes").request_hash != request(quantity_unit="correos").request_hash


@pytest.mark.parametrize(
    ("quantity", "quantity_unit"),
    [(1, None), (None, "messages")],
)
def test_request_requires_quantity_and_unit_together(quantity, quantity_unit):
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=quantity, quantity_unit=quantity_unit)


@pytest.mark.parametrize(
    "limits",
    [
        {"quantity": 1},
        {"quantity_unit": "messages"},
    ],
)
def test_rule_limits_require_quantity_and_unit_together(limits):
    with pytest.raises(AuthorityEventValidationError):
        RuleLimits(catalogo=CATALOG, **limits)


@pytest.mark.parametrize("arguments", [{"nested": object()}])
def test_request_rejects_opaque_arguments(arguments):
    with pytest.raises(AuthorityEventValidationError):
        request(arguments=arguments)


def test_request_validates_uuid7_and_exactly_one_action_or_money_limit():
    with pytest.raises(AuthorityEventValidationError):
        request(request_id="not-a-uuid")
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=0)
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=1, amount=50, currency="USD")


def test_rule_limits_allow_only_action_money_frequency_duration_and_cost():
    assert RuleLimits(catalogo=CATALOG, quantity=2, quantity_unit="mensajes",
                      frequency_count=2, frequency_window_seconds=3600,
                      frequency_unit="por_hora", duration_seconds=30, tokens=400,
                      cost_minor_units=10, cost_currency="usd")


def test_limits_require_snapshot_catalog_and_nonempty_values():
    with pytest.raises(AuthorityEventValidationError, match="catálogo cargado"):
        RuleLimits(quantity=1, quantity_unit="mensajes")
    with pytest.raises(AuthorityEventValidationError, match="vacío"):
        RuleLimits(catalogo=CATALOG)


@pytest.mark.parametrize("bad", ["conexiones", "recipients"])
def test_limits_reject_units_outside_loaded_catalog(bad):
    with pytest.raises(AuthorityEventValidationError, match="catálogo"):
        RuleLimits(catalogo=CATALOG, quantity=1, quantity_unit=bad)


@pytest.mark.parametrize("kwargs", [
    {"quantity": True, "quantity_unit": "mensajes"},
    {"amount": 2**53 + 1, "currency": "usd"},
    {"currency": "usd"},
])
def test_limits_reject_bool_overflow_and_currency_without_amount(kwargs):
    with pytest.raises(AuthorityEventValidationError):
        RuleLimits(catalogo=CATALOG, **kwargs)


def test_request_rejects_currency_without_amount():
    with pytest.raises(AuthorityEventValidationError, match="amount y currency"):
        request(currency="usd")


def test_permit_has_no_persistable_reason_and_storage_failure_is_not_a_reason():
    with pytest.raises(AuthorityEventValidationError):
        RuleDecision("0199f8a1-8c00-7000-8000-000000000001", "sha256:"+"a"*64,
                     RuleDecisionStatus.PERMIT, "rule-one", "STORAGE_FAILURE",
                     datetime(2026, 10, 6, tzinfo=timezone.utc))
    with pytest.raises(AuthorityEventValidationError):
        RuleDecision("0199f8a1-8c00-7000-8000-000000000001", "sha256:"+"a"*64,
                     RuleDecisionStatus.DENY, "rule-one", "STORAGE_FAILURE",
                     datetime(2026, 10, 6, tzinfo=timezone.utc))


def test_rule_evaluation_request_limits_use_catalog_units_and_reject_bool():
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=True)
    with pytest.raises(AuthorityEventValidationError, match="catálogo"):
        request(quantity_unit="recipients")


@pytest.mark.parametrize("field", FORBIDDEN_LIMIT_FIELDS)
def test_rule_limits_cannot_represent_infrastructure_limits(field):
    with pytest.raises(TypeError):
        RuleLimits(**{field: 2})


def test_memory_decision_store_is_idempotent_by_request_id_and_hash():
    store = InMemoryRuleDecisionStore()
    submitted = request()
    decision = RuleDecision(
        request_id=submitted.request_id,
        request_hash=submitted.request_hash,
        status=RuleDecisionStatus.DENY,
        required_rule_id=submitted.rule_id,
        reason_code="STOP_ACTIVE",
        decided_at_utc=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )

    assert store.record(submitted, decision) == decision
    assert store.record(request(), decision) == decision
    assert store.get(submitted) == decision
    assert store.get(request(arguments={"recipient": "different"})) is None
    changed = request(arguments={"recipient": "different"})
    changed_decision = RuleDecision(
        request_id=changed.request_id,
        request_hash=changed.request_hash,
        status=RuleDecisionStatus.DENY,
        required_rule_id=changed.rule_id,
        reason_code="STOP_ACTIVE",
        decided_at_utc=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )
    with pytest.raises(AuthorityStateError):
        store.record(changed, changed_decision)


def test_memory_store_cannot_inject_decisions_in_constructor():
    with pytest.raises(TypeError):
        InMemoryRuleDecisionStore(_decisions={})


def test_memory_store_rejects_decision_for_different_request():
    store = InMemoryRuleDecisionStore()
    submitted = request()
    other = request(resource_id="message:invoice-43")
    decision = RuleDecision(
        request_id=other.request_id,
        request_hash=other.request_hash,
        status=RuleDecisionStatus.MISSING_RULE,
        required_rule_id=other.rule_id,
        reason_code="RULE_NOT_FOUND",
        decided_at_utc=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )

    with pytest.raises(AuthorityStateError):
        store.record(submitted, decision)
    assert store.get(submitted) is None


@pytest.mark.parametrize(
    ("status", "reason_code"),
    [
        (RuleDecisionStatus.MISSING_RULE, "STOP_ACTIVE"),
        (RuleDecisionStatus.DENY, "RULE_NOT_FOUND"),
    ],
)
def test_decision_status_and_reason_code_cannot_contradict(status, reason_code):
    with pytest.raises(AuthorityEventValidationError):
        RuleDecision(
            request_id="0199f8a1-8c00-7000-8000-000000000001",
            request_hash="sha256:" + "a" * 64,
            status=status,
            required_rule_id="rule-one",
            reason_code=reason_code,
            decided_at_utc=datetime(2026, 10, 6, tzinfo=timezone.utc),
        )
