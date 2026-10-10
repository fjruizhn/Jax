from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.policy.catalogo_pin import catalogo_del_pin
from policy.authority_ledger.errors import AuthorityStateError
from policy.rule_authority.errors import RuleAuthorityError
from policy.rule_authority.models import RuleDecision, RuleDecisionStatus, RuleEvaluationRequest
from policy.rule_authority.permit import RulePermitDraft, _permit_after_kernel_evaluation
from policy.rule_authority.store import InMemoryRuleDecisionStore


CATALOG = catalogo_del_pin()


def _request(request_id="0199f8a1-8c00-7000-8000-000000000501", *, recipient="a@example.test"):
    return RuleEvaluationRequest(
        request_id=request_id, rule_id="rule-one", subject="actor:fernando",
        capability="MAIL_SEND", objective="send-notice", resource_id="mail:501",
        arguments={"recipient": recipient}, catalogo=CATALOG,
        quantity=1, quantity_unit="mensajes",
    )


def _decision(request):
    return RuleDecision(
        request_id=request.request_id, request_hash=request.request_hash,
        status=RuleDecisionStatus.PERMIT, required_rule_id=request.rule_id,
        reason_code=None, decided_at_utc=datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc),
    )


def _draft(request, decision, permit_id="0199f8a1-8c00-7000-8000-000000000502"):
    return RulePermitDraft(
        permit_id=permit_id, request_id=request.request_id, request_hash=request.request_hash,
        rule_id=request.rule_id, rule_path="policy/faro/rule-one.yaml",
        rule_blob_oid="a" * 40, rule_content_hash="sha256:" + "1" * 64,
        policy_revision="b" * 40, policy_tree_oid="c" * 40,
        policy_snapshot_hash="sha256:" + "2" * 64,
        ratification_event_id="0199f8a1-8c00-7000-8000-000000000503",
        authority_ledger_checkpoint={"sequence": 4, "head_event_hash": "sha256:" + "3" * 64},
        stop_checkpoint={"version": 0, "fingerprint": "stop:0"},
        capability_id=request.capability, capability_version="1",
        capability_class="OBLIGATING",
        capability_limits={"form": "CANTIDAD", "unit": "mensajes", "max": 4},
        issued_at_utc=decision.decided_at_utc,
        expires_at_utc=decision.decided_at_utc + timedelta(seconds=30),
    )


def test_memoria_sella_y_reusa_solo_la_misma_decision_y_el_mismo_permit():
    request = _request()
    decision = _decision(request)
    draft = _draft(request, decision)
    evaluated = _permit_after_kernel_evaluation(draft)
    store = InMemoryRuleDecisionStore()

    result = store.record_permit(request, decision, evaluated)
    retry = store.record_permit(request, decision, evaluated)

    assert result._is_store_sealed()
    assert retry.projection() == result.projection()
    assert store.get(request) == decision


def test_memoria_rechaza_draft_sin_sello_kernel_y_permit_distinto_en_retry():
    request = _request()
    decision = _decision(request)
    store = InMemoryRuleDecisionStore()
    with pytest.raises((TypeError, RuleAuthorityError)):
        store.record_permit(request, decision, _draft(request, decision))

    first = _draft(request, decision)
    store.record_permit(request, decision, _permit_after_kernel_evaluation(first))
    conflicting = _draft(request, decision, "0199f8a1-8c00-7000-8000-000000000504")
    with pytest.raises(AuthorityStateError):
        store.record_permit(request, decision, _permit_after_kernel_evaluation(conflicting))
