"""Decision-store boundary with a deterministic in-memory TDD implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol

from policy.authority_ledger.errors import AuthorityStateError

from .models import RuleDecision, RuleDecisionStatus, RuleEvaluationRequest
from .permit import (
    RulePermit,
    RulePermitDraft,
    _EvaluatedPermit,
    _trusted_permit,
)


class RuleDecisionStore(Protocol):
    def get(self, request: RuleEvaluationRequest) -> RuleDecision | None: ...
    def record(self, request: RuleEvaluationRequest, decision: RuleDecision) -> RuleDecision: ...
    def record_permit(self, request: RuleEvaluationRequest, decision: RuleDecision,
                      evaluated_permit: _EvaluatedPermit) -> RulePermit: ...


@dataclass
class InMemoryRuleDecisionStore:
    """Thread-safe unique-request store for tests; not durable authority storage."""

    _decisions: dict[str, RuleDecision] = field(default_factory=dict, init=False, repr=False)
    _permits: dict[str, RulePermit] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def get(self, request: RuleEvaluationRequest) -> RuleDecision | None:
        if not isinstance(request, RuleEvaluationRequest):
            raise TypeError("get requiere RuleEvaluationRequest para ligar request_hash")
        with self._lock:
            decision = self._decisions.get(request.request_id)
            if decision is None or decision.request_hash != request.request_hash:
                return None
            return decision

    def record(self, request: RuleEvaluationRequest, decision: RuleDecision) -> RuleDecision:
        if type(request) is not RuleEvaluationRequest or type(decision) is not RuleDecision:
            raise TypeError("record requiere RuleEvaluationRequest y RuleDecision")
        if (decision.request_id != request.request_id or
                decision.request_hash != request.request_hash or
                decision.required_rule_id != request.rule_id):
            raise AuthorityStateError("decisión no corresponde a la solicitud")
        with self._lock:
            existing = self._decisions.get(request.request_id)
            if existing is not None:
                if existing.request_hash != request.request_hash:
                    raise AuthorityStateError("request_id reutilizado con otro hash")
                return existing
            self._decisions[request.request_id] = decision
            return decision

    def record_permit(self, request: RuleEvaluationRequest, decision: RuleDecision,
                      evaluated_permit: _EvaluatedPermit) -> RulePermit:
        if (type(request) is not RuleEvaluationRequest or type(decision) is not RuleDecision
                or type(evaluated_permit) is not _EvaluatedPermit
                or not evaluated_permit._is_kernel_sealed()):
            raise TypeError("record_permit requiere una evaluación sellada del kernel")
        draft = evaluated_permit._draft
        if type(draft) is not RulePermitDraft:
            raise TypeError("evaluación sellada no contiene RulePermitDraft")
        if (decision.status is not RuleDecisionStatus.PERMIT or decision.reason_code is not None
                or decision.request_id != request.request_id
                or decision.request_hash != request.request_hash
                or decision.required_rule_id != request.rule_id
                or draft.request_id != request.request_id
                or draft.request_hash != request.request_hash
                or draft.rule_id != request.rule_id
                or draft.issued_at_utc.astimezone(decision.decided_at_utc.tzinfo)
                != decision.decided_at_utc):
            raise AuthorityStateError("RulePermit no corresponde a la decisión/request")

        # Build before mutating so malformed drafts cannot leave a decision behind.
        permit = _trusted_permit(draft)
        with self._lock:
            existing = self._decisions.get(request.request_id)
            stored_permit = self._permits.get(request.request_id)
            if existing is not None:
                if existing.request_hash != request.request_hash:
                    raise AuthorityStateError("request_id reutilizado con otro hash")
                if existing != decision or stored_permit is None:
                    raise AuthorityStateError("request_id ya persistido con otro resultado")
                if dict(stored_permit.projection()) != dict(permit.projection()):
                    raise AuthorityStateError("request_id reintentado con RulePermit distinto")
                return stored_permit
            if stored_permit is not None:
                raise AuthorityStateError("RulePermit huérfano sin decisión")
            self._decisions[request.request_id] = decision
            self._permits[request.request_id] = permit
            return permit
