"""Decision-store boundary with a deterministic in-memory TDD implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol

from policy.authority_ledger.errors import AuthorityStateError

from .models import RuleDecision, RuleEvaluationRequest


class RuleDecisionStore(Protocol):
    def get(self, request_id: str) -> RuleDecision | None: ...
    def record(self, request: RuleEvaluationRequest, decision: RuleDecision) -> RuleDecision: ...


@dataclass
class InMemoryRuleDecisionStore:
    """Thread-safe unique-request store for tests; not durable authority storage."""

    _decisions: dict[str, RuleDecision] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def get(self, request_id: str) -> RuleDecision | None:
        with self._lock:
            return self._decisions.get(request_id)

    def record(self, request: RuleEvaluationRequest, decision: RuleDecision) -> RuleDecision:
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
