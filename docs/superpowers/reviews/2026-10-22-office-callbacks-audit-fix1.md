# Task4C2 fix1 · independent Sol review

2026-10-22 · Codex records delivery_path read-only verdict on exact `adf2db7076fab55e440c6e9efcac76b29e3942fc`, base approved C1 `9eb3b747bcaefff863e641c1205f0c8f820b0e8a`. Spec RECHAZADO; quality RECHAZADO. Reviewer inspected clean exact tree, source and pure JSON/UTF-8 reproduction, no DB/network mutations. Root independently ran broad private C1 suite: 330 passed, 316 warnings, 7.57s. Those tests do not close this finding.

## BLOCK: invalid strict UTF-8 escapes reconciliation

Provider JSON may legally escape an unpaired surrogate; `parse_office_provider_response` currently returns it as Python str. After DELIVERY_IN_FLIGHT CAS the hash raises, and exception handlers repeat the same hash before calling `_office_failure`, raising again and leaving delivery unresolved. Sources: `jacobs/office_runtime.py:145-191`, `jacobs/executor.py:1415-1475`, `jacobs/office_contract.py:54-61`. Pure reproduction confirmed JSON decoding returns U+D800 and strict hash raises ValueError.

Fix2/5 ruling, Codex accepting the existing callback-boundary and reconciliation requirements: initialize output_sha256=None; validate/hash exactly once; map strict UTF-8 failure to stable OFFICE_PROVIDER_RESPONSE_INVALID / GENERATION / provider_call_made=true with no content or cause. Exception paths use the already computed variable, never rehash output. Do not call complete; call fail with GENERATION/model_invoked=true/hash=None and honor typed terminal result from DELIVERY_IN_FLIGHT. No new schema/protocol or fallback.

## Required effective coverage

Runner regression must obtain surrogate from actual JSON/parser, win delivery claim, assert no complete, one fail with exact fields, typed local terminal and no content in logs/errors/durable storage. Existing outcome matrix only inspects outgoing request: assert local FAILED/CANCELLED_REVOKED/OUTCOME_UNKNOWN as appropriate. Existing complete error matrix mocks `_post`: exercise real MockTransport HTTP409/422/503/malformed200 and preserve delivery ambiguity.

## Original findings

All original ten source defects were closed: RESERVED preparation/replay, enum DTO, materialize before paid preflight, canonical capability gate, phase-correct READY exits, typed outcome/receipt reconciliation, kill-switch, catalog model/hash truth, periodic recovery/exception observation, URL validation. Original reconciliation point6 is effectively reopened for the strict UTF-8 edge above. No other blocking source defect found. Pair gate stays false; no consumer before exact-SHA clean re-review.
