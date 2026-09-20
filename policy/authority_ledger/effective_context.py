"""Construct Block 4 effective context/envelope without changing Block 3."""
from __future__ import annotations

from datetime import datetime, timezone

from policy.authority_resolution.resolver import resolve_static_authority

from .errors import AuthorityStateError
from .models import EffectiveAuthorityContext, EffectiveAuthorityEnvelope
from .replay import ReconstructedAuthorityState, effective_overlays


def build_effective_authority_context(state: ReconstructedAuthorityState, context, evaluation_time_utc: datetime) -> EffectiveAuthorityContext:
    if state.active_policy_corpus_hash is None:
        raise AuthorityStateError("no hay corpus activo")
    if evaluation_time_utc.tzinfo is None:
        raise AuthorityStateError("evaluation_time_utc debe ser timezone-aware")
    return EffectiveAuthorityContext("1.0", "JAX_EFFECTIVE_AUTHORITY_CONTEXT", state.active_policy_corpus_hash, "JAX-AUTHORITY-RESOLVER/1", "1.0", effective_overlays(state, context, evaluation_time_utc.astimezone(timezone.utc)))


def build_effective_authority_envelope(state: ReconstructedAuthorityState, static_policy_view, context, evaluation_time_utc: datetime) -> EffectiveAuthorityEnvelope:
    effective = build_effective_authority_context(state, context, evaluation_time_utc)
    if static_policy_view.policy_corpus_hash != effective.active_policy_corpus_hash:
        raise AuthorityStateError("static view no corresponde al corpus activo")
    resolution = resolve_static_authority(static_policy_view, context)
    overlays = tuple(sorted(x.overlay_id for x in effective.effective_overlays))
    return EffectiveAuthorityEnvelope("1.0", "JAX_EFFECTIVE_AUTHORITY_ENVELOPE", resolution, effective.active_policy_corpus_hash, effective.effective_authority_context_hash, state.checkpoint.authority_ledger_checkpoint_hash, overlays, "EFFECTIVE_STATIC_AUTHORITY_ONLY")
