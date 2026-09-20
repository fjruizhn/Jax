"""Construct Block 4 effective context/envelope without changing Block 3."""
from __future__ import annotations

from datetime import datetime, timezone

from policy.authority_resolution.resolver import resolve_static_authority
from policy.authority_resolution.models import (FrozenAuthorityMetaContract, FrozenExternalConstraints,
    FrozenFailClosed, FrozenNormativeDocument, FrozenNormativeSources, FrozenPrecedence,
    FrozenRelationships, FrozenScope, ProtectedConstraint, ValidatedStaticPolicyView)

from .errors import AuthorityStateError
from .models import EffectiveAuthorityContext, EffectiveAuthorityEnvelope
from .replay import ReconstructedAuthorityState, effective_overlays


def build_effective_authority_context(state: ReconstructedAuthorityState, context, evaluation_time_utc: datetime) -> EffectiveAuthorityContext:
    if not isinstance(state, ReconstructedAuthorityState) or not state._is_verified():
        raise AuthorityStateError("authority state no verificado")
    if state.active_policy_corpus_hash is None:
        raise AuthorityStateError("no hay corpus activo")
    if evaluation_time_utc.tzinfo is None:
        raise AuthorityStateError("evaluation_time_utc debe ser timezone-aware")
    return EffectiveAuthorityContext("1.0", "JAX_EFFECTIVE_AUTHORITY_CONTEXT", state.active_policy_corpus_hash, "JAX-AUTHORITY-RESOLVER/1", "1.0", effective_overlays(state, context, evaluation_time_utc.astimezone(timezone.utc)))


def _snapshot_view(projection) -> ValidatedStaticPolicyView:
    """Rehydrate only the signed ratification snapshot, never caller content."""
    try:
        authority = projection["authority_meta_contract"]
        precedence = FrozenPrecedence(**authority["precedence"])
        sources = FrozenNormativeSources(**authority["normative_sources"])
        protected = tuple(ProtectedConstraint(**entry) for entry in authority["protected_metanorms"])
        external = FrozenExternalConstraints(**authority["external_constraints"])
        fail_closed = FrozenFailClosed(**authority["fail_closed"])
        frozen_authority = FrozenAuthorityMetaContract(authority["scope_jurisdiction"], precedence, sources, protected, external, fail_closed)
        documents = tuple(FrozenNormativeDocument(
            entry["id"], entry["document_class"], entry["normative_layer"], entry["normative_effect"],
            FrozenScope(**entry["scope"]), FrozenRelationships(**entry["relationships"])
        ) for entry in projection["ordinary_documents"])
        return ValidatedStaticPolicyView(projection["schema_version"], projection["kind"], projection["policy_corpus_hash"], projection["canonicalizer_identity"], projection["bootstrap_bundle_id"], frozen_authority, documents)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityStateError("snapshot de ratificación inválido") from exc


def build_effective_authority_envelope(state: ReconstructedAuthorityState, context, evaluation_time_utc: datetime) -> EffectiveAuthorityEnvelope:
    effective = build_effective_authority_context(state, context, evaluation_time_utc)
    if state.active_ratification_event_id is None:
        raise AuthorityStateError("no hay ratificación activa")
    intent = state.ratifications[state.active_ratification_event_id].intent
    static_policy_view = _snapshot_view(intent.static_policy_view_projection)
    if static_policy_view.policy_corpus_hash != effective.active_policy_corpus_hash:
        raise AuthorityStateError("snapshot no corresponde al corpus activo")
    resolution = resolve_static_authority(static_policy_view, context)
    overlays = tuple(sorted(x.overlay_id for x in effective.effective_overlays))
    return EffectiveAuthorityEnvelope("1.0", "JAX_EFFECTIVE_AUTHORITY_ENVELOPE", resolution, effective.active_policy_corpus_hash, effective.effective_authority_context_hash, state.checkpoint.authority_ledger_checkpoint_hash, overlays, "EFFECTIVE_STATIC_AUTHORITY_ONLY")
