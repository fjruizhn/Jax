"""Pure replay, event-chain verification, and effective overlay selection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .canonical import canonical_bytes, domain_hash
from .errors import (AuthorityAuthenticationError, AuthorityStateError,
                     LedgerIntegrityError, OverlayApplicabilityIndeterminateError,
                     OverlayConflictError, TrustedRootMismatchError)
from .models import (AuthorityEvent, AuthorityEventType, AuthorityLedgerCheckpoint,
                     AuthorityLedgerGenesis, AuthorityEventIntent, OverlayApplicability,
                     OverlayPayload, OverlayType)
from .signatures import decode_public_key, public_key_bytes, public_key_fingerprint, verify
from .trusted_root import TrustedAuthorityRoot


def genesis_hash(genesis: AuthorityLedgerGenesis) -> str:
    return domain_hash("JAX-AUTHORITY-LEDGER-GENESIS", "1.0", genesis.projection())


def event_unsigned_bytes(event: AuthorityEvent) -> bytes:
    return canonical_bytes(event.unsigned_projection())


def event_hash(event: AuthorityEvent) -> str:
    return domain_hash("JAX-AUTHORITY-EVENT", "1.0", event.unsigned_projection() | {"signature": event.signature})


@dataclass(frozen=True)
class ReconstructedAuthorityState:
    ratifications: dict[str, AuthorityEvent]
    revoked_ratifications: frozenset[str]
    active_ratification_event_id: str | None
    overlays: dict[str, OverlayPayload]
    revoked_overlays: frozenset[str]
    checkpoint: AuthorityLedgerCheckpoint

    @property
    def active_policy_corpus_hash(self) -> str | None:
        if self.active_ratification_event_id is None:
            return None
        return self.ratifications[self.active_ratification_event_id].intent.policy_corpus_hash


def verify_authority_ledger(genesis: AuthorityLedgerGenesis, events: Iterable[AuthorityEvent], trusted_root: TrustedAuthorityRoot) -> ReconstructedAuthorityState:
    """Verify external genesis anchor before replaying a single ledger event."""
    if trusted_root.ledger_identity != genesis.ledger_identity or trusted_root.genesis_hash != genesis_hash(genesis):
        raise TrustedRootMismatchError("genesis no coincide con trusted root")
    public = decode_public_key(genesis.constitutional_public_key)
    if trusted_root.constitutional_key_id != genesis.constitutional_key_id or trusted_root.constitutional_public_key_fingerprint != public_key_fingerprint(public_key_bytes(public)):
        raise TrustedRootMismatchError("clave constitucional no coincide con trusted root")

    ordered = tuple(events)
    previous: str | None = None
    ratifications: dict[str, AuthorityEvent] = {}
    revoked_ratifications: set[str] = set()
    overlays: dict[str, OverlayPayload] = {}
    revoked_overlays: set[str] = set()
    active: str | None = None
    for expected_sequence, event in enumerate(ordered, 1):
        if event.sequence != expected_sequence or event.previous_event_hash != previous:
            raise LedgerIntegrityError("chain sequence/predecessor inválida")
        if event.intent.actor_id != genesis.constitutional_actor_id:
            raise AuthorityAuthenticationError("actor no es ratificador constitucional")
        verify(public, event_unsigned_bytes(event), event.signature)
        if event.event_hash != event_hash(event):
            raise LedgerIntegrityError("event hash inválido")
        intent = event.intent
        if intent.event_type is AuthorityEventType.RATIFICATION_GRANTED:
            ratifications[event.event_id] = event
        elif intent.event_type is AuthorityEventType.RATIFICATION_REVOKED:
            target = intent.ratification_event_id
            if target not in ratifications:
                raise AuthorityStateError("revocación de ratificación desconocida")
            revoked_ratifications.add(target)
            if active == target:
                active = None
        elif intent.event_type is AuthorityEventType.ACTIVATION_GRANTED:
            target = intent.ratification_event_id
            if target not in ratifications or target in revoked_ratifications:
                raise AuthorityStateError("activación requiere ratificación vigente")
            active = target
        elif intent.event_type is AuthorityEventType.ACTIVATION_DEACTIVATED:
            active = None
        elif intent.event_type is AuthorityEventType.OVERLAY_ISSUED:
            assert intent.overlay is not None
            if intent.overlay.overlay_id in overlays:
                raise AuthorityStateError("overlay_id duplicado")
            overlays[intent.overlay.overlay_id] = intent.overlay
        elif intent.event_type is AuthorityEventType.OVERLAY_REVOKED:
            assert intent.overlay_id is not None
            if intent.overlay_id not in overlays:
                raise AuthorityStateError("revocación de overlay desconocido")
            revoked_overlays.add(intent.overlay_id)
        else:  # defensive against enum extension without replay semantics
            raise LedgerIntegrityError("authority event type desconocido")
        previous = event.event_hash
    checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, len(ordered), ordered[-1].event_id if ordered else None, previous)
    return ReconstructedAuthorityState(ratifications, frozenset(revoked_ratifications), active, overlays, frozenset(revoked_overlays), checkpoint)


def overlay_applicability(overlay: OverlayPayload, context, evaluation_time_utc: datetime) -> OverlayApplicability:
    # Context is Block 3's closed EvaluationContext; no coercion/fuzzy semantics.
    if context.subject not in overlay.scope.subjects or context.action not in overlay.scope.actions:
        return OverlayApplicability.NOT_APPLICABLE
    if evaluation_time_utc < overlay.valid_from_utc or (overlay.valid_until_utc is not None and evaluation_time_utc >= overlay.valid_until_utc):
        return OverlayApplicability.NOT_APPLICABLE
    facts = {item.id: item.value for item in context.conditions}
    for condition in overlay.scope.conditions_all:
        if condition not in facts:
            return OverlayApplicability.INDETERMINATE
        if facts[condition] is False:
            return OverlayApplicability.NOT_APPLICABLE
    return OverlayApplicability.APPLICABLE


def _same_semantics(left: OverlayPayload, right: OverlayPayload) -> bool:
    return canonical_bytes(left.semantic_projection()) == canonical_bytes(right.semantic_projection())


def effective_overlays(state: ReconstructedAuthorityState, context, evaluation_time_utc: datetime) -> tuple[OverlayPayload, ...]:
    """Apply the frozen pairwise matrix for one explicit evaluation."""
    candidates: list[OverlayPayload] = []
    for overlay_id, overlay in state.overlays.items():
        if overlay_id in state.revoked_overlays:
            continue
        result = overlay_applicability(overlay, context, evaluation_time_utc)
        if result is OverlayApplicability.INDETERMINATE:
            raise OverlayApplicabilityIndeterminateError(f"overlay {overlay_id} tiene condición requerida ausente")
        if result is OverlayApplicability.APPLICABLE:
            candidates.append(overlay)
    # Suspensions are first and remove only their targeted overlay.
    suspended = {x.target_overlay_id for x in candidates if x.overlay_type is OverlayType.SUSPENSION}
    active = [x for x in candidates if x.overlay_type is not OverlayType.SUSPENSION and x.overlay_id not in suspended]
    result: list[OverlayPayload] = []
    for overlay in sorted(active, key=lambda x: x.overlay_id):
        duplicate = False
        for existing in result:
            if overlay.overlay_type is not existing.overlay_type:
                continue
            conflict_key = None
            if overlay.overlay_type is OverlayType.EXCEPTION:
                conflict_key = bool(set(overlay.target_rule_ids) & set(existing.target_rule_ids))
            elif overlay.overlay_type is OverlayType.DELEGATION:
                conflict_key = overlay.delegate_actor_id == existing.delegate_actor_id
            elif overlay.overlay_type is OverlayType.BINDING_PARTICULAR_INTERPRETATION:
                conflict_key = bool(set(overlay.target_rule_ids) & set(existing.target_rule_ids))
            if conflict_key:
                if _same_semantics(overlay, existing):
                    duplicate = True
                    break
                raise OverlayConflictError(f"overlays incompatibles: {existing.overlay_id}/{overlay.overlay_id}")
        if not duplicate:
            result.append(overlay)
    return tuple(sorted(result, key=lambda x: (x.overlay_type.value, canonical_bytes(x.semantic_projection()))))
