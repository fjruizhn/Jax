"""Pure replay, event-chain verification, and effective overlay selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping
from types import MappingProxyType

from .canonical import canonical_bytes, domain_hash
from .errors import (AuthorityAuthenticationError, AuthorityStateError,
                     LedgerIntegrityError, OverlayApplicabilityIndeterminateError,
                     OverlayConflictError, TrustedRootMismatchError)
from .models import (AuthorityEvent, AuthorityEventType, AuthorityLedgerCheckpoint,
                     AuthorityLedgerGenesis, AuthorityEventIntent, OverlayApplicability,
                     OverlayPayload, OverlayType)
from .signatures import decode_public_key, public_key_bytes, public_key_fingerprint, verify
from .trusted_root import TrustedAuthorityRoot
from .trusted_checkpoint import TrustedCheckpointStore, require_trusted_checkpoint_store
from .errors import LedgerRollbackError, UnanchoredLedgerHeadError


def genesis_hash(genesis: AuthorityLedgerGenesis) -> str:
    return domain_hash("JAX-AUTHORITY-LEDGER-GENESIS", "1.0", genesis.projection())


def event_unsigned_bytes(event: AuthorityEvent) -> bytes:
    return canonical_bytes(event.unsigned_projection())


def event_hash(event: AuthorityEvent) -> str:
    return domain_hash("JAX-AUTHORITY-EVENT", "1.0", event.unsigned_projection() | {"signature": event.signature})


@dataclass(frozen=True)
class ReconstructedAuthorityState:
    ratifications: Mapping[str, AuthorityEvent]
    revoked_ratifications: frozenset[str]
    active_ratification_event_id: str | None
    overlays: Mapping[str, OverlayPayload]
    revoked_overlays: frozenset[str]
    checkpoint: AuthorityLedgerCheckpoint
    _rule_ratification_grants: Mapping[str, AuthorityEvent] = field(default_factory=dict, repr=False)
    _latest_rule_ratifications: Mapping[str, AuthorityEvent] = field(default_factory=dict, repr=False)
    _revoked_rule_ratifications: frozenset[str] = field(default_factory=frozenset, repr=False)
    _verified_seal: object | None = None
    quarantined_overlays: frozenset[str] = field(default_factory=frozenset)

    @property
    def active_policy_corpus_hash(self) -> str | None:
        if self.active_ratification_event_id is None:
            return None
        return self.ratifications[self.active_ratification_event_id].intent.policy_corpus_hash

    def _is_verified(self) -> bool:
        return self._verified_seal is _REPLAY_SEAL

    def latest_unrevoked_rule_ratification(self, rule_id: str) -> AuthorityEvent | None:
        """Return the latest unrevoked grant; temporal validity belongs to rule evaluation."""
        if not self._is_verified():
            raise AuthorityStateError("rule ratification requiere replay verificado")
        event = self._latest_rule_ratifications.get(rule_id)
        if event is None or event.event_id in self._revoked_rule_ratifications:
            return None
        return event


@dataclass(frozen=True)
class HistoricalAuthorityState:
    """Verified replay of a prefix; structurally distinct from current authority."""
    ratifications: Mapping[str, AuthorityEvent]
    revoked_ratifications: frozenset[str]
    active_ratification_event_id: str | None
    overlays: Mapping[str, OverlayPayload]
    revoked_overlays: frozenset[str]
    checkpoint: AuthorityLedgerCheckpoint
    _rule_ratification_grants: Mapping[str, AuthorityEvent] = field(default_factory=dict, repr=False)
    _latest_rule_ratifications: Mapping[str, AuthorityEvent] = field(default_factory=dict, repr=False)
    _revoked_rule_ratifications: frozenset[str] = field(default_factory=frozenset, repr=False)
    quarantined_overlays: frozenset[str] = field(default_factory=frozenset)
    _history_seal: object | None = field(default=None, repr=False)

    @property
    def active_policy_corpus_hash(self) -> str | None:
        if self.active_ratification_event_id is None:
            return None
        return self.ratifications[self.active_ratification_event_id].intent.policy_corpus_hash

    def _is_verified_history(self) -> bool:
        return self._history_seal is _HISTORY_SEAL


_HISTORY_SEAL = object()


_REPLAY_SEAL = object()


def verify_authority_ledger(genesis: AuthorityLedgerGenesis, events: Iterable[AuthorityEvent], trusted_root: TrustedAuthorityRoot, checkpoint_store: TrustedCheckpointStore) -> ReconstructedAuthorityState:
    """Verify current authority against its mandatory external checkpoint."""
    checkpoint_store = require_trusted_checkpoint_store(checkpoint_store)
    return _replay_authority_ledger(genesis, events, trusted_root, checkpoint_store, historical=False)


def replay_authority_history(genesis: AuthorityLedgerGenesis, events: Iterable[AuthorityEvent], trusted_root: TrustedAuthorityRoot) -> HistoricalAuthorityState:
    """Verify historical signed history for diagnosis; never returns current authority."""
    return _replay_authority_ledger(genesis, events, trusted_root, None, historical=True)


def _replay_authority_ledger(genesis: AuthorityLedgerGenesis, events: Iterable[AuthorityEvent], trusted_root: TrustedAuthorityRoot, checkpoint_store: TrustedCheckpointStore | None, *, historical: bool):
    """Verify the external genesis and replay an explicitly current or historical view."""
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
    quarantined_overlays: set[str] = set()
    rule_grants_by_event_id: dict[str, AuthorityEvent] = {}
    latest_rule_ratifications: dict[str, AuthorityEvent] = {}
    revoked_rule_ratifications: set[str] = set()
    event_ids_seen: set[str] = set()
    active: str | None = None
    for expected_sequence, event in enumerate(ordered, 1):
        if event.sequence != expected_sequence or event.previous_event_hash != previous:
            raise LedgerIntegrityError("chain sequence/predecessor inválida")
        if event.event_id in event_ids_seen:
            raise LedgerIntegrityError("event_id duplicado")
        event_ids_seen.add(event.event_id)
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
            # Historical append-only streams may contain an overlay issued
            # before its target corpus had a live ratification.  Preserve its
            # signed history and allow a later revoke, but quarantine it
            # permanently: later ratification never makes it effective.
            if not any(event.intent.policy_corpus_hash == intent.overlay.policy_corpus_hash
                       for event_id, event in ratifications.items()
                       if event_id not in revoked_ratifications):
                quarantined_overlays.add(intent.overlay.overlay_id)
            overlays[intent.overlay.overlay_id] = intent.overlay
        elif intent.event_type is AuthorityEventType.OVERLAY_REVOKED:
            assert intent.overlay_id is not None
            if intent.overlay_id not in overlays:
                raise AuthorityStateError("revocación de overlay desconocido")
            revoked_overlays.add(intent.overlay_id)
        elif intent.event_type is AuthorityEventType.RULE_RATIFICATION_GRANTED:
            assert intent.rule_ratification is not None
            rule_grants_by_event_id[event.event_id] = event
            latest_rule_ratifications[intent.rule_ratification.rule_id] = event
        elif intent.event_type is AuthorityEventType.RULE_RATIFICATION_REVOKED:
            target = intent.rule_ratification_event_id
            if target not in rule_grants_by_event_id:
                raise AuthorityStateError("revocación de rule ratification desconocida")
            if target in revoked_rule_ratifications:
                raise AuthorityStateError("rule ratification revocada más de una vez")
            revoked_rule_ratifications.add(target)
        else:  # defensive against enum extension without replay semantics
            raise LedgerIntegrityError("authority event type desconocido")
        previous = event.event_hash
    checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, len(ordered), ordered[-1].event_id if ordered else None, previous)
    if checkpoint_store is not None:
        root_projection = {"schema_version": trusted_root.schema_version, "kind": trusted_root.kind,
            "ledger_identity": trusted_root.ledger_identity, "genesis_hash": trusted_root.genesis_hash,
            "constitutional_key_id": trusted_root.constitutional_key_id,
            "constitutional_public_key_fingerprint": trusted_root.constitutional_public_key_fingerprint}
        genesis_checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, 0, None, None)
        receipt = {"schema_version": "1.0", "kind": "JAX_AUTHORITY_LEDGER_BOOTSTRAP_RECEIPT",
            "ledger_identity": genesis.ledger_identity, "genesis_hash": genesis_hash(genesis),
            "trusted_root_hash": domain_hash("JAX-TRUSTED-AUTHORITY-ROOT", "1.0", root_projection),
            "checkpoint_hash": genesis_checkpoint.authority_ledger_checkpoint_hash}
        checkpoint_store.validate_bootstrap_receipt(receipt)
        anchored = checkpoint_store.latest()
        if checkpoint.sequence < anchored.sequence:
            raise LedgerRollbackError("DB ledger truncado antes del checkpoint externo")
        if checkpoint.sequence > anchored.sequence:
            raise UnanchoredLedgerHeadError(
                "cabeza sin checkpoint: DB ledger adelante del checkpoint externo — "
                "reconciliar re-anclando el checkpoint al head existente "
                "(reanchor_authority_checkpoint); el head verificado no queda "
                "inverificable para siempre")
        if checkpoint.projection() != anchored.projection():
            raise LedgerRollbackError("head DB no coincide con checkpoint externo")
    common = (
        MappingProxyType(dict(ratifications)), frozenset(revoked_ratifications), active,
        MappingProxyType(dict(overlays)), frozenset(revoked_overlays), checkpoint,
        MappingProxyType(dict(rule_grants_by_event_id)), MappingProxyType(dict(latest_rule_ratifications)),
        frozenset(revoked_rule_ratifications),
    )
    if historical:
        return HistoricalAuthorityState(*common, quarantined_overlays=frozenset(quarantined_overlays), _history_seal=_HISTORY_SEAL)
    return ReconstructedAuthorityState(*common, quarantined_overlays=frozenset(quarantined_overlays), _verified_seal=_REPLAY_SEAL)


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
    if not isinstance(state, ReconstructedAuthorityState) or not state._is_verified():
        raise AuthorityStateError("effective state requiere replay verificado")
    return _select_effective_overlays(state, context, evaluation_time_utc)


def _effective_overlays_for_history(state: HistoricalAuthorityState, context, evaluation_time_utc: datetime) -> tuple[OverlayPayload, ...]:
    if not isinstance(state, HistoricalAuthorityState) or not state._is_verified_history():
        raise AuthorityStateError("historical state requiere replay histórico verificado")
    return _select_effective_overlays(state, context, evaluation_time_utc)


def _select_effective_overlays(state, context, evaluation_time_utc: datetime) -> tuple[OverlayPayload, ...]:
    active_hash = state.active_policy_corpus_hash
    if active_hash is None:
        return ()
    candidates: list[OverlayPayload] = []
    for overlay_id, overlay in state.overlays.items():
        if overlay_id in state.revoked_overlays:
            continue
        if overlay_id in state.quarantined_overlays:
            continue
        if overlay.policy_corpus_hash != active_hash:
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
