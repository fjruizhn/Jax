"""Authorized append service: the only Block 4 writer boundary."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .errors import (AuthorityStateError, LedgerCheckpointError, LedgerIntegrityError,
                     LedgerRollbackError, UnanchoredLedgerHeadError, TrustedRootMismatchError)
from .models import (
    AuthorityEvent, AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    _RATIFICATION_SNAPSHOT_SEAL, _RATIFICATION_STORAGE_SEAL,
    _RULE_RATIFICATION_SNAPSHOT_SEAL,
)
from .replay import event_hash, event_unsigned_bytes, genesis_hash, verify_authority_ledger
from .signatures import decode_public_key, public_key_bytes, public_key_fingerprint, sign
from .storage import AuthorityLedgerStore
from .trusted_root import TrustedAuthorityRoot
from .canonical import plain
from .trusted_checkpoint import TrustedCheckpointStore


def _uuid7() -> str:
    creator = getattr(uuid, "uuid7", None)
    if creator is None:  # v1 contract requires UUIDv7, never silently downgrade.
        raise LedgerIntegrityError("runtime no provee UUIDv7")
    return str(creator())


def initialize_authority_ledger(store: AuthorityLedgerStore, genesis: AuthorityLedgerGenesis, trusted_root: TrustedAuthorityRoot) -> None:
    """Verify the externally pinned root; it does not write or bless a root."""
    existing = store.get_genesis()
    if existing != genesis:
        raise TrustedRootMismatchError("genesis de storage no es el genesis solicitado")
    verify_authority_ledger(existing, (), trusted_root)


def ratification_intent_from_candidate(corpus, evidence_refs: tuple[str, ...] = ()) -> AuthorityEventIntent:
    """Mint a ratification intent only from Block 3's sealed candidate boundary.

    The static projection is captured from that same atomic object; callers
    cannot pass a corpus hash and unrelated authority/documents separately.
    """
    from policy.authority_resolution.adapter import to_static_policy_view
    from policy.authority_resolution.models import ValidatedCandidateCorpus
    if not isinstance(corpus, ValidatedCandidateCorpus) or not corpus._was_loader_validated():
        raise AuthorityStateError("ratificación requiere ValidatedCandidateCorpus atómico")
    view = to_static_policy_view(corpus)
    projection = plain(view)
    return AuthorityEventIntent._from_validated_snapshot(
        corpus.policy_corpus_hash, projection, evidence_refs
    )


def append_authority_event(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, private_key: Ed25519PrivateKey, intent: AuthorityEventIntent, *, event_id: str | None = None, recorded_at_utc: datetime | None = None, checkpoint_store: TrustedCheckpointStore | None = None) -> AuthorityEvent:
    """Append one signed Fernando event after verifying the complete ledger."""
    if checkpoint_store is None:
        return _append_authority_event_unlocked(
            store, trusted_root, private_key, intent, event_id=event_id,
            recorded_at_utc=recorded_at_utc, checkpoint_store=None,
        )
    # The checkpoint sidecar is the cross-process writer boundary for this
    # deployment's single host/local filesystem.  It spans every ledger read,
    # the DB append+commit, durable checkpoint publication, and its reread.
    with checkpoint_store.locked():
        return _append_authority_event_unlocked(
            store, trusted_root, private_key, intent, event_id=event_id,
            recorded_at_utc=recorded_at_utc, checkpoint_store=checkpoint_store,
        )


def _append_authority_event_unlocked(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, private_key: Ed25519PrivateKey, intent: AuthorityEventIntent, *, event_id: str | None, recorded_at_utc: datetime | None, checkpoint_store: TrustedCheckpointStore | None) -> AuthorityEvent:
    if intent._ratification_snapshot_seal is _RATIFICATION_STORAGE_SEAL:
        raise AuthorityStateError("ratificación rehidratada desde storage no se puede volver a anexar")
    if (intent.event_type is AuthorityEventType.RATIFICATION_GRANTED
            and intent._ratification_snapshot_seal is not _RATIFICATION_SNAPSHOT_SEAL):
        raise AuthorityStateError("ratificación requiere snapshot sellado del candidate boundary")
    existing_events = store.events()
    use_anchor = checkpoint_store is not None and (bool(existing_events) or checkpoint_store.path.exists())
    state = verify_authority_ledger(store.get_genesis(), existing_events, trusted_root,
                                    checkpoint_store if use_anchor else None)
    if intent.event_type is AuthorityEventType.OVERLAY_ISSUED:
        assert intent.overlay is not None
        if not any(event.intent.policy_corpus_hash == intent.overlay.policy_corpus_hash
                   for event_id, event in state.ratifications.items()
                   if event_id not in state.revoked_ratifications):
            raise AuthorityStateError("overlay exige ratificación vigente del corpus objetivo")
    genesis = store.get_genesis()
    public = decode_public_key(genesis.constitutional_public_key)
    # A key mismatch is never an actor-id workaround.
    if public_key_bytes(private_key.public_key()) != public_key_bytes(public):
        raise AuthorityStateError("private key no corresponde al ratificador constitucional")
    if intent.actor_id != "human:fernando":
        raise AuthorityStateError("agentes/no-Fernando no pueden emitir authority events")
    if (intent.event_type is AuthorityEventType.RULE_RATIFICATION_GRANTED and
            intent._rule_ratification_snapshot_seal is not _RULE_RATIFICATION_SNAPSHOT_SEAL):
        raise AuthorityStateError("rule grant requires a sealed Faro snapshot")
    if intent.event_type is AuthorityEventType.RATIFICATION_GRANTED:
        # Compares fields: the hash field against the one carried inside the
        # frozen projection. It does NOT recompute the hash from the projection.
        # In-process `object.__setattr__` on both fields is outside the threat
        # model (the seal is a guard against public paths, not a cryptographic
        # boundary).
        projection = intent.static_policy_view_projection
        if projection is None or projection.get("policy_corpus_hash") != intent.policy_corpus_hash:
            raise AuthorityStateError("policy_corpus_hash no coincide con la proyección congelada del snapshot")
    events = store.events()
    provisional = AuthorityEvent(event_id or _uuid7(), len(events) + 1, events[-1].event_hash if events else None, intent, recorded_at_utc or datetime.now(timezone.utc), "", "sha256:" + "0" * 64)
    signature = sign(private_key, event_unsigned_bytes(provisional))
    signed = AuthorityEvent(provisional.event_id, provisional.sequence, provisional.previous_event_hash, provisional.intent, provisional.recorded_at_utc, signature, "sha256:" + "0" * 64)
    complete = AuthorityEvent(signed.event_id, signed.sequence, signed.previous_event_hash, signed.intent, signed.recorded_at_utc, signed.signature, event_hash(signed))
    # Full replay WITH the new event: the signature is computed over ephemeral
    # bytes precisely so this gate can cryptographically verify the candidate
    # against the reconstructed state. If replay rejects it, nothing reaches
    # storage and the append-only ledger cannot be poisoned.
    verify_authority_ledger(genesis, events + (complete,), trusted_root)
    store.append(complete)
    if checkpoint_store is not None:
        from .models import AuthorityLedgerCheckpoint
        # El evento NO se considera aceptado hasta que su checkpoint quedó
        # escrito. Si el checkpoint falla, el evento ya es append-only y no se
        # retira: se reporta como huérfano con error tipado y el verificador
        # delata la cabeza sin anclar hasta la reconciliación.
        checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, complete.sequence, complete.event_id, complete.event_hash)
        try:
            checkpoint_store._append_locked(checkpoint)
        except Exception as exc:
            raise LedgerCheckpointError(
                f"evento huérfano {complete.event_id} (secuencia {complete.sequence}): "
                "escrito en el ledger sin checkpoint externo — reconciliar con "
                "reanchor_authority_checkpoint(store, trusted_root, checkpoint_store)"
            ) from exc
    return complete


def reanchor_authority_checkpoint(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, checkpoint_store) -> "AuthorityLedgerCheckpoint":
    """Reconciliación de una cabeza sin checkpoint: re-ancla el checkpoint al
    head EXISTENTE.

    Procedimiento (auditor de #377): el evento huérfano no se retira (el ledger
    es append-only); se verifica el stream COMPLETO sin ancla externa y, si el
    head es válido, se escribe su checkpoint. Tras esto, el verificador con
    ``checkpoint_store`` vuelve a pasar: la cabeza queda anclada y el ledger no
    quedó inverificable para siempre."""
    with checkpoint_store.locked():
        return _reanchor_authority_checkpoint_unlocked(store, trusted_root, checkpoint_store)


def _reanchor_authority_checkpoint_unlocked(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, checkpoint_store) -> "AuthorityLedgerCheckpoint":
    from .models import AuthorityLedgerCheckpoint
    genesis = store.get_genesis()
    events = store.events()
    if not events:
        raise AuthorityStateError("reconciliación exige un head existente")
    try:
        verify_authority_ledger(genesis, events, trusted_root, checkpoint_store)
    except UnanchoredLedgerHeadError:
        anchored = checkpoint_store.latest()
        if (len(events) < anchored.sequence
                or events[anchored.sequence - 1].event_id != anchored.head_event_id
                or events[anchored.sequence - 1].event_hash != anchored.head_event_hash):
            raise LedgerRollbackError("stream no conserva el checkpoint vigente como prefijo")
    else:
        raise AuthorityStateError("reconciliación solo procede ante una cabeza sin checkpoint")
    verify_authority_ledger(genesis, events, trusted_root)
    head = events[-1]
    checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, head.sequence, head.event_id, head.event_hash)
    checkpoint_store._append_locked(checkpoint)
    return checkpoint
