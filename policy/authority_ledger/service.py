"""Authorized append service: the only Block 4 writer boundary."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .errors import (AuthorityStateError, CheckpointPublicationOutcomeUnknownError,
                     LedgerAlreadyInitializedError, LedgerCheckpointError, LedgerIntegrityError,
                     LedgerRollbackError, UnanchoredLedgerHeadError, TrustedRootMismatchError)
from .models import (
    AuthorityEvent, AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    _RULE_RATIFICATION_SNAPSHOT_SEAL,
)
from .replay import (event_hash, event_unsigned_bytes, genesis_hash,
                     replay_authority_history, verify_authority_ledger)
from .signatures import decode_public_key, public_key_bytes, public_key_fingerprint, sign
from .storage import AuthorityLedgerStore
from .trusted_root import TrustedAuthorityRoot
from .canonical import domain_hash, plain
from .trusted_checkpoint import TrustedCheckpointStore, require_trusted_checkpoint_store


def _uuid7() -> str:
    creator = getattr(uuid, "uuid7", None)
    if creator is None:  # v1 contract requires UUIDv7, never silently downgrade.
        raise LedgerIntegrityError("runtime no provee UUIDv7")
    return str(creator())


def initialize_authority_ledger(store: AuthorityLedgerStore, genesis: AuthorityLedgerGenesis, trusted_root: TrustedAuthorityRoot, checkpoint_store: TrustedCheckpointStore) -> None:
    """Bootstrap an empty ledger once; use `verify_authority_ledger` afterward."""
    checkpoint_store = require_trusted_checkpoint_store(checkpoint_store)
    if store.get_genesis() != genesis:
        raise TrustedRootMismatchError("genesis de storage no es el genesis solicitado")
    with checkpoint_store.locked():
        if checkpoint_store.path.exists():
            raise LedgerAlreadyInitializedError("ledger authority ya inicializado; usar verify_authority_ledger")
        events = store.events()
        if checkpoint_store.bootstrap_receipt_path.exists():
            raise LedgerAlreadyInitializedError("recibo bootstrap existe pero falta checkpoint; restaurar ambos artifacts, no reinicializar")
        if events:
            raise LedgerCheckpointError("bootstrap rechazado: ledger legacy con eventos sin checkpoint; conservar y recuperar artifacts")
        if trusted_root.ledger_identity != genesis.ledger_identity or trusted_root.genesis_hash != genesis_hash(genesis):
            raise TrustedRootMismatchError("genesis no coincide con trusted root")
        public = decode_public_key(genesis.constitutional_public_key)
        if trusted_root.constitutional_key_id != genesis.constitutional_key_id or trusted_root.constitutional_public_key_fingerprint != public_key_fingerprint(public_key_bytes(public)):
            raise TrustedRootMismatchError("clave constitucional no coincide con trusted root")
        from .models import AuthorityLedgerCheckpoint
        checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, 0, None, None)
        root_projection = {"schema_version": trusted_root.schema_version, "kind": trusted_root.kind, "ledger_identity": trusted_root.ledger_identity, "genesis_hash": trusted_root.genesis_hash, "constitutional_key_id": trusted_root.constitutional_key_id, "constitutional_public_key_fingerprint": trusted_root.constitutional_public_key_fingerprint}
        receipt = {"schema_version": "1.0", "kind": "JAX_AUTHORITY_LEDGER_BOOTSTRAP_RECEIPT", "ledger_identity": genesis.ledger_identity, "genesis_hash": genesis_hash(genesis), "trusted_root_hash": domain_hash("JAX-TRUSTED-AUTHORITY-ROOT", "1.0", root_projection), "checkpoint_hash": checkpoint.authority_ledger_checkpoint_hash}
        checkpoint_store.bootstrap(checkpoint, receipt)
        verify_authority_ledger(genesis, (), trusted_root, checkpoint_store)


def ratification_intent_from_candidate(corpus, evidence_refs: tuple[str, ...] = ()) -> AuthorityEventIntent:
    """Build a non-signable ratification value for serialization consumers.

    This helper never appends. Production writes must use
    :func:`append_ratification_from_candidate`, which keeps validation,
    projection capture, and signing in one public flow.
    """
    from policy.authority_resolution.adapter import to_static_policy_view

    projection = plain(to_static_policy_view(corpus))
    return AuthorityEventIntent(
        AuthorityEventType.RATIFICATION_GRANTED,
        "human:fernando",
        evidence_refs,
        projection["policy_corpus_hash"],
        projection,
    )


def append_ratification_from_candidate(
    store: AuthorityLedgerStore,
    trusted_root: TrustedAuthorityRoot,
    private_key: Ed25519PrivateKey,
    corpus,
    evidence_refs: tuple[str, ...] = (),
    *,
    event_id: str | None = None,
    recorded_at_utc: datetime | None = None,
    checkpoint_store: TrustedCheckpointStore,
) -> AuthorityEvent:
    """Validate one candidate, derive its view, and append its ratification.

    This is deliberately the only public RATIFICATION_GRANTED write path. It
    captures one locally validated projection, takes the hash from that same
    projection, then signs it without accepting a caller-constructed intent.
    """
    from policy.authority_resolution.adapter import to_static_policy_view
    from policy.authority_resolution.errors import ResolverContractError

    try:
        view = to_static_policy_view(corpus)
    except ResolverContractError as exc:
        raise AuthorityStateError("ratificación requiere ValidatedCandidateCorpus atómico") from exc
    projection = plain(view)
    intent = AuthorityEventIntent(
        AuthorityEventType.RATIFICATION_GRANTED,
        "human:fernando",
        evidence_refs,
        projection["policy_corpus_hash"],
        projection,
    )
    checkpoint_store = require_trusted_checkpoint_store(checkpoint_store)
    with checkpoint_store.locked():
        # Keep the signing path here: no importable helper accepts an
        # AuthorityEventIntent and can turn caller supplied authority into a
        # signed ledger event.
        existing_events = store.events()
        verify_authority_ledger(store.get_genesis(), existing_events, trusted_root, checkpoint_store)
        genesis = store.get_genesis()
        public = decode_public_key(genesis.constitutional_public_key)
        if public_key_bytes(private_key.public_key()) != public_key_bytes(public):
            raise AuthorityStateError("private key no corresponde al ratificador constitucional")
        provisional = AuthorityEvent(event_id or _uuid7(), len(existing_events) + 1, existing_events[-1].event_hash if existing_events else None, intent, recorded_at_utc or datetime.now(timezone.utc), "", "sha256:" + "0" * 64)
        signature = sign(private_key, event_unsigned_bytes(provisional))
        signed = AuthorityEvent(provisional.event_id, provisional.sequence, provisional.previous_event_hash, provisional.intent, provisional.recorded_at_utc, signature, "sha256:" + "0" * 64)
        complete = AuthorityEvent(signed.event_id, signed.sequence, signed.previous_event_hash, signed.intent, signed.recorded_at_utc, signed.signature, event_hash(signed))
        replay_authority_history(genesis, existing_events + (complete,), trusted_root)
        store.append(complete)
        from .models import AuthorityLedgerCheckpoint
        checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, complete.sequence, complete.event_id, complete.event_hash)
        try:
            checkpoint_store._append_locked(checkpoint)
        except CheckpointPublicationOutcomeUnknownError as exc:
            raise CheckpointPublicationOutcomeUnknownError(
                f"evento {complete.event_id} (secuencia {complete.sequence}): resultado de publicación "
                "desconocido después de os.replace; verificar/reconciliar con "
                "reanchor_authority_checkpoint(store, trusted_root, checkpoint_store)"
            ) from exc
        except Exception as exc:
            raise LedgerCheckpointError(
                f"evento huérfano {complete.event_id} (secuencia {complete.sequence}): "
                "escrito en el ledger sin checkpoint externo — reconciliar con "
                "reanchor_authority_checkpoint(store, trusted_root, checkpoint_store)"
            ) from exc
        return complete


def append_authority_event(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, private_key: Ed25519PrivateKey, intent: AuthorityEventIntent, *, event_id: str | None = None, recorded_at_utc: datetime | None = None, checkpoint_store: TrustedCheckpointStore) -> AuthorityEvent:
    """Append a non-ratification Fernando event after full ledger replay."""
    if intent.event_type is AuthorityEventType.RATIFICATION_GRANTED:
        raise AuthorityStateError(
            "RATIFICATION_GRANTED requiere append_ratification_from_candidate"
        )
    checkpoint_store = require_trusted_checkpoint_store(checkpoint_store)
    with checkpoint_store.locked():
        return _append_authority_event_unlocked(
            store, trusted_root, private_key, intent,
            event_id=event_id, recorded_at_utc=recorded_at_utc,
            checkpoint_store=checkpoint_store,
        )


def _append_authority_event(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, private_key: Ed25519PrivateKey, intent: AuthorityEventIntent, *, event_id: str | None = None, recorded_at_utc: datetime | None = None, checkpoint_store: TrustedCheckpointStore) -> AuthorityEvent:
    """Internal non-ratification writer; it also rejects direct bypasses."""
    if intent.event_type is AuthorityEventType.RATIFICATION_GRANTED:
        raise AuthorityStateError(
            "RATIFICATION_GRANTED requiere append_ratification_from_candidate"
        )
    return append_authority_event(
        store, trusted_root, private_key, intent,
        event_id=event_id,
        recorded_at_utc=recorded_at_utc,
        checkpoint_store=checkpoint_store,
    )


def _append_authority_event_unlocked(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, private_key: Ed25519PrivateKey, intent: AuthorityEventIntent, *, event_id: str | None, recorded_at_utc: datetime | None, checkpoint_store: TrustedCheckpointStore) -> AuthorityEvent:
    if intent.event_type is AuthorityEventType.RATIFICATION_GRANTED:
        raise AuthorityStateError(
            "RATIFICATION_GRANTED requiere append_ratification_from_candidate"
        )
    existing_events = store.events()
    state = verify_authority_ledger(store.get_genesis(), existing_events, trusted_root, checkpoint_store)
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
    events = existing_events
    provisional = AuthorityEvent(event_id or _uuid7(), len(events) + 1, events[-1].event_hash if events else None, intent, recorded_at_utc or datetime.now(timezone.utc), "", "sha256:" + "0" * 64)
    signature = sign(private_key, event_unsigned_bytes(provisional))
    signed = AuthorityEvent(provisional.event_id, provisional.sequence, provisional.previous_event_hash, provisional.intent, provisional.recorded_at_utc, signature, "sha256:" + "0" * 64)
    complete = AuthorityEvent(signed.event_id, signed.sequence, signed.previous_event_hash, signed.intent, signed.recorded_at_utc, signed.signature, event_hash(signed))
    # Full replay WITH the new event: the signature is computed over ephemeral
    # bytes precisely so this gate can cryptographically verify the candidate
    # against the reconstructed state. If replay rejects it, nothing reaches
    # storage and the append-only ledger cannot be poisoned.
    replay_authority_history(genesis, events + (complete,), trusted_root)
    store.append(complete)
    from .models import AuthorityLedgerCheckpoint
    # The append is not accepted until its checkpoint is durably published.
    checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, complete.sequence, complete.event_id, complete.event_hash)
    try:
        checkpoint_store._append_locked(checkpoint)
    except CheckpointPublicationOutcomeUnknownError as exc:
        raise CheckpointPublicationOutcomeUnknownError(
            f"evento {complete.event_id} (secuencia {complete.sequence}): resultado de publicación "
            "desconocido después de os.replace; verificar/reconciliar con "
            "reanchor_authority_checkpoint(store, trusted_root, checkpoint_store)"
        ) from exc
    except Exception as exc:
        raise LedgerCheckpointError(
            f"evento huérfano {complete.event_id} (secuencia {complete.sequence}): "
            "escrito en el ledger sin checkpoint externo — reconciliar con "
            "reanchor_authority_checkpoint(store, trusted_root, checkpoint_store)"
        ) from exc
    return complete


def reanchor_authority_checkpoint(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, checkpoint_store) -> "AuthorityLedgerCheckpoint":
    """Advance an existing checkpoint over a verified append-only DB suffix.

    This cannot recreate missing trust evidence. It accepts only a valid-prefix
    database head, and is idempotent when the checkpoint already matches.
    """
    checkpoint_store = require_trusted_checkpoint_store(checkpoint_store)
    with checkpoint_store.locked():
        return _reanchor_authority_checkpoint_unlocked(store, trusted_root, checkpoint_store)


def _reanchor_authority_checkpoint_unlocked(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, checkpoint_store) -> "AuthorityLedgerCheckpoint":
    from .models import AuthorityLedgerCheckpoint
    events = store.events()
    if not events:
        raise AuthorityStateError("reconciliación exige un head existente")
    if not checkpoint_store.path.exists():
        raise LedgerIntegrityError("reanchor exige checkpoint externo vigente; no puede reconstruir confianza ausente")
    genesis = store.get_genesis()
    head = events[-1]
    checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, head.sequence, head.event_id, head.event_hash)
    try:
        verify_authority_ledger(genesis, events, trusted_root, checkpoint_store)
    except UnanchoredLedgerHeadError:
        anchored = checkpoint_store.latest()
        if anchored.ledger_identity != genesis.ledger_identity:
            raise LedgerRollbackError("checkpoint vigente pertenece a otro ledger")
        if anchored.sequence == 0:
            if anchored.head_event_id is not None or anchored.head_event_hash is not None:
                raise LedgerRollbackError("checkpoint genesis inválido")
        elif (len(events) < anchored.sequence
                or events[anchored.sequence - 1].event_id != anchored.head_event_id
                or events[anchored.sequence - 1].event_hash != anchored.head_event_hash):
            raise LedgerRollbackError("stream no conserva el checkpoint vigente como prefijo")
    else:
        # `os.replace` may have completed before the previous writer lost its
        # directory fsync or reread.  The matching head is already published;
        # finish its durability/re-read obligations idempotently.
        checkpoint_store._fsync_parent_directory()
        published = checkpoint_store.latest()
        if published.projection() != checkpoint.projection():
            raise LedgerRollbackError("checkpoint vigente no coincide con el head DB")
        verify_authority_ledger(genesis, events, trusted_root, checkpoint_store)
        return checkpoint
    replay_authority_history(genesis, events, trusted_root)
    checkpoint_store._append_locked(checkpoint)
    return checkpoint
