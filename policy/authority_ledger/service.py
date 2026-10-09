"""Authorized append service: the only Block 4 writer boundary."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .errors import AuthorityStateError, LedgerIntegrityError, TrustedRootMismatchError
from .models import (
    AuthorityEvent, AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
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
    checkpoint_store: TrustedCheckpointStore | None = None,
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
    intent = AuthorityEventIntent(AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", evidence_refs, projection["policy_corpus_hash"], projection)
    # Keep this write path separate from `_append_authority_event`: that
    # helper rejects RATIFICATION_GRANTED even when called directly. Reusing a
    # raw-intent helper here would recreate the bypass this boundary closes.
    verify_authority_ledger(store.get_genesis(), store.events(), trusted_root)
    genesis = store.get_genesis()
    public = decode_public_key(genesis.constitutional_public_key)
    if public_key_bytes(private_key.public_key()) != public_key_bytes(public):
        raise AuthorityStateError("private key no corresponde al ratificador constitucional")
    events = store.events()
    provisional = AuthorityEvent(event_id or _uuid7(), len(events) + 1, events[-1].event_hash if events else None, intent, recorded_at_utc or datetime.now(timezone.utc), "", "sha256:" + "0" * 64)
    signature = sign(private_key, event_unsigned_bytes(provisional))
    signed = AuthorityEvent(provisional.event_id, provisional.sequence, provisional.previous_event_hash, provisional.intent, provisional.recorded_at_utc, signature, "sha256:" + "0" * 64)
    complete = AuthorityEvent(signed.event_id, signed.sequence, signed.previous_event_hash, signed.intent, signed.recorded_at_utc, signed.signature, event_hash(signed))
    verify_authority_ledger(genesis, events + (complete,), trusted_root)
    store.append(complete)
    if checkpoint_store is not None:
        from .models import AuthorityLedgerCheckpoint
        checkpoint_store.append(AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, complete.sequence, complete.event_id, complete.event_hash))
    return complete


def append_authority_event(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, private_key: Ed25519PrivateKey, intent: AuthorityEventIntent, *, event_id: str | None = None, recorded_at_utc: datetime | None = None, checkpoint_store: TrustedCheckpointStore | None = None) -> AuthorityEvent:
    """Append a non-ratification Fernando event after full ledger replay."""
    if intent.event_type is AuthorityEventType.RATIFICATION_GRANTED:
        raise AuthorityStateError(
            "RATIFICATION_GRANTED requiere append_ratification_from_candidate"
        )
    return _append_authority_event(
        store, trusted_root, private_key, intent,
        event_id=event_id,
        recorded_at_utc=recorded_at_utc,
        checkpoint_store=checkpoint_store,
    )


def _append_authority_event(store: AuthorityLedgerStore, trusted_root: TrustedAuthorityRoot, private_key: Ed25519PrivateKey, intent: AuthorityEventIntent, *, event_id: str | None = None, recorded_at_utc: datetime | None = None, checkpoint_store: TrustedCheckpointStore | None = None) -> AuthorityEvent:
    """Internal signer for non-ratification public writer boundaries.

    The Python module is one trusted in-process component; private names do
    not defend against code that can inspect memory or import internals. This
    helper nevertheless fails closed for direct accidental/internal calls so
    its callable surface cannot authorize a caller-created ratification.
    """
    if intent.event_type is AuthorityEventType.RATIFICATION_GRANTED:
        raise AuthorityStateError(
            "RATIFICATION_GRANTED requiere append_ratification_from_candidate"
        )
    state = verify_authority_ledger(store.get_genesis(), store.events(), trusted_root)
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
        checkpoint_store.append(AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, complete.sequence, complete.event_id, complete.event_hash))
    return complete
