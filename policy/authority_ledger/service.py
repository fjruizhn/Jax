"""Authorized append service: the only Block 4 writer boundary."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .errors import AuthorityStateError, LedgerIntegrityError, TrustedRootMismatchError
from .models import AuthorityEvent, AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis
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
    state = verify_authority_ledger(store.get_genesis(), store.events(), trusted_root)
    genesis = store.get_genesis()
    public = decode_public_key(genesis.constitutional_public_key)
    # A key mismatch is never an actor-id workaround.
    if public_key_bytes(private_key.public_key()) != public_key_bytes(public):
        raise AuthorityStateError("private key no corresponde al ratificador constitucional")
    if intent.actor_id != "human:fernando":
        raise AuthorityStateError("agentes/no-Fernando no pueden emitir authority events")
    events = store.events()
    provisional = AuthorityEvent(event_id or _uuid7(), len(events) + 1, events[-1].event_hash if events else None, intent, recorded_at_utc or datetime.now(timezone.utc), "", "sha256:" + "0" * 64)
    signature = sign(private_key, event_unsigned_bytes(provisional))
    signed = AuthorityEvent(provisional.event_id, provisional.sequence, provisional.previous_event_hash, provisional.intent, provisional.recorded_at_utc, signature, "sha256:" + "0" * 64)
    complete = AuthorityEvent(signed.event_id, signed.sequence, signed.previous_event_hash, signed.intent, signed.recorded_at_utc, signed.signature, event_hash(signed))
    store.append(complete)
    if checkpoint_store is not None:
        from .models import AuthorityLedgerCheckpoint
        checkpoint_store.append(AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", genesis.ledger_identity, complete.sequence, complete.event_id, complete.event_hash))
    return complete
