"""Append-only authority event storage boundary.

The in-memory implementation is for deterministic unit tests.  A production
MariaDB adapter must implement the same locked append contract; replay, not a
cache, remains authoritative.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol, Callable, Any, ContextManager, TYPE_CHECKING

from .errors import AuthorityStateError
from .models import AuthorityEvent, AuthorityLedgerGenesis

if TYPE_CHECKING:
    from .replay import ReconstructedAuthorityState
    from .trusted_root import TrustedAuthorityRoot
    from policy.rule_authority.permit import (
        RulePermit,
        RulePermitConsumption,
        RulePermitConsumptionDraft,
    )
    from policy.rule_authority.storage import MariaDBRuleDecisionStore


class AuthorityLedgerStore(Protocol):
    def get_genesis(self) -> AuthorityLedgerGenesis: ...
    def events(self) -> tuple[AuthorityEvent, ...]: ...
    def append(self, event: AuthorityEvent) -> None: ...


class AuthorityLedgerTransactionOwner(Protocol):
    """One already-started DB transaction shared by both schemas."""
    @property
    def active(self) -> bool: ...
    def lock_permit_for_consumption(
        self, permit_id: str, request_hash: str
    ) -> "RulePermit": ...
    def consume_rule_permit(
        self,
        rule_authority_store: "MariaDBRuleDecisionStore",
        consumption_draft: "RulePermitConsumptionDraft",
    ) -> "RulePermitConsumption": ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


class AuthorityLedgerVerificationFence(Protocol):
    """Fence contract for current, externally anchored Block 4 authority."""
    def verify_for_update(
        self,
        transaction_owner: AuthorityLedgerTransactionOwner,
        trusted_root: "TrustedAuthorityRoot",
    ) -> "ReconstructedAuthorityState": ...
    def transaction_owner(
        self, rule_authority_decision_store
    ) -> ContextManager[AuthorityLedgerTransactionOwner]: ...


class AuthorityLedgerTransactionProvider(AuthorityLedgerStore, Protocol):
    """Production contract for Block 4's external checkpoint fence.

    The shared transaction principal also needs read-only SELECT on the three
    Block 4 genesis/events/head tables; provisioning those cross-schema grants
    belongs to Rule Authority wiring, not this storage adapter.
    """
    def checkpoint_fence(self, checkpoint_store) -> ContextManager[AuthorityLedgerVerificationFence]: ...


@dataclass
class InMemoryAuthorityLedgerStore:
    """Test-only transactional append model with one linear predecessor."""
    genesis: AuthorityLedgerGenesis
    _events: list[AuthorityEvent] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def get_genesis(self) -> AuthorityLedgerGenesis:
        return self.genesis

    def events(self) -> tuple[AuthorityEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def append(self, event: AuthorityEvent) -> None:
        with self._lock:
            expected_sequence = len(self._events) + 1
            predecessor = self._events[-1].event_hash if self._events else None
            if event.sequence != expected_sequence or event.previous_event_hash != predecessor:
                raise AuthorityStateError("append fuera de secuencia/predecesor")
            if any(existing.event_id == event.event_id for existing in self._events):
                raise AuthorityStateError("event_id duplicado")
            self._events.append(event)


class MariaDBAuthorityLedgerStore:
    """Production DB-API adapter. Connection factory owns repository deployment wiring.

    The adapter deliberately uses only INSERT for events and locks the singleton
    head row while checking the signed caller's expected predecessor.
    """
    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connect = connection_factory

    @contextmanager
    def checkpoint_fence(self, checkpoint_store):
        """Hold Block 4's checkpoint flock before RA's lock and shared DB transaction.

        Required order is Block 4 checkpoint flock, Rule Authority checkpoint
        flock, then the shared MariaDB transaction. Use the yielded fence's
        ``transaction_owner(rule_authority_decision_store)`` context manager:
        it acquires the canonical Rule Authority checkpoint flock before BEGIN
        and keeps both locks until commit/rollback and connection close.  The
        decision store supplies both its canonical checkpoint resource and the
        connection principal; callers cannot substitute either one.
        """
        from .trusted_checkpoint import require_trusted_checkpoint_store

        checkpoint_store = require_trusted_checkpoint_store(checkpoint_store)
        with checkpoint_store.locked():
            fence = _AuthorityLedgerCheckpointFence(checkpoint_store)
            fence._active = True
            try:
                yield fence
            except BaseException:
                if fence._owner is not None and fence._owner.active:
                    fence._owner.rollback()
                raise
            else:
                if fence._owner is not None and fence._owner.active:
                    fence._owner.rollback()
                    raise AuthorityStateError(
                        "transaction debe confirmar o revertir antes de salir del checkpoint fence"
                    )
            finally:
                fence._active = False

    def get_genesis(self) -> AuthorityLedgerGenesis:
        from .serialization import genesis_from_projection
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT canonical_genesis FROM jax_authority.authority_ledger_genesis WHERE singleton=1")
            row = cursor.fetchone()
            if row is None:
                raise AuthorityStateError("genesis ausente")
            return genesis_from_projection(row[0])

    def events(self) -> tuple[AuthorityEvent, ...]:
        from .serialization import event_from_storage_row
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT sequence,event_id,event_type,actor_id,canonical_intent,canonical_event,"
                "evidence_refs,previous_event_hash,event_hash,signature,recorded_at_utc "
                "FROM jax_authority.authority_events ORDER BY sequence ASC"
            )
            return tuple(event_from_storage_row(row) for row in cursor.fetchall())

    def append(self, event: AuthorityEvent) -> None:
        from .canonical import canonical_bytes
        from .serialization import event_projection, intent_projection
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT sequence,head_event_hash FROM jax_authority.authority_ledger_head WHERE singleton=1 FOR UPDATE")
            row = cursor.fetchone()
            if row is None:
                raise AuthorityStateError("authority ledger head ausente")
            sequence, predecessor = row
            if event.sequence != sequence + 1 or event.previous_event_hash != predecessor:
                raise AuthorityStateError("append concurrente/stale")
            cursor.execute(
                "INSERT INTO jax_authority.authority_events "
                "(sequence,event_id,event_type,actor_id,canonical_intent,canonical_event,evidence_refs,previous_event_hash,event_hash,signature,recorded_at_utc) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    event.sequence, event.event_id, event.intent.event_type.value,
                    event.intent.actor_id, canonical_bytes(intent_projection(event.intent)),
                    canonical_bytes(event_projection(event)),
                    canonical_bytes(event.intent.evidence_refs), event.previous_event_hash,
                    event.event_hash, event.signature, event.recorded_at_utc,
                ),
            )
            cursor.execute("UPDATE jax_authority.authority_ledger_head SET sequence=%s,head_event_id=%s,head_event_hash=%s WHERE singleton=1", (event.sequence,event.event_id,event.event_hash))
            if cursor.rowcount != 1:
                raise AuthorityStateError("actualización del authority ledger head no tocó exactamente una fila")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


_TRANSACTION_OWNER_SEAL = object()


class MariaDBAuthorityLedgerTransactionOwner:
    """Sealed owner for one MariaDB transaction started after caller-held locks.

    Constructed only by the checkpoint fence after it has acquired both external
    locks. The connection stays private and the cursor stays inaccessible until
    Block 4 verification succeeds.
    """

    def __init__(self, connection, seal, checkpoint_fence) -> None:
        if seal is not _TRANSACTION_OWNER_SEAL:
            raise AuthorityStateError("transaction owner debe crearse desde checkpoint fence")
        self._connection = connection
        self._checkpoint_fence = checkpoint_fence
        self._cursor = None
        self._active = True
        self._verified = False
        self._permit_locked = False
        self._audit_head_locked = False
        self._consumption_persisted = False
        self._locked_permit = None
        self._locked_permit_identity = None
        self._rule_authority_decision_store = None
        self._resources_closed = False
        self._post_commit_close_error = None

    @classmethod
    def _begin(
        cls,
        connection_factory: Callable[[], Any],
        checkpoint_fence: "_AuthorityLedgerCheckpointFence",
        rule_authority_lock_token,
    ) -> "MariaDBAuthorityLedgerTransactionOwner":
        from pymysql.constants import SERVER_STATUS

        if not callable(connection_factory):
            raise AuthorityStateError("transaction owner requiere connection factory privada")
        if not isinstance(checkpoint_fence, _AuthorityLedgerCheckpointFence) or not checkpoint_fence.active:
            raise AuthorityStateError("BEGIN requiere checkpoint fence Block 4 activo")
        from policy.rule_authority.errors import CheckpointInvalido
        from policy.rule_authority.trusted_checkpoint import (
            require_active_authority_ledger_lock_token,
        )

        try:
            require_active_authority_ledger_lock_token(
                rule_authority_lock_token, checkpoint_fence
            )
        except CheckpointInvalido as exc:
            raise AuthorityStateError(
                "BEGIN requiere token Rule Authority activo dentro del fence Block 4"
            ) from exc
        connection = connection_factory()
        get_autocommit = getattr(connection, "get_autocommit", None)
        begin_attempted = False
        try:
            if not callable(get_autocommit) or get_autocommit():
                raise AuthorityStateError("transaction owner requiere conexión MariaDB con autocommit desactivado")
            status = getattr(connection, "server_status", None)
            if type(status) is not int:
                raise AuthorityStateError("transaction owner requiere estado MariaDB verificable")
            if status & SERVER_STATUS.SERVER_STATUS_IN_TRANS:
                raise AuthorityStateError("transaction owner no puede anidar sobre una transacción previa")
            begin_attempted = True
            connection.begin()
            status = getattr(connection, "server_status", None)
            if type(status) is not int or not status & SERVER_STATUS.SERVER_STATUS_IN_TRANS:
                connection.rollback()
                raise AuthorityStateError("MariaDB no confirmó el inicio de la transacción")
            owner = cls(connection, _TRANSACTION_OWNER_SEAL, checkpoint_fence)
            owner._rule_authority_lock_token = rule_authority_lock_token
            owner._cursor = connection.cursor()
            checkpoint_fence._register_owner(owner)
        except Exception:
            try:
                status = getattr(connection, "server_status", None)
                if (begin_attempted and type(status) is int
                        and status & SERVER_STATUS.SERVER_STATUS_IN_TRANS):
                    connection.rollback()
            finally:
                connection.close()
            raise
        return owner

    @property
    def active(self) -> bool:
        return self._active

    def lock_permit_for_consumption(
        self, permit_id: str, request_hash: str
    ) -> "RulePermit":
        """Lock the exact permit before Block 4 locks authority_ledger_head.

        The filesystem flock order is necessarily Block 4 then Rule Authority
        before BEGIN, because both checkpoints must remain durable for the
        whole transaction.  This method defines the independent *InnoDB row*
        order for consumption: permit row, then Block 4's authority head (in
        ``verify_for_update``), then Rule Authority's audit-head row.  No raw
        cursor is exposed until Block 4 verification has completed.
        """
        if not self._active:
            raise AuthorityStateError("lock de permit requiere transaction owner activo")
        if self._verified:
            raise AuthorityStateError("lock de permit debe ocurrir antes de Block 4")
        if self._permit_locked:
            raise AuthorityStateError("consumo solo permite un permit bloqueado por transacción")
        if type(permit_id) is not str or type(request_hash) is not str:
            raise AuthorityStateError("lock de permit requiere identificadores string exactos")
        self._cursor.execute(
            "SELECT permit_id,request_id,request_hash,decision_status,rule_id,rule_path,"
            "rule_blob_oid,rule_content_hash,policy_revision,policy_tree_oid,"
            "policy_snapshot_hash,ratification_event_id,authority_ledger_checkpoint,"
            "stop_checkpoint,capability_id,capability_version,capability_class,"
            "capability_limits,issued_at_utc,expires_at_utc,permit_hash,canonical_payload,"
            "previous_record_hash,record_hash,audit_sequence "
            "FROM jax_rule_authority.rule_permits "
            "WHERE permit_id=%s AND request_hash=%s FOR UPDATE",
            (permit_id, request_hash),
        )
        row = self._cursor.fetchone()
        if row is None:
            raise AuthorityStateError("permit ausente o no coincide con request_hash")
        try:
            from policy.rule_authority.permit import RulePermit
            from policy.rule_authority.storage import MariaDBRuleDecisionStore
            permit = MariaDBRuleDecisionStore._decode_permit_row(row)
        except Exception as exc:
            raise AuthorityStateError("permit bloqueado no es canónico") from exc
        if (type(permit) is not RulePermit or permit._is_store_sealed() is not True
                or permit.permit_id != permit_id or permit.request_hash != request_hash):
            raise AuthorityStateError("permit ausente o no coincide con request_hash")
        self._permit_locked = True
        self._locked_permit = permit
        self._locked_permit_identity = (permit_id, request_hash)
        return permit

    def consume_rule_permit(
        self,
        rule_authority_store: "MariaDBRuleDecisionStore",
        consumption_draft: "RulePermitConsumptionDraft",
    ) -> "RulePermitConsumption":
        """Ask the bound Rule Authority store to append the one consumption record.

        The owner exposes no general cursor: the Rule Authority adapter owns the
        row shape, canonical payload and hash-chain update.  This wrapper only
        enforces the cross-schema transaction state and binds the operation to
        the exact store which acquired the Rule Authority checkpoint fence.
        """
        if not self._active or not self._verified:
            raise AuthorityStateError("consumo requiere verificación Block 4 previa")
        if not self._permit_locked or self._locked_permit_identity is None:
            raise AuthorityStateError("consumo requiere lock de permit previo")
        if self._consumption_persisted:
            raise AuthorityStateError("consumo ya fue persistido por esta transacción")
        if rule_authority_store is not self._rule_authority_decision_store:
            raise AuthorityStateError("consumo requiere el decision-store ligado al transaction owner")
        return rule_authority_store._finalize_consumption_in_transaction(self, consumption_draft)

    def _mark_consumption_persisted(self) -> None:
        if (not self._active or not self._verified or not self._permit_locked
                or not self._audit_head_locked or self._consumption_persisted):
            raise AuthorityStateError("estado inválido al sellar consumo atómico")
        self._consumption_persisted = True

    def _commit_consumption_for_store(self) -> BaseException | None:
        """Private completion path; callers cannot commit a consumption half."""
        if not self._consumption_persisted:
            raise AuthorityStateError("consumo no fue persistido para finalizar")
        self._commit_db()
        return self._post_commit_close_error

    def _commit_db(self) -> None:
        if not self._active:
            raise AuthorityStateError("transacción Rule Authority ya finalizada")
        try:
            self._connection.commit()
        except BaseException:
            try:
                self._connection.rollback()
            finally:
                self._active = False
                try:
                    self._close_resources()
                except BaseException:
                    # A failed commit never becomes a committed outcome merely
                    # because cleanup also failed. Preserve the commit failure.
                    pass
            raise
        self._active = False
        try:
            self._close_resources()
        except BaseException as exc:
            # COMMIT already succeeded.  The external Rule Authority anchor
            # must still be attempted by the consumption store while both
            # filesystem fences remain held; it reports this as committed
            # state after that attempt.
            self._post_commit_close_error = exc

    def lock_rule_authority_audit_head_for_update(self) -> tuple[int, str | None]:
        """Lock Rule Authority's audit head after the permit and Block 4 head.

        This is the consumption path's final InnoDB lock.  It is intentionally
        unavailable before verification, so its SQL cannot invert the required
        permit → authority_ledger_head → audit_head row order.
        """
        if not self._active or not self._verified:
            raise AuthorityStateError("lock de audit head requiere verificación Block 4 previa")
        if not self._permit_locked:
            raise AuthorityStateError("lock de audit head requiere lock de permit previo")
        if self._audit_head_locked:
            raise AuthorityStateError("audit head ya fue bloqueado por esta transacción")
        self._cursor.execute(
            "SELECT audit_sequence,head_hash FROM jax_rule_authority.rule_authority_audit_head "
            "WHERE singleton=1 FOR UPDATE"
        )
        row = self._cursor.fetchone()
        if (not isinstance(row, tuple) or len(row) != 2
                or type(row[0]) is not int or (row[1] is not None and type(row[1]) is not str)):
            raise AuthorityStateError("rule authority audit head ausente/ inválido")
        self._audit_head_locked = True
        return row

    def commit(self) -> None:
        if not self._active:
            raise AuthorityStateError("transacción Rule Authority ya finalizada")
        if self._permit_locked:
            self.rollback()
            raise AuthorityStateError(
                "consumo RulePermit se finaliza solo mediante consume_permit"
            )
        self._commit_db()

    def rollback(self) -> None:
        if not self._active:
            raise AuthorityStateError("transacción Rule Authority ya finalizada")
        try:
            self._connection.rollback()
        finally:
            self._active = False
            self._close_resources()

    def close(self) -> None:
        """Close the cursor; rollback first if the owner was not finalized."""
        if self._active:
            self.rollback()
        else:
            self._close_resources()

    def _close_resources(self) -> None:
        if self._resources_closed:
            return
        self._resources_closed = True
        try:
            if self._cursor is not None:
                close = getattr(self._cursor, "close", None)
                if callable(close):
                    close()
                self._cursor = None
        finally:
            self._connection.close()

    def _verify_for_update(
        self, checkpoint_store, trusted_root: "TrustedAuthorityRoot"
    ) -> "ReconstructedAuthorityState":
        """Internal verifier, called only by an active checkpoint-fence object."""
        if not self._active or self._verified:
            raise AuthorityStateError("verificación requiere owner activo sin verificar")
        cursor = self._cursor
        from .replay import verify_authority_ledger
        from .serialization import event_from_storage_row, genesis_from_projection
        from .replay import genesis_hash
        from .canonical import canonical_bytes
        from .errors import TrustedRootMismatchError
        from .signatures import decode_public_key, public_key_bytes, public_key_fingerprint

        cursor.execute(
            "SELECT canonical_genesis,genesis_hash "
            "FROM jax_authority.authority_ledger_genesis WHERE singleton=1"
        )
        genesis_row = cursor.fetchone()
        if genesis_row is None or len(genesis_row) != 2:
            raise AuthorityStateError("genesis ausente/ inválido")
        genesis = genesis_from_projection(genesis_row[0])
        stored_genesis = genesis_row[0]
        if isinstance(stored_genesis, str):
            stored_genesis = stored_genesis.encode("utf-8")
        if stored_genesis != canonical_bytes(genesis.projection()):
            raise AuthorityStateError("canonical_genesis no es canónico")
        if genesis_row[1] != genesis_hash(genesis):
            raise AuthorityStateError("genesis hash almacenado no coincide")

        public = decode_public_key(genesis.constitutional_public_key)
        if (trusted_root.ledger_identity != genesis.ledger_identity
                or trusted_root.genesis_hash != genesis_hash(genesis)
                or trusted_root.constitutional_key_id != genesis.constitutional_key_id
                or trusted_root.constitutional_public_key_fingerprint
                != public_key_fingerprint(public_key_bytes(public))):
            raise TrustedRootMismatchError("genesis no coincide con trusted root")

        # Root authentication precedes filesystem sync, matching replay.py's
        # ordering and preventing an invalid root from masking its own failure.
        checkpoint_store.sync_for_verification()

        cursor.execute(
            "SELECT sequence,head_event_id,head_event_hash "
            "FROM jax_authority.authority_ledger_head WHERE singleton=1 FOR UPDATE"
        )
        head = cursor.fetchone()
        if head is None or len(head) != 3:
            raise AuthorityStateError("authority ledger head ausente/ inválido")

        # Replay is O(n) by design until a measured load test justifies a safe
        # acceleration. The locked head serializes writers; immutable event rows
        # therefore need no per-row FOR UPDATE locks.
        cursor.execute(
            "SELECT sequence,event_id,event_type,actor_id,canonical_intent,canonical_event,"
            "evidence_refs,previous_event_hash,event_hash,signature,recorded_at_utc "
            "FROM jax_authority.authority_events ORDER BY sequence ASC"
        )
        events = tuple(event_from_storage_row(row) for row in cursor.fetchall())
        state = verify_authority_ledger(
            genesis, events, trusted_root, checkpoint_store
        )

        sequence, event_id, head_hash = head
        expected_head = (
            state.checkpoint.sequence,
            events[-1].event_id if events else None,
            state.checkpoint.head_event_hash,
        )
        if (type(sequence) is not int or (sequence, event_id, head_hash) != expected_head):
            raise AuthorityStateError("authority ledger head no coincide con replay verificado")
        return state

    def _mark_verified(self) -> None:
        if not self._active:
            raise AuthorityStateError("owner finalizado durante verificación")
        self._verified = True


class _AuthorityLedgerCheckpointFence:
    """Active Block 4 lock scope that verifies a sealed transaction owner."""

    def __init__(self, checkpoint_store) -> None:
        self._checkpoint_store = checkpoint_store
        self._active = False
        self._owner = None

    @property
    def active(self) -> bool:
        return self._active

    def _register_owner(self, owner) -> None:
        if not self._active or self._owner is not None or owner._checkpoint_fence is not self:
            raise AuthorityStateError("transaction owner no corresponde al checkpoint fence activo")
        self._owner = owner

    @contextmanager
    def transaction_owner(self, rule_authority_decision_store):
        """Acquire the wired RA fence, begin, and finalize before unlock.

        The caller passes the real MariaDB decision store instead of a free
        checkpoint path or connection factory.  That binds this transaction to
        the exact durable audit resource and DB principal that persist permits.
        """
        if not self._active:
            raise AuthorityStateError("transaction owner requiere checkpoint fence Block 4 activo")
        from policy.rule_authority.trusted_checkpoint import RuleAuditCheckpointStore
        from policy.rule_authority.storage import MariaDBRuleDecisionStore

        if type(rule_authority_decision_store) is not MariaDBRuleDecisionStore:
            raise AuthorityStateError(
                "transaction owner requiere decision-store MariaDB canónico"
            )
        rule_authority_checkpoint_store = rule_authority_decision_store._checkpoint_store
        if type(rule_authority_checkpoint_store) is not RuleAuditCheckpointStore:
            raise AuthorityStateError("decision-store no tiene checkpoint canónico RuleAuditCheckpointStore")
        connection_factory = rule_authority_decision_store._connect
        with rule_authority_checkpoint_store.locked():
            rule_authority_lock_token = (
                rule_authority_checkpoint_store.issue_authority_ledger_lock_token(self)
            )
            owner = None
            try:
                owner = MariaDBAuthorityLedgerTransactionOwner._begin(
                    connection_factory, self, rule_authority_lock_token
                )
                owner._rule_authority_decision_store = rule_authority_decision_store
                yield owner
            except BaseException:
                if owner is not None and owner.active:
                    owner.rollback()
                raise
            else:
                if owner.active:
                    owner.rollback()
                    raise AuthorityStateError(
                        "transaction debe confirmar o revertir antes de salir del lock Rule Authority"
                    )
            finally:
                rule_authority_lock_token._deactivate()

    def verify_for_update(
        self,
        transaction_owner: AuthorityLedgerTransactionOwner,
        trusted_root: "TrustedAuthorityRoot",
    ) -> "ReconstructedAuthorityState":
        if not self._active:
            raise AuthorityStateError("checkpoint fence fuera de su sección crítica")
        if (not isinstance(transaction_owner, MariaDBAuthorityLedgerTransactionOwner)
                or not transaction_owner.active
                or transaction_owner._checkpoint_fence is not self
                or transaction_owner._verified):
            raise AuthorityStateError("verify_for_update requiere transaction owner MariaDB activo y sellado")
        if self._owner is not transaction_owner:
            raise AuthorityStateError("transaction owner no fue creado dentro de este checkpoint fence")
        from policy.rule_authority.errors import CheckpointInvalido
        from policy.rule_authority.trusted_checkpoint import (
            require_active_authority_ledger_lock_token,
        )
        try:
            require_active_authority_ledger_lock_token(
                transaction_owner._rule_authority_lock_token, self
            )
        except (AttributeError, CheckpointInvalido) as exc:
            raise AuthorityStateError(
                "verify_for_update requiere token Rule Authority activo y sellado"
            ) from exc
        state = transaction_owner._verify_for_update(self._checkpoint_store, trusted_root)
        transaction_owner._mark_verified()
        return state
