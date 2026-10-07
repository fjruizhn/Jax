import threading

import pytest

from policy.authority_ledger.errors import (AuthorityStateError, LedgerCheckpointError,
                                            LedgerIntegrityError, LedgerRollbackError,
                                            UnanchoredLedgerHeadError)
from policy.authority_ledger.errors import CheckpointPublicationOutcomeUnknownError
from policy.authority_ledger.models import AuthorityEvent, AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import event_hash, event_unsigned_bytes
from tests.policy.test_authority_ledger_events import verify_authority_ledger
from tests.policy.test_authority_ledger_events import append_authority_event
from policy.authority_ledger.service import reanchor_authority_checkpoint
from policy.authority_ledger.signatures import sign
from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore
from tests.policy.test_authority_ledger_events import setup_ledger


class _ObservingStore:
    """Delegates to a real store and signals the first ledger read."""

    def __init__(self, inner):
        self._inner = inner
        self.events_read = threading.Event()

    def get_genesis(self):
        return self._inner.get_genesis()

    def events(self):
        self.events_read.set()
        return self._inner.events()

    def append(self, event):
        return self._inner.append(event)


def _checkpoint(ledger_identity, sequence, event_id, event_hash):
    from policy.authority_ledger.models import AuthorityLedgerCheckpoint
    return AuthorityLedgerCheckpoint(
        "1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", ledger_identity,
        sequence, event_id, event_hash,
    )


class _CheckpointRoto:
    """TrustedCheckpointStore cuyo append SIEMPRE falla (disco lleno/fsync)."""

    def __init__(self, inner: TrustedCheckpointStore) -> None:
        self._inner = inner

    def append(self, checkpoint) -> None:
        raise OSError("checkpoint: disco lleno")

    def _append_locked(self, checkpoint) -> None:
        raise OSError("checkpoint: disco lleno")

    def locked(self):
        return self._inner.locked()

    def latest(self):
        return self._inner.latest()


class _FsyncDirectoryFailsOnce(TrustedCheckpointStore):
    def __init__(self, path):
        super().__init__(path)
        self.fail_next_directory_fsync = False

    def _fsync_parent_directory(self):
        if self.fail_next_directory_fsync:
            self.fail_next_directory_fsync = False
            raise OSError("fsync directory failed")
        return super()._fsync_parent_directory()


class _RereadFailsOnce(TrustedCheckpointStore):
    def __init__(self, path):
        super().__init__(path)
        self.fail_next_reread = False
        self._publishing = False

    def _append_locked(self, checkpoint):
        self._publishing = True
        try:
            return super()._append_locked(checkpoint)
        finally:
            self._publishing = False

    def latest(self):
        if self._publishing and self.fail_next_reread:
            self.fail_next_reread = False
            raise OSError("reread failed")
        return super().latest()


class _BootstrapReceiptDirectoryFsyncFails(TrustedCheckpointStore):
    def __init__(self, path, *, bootstrap_receipt_path):
        super().__init__(path, bootstrap_receipt_path=bootstrap_receipt_path)
        self._directory_fsyncs = 0

    def _fsync_directory(self, path):
        self._directory_fsyncs += 1
        if self._directory_fsyncs == 2:
            raise OSError("receipt directory fsync failed")
        return super()._fsync_directory(path)


def _signed_event(key, event_id, sequence, previous_event_hash, intent):
    from tests.policy.test_authority_ledger_events import base_time
    provisional = AuthorityEvent(event_id, sequence, previous_event_hash, intent, base_time(), "", "sha256:" + "0" * 64)
    signed = AuthorityEvent(event_id, sequence, previous_event_hash, intent, base_time(), sign(key, event_unsigned_bytes(provisional)), "sha256:" + "0" * 64)
    return AuthorityEvent(event_id, sequence, previous_event_hash, intent, base_time(), signed.signature, event_hash(signed))


class _RevokesAfterSnapshot:
    """Simulates a DB writer committing a revocation between read and append."""

    def __init__(self, inner, key, revocation):
        self._inner, self._key, self._revocation = inner, key, revocation
        self._snapshot_read = False
        self._revoked = False

    def get_genesis(self):
        if self._snapshot_read and not self._revoked:
            prior = self._inner.events()[-1]
            self._inner.append(_signed_event(self._key, "018cc251-f400-7000-8000-000000000002", 2, prior.event_hash, self._revocation))
            self._revoked = True
        return self._inner.get_genesis()

    def events(self):
        self._snapshot_read = True
        return self._inner.events()

    def append(self, event):
        return self._inner.append(event)


def test_checkpoint_changes_when_ledger_advances():
    store, root, key = setup_ledger()
    before = verify_authority_ledger(store.get_genesis(), (), root).checkpoint.authority_ledger_checkpoint_hash
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    assert verify_authority_ledger(store.get_genesis(), store.events(), root).checkpoint.authority_ledger_checkpoint_hash != before


# ---------------- checkpoint tras append: fallo cerrado y reconciliación (auditor de #377)

def test_checkpoint_fallo_deja_huerfano_tipado_y_el_verificador_delata_la_cabeza(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000001", checkpoint_store=anchor)
    # La segunda escritura ancla el evento 2 pero el checkpoint revienta: el
    # evento es append-only y NO se retira — queda huérfano, con error tipado
    # que lo nombra.
    with pytest.raises(LedgerCheckpointError, match="018cc251-f400-7000-8000-000000000002") as excinfo:
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000002", checkpoint_store=_CheckpointRoto(anchor))
    assert "huérfano" in str(excinfo.value)
    assert "reanchor_authority_checkpoint" in str(excinfo.value)
    assert len(store.events()) == 2
    # El verificador reporta la cabeza sin checkpoint y documenta la salida.
    with pytest.raises(UnanchoredLedgerHeadError, match="cabeza sin checkpoint"):
        verify_authority_ledger(store.get_genesis(), store.events(), root, anchor)


def test_reconciliacion_reancla_el_checkpoint_del_head_y_el_verificador_pasa(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000001", checkpoint_store=anchor)
    with pytest.raises(LedgerCheckpointError):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000002", checkpoint_store=_CheckpointRoto(anchor))
    # Reconciliación: verificar el head existente SIN ancla y re-anclarlo.
    checkpoint = reanchor_authority_checkpoint(store, root, anchor)
    assert checkpoint.sequence == 2
    state = verify_authority_ledger(store.get_genesis(), store.events(), root, anchor)
    assert state.checkpoint.sequence == 2


def test_reanchor_advances_from_genesis_checkpoint_after_first_db_append_orphans():
    store, root, key = setup_ledger()
    anchor = store._checkpoint_store
    assert anchor.latest().sequence == 0
    with pytest.raises(LedgerCheckpointError):
        append_authority_event(store, root, key,
            AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"),
            checkpoint_store=_CheckpointRoto(anchor))
    assert len(store.events()) == 1
    assert anchor.latest().sequence == 0
    reconciled = reanchor_authority_checkpoint(store, root, anchor)
    assert reconciled.sequence == 1
    assert verify_authority_ledger(store.get_genesis(), store.events(), root, anchor).checkpoint.sequence == 1


def test_reconciliacion_exige_un_head_existente_y_un_stream_valido(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    with pytest.raises(AuthorityStateError, match="head existente"):
        reanchor_authority_checkpoint(store, root, anchor)
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000001", checkpoint_store=anchor)
    # Un stream corrupto (hash de cadena roto) no se re-ancla: verify niega antes.
    corrupto = list(store.events())
    object.__setattr__(corrupto[0], "event_hash", "sha256:" + "0" * 64)
    class _StoreVista:
        def __init__(self, inner, eventos):
            self._inner, self._events = inner, eventos
        def get_genesis(self):
            return self._inner.get_genesis()
        def events(self):
            return tuple(self._events)
    with pytest.raises(LedgerIntegrityError):
        reanchor_authority_checkpoint(_StoreVista(store, corrupto), root, anchor)


def test_reanchor_no_lava_rollback_de_un_head_anclado(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    for n in range(1, 4):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id=f"018cc251-f400-7000-8000-{n:012d}", checkpoint_store=anchor)
    class _StoreVista:
        def get_genesis(self): return store.get_genesis()
        def events(self): return tuple(store.events()[:2])
    with pytest.raises(LedgerRollbackError):
        reanchor_authority_checkpoint(_StoreVista(), root, anchor)
    assert anchor.latest().sequence == 3


def test_append_con_ancla_rechaza_head_huerfano_antes_de_escribir(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000001", checkpoint_store=anchor)
    with pytest.raises(LedgerCheckpointError):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000002", checkpoint_store=_CheckpointRoto(anchor))
    before = len(store.events())
    with pytest.raises(UnanchoredLedgerHeadError):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000003", checkpoint_store=anchor)
    assert len(store.events()) == before


def test_checkpoint_store_rechaza_rollback_y_head_conflictivo(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    event = append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    from policy.authority_ledger.models import AuthorityLedgerCheckpoint
    checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", store.get_genesis().ledger_identity, 1, event.event_id, event.event_hash)
    anchor.append(checkpoint)
    with pytest.raises(LedgerRollbackError):
        anchor.append(AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", store.get_genesis().ledger_identity, 1, "different", event.event_hash))
    with pytest.raises(LedgerRollbackError):
        anchor.append(AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", store.get_genesis().ledger_identity, 0, None, None))


def test_checkpoint_store_falla_cerrado_con_linea_parcial(tmp_path):
    path = tmp_path / "checkpoints.log"
    from policy.authority_ledger.models import AuthorityLedgerCheckpoint
    from policy.authority_ledger.canonical import canonical_bytes
    checkpoint = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", "ledger", 2, "next", "sha256:" + "1" * 64)
    prior = AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", "ledger", 1, "event", "sha256:" + "0" * 64)
    path.write_bytes(canonical_bytes(prior.projection()))
    with pytest.raises(LedgerIntegrityError, match="línea parcial"):
        TrustedCheckpointStore(path).append(checkpoint)


def test_append_normal_no_reescribe_un_head_db_retrocedido(tmp_path):
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    for n in range(1, 3):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id=f"018cc251-f400-7000-8000-{n:012d}", checkpoint_store=anchor)
    class _StoreVista:
        def get_genesis(self): return store.get_genesis()
        def events(self): return tuple(store.events()[:1])
        def append(self, event): raise AssertionError("no debe escribir")
    with pytest.raises(LedgerRollbackError):
        append_authority_event(_StoreVista(), root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), checkpoint_store=anchor)


def test_append_holds_checkpoint_lock_before_its_first_ledger_read(tmp_path):
    """Removing the service-wide lock lets this append inspect stale state."""
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000001", checkpoint_store=anchor)
    observed = _ObservingStore(store)
    result, failure = [], []

    def append_later():
        try:
            result.append(append_authority_event(observed, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000002", checkpoint_store=anchor))
        except Exception as exc:  # fail-closed: captured and asserted by the parent thread
            failure.append(exc)

    with anchor.locked():
        thread = threading.Thread(target=append_later)
        thread.start()
        assert not observed.events_read.wait(0.15), "append leyó el ledger sin el candado compartido"
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not failure
    assert [event.sequence for event in result] == [2]


def test_reanchor_holds_checkpoint_lock_before_its_first_ledger_read(tmp_path):
    """Reanchor shares the writer lock, so it cannot bless a moving head."""
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000001", checkpoint_store=anchor)
    with pytest.raises(LedgerCheckpointError):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000002", checkpoint_store=_CheckpointRoto(anchor))
    observed = _ObservingStore(store)
    result, failure = [], []

    def reanchor_later():
        try:
            result.append(reanchor_authority_checkpoint(observed, root, anchor))
        except Exception as exc:  # fail-closed: captured and asserted by the parent thread
            failure.append(exc)

    with anchor.locked():
        thread = threading.Thread(target=reanchor_later)
        thread.start()
        assert not observed.events_read.wait(0.15), "reanchor leyó el ledger sin el candado compartido"
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not failure
    assert [checkpoint.sequence for checkpoint in result] == [2]


def test_direct_checkpoint_append_waits_for_the_same_writer_lock(tmp_path):
    """Removing the store lock makes the replacement race lose a writer."""
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    first = _checkpoint("ledger", 1, "event-1", "sha256:" + "1" * 64)
    second = _checkpoint("ledger", 2, "event-2", "sha256:" + "2" * 64)
    anchor.append(first)
    done = threading.Event()

    def append_later():
        anchor.append(second)
        done.set()

    with anchor.locked():
        thread = threading.Thread(target=append_later)
        thread.start()
        assert not done.wait(0.15), "append directo reemplazó el archivo sin el candado compartido"
        assert anchor.latest().sequence == 1
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert done.is_set()
    assert anchor.latest().sequence == 2


@pytest.mark.parametrize(
    "first, second",
    [
        (_checkpoint("ledger-a", 1, "event-1", "sha256:" + "1" * 64), _checkpoint("ledger-b", 2, "event-2", "sha256:" + "2" * 64)),
        (_checkpoint("ledger", 1, "event-1", "sha256:" + "1" * 64), _checkpoint("ledger", 1, "event-2", "sha256:" + "2" * 64)),
    ],
)
def test_checkpoint_append_validates_every_existing_row(tmp_path, first, second):
    """Validating only latest silently accepts an identity switch or a sequence gap."""
    from policy.authority_ledger.canonical import canonical_bytes
    path = tmp_path / "checkpoints.log"
    path.write_bytes(canonical_bytes(first.projection()) + b"\n" + canonical_bytes(second.projection()) + b"\n")
    with pytest.raises(LedgerIntegrityError):
        TrustedCheckpointStore(path).append(_checkpoint(second.ledger_identity, 4, "event-4", "sha256:" + "4" * 64))


def test_checkpoint_append_rejects_an_invalid_candidate_before_publication(tmp_path):
    """A malformed row must never become the trusted first checkpoint."""
    from policy.authority_ledger.models import AuthorityLedgerCheckpoint
    anchor = TrustedCheckpointStore(tmp_path / "checkpoints.log")
    invalid = AuthorityLedgerCheckpoint("1.0", "WRONG_KIND", "ledger", 1, "event-1", "sha256:" + "1" * 64)
    with pytest.raises(LedgerIntegrityError):
        anchor.append(invalid)
    assert not anchor.path.exists()


def test_append_uses_one_snapshot_and_stale_store_rejects_candidate_after_concurrent_revocation():
    """A concurrent revoke cannot change a candidate's replay input mid-append."""
    from tests.policy.test_authority_ledger_events import overlay, ratification_intent
    store, root, key = setup_ledger()
    ratification = append_authority_event(store, root, key, ratification_intent(), event_id="018cc251-f400-7000-8000-000000000001")
    revocation = AuthorityEventIntent(AuthorityEventType.RATIFICATION_REVOKED, "human:fernando", ratification_event_id=ratification.event_id)
    racing_store = _RevokesAfterSnapshot(store, key, revocation)
    with pytest.raises(AuthorityStateError, match="append fuera de secuencia/predecesor"):
        append_authority_event(racing_store, root, key, AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=overlay("racing-overlay")), event_id="018cc251-f400-7000-8000-000000000003")
    assert [event.intent.event_type for event in store.events()] == [AuthorityEventType.RATIFICATION_GRANTED, AuthorityEventType.RATIFICATION_REVOKED]


@pytest.mark.parametrize("store_type, failure", [(_FsyncDirectoryFailsOnce, "directory"), (_RereadFailsOnce, "reread")])
def test_post_replace_checkpoint_failure_has_unknown_outcome_and_reanchor_is_idempotent(tmp_path, store_type, failure):
    """After replace, a transient durability/read failure may already have published the head."""
    store, root, key = setup_ledger()
    anchor = store_type(tmp_path / "checkpoints.log")
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000001", checkpoint_store=anchor)
    if failure == "directory":
        anchor.fail_next_directory_fsync = True
    else:
        anchor.fail_next_reread = True
    with pytest.raises(CheckpointPublicationOutcomeUnknownError, match="resultado de publicación desconocido") as excinfo:
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), event_id="018cc251-f400-7000-8000-000000000002", checkpoint_store=anchor)
    assert "sin checkpoint externo" not in str(excinfo.value)
    assert anchor.latest().sequence == 2
    assert reanchor_authority_checkpoint(store, root, anchor).sequence == 2
    assert verify_authority_ledger(store.get_genesis(), store.events(), root, anchor).checkpoint.sequence == 2


def test_current_replay_requires_checkpoint_while_history_has_distinct_type():
    from policy.authority_ledger.errors import AuthorityStateError
    from policy.authority_ledger.replay import HistoricalAuthorityState, ReconstructedAuthorityState, replay_authority_history
    from policy.authority_ledger.service import append_authority_event as append_current
    from tests.policy.test_decision_input import context, instant
    from policy.authority_ledger.effective_context import build_effective_authority_context
    store, root, key = setup_ledger()
    with pytest.raises(TypeError):
        append_current(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    assert store.events() == ()
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    from policy.authority_ledger.replay import verify_authority_ledger as verify_current
    with pytest.raises(TypeError):
        verify_current(store.get_genesis(), store.events(), root)
    historical = replay_authority_history(store.get_genesis(), store.events(), root)
    assert type(historical) is HistoricalAuthorityState
    assert not isinstance(historical, ReconstructedAuthorityState)
    with pytest.raises(AuthorityStateError, match="no verificado"):
        build_effective_authority_context(historical, context(), instant())


def test_bootstrap_receipt_is_required_only_for_sequence_zero_anchor(tmp_path):
    from policy.authority_ledger.models import AuthorityLedgerCheckpoint
    from policy.authority_ledger.errors import LedgerIntegrityError
    store, root, key = setup_ledger()
    anchor = store._checkpoint_store
    verify_authority_ledger(store.get_genesis(), (), root, anchor)
    original_receipt = anchor.bootstrap_receipt_path.read_bytes()
    anchor.bootstrap_receipt_path.write_bytes(original_receipt.replace(b"sha256:", b"sha256:x", 1))
    with pytest.raises(LedgerIntegrityError, match="recibo bootstrap"):
        verify_authority_ledger(store.get_genesis(), (), root, anchor)
    anchor.bootstrap_receipt_path.write_bytes(original_receipt)
    event = append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"))
    anchor.bootstrap_receipt_path.unlink()
    with pytest.raises(LedgerIntegrityError, match="recibo bootstrap"):
        verify_authority_ledger(store.get_genesis(), store.events(), root, anchor)

    legacy = TrustedCheckpointStore(tmp_path / "legacy.log", bootstrap_receipt_path=tmp_path / "no-receipt.json")
    legacy.append(AuthorityLedgerCheckpoint("1.0", "JAX_AUTHORITY_LEDGER_CHECKPOINT", store.get_genesis().ledger_identity, 1, event.event_id, event.event_hash))
    assert verify_authority_ledger(store.get_genesis(), store.events(), root, legacy).checkpoint.sequence == 1


def test_initialize_existing_fails_without_modifying_checkpoint_or_receipt():
    from policy.authority_ledger.service import initialize_authority_ledger
    from policy.authority_ledger.errors import LedgerAlreadyInitializedError, LedgerIntegrityError
    store, root, _ = setup_ledger()
    anchor = store._checkpoint_store
    checkpoint_bytes = anchor.path.read_bytes()
    receipt_bytes = anchor.bootstrap_receipt_path.read_bytes()
    with pytest.raises(LedgerAlreadyInitializedError, match="ya inicializado"):
        initialize_authority_ledger(store, store.get_genesis(), root, anchor)
    assert anchor.path.read_bytes() == checkpoint_bytes
    assert anchor.bootstrap_receipt_path.read_bytes() == receipt_bytes
    anchor.path.write_bytes(b"corrupt\n")
    with pytest.raises(LedgerAlreadyInitializedError):
        initialize_authority_ledger(store, store.get_genesis(), root, anchor)
    assert anchor.path.read_bytes() == b"corrupt\n"


def test_missing_checkpoint_cannot_be_recreated_by_append_or_reanchor(tmp_path):
    from policy.authority_ledger.errors import LedgerIntegrityError
    store, root, key = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "trusted.log", bootstrap_receipt_path=tmp_path / "receipt.json")
    append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), checkpoint_store=anchor)
    receipt = anchor.bootstrap_receipt_path.read_bytes()
    events_before = store.events()
    anchor.path.unlink()
    from policy.authority_ledger.service import initialize_authority_ledger
    with pytest.raises(LedgerIntegrityError, match="falta checkpoint"):
        initialize_authority_ledger(store, store.get_genesis(), root, anchor)
    with pytest.raises(LedgerIntegrityError):
        append_authority_event(store, root, key, AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"), checkpoint_store=anchor)
    with pytest.raises(LedgerIntegrityError, match="no puede reconstruir confianza ausente"):
        reanchor_authority_checkpoint(store, root, anchor)
    assert not anchor.path.exists()
    assert store.events() == events_before
    assert anchor.bootstrap_receipt_path.read_bytes() == receipt


def test_concurrent_initializers_publish_one_genesis_anchor(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from policy.authority_ledger.service import initialize_authority_ledger
    store, root, _ = setup_ledger()
    anchor = TrustedCheckpointStore(tmp_path / "trusted.log", bootstrap_receipt_path=tmp_path / "receipt.json")
    def initialize():
        try:
            return initialize_authority_ledger(store, store.get_genesis(), root, anchor)
        except Exception as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: initialize(), range(2)))
    from policy.authority_ledger.errors import LedgerAlreadyInitializedError
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, LedgerAlreadyInitializedError) for result in results) == 1
    assert len(anchor.path.read_bytes().splitlines()) == 1
    assert anchor.latest().sequence == 0
    assert anchor.bootstrap_receipt_path.exists()


def test_bootstrap_post_publish_failure_has_unknown_outcome_and_preserves_artifacts(tmp_path):
    from policy.authority_ledger.service import initialize_authority_ledger
    store, root, _ = setup_ledger()
    anchor = _BootstrapReceiptDirectoryFsyncFails(
        tmp_path / "trusted.log", bootstrap_receipt_path=tmp_path / "receipt.json")
    with pytest.raises(CheckpointPublicationOutcomeUnknownError, match="bootstrap publicó checkpoint cero"):
        initialize_authority_ledger(store, store.get_genesis(), root, anchor)
    assert anchor.path.exists()
    assert anchor.bootstrap_receipt_path.exists()
    assert anchor.latest().sequence == 0
