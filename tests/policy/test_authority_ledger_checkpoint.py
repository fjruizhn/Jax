import pytest

from policy.authority_ledger.errors import (AuthorityStateError, LedgerCheckpointError,
                                            LedgerIntegrityError, LedgerRollbackError,
                                            UnanchoredLedgerHeadError)
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType
from policy.authority_ledger.replay import verify_authority_ledger
from policy.authority_ledger.service import append_authority_event, reanchor_authority_checkpoint
from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore
from tests.policy.test_authority_ledger_events import setup_ledger


class _CheckpointRoto:
    """TrustedCheckpointStore cuyo append SIEMPRE falla (disco lleno/fsync)."""

    def __init__(self, inner: TrustedCheckpointStore) -> None:
        self._inner = inner

    def append(self, checkpoint) -> None:
        raise OSError("checkpoint: disco lleno")

    def latest(self):
        return self._inner.latest()


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
