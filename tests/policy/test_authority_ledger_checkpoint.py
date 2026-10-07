import pytest

from policy.authority_ledger.errors import (AuthorityStateError, LedgerCheckpointError,
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
    with pytest.raises(Exception):
        reanchor_authority_checkpoint(_StoreVista(store, corrupto), root, anchor)
