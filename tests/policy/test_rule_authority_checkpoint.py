from __future__ import annotations

import json
import multiprocessing
import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore
from policy.rule_authority.errors import (
    CheckpointDurabilityUnknown, CheckpointInvalido,
)
from policy.rule_authority.storage import MariaDBRuleDecisionStore
from policy.rule_authority.trusted_checkpoint import (
    RuleAuditCheckpointStore,
    _checkpoint_hash,
)
from policy.rule_authority.providers import (
    AlmacenCheckpointAuditoria,
    AlmacenCheckpointAuditoriaBloqueable,
)
from tests.policy.proveedores_dobles import CheckpointsEnMemoria


def _bootstrapped_store(path):
    store = RuleAuditCheckpointStore(path)
    store.bootstrap()
    return store


def test_publica_y_confirma_heads_exactos_incluso_tras_un_descendiente(tmp_path):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")

    store.publicar("sha256:" + "1" * 64, anterior=store.head_actual())
    store.publicar("sha256:" + "2" * 64, anterior="sha256:" + "1" * 64)

    assert store.head_actual() == "sha256:" + "2" * 64
    assert store.confirmar("sha256:" + "1" * 64) is True
    assert store.confirmar("sha256:" + "2" * 64) is True
    assert store.confirmar("sha256:" + "3" * 64) is False


def test_solo_idempotencia_cas_exacta_y_sin_retroceso(tmp_path):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    first = "sha256:" + "1" * 64
    second = "sha256:" + "2" * 64

    genesis = store.head_actual()
    store.publicar(first, anterior=genesis)
    store.publicar(first, anterior=genesis)
    store.publicar(second, anterior=first)

    with pytest.raises(CheckpointInvalido):
        store.publicar(first, anterior=second)
    with pytest.raises(CheckpointInvalido):
        store.publicar(second, anterior="")
    with pytest.raises(CheckpointInvalido):
        store.publicar("sha256:" + "3" * 64, anterior=first)


def test_archivo_no_canonico_truncado_o_con_cadena_rota_falla_cerrado(tmp_path):
    path = tmp_path / "audit-checkpoints.jsonl"
    store = _bootstrapped_store(path)
    store.publicar("sha256:" + "1" * 64, anterior=store.head_actual())
    original = path.read_bytes()

    path.write_bytes(original[:-1])
    with pytest.raises(CheckpointInvalido):
        store.head_actual()

    path.write_bytes(b"{}\n")
    with pytest.raises(CheckpointInvalido):
        store.confirmar("sha256:" + "1" * 64)

    path.write_bytes(original.replace(b"sha256:", b"sha256x:", 1))
    with pytest.raises(CheckpointInvalido):
        store.head_actual()


def test_concurrencia_cas_no_pierde_o_sobrescribe_heads(tmp_path):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    store.publicar("sha256:" + "0" * 64, anterior=store.head_actual())

    def publish(digit: str):
        try:
            store.publicar("sha256:" + digit * 64, anterior="sha256:" + "0" * 64)
            return "ok"
        except CheckpointInvalido:
            return "rejected"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(publish, "12345678"))

    assert outcomes.count("ok") == 1
    assert outcomes.count("rejected") == 7
    assert store.head_actual() in {"sha256:" + digit * 64 for digit in "12345678"}


def test_reinicio_relee_log_y_no_reinicializa_un_head_existente(tmp_path):
    path = tmp_path / "audit-checkpoints.jsonl"
    first = _bootstrapped_store(path)
    first.publicar("sha256:" + "1" * 64, anterior=first.head_actual())

    second = RuleAuditCheckpointStore(path)
    assert second.head_actual() == "sha256:" + "1" * 64
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["sequence"] == 0


def _bloquear_checkpoint(path, ready, acquired):
    store = RuleAuditCheckpointStore(path)
    ready.set()
    with store.locked():
        acquired.set()


def test_flock_serializa_procesos_distintos(tmp_path):
    path = tmp_path / "audit-checkpoints.jsonl"
    store = _bootstrapped_store(path)
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    acquired = context.Event()
    with store.locked():
        process = context.Process(target=_bloquear_checkpoint, args=(path, ready, acquired))
        process.start()
        assert ready.wait(5)
        assert not acquired.wait(0.2)
    assert acquired.wait(5)
    process.join(5)
    assert process.exitcode == 0


def test_lock_se_crea_con_modo_restringido_antes_de_aplicar_umask(tmp_path, monkeypatch):
    store = RuleAuditCheckpointStore(tmp_path / "audit-checkpoints.jsonl")
    real_open = os.open
    observed = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(path) == os.fspath(store.lock_path):
            observed.append(mode)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", recording_open)
    previous = os.umask(0)
    try:
        with store.locked():
            assert stat.S_IMODE(store.lock_path.stat().st_mode) == 0o600
    finally:
        os.umask(previous)
    assert observed == [0o600]


def test_lock_limpia_temporales_de_publicacion_abandonados(tmp_path):
    path = tmp_path / "audit-checkpoints.jsonl"
    store = _bootstrapped_store(path)
    abandoned = tmp_path / f".{path.name}.tmp-{'a' * 32}"
    abandoned.write_bytes(b"partial checkpoint")
    abandoned.chmod(0o600)

    assert store.head_actual()
    assert not abandoned.exists()


def test_protocol_declara_locked_y_store_real_lo_implementa(tmp_path):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    assert isinstance(store, AlmacenCheckpointAuditoria)
    assert isinstance(store, AlmacenCheckpointAuditoriaBloqueable)
    memory = CheckpointsEnMemoria()
    assert isinstance(memory, AlmacenCheckpointAuditoria)
    assert not isinstance(memory, AlmacenCheckpointAuditoriaBloqueable)


def test_storage_mariadb_exige_checkpoint_bloqueable(tmp_path):
    with pytest.raises(TypeError, match="contrato durable bloqueable"):
        MariaDBRuleDecisionStore(lambda: None, checkpoint_store=CheckpointsEnMemoria())
    checkpoint_block4 = TrustedCheckpointStore(
        tmp_path / "block4-checkpoints.jsonl",
        bootstrap_receipt_path=tmp_path / "block4-receipt.json",
    )
    with pytest.raises(TypeError, match="contrato durable bloqueable"):
        MariaDBRuleDecisionStore(lambda: None, checkpoint_store=checkpoint_block4)
    MariaDBRuleDecisionStore(
        lambda: None,
        checkpoint_store=RuleAuditCheckpointStore(tmp_path / "rule-authority-checkpoint-contract.jsonl"),
    )


def test_lector_rechaza_head_historico_repetido_aunque_la_cadena_sea_valida(tmp_path):
    path = tmp_path / "audit-checkpoints.jsonl"
    store = _bootstrapped_store(path)
    first = "sha256:" + "1" * 64
    store.publicar(first, anterior=store.head_actual())
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    previous = rows[-1]
    repeated = {
        "schema_version": previous["schema_version"],
        "kind": previous["kind"],
        "sequence": previous["sequence"] + 1,
        "head": first,
        "previous_head": first,
        "previous_checkpoint_hash": previous["checkpoint_hash"],
    }
    repeated["checkpoint_hash"] = _checkpoint_hash(repeated)
    path.write_bytes(path.read_bytes() + canonical_bytes(repeated) + b"\n")

    with pytest.raises(CheckpointInvalido, match="repite un head histórico"):
        store.head_actual()


def test_fallo_fsync_del_archivo_conserva_el_head_anterior(tmp_path, monkeypatch):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    genesis = store.head_actual()

    def fail_fsync(_fd):
        raise OSError("injected file fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(CheckpointInvalido, match="preparar checkpoint antes de replace"):
        store.publicar("sha256:" + "1" * 64, anterior=genesis)

    assert store.head_actual() == genesis


def test_fallo_fsync_directorio_despues_de_replace_es_publicacion_incierta(tmp_path, monkeypatch):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    genesis = store.head_actual()
    new_head = "sha256:" + "1" * 64

    def fail_directory_fsync(_path):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(RuleAuditCheckpointStore, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(CheckpointDurabilityUnknown, match="desconocido tras replace"):
        store.publicar(new_head, anterior=genesis)

    assert store._durability_unknown_head == new_head
    from policy.rule_authority import RuleAuthorityDurabilityUnknown as public_error
    from policy.rule_authority.errors import RuleAuthorityDurabilityUnknown as error_type
    assert public_error is error_type
    from policy.rule_authority.storage import (
        RuleAuthorityCommittedCheckpointError,
        _confirmar_checkpoint_durable,
        _head_actual_checkpoint_durable,
        _publicar_checkpoint_durable,
    )

    class GenericPublicationFailure:
        def publicar(self, _head, *, anterior):
            raise CheckpointInvalido("fallo determinista antes de replace")

    with pytest.raises(RuleAuthorityCommittedCheckpointError) as generic_failure:
        _publicar_checkpoint_durable(
            GenericPublicationFailure(), new_head, genesis,
            error_message="DB committed",
        )
    assert not isinstance(generic_failure.value, public_error)

    class UncertainConfirmation:
        def confirmar(self, _head):
            raise CheckpointDurabilityUnknown("fsync incierto")

    with pytest.raises(public_error, match="DURABILITY_UNKNOWN"):
        _confirmar_checkpoint_durable(UncertainConfirmation(), new_head)

    class UncertainHead:
        def head_actual(self):
            raise CheckpointDurabilityUnknown("fsync incierto al abrir proceso")

    with pytest.raises(public_error, match="DURABILITY_UNKNOWN"):
        _head_actual_checkpoint_durable(UncertainHead())
    # The in-process marker makes even a head read reconcile before exposing
    # it as usable. A visible rename alone is not an acknowledgement.
    with pytest.raises(CheckpointDurabilityUnknown):
        store.head_actual()
    monkeypatch.undo()
    assert store.confirmar(new_head) is True
    assert store._durability_unknown_head is None
    assert store.head_actual() == new_head


def test_publicar_no_retorna_exito_si_relectura_final_no_contiene_el_head(
    tmp_path, monkeypatch
):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    genesis = store.head_actual()
    previous_payload = store.path.read_bytes()
    original_replace = RuleAuditCheckpointStore._replace_log

    def replace_then_restore_previous_log(instance, rows):
        original_replace(instance, rows)
        instance.path.write_bytes(previous_payload)

    monkeypatch.setattr(
        RuleAuditCheckpointStore, "_replace_log", replace_then_restore_previous_log
    )
    with pytest.raises(CheckpointDurabilityUnknown, match="relectura exacta"):
        store.publicar("sha256:" + "1" * 64, anterior=genesis)

    monkeypatch.setattr(RuleAuditCheckpointStore, "_replace_log", original_replace)
    with pytest.raises(CheckpointDurabilityUnknown, match="DURABILITY_UNKNOWN"):
        store.head_actual()


def test_confirmar_rechaza_cambio_de_inode_con_mismo_hash_durante_reconciliacion(
        tmp_path, monkeypatch
):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    genesis = store.head_actual()
    new_head = "sha256:" + "1" * 64

    def fail_directory_fsync(_path):
        raise OSError("injected post-replace directory fsync failure")

    with monkeypatch.context() as failed_fsync:
        failed_fsync.setattr(
            RuleAuditCheckpointStore, "_fsync_directory", fail_directory_fsync
        )
        with pytest.raises(CheckpointDurabilityUnknown):
            store.publicar(new_head, anterior=genesis)

    assert store._durability_unknown_head == new_head
    original_fsync = store._fsync_checkpoint_file

    def fsync_then_replace_with_identical_bytes():
        identity = original_fsync()
        replacement = store.path.with_name("checkpoint-identical-replacement.jsonl")
        replacement.write_bytes(store.path.read_bytes())
        os.chmod(replacement, 0o600)
        os.replace(replacement, store.path)
        return identity

    monkeypatch.setattr(
        store, "_fsync_checkpoint_file", fsync_then_replace_with_identical_bytes
    )
    with pytest.raises(CheckpointDurabilityUnknown, match="identidad/hash"):
        store.confirmar(new_head)
    assert store._durability_unknown_head == new_head
    assert store._durability_unknown is True


def test_reintento_no_confirma_head_visible_hasta_reconciliar_durabilidad(tmp_path, monkeypatch):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    genesis = store.head_actual()
    new_head = "sha256:" + "1" * 64

    def fail_directory_fsync(_path):
        raise OSError("injected persistent directory fsync failure")

    with monkeypatch.context() as failed_directory:
        failed_directory.setattr(
            RuleAuditCheckpointStore, "_fsync_directory", fail_directory_fsync
        )
        with pytest.raises(CheckpointInvalido) as publication_failure:
            store.publicar(new_head, anterior=genesis)
        assert isinstance(publication_failure.value, CheckpointDurabilityUnknown)

        # Re-reading a rename that is visible in this process cannot prove that
        # its directory entry survived a crash. Recovery and the identical CAS
        # retry from a recreated store must fail while that durability barrier
        # still fails. Recreating the store models process restart; the DB's
        # committed head remains authoritative in the integration regression.
        restarted_store = RuleAuditCheckpointStore(store.path)
        with pytest.raises(CheckpointDurabilityUnknown, match="DURABILITY_UNKNOWN"):
            restarted_store.head_actual()
        with pytest.raises(CheckpointInvalido) as confirmation_failure:
            restarted_store.confirmar(new_head)
        assert isinstance(confirmation_failure.value, CheckpointDurabilityUnknown)
        with pytest.raises(CheckpointInvalido) as retry_failure:
            restarted_store.publicar(new_head, anterior=genesis)
        assert isinstance(retry_failure.value, CheckpointDurabilityUnknown)

    real_fsync = os.fsync

    def fail_checkpoint_file_fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("injected checkpoint file fsync failure")
        real_fsync(fd)

    with monkeypatch.context() as failed_file:
        failed_file.setattr(os, "fsync", fail_checkpoint_file_fsync)
        with pytest.raises(CheckpointInvalido) as file_failure:
            restarted_store.confirmar(new_head)
        assert isinstance(file_failure.value, CheckpointDurabilityUnknown)

    assert restarted_store.confirmar(new_head) is True
    assert restarted_store._durability_unknown_head is None
    restarted_store.publicar(new_head, anterior=genesis)


def test_no_se_infiere_genesis_desde_un_log_ausente_y_bootstrap_es_unico(tmp_path):
    path = tmp_path / "audit-checkpoints.jsonl"
    store = RuleAuditCheckpointStore(path)
    with pytest.raises(CheckpointInvalido, match="bootstrap"):
        store.publicar("sha256:" + "1" * 64, anterior="")

    store.bootstrap()
    assert isinstance(store, AlmacenCheckpointAuditoria)
    assert store.confirmar(store.head_actual()) is True
    with pytest.raises(CheckpointInvalido, match="ya existe"):
        store.bootstrap()
