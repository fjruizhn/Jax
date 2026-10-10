from __future__ import annotations

import json
import multiprocessing
import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from policy.authority_ledger.canonical import canonical_bytes
from policy.rule_authority.errors import CheckpointInvalido
from policy.rule_authority.storage import MariaDBRuleDecisionStore
from policy.rule_authority.trusted_checkpoint import (
    RuleAuditCheckpointStore,
    _checkpoint_hash,
)
from policy.rule_authority.providers import AlmacenCheckpoints, AlmacenCheckpointsBloqueable
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
    assert isinstance(store, AlmacenCheckpoints)
    assert isinstance(store, AlmacenCheckpointsBloqueable)
    memory = CheckpointsEnMemoria()
    assert isinstance(memory, AlmacenCheckpoints)
    assert not isinstance(memory, AlmacenCheckpointsBloqueable)


def test_storage_mariadb_exige_checkpoint_bloqueable(tmp_path):
    with pytest.raises(TypeError, match="contrato durable bloqueable"):
        MariaDBRuleDecisionStore(lambda: None, checkpoint_store=CheckpointsEnMemoria())
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
    with pytest.raises(CheckpointInvalido, match="resultado de publicación durable desconocido"):
        store.publicar("sha256:" + "1" * 64, anterior=genesis)

    assert store.head_actual() == genesis


def test_fallo_fsync_directorio_despues_de_replace_es_publicacion_incierta(tmp_path, monkeypatch):
    store = _bootstrapped_store(tmp_path / "audit-checkpoints.jsonl")
    genesis = store.head_actual()
    new_head = "sha256:" + "1" * 64

    def fail_directory_fsync(_path):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(RuleAuditCheckpointStore, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(CheckpointInvalido, match="resultado de publicación durable desconocido"):
        store.publicar(new_head, anterior=genesis)

    # The rename is visible, but the failed directory fsync means callers cannot
    # treat publication as acknowledged; a fresh read must validate the file.
    assert store.head_actual() == new_head


def test_no_se_infiere_genesis_desde_un_log_ausente_y_bootstrap_es_unico(tmp_path):
    path = tmp_path / "audit-checkpoints.jsonl"
    store = RuleAuditCheckpointStore(path)
    with pytest.raises(CheckpointInvalido, match="bootstrap"):
        store.publicar("sha256:" + "1" * 64, anterior="")

    store.bootstrap()
    assert isinstance(store, AlmacenCheckpoints)
    assert store.confirmar(store.head_actual()) is True
    with pytest.raises(CheckpointInvalido, match="ya existe"):
        store.bootstrap()
