from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from policy.rule_authority.errors import CheckpointInvalido
from policy.rule_authority.trusted_checkpoint import RuleAuditCheckpointStore
from policy.rule_authority.providers import AlmacenCheckpoints


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
