"""
Test de policy/governance/loaders.py — I/O de config ESTÁTICA de policy
(predicates.yaml, closed_vocabulary.yaml, render_templates.yaml + hash).

El primer test es el fail-closed de load_templates(): si el hash de
render_templates.yaml no coincide con el registrado en VERSION, el
subsistema no debe cargar nada — ni parcial, ni con warning. Va primero
a propósito (ver plan, Task 3): es la prueba que faltó en
backup-hall9000.sh y no debe ser la que se recorta bajo presión.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import loaders  # noqa: E402


def test_load_templates_fails_closed_on_hash_mismatch(tmp_path, monkeypatch):
    templates_file = tmp_path / "render_templates.yaml"
    templates_file.write_text(
        "templates:\n  FOO:\n    status: definida\n    template: 'x'\n",
        encoding="utf-8",
    )
    version_file = tmp_path / "VERSION"
    version_file.write_text(
        "version: 0.1.0\nsha256: deadbeef\ntemplates_sha256: 0000000000\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(loaders, "TEMPLATES_FILE", templates_file)
    monkeypatch.setattr(loaders, "VERSION_FILE", version_file)

    with pytest.raises(RuntimeError, match="Hash de"):
        loaders.load_templates()


def test_load_templates_happy_path_returns_real_specs():
    templates = loaders.load_templates()
    assert templates["CAPABILITY_AVAILABLE"].status == "definida"
    assert templates["CAPABILITY_AVAILABLE"].template == (
        "La capability {name} está disponible en modo {mode}."
    )
    assert templates["FACET_EXISTS"].status == "pendiente"
    assert templates["FACET_EXISTS"].template is None


def test_load_predicates_returns_all_eight():
    predicates = loaders.load_predicates()
    assert len(predicates) == 8
    assert predicates["CAPABILITY_AVAILABLE"].args == ("name", "mode")
    assert predicates["FILE_EXISTS"].args == ("path", "hash")
    assert predicates["MEMORY_ENTRY_EXISTS"].source_of_truth == "MariaDB jax_memory"


def test_load_vocabulary_flattens_categories_and_keeps_config_paths_separate():
    vocab = loaders.load_vocabulary()
    assert "code_swarm" in vocab.flattened   # capabilities
    assert "ssh_exec" in vocab.flattened     # ops
    assert "hyde" in vocab.flattened         # facets_las_manos / facets_jax
    assert "policy/" in vocab.config_paths
    assert "las_manos/config.toml" in vocab.config_paths
    # config_paths no debe filtrarse al vocabulario léxico plano
    assert "policy/" not in vocab.flattened


def test_load_vocabulary_includes_list_shaped_categories(tmp_path, monkeypatch):
    vocab_file = tmp_path / "closed_vocabulary.yaml"
    vocab_file.write_text(
        "capabilities:\n  code_swarm: {}\n"
        "commands:\n  - trae a hyde\n  - invoca a ada\n"
        "config_paths:\n  - some/path\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loaders, "VOCABULARY_FILE", vocab_file)

    vocab = loaders.load_vocabulary()

    assert "trae a hyde" in vocab.flattened
    assert "invoca a ada" in vocab.flattened
    assert "code_swarm" in vocab.flattened


def test_load_vocabulary_raises_for_malformed_category(tmp_path, monkeypatch):
    vocab_file = tmp_path / "closed_vocabulary.yaml"
    vocab_file.write_text(
        "capabilities:\n  code_swarm: {}\n"
        "broken_category: 'not a dict or list'\n"
        "config_paths:\n  - some/path\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loaders, "VOCABULARY_FILE", vocab_file)

    with pytest.raises(RuntimeError, match="broken_category"):
        loaders.load_vocabulary()


def test_load_vocabulary_tracks_term_categories():
    vocab = loaders.load_vocabulary()
    assert "capabilities" in vocab.term_categories["code_swarm"]
    assert "ops" in vocab.term_categories["ssh_exec"]


def test_load_vocabulary_term_in_multiple_categories():
    vocab = loaders.load_vocabulary()
    ada_categories = vocab.term_categories["ada"]
    assert "facets_las_manos" in ada_categories
    assert "facets_jax" in ada_categories
    assert "motors" in ada_categories


def test_load_vocabulary_config_paths_not_in_term_categories():
    vocab = loaders.load_vocabulary()
    assert "policy/" not in vocab.term_categories
