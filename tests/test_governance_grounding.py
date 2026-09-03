"""
Tests de policy/governance/grounding.py — REFORMAS Fase 2 SP3.

Todo acá es PURO: sin I/O, sin DB, sin red. Los ValidationContext se arman a
mano. Numeración de pruebas = spec §9.1 / §9.1b.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOVERNANCE = REPO_ROOT / "policy" / "governance"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GOVERNANCE))

import grounding  # noqa: E402
import validator  # noqa: E402
from las_manos.motor_registry.catalog import MotorCatalog  # noqa: E402


def _ctx(ops: set[str], mutating: set[str] = frozenset({"write_file"})) -> validator.ValidationContext:
    return validator.ValidationContext(
        ops=frozenset(ops),
        mutating_capabilities=frozenset(mutating),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )


# Contexto A del spec: write_file es mutante. Con orden por name,
# 'read_file' < 'write_file', así que write_file queda en /capabilities/1.
CTX_A = _ctx({"write_file", "read_file"})


# --- normalize_args ---------------------------------------------------------

def test_normalize_args_str_and_strip_every_value():
    assert grounding.normalize_args({"name": " write_file ", "mode": 1}) == {"name": "write_file", "mode": "1"}


def test_normalize_args_keeps_keys_untouched():
    # Las claves NO se normalizan: ARGS_MISMATCH (paso 2) es quien juzga claves.
    assert grounding.normalize_args({" Name": "x"}) == {" Name": "x"}


# --- build_snapshot: 7a determinismo -----------------------------------------

def test_7a_same_ctx_same_sha256():
    assert grounding.build_snapshot(CTX_A).sha256 == grounding.build_snapshot(CTX_A).sha256


def test_7a_different_ctx_different_sha256():
    ctx_b = _ctx({"read_file"})
    assert grounding.build_snapshot(CTX_A).sha256 != grounding.build_snapshot(ctx_b).sha256


def test_sha256_is_over_the_canonical_json():
    import hashlib
    snap = grounding.build_snapshot(CTX_A)
    assert snap.sha256 == hashlib.sha256(snap.canonical_json.encode("utf-8")).hexdigest()
    # canónico: sort_keys + separators compactos
    assert snap.canonical_json == json.dumps(
        json.loads(snap.canonical_json), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


# --- 7c: orden por name, no por orden de llegada -----------------------------

def test_7c_entries_ordered_by_name_regardless_of_input_order():
    a = grounding.build_snapshot(_ctx({"write_file", "read_file", "list_dir"}))
    b = grounding.build_snapshot(_ctx({"list_dir", "write_file", "read_file"}))
    assert a.sha256 == b.sha256
    assert [e.pointer for e in a.entries] == ["/capabilities/0", "/capabilities/1", "/capabilities/2"]
    assert [e.args["name"] for e in a.entries] == ["list_dir", "read_file", "write_file"]


def test_snapshot_entry_carries_predicate_and_normalized_args():
    snap = grounding.build_snapshot(CTX_A)
    e = snap.lookup("/capabilities/1")
    assert e is not None
    assert e.predicate == "CAPABILITY_AVAILABLE"
    assert e.args == {"name": "write_file", "mode": "mutating"}
    assert snap.lookup("/capabilities/0").args == {"name": "read_file", "mode": "read_only"}


def test_lookup_returns_none_for_unknown_pointer():
    snap = grounding.build_snapshot(CTX_A)
    assert snap.lookup("/capabilities/99") is None
    assert snap.lookup("/facets/0") is None


# --- 7b: render sin hash ------------------------------------------------------

def test_7b_render_does_not_contain_the_hash_nor_any_prefix_of_it():
    snap = grounding.build_snapshot(CTX_A)
    text = grounding.render(snap)
    assert snap.sha256 not in text
    assert snap.sha256[:12] not in text
    assert "sha256" not in text


def test_render_lists_every_entry_with_its_pointer_and_args():
    snap = grounding.build_snapshot(CTX_A)
    text = grounding.render(snap)
    assert "/capabilities/0: name=read_file, mode=read_only" in text
    assert "/capabilities/1: name=write_file, mode=mutating" in text
    assert "evidence_pointer" in text  # la instrucción de cómo citar


# --- 7d: fallo ruidoso (P10) ---------------------------------------------------

def test_7d_build_snapshot_raises_on_broken_ctx_never_returns_empty():
    class Broken:
        @property
        def ops(self):
            raise OSError("config ilegible")
        mutating_capabilities = frozenset()
    with pytest.raises(grounding.GroundingBuildError):
        grounding.build_snapshot(Broken())


def test_empty_ops_is_a_valid_snapshot_not_an_error():
    # Cero capabilities es una OBSERVACIÓN válida (spec: "predicado con
    # resolver y cero entradas = observación válida, no UNGROUNDED").
    snap = grounding.build_snapshot(_ctx(set()))
    assert snap.entries == ()
    assert len(snap.sha256) == 64
