"""
MotorCatalog lleva el modo de cada capability (tanda A v2, spec 2026-09-14 §3.2).

`mode` sale de la columna capability.mode (jax-platform, NOT NULL sin
default). El constructor por dict (solo tests) usa 'mutating' si no se
declara: fail-closed, en la línea de risk_level='high' y
requires_human_gate=True. Un claim de solo lectura sobre una capability sin
modo declarado da FACT_MISMATCH, nunca VALID.
"""
from __future__ import annotations

import pytest

from las_manos.motor_registry.catalog import CAPABILITY_MODES, MotorCatalog


def test_los_modos_son_los_del_check_de_la_columna():
    assert CAPABILITY_MODES == frozenset({"read_only", "mutating"})


def test_sin_mode_declarado_el_constructor_por_dict_es_fail_closed():
    assert MotorCatalog({"capabilities": {"x": {}}}).get_capability("x").mode == "mutating"


def test_el_mode_declarado_se_respeta():
    assert MotorCatalog({"capabilities": {"x": {"mode": "read_only"}}}).get_capability("x").mode == "read_only"


def test_un_mode_invalido_lanza_con_el_nombre_de_la_capability():
    with pytest.raises(RuntimeError, match="'x'"):
        MotorCatalog({"capabilities": {"x": {"mode": "escritura"}}})


def test_capabilities_sale_ordenado_por_nombre():
    cat = MotorCatalog({"capabilities": {"b": {}, "a": {}, "c": {}}})
    assert [c.name for c in cat.capabilities()] == ["a", "b", "c"]
