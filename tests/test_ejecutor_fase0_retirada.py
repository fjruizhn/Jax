# tests/test_ejecutor_fase0_retirada.py
"""El lanzador de la Fase 0 ya no lanza: esquivaba la jaula, el gancho y el registro."""
import importlib.util
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_fase0" / "harness.py"


def _harness():
    spec = importlib.util.spec_from_file_location("fase0_harness_retirada", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_correr_ya_no_lanza():
    h = _harness()
    with pytest.raises(h.Fase0Retirada):
        h.correr("http://127.0.0.1:18435", "m", "hola")


def test_el_modulo_no_llama_subprocesos():
    fuente = RUTA.read_text(encoding="utf-8")
    assert "subprocess" not in fuente
