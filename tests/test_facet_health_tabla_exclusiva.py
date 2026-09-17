"""Compuerta de los tests globales de jacobs/_facet_health_io_test.py (Ruling
R50). Pura: importa el módulo con y sin la variable y mira el marcador, y
colecta con pytest --collect-only (sin tocar la base).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "jacobs" / "_facet_health_io_test.py"
VARIABLE = "JAX_TEST_FACET_HEALTH_TABLA_EXCLUSIVA"
GLOBALES = (
    "test_la_alerta_va_bajo___system___y_no_es_lista_vacia",
    "test_tabla_VACIA_alerta_igual_bajo___system__",
    "test_la_supresion_de_6h_se_respeta_en_la_alerta_agregada",
)
ACOTADOS = ("test_eventos_vencidos_dan_unknown_y_NUNCA_ok", "test_eventos_frescos_no_producen___system__")


def _cargar(monkeypatch, valor):
    if valor is None:
        monkeypatch.delenv(VARIABLE, raising=False)
    else:
        monkeypatch.setenv(VARIABLE, valor)
    spec = importlib.util.spec_from_file_location(f"_fh_io_{valor}", ARCHIVO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _condiciones_de_la_compuerta(fn):
    return [m.args[0] for m in getattr(fn, "pytestmark", [])
            if m.name == "skipif" and "R50" in m.kwargs.get("reason", "")]


@pytest.mark.parametrize("valor", [None, "0", "true", ""])
def test_sin_la_variable_en_1_los_tres_globales_se_saltan(monkeypatch, valor):
    """Expected contra 49f4faa: [] == [True] (no había compuerta)."""
    m = _cargar(monkeypatch, valor)
    for nombre in GLOBALES:
        assert _condiciones_de_la_compuerta(getattr(m, nombre)) == [True], nombre
    for nombre in ACOTADOS:
        assert _condiciones_de_la_compuerta(getattr(m, nombre)) == [], nombre


def test_con_la_variable_en_1_los_tres_globales_no_se_saltan_y_se_colectan(monkeypatch):
    m = _cargar(monkeypatch, "1")
    for nombre in GLOBALES:
        assert _condiciones_de_la_compuerta(getattr(m, nombre)) == [False], nombre
    entorno = {**os.environ, VARIABLE: "1", "PYTHONPATH": ".:las_manos"}
    salida = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ARCHIVO)],
        cwd=RAIZ, env=entorno, capture_output=True, text=True, timeout=60,
    ).stdout
    for nombre in GLOBALES + ACOTADOS:
        assert f"::{nombre}" in salida, salida[-500:]


def test_el_skip_nombra_la_regla_y_el_por_que(monkeypatch):
    m = _cargar(monkeypatch, None)
    razones = [mk.kwargs["reason"] for mk in m.test_tabla_VACIA_alerta_igual_bajo___system__.pytestmark
               if mk.name == "skipif" and "R50" in mk.kwargs.get("reason", "")]
    assert len(razones) == 1
    assert VARIABLE in razones[0] and "borraría filas ajenas" in razones[0]


def test_solo_el_job_con_mariadb_propia_define_la_variable():
    """La variable vive SOLO en el job facet-health-io, que tiene su propio
    service container de MariaDB (base exclusiva)."""
    texto = (RAIZ / ".github" / "workflows" / "policy.yml").read_text(encoding="utf-8")
    trabajos = texto.split("\n  facet-health-io:\n", 1)
    assert len(trabajos) == 2
    job = trabajos[1].split("\n\n  #", 1)[0]
    assert f"{VARIABLE}: \"1\"" in job and "services:" in job and "mariadb:" in job
    assert texto.count(f"{VARIABLE}:") == 1
