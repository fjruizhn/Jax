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
    # Condición 1 de la sesión principal sobre R50: el motivo dice POR QUÉ es
    # global y QUÉ variable lo habilita. Expected contra 6f8bbd5: no nombraba
    # check_facet_health ni que lee y reescribe la tabla entera.
    for parte in ("check_facet_health()", "lee y reescribe", "TODA la tabla",
                  f"{VARIABLE}=1", "base exclusiva", "borraría filas ajenas"):
        assert parte in razones[0], (parte, razones[0])


def test_el_motivo_aparece_en_el_resumen_rs_de_pytest(tmp_path):
    """El skip se ve en `pytest -rs` con el motivo completo (sin base: el
    archivo se colecta y los tres globales se saltan antes de tocar la DB).
    Sin JAX_DB_HOST los cinco quedan por el skip de "necesita MariaDB"; con uno
    de mentira que termina en _test, sólo los tres de R50 se saltan por R50 y
    los demás no se ejecutan (se deseleccionan con -k)."""
    # JAX_TEST_DB_SUFIJO se saca junto con VARIABLE: este test arma el entorno
    # del subproceso A MANO y fija `JAX_DB_NAME`; si el sufijo de la sesión se
    # heredara, el conftest del hijo lo pisaría con la base de la sesión y el
    # subproceso no quedaría en el estado que este test dice estar probando
    # (visto en rojo el 2026-09-18 corriendo la suite con base propia).
    entorno = {k: v for k, v in os.environ.items()
               if k not in (VARIABLE, "JAX_TEST_DB_SUFIJO")}
    entorno.update({"PYTHONPATH": ".:las_manos", "JAX_DB_HOST": "127.0.0.1", "JAX_DB_PORT": "1",
                    "JAX_DB_NAME": "jax_memory_test", "COLUMNS": "400"})
    seleccion = " or ".join(GLOBALES)
    salida = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "-k", seleccion, str(ARCHIVO)],
        cwd=RAIZ, env=entorno, capture_output=True, text=True, timeout=60,
    ).stdout
    lineas = [l for l in salida.splitlines() if l.startswith("SKIPPED")]
    assert len(lineas) == 3, salida[-800:]
    for linea in lineas:
        assert "check_facet_health()" in linea and f"{VARIABLE}=1" in linea, linea


def problemas_del_workflow(ruta: Path) -> list[str]:
    """Condición 2 de la sesión principal sobre R50, leyendo el YAML (no el
    texto): en el job facet-health-io, CADA paso que corre
    jacobs/_facet_health_io_test.py ve JAX_TEST_FACET_HEALTH_TABLA_EXCLUSIVA="1"
    en su entorno efectivo (env del workflow, del job y del paso, en ese orden
    de precedencia de GitHub Actions); el job tiene su propio service de
    MariaDB; y ningún OTRO job ni paso la define. Devuelve la lista de
    problemas (vacía = bien)."""
    import yaml

    flujo = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    problemas = []
    job = flujo.get("jobs", {}).get("facet-health-io")
    if job is None:
        return ["no existe el job facet-health-io"]
    if "mariadb" not in (job.get("services") or {}):
        problemas.append("facet-health-io no tiene service propio de mariadb (la base no es exclusiva)")
    pasos = [p for p in job.get("steps", []) if "_facet_health_io_test.py" in str(p.get("run", ""))]
    if not pasos:
        problemas.append("ningún paso de facet-health-io corre _facet_health_io_test.py")
    for paso in pasos:
        efectivo = {**(flujo.get("env") or {}), **(job.get("env") or {}), **(paso.get("env") or {})}
        if str(efectivo.get(VARIABLE)) != "1":
            problemas.append(f"el paso {str(paso.get('run'))[:60]!r} no ve {VARIABLE}=1 (ve {efectivo.get(VARIABLE)!r})")
    if VARIABLE in (flujo.get("env") or {}):
        problemas.append(f"{VARIABLE} definida a nivel workflow: alcanzaría a jobs con base compartida")
    for nombre, otro in flujo.get("jobs", {}).items():
        if nombre == "facet-health-io":
            continue
        if VARIABLE in (otro.get("env") or {}) or any(VARIABLE in (p.get("env") or {}) for p in otro.get("steps", [])):
            problemas.append(f"el job {nombre} también define {VARIABLE}")
    return problemas


def test_solo_el_job_con_mariadb_propia_define_la_variable_y_alcanza_a_sus_pasos():
    """Rojo verificado con una copia de policy.yml sin la variable (ver
    r38-report.md): 'no ve JAX_TEST_FACET_HEALTH_TABLA_EXCLUSIVA=1' en los dos
    pasos del job."""
    assert problemas_del_workflow(RAIZ / ".github" / "workflows" / "policy.yml") == []
