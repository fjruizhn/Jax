"""Los pisos de CI viven en `ci/pisos.json`, no dentro de `.github/workflows/policy.yml`.

Por qué existe: GitHub deja de ejecutar un workflow cuyo archivo supera unos 512 000
bytes («workflow file issue»); cada PR que subía un piso le sumaba comentarios y
policy.yml ya rozaba el límite. Los números salen a un archivo de datos y el
workflow los lee en tiempo de ejecución con `ci/piso.py`.

Qué fija este archivo (permanente, no depende de ninguna fotografía de master):
  * el lector falla CERRADO: sin archivo, sin parsear o sin la clave, sale con 2 y
    nunca da por bueno un piso;
  * el workflow no conserva ningún piso literal (`grep -qE "^N passed`);
  * cada clave que el workflow usa existe, y cada clave de los datos se usa una vez
    (un piso huérfano sería un piso que nadie compara).
La equivalencia byte a byte con master está en test_pisos_migracion_desde_master.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[2]
WORKFLOW = RAIZ / ".github" / "workflows" / "policy.yml"
PISOS = RAIZ / "ci" / "pisos.json"
PISO_PY = RAIZ / "ci" / "piso.py"

USO = re.compile(r"python3 ci/piso\.py (verificar|minimo) ([^\s)]+)")


def _modulo():
    spec = importlib.util.spec_from_file_location("ci_piso", PISO_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _correr(directorio: Path, *args: str) -> subprocess.CompletedProcess:
    """Corre una copia de piso.py junto al pisos.json que haya en `directorio`."""
    shutil.copy(PISO_PY, directorio / "piso.py")
    return subprocess.run([sys.executable, str(directorio / "piso.py"), *args],
                          capture_output=True, text=True, cwd=directorio)


@pytest.fixture
def salida(tmp_path):
    f = tmp_path / "salida.txt"
    f.write_text("....\n5 passed in 0.10s\n", encoding="utf-8")
    return f


def _escribir(directorio: Path, datos) -> None:
    (directorio / "pisos.json").write_text(
        datos if isinstance(datos, str) else json.dumps(datos), encoding="utf-8")


BUENO = {"version": 1,
         "pisos": {"j/x": {"patron": "^5 passed", "mensaje": "PISO ROTO: se esperaban 5."}},
         "minimos": {"j/m": 7}}


# --- el lector falla cerrado ---------------------------------------------------

def test_cumple_el_piso(tmp_path, salida):
    _escribir(tmp_path, BUENO)
    r = _correr(tmp_path, "verificar", "j/x", str(salida))
    assert r.returncode == 0, r.stderr


def test_piso_roto_sale_1_y_dice_el_mensaje(tmp_path):
    _escribir(tmp_path, BUENO)
    f = tmp_path / "s.txt"
    f.write_text("4 passed in 0.1s\n", encoding="utf-8")
    r = _correr(tmp_path, "verificar", "j/x", str(f))
    assert r.returncode == 1
    assert "PISO ROTO: se esperaban 5." in r.stdout


def test_el_patron_se_ancla_al_inicio_de_linea_como_grep(tmp_path):
    _escribir(tmp_path, BUENO)
    f = tmp_path / "s.txt"
    f.write_text("sobran 15 passed\n", encoding="utf-8")
    assert _correr(tmp_path, "verificar", "j/x", str(f)).returncode == 1


def test_sin_archivo_de_pisos_sale_2(tmp_path, salida):
    r = _correr(tmp_path, "verificar", "j/x", str(salida))
    assert r.returncode == 2
    assert "ERROR" in r.stderr


@pytest.mark.parametrize("contenido", [
    "{no es json",
    "",
    "[]",
    json.dumps({"version": 2, "pisos": {}, "minimos": {}}),
    json.dumps({"version": 1, "pisos": {}}),
    json.dumps({"version": 1, "minimos": {}}),
])
def test_archivo_de_pisos_ilegible_sale_2(tmp_path, salida, contenido):
    _escribir(tmp_path, contenido)
    assert _correr(tmp_path, "verificar", "j/x", str(salida)).returncode == 2
    assert _correr(tmp_path, "minimo", "j/m").returncode == 2


def test_clave_ausente_sale_2(tmp_path, salida):
    _escribir(tmp_path, BUENO)
    assert _correr(tmp_path, "verificar", "j/no-existe", str(salida)).returncode == 2
    assert _correr(tmp_path, "minimo", "j/no-existe").returncode == 2


@pytest.mark.parametrize("entrada", [
    {"patron": "^5 passed"},                      # sin mensaje
    {"mensaje": "x"},                             # sin patron
    {"patron": "", "mensaje": "x"},               # patron vacio: casaria con todo
    {"patron": "^5 passed", "mensaje": ""},
    {"patron": "(sin cerrar", "mensaje": "x"},
    "^5 passed",
    None,
])
def test_piso_mal_formado_sale_2(tmp_path, salida, entrada):
    datos = dict(BUENO, pisos={"j/x": entrada})
    _escribir(tmp_path, datos)
    assert _correr(tmp_path, "verificar", "j/x", str(salida)).returncode == 2


def test_salida_de_pytest_inexistente_sale_2(tmp_path):
    _escribir(tmp_path, BUENO)
    assert _correr(tmp_path, "verificar", "j/x", str(tmp_path / "no-hay.txt")).returncode == 2


@pytest.mark.parametrize("valor", [0, -1, "7", 7.5, True, None])
def test_minimo_invalido_sale_2(tmp_path, valor):
    _escribir(tmp_path, dict(BUENO, minimos={"j/m": valor}))
    assert _correr(tmp_path, "minimo", "j/m").returncode == 2


def test_minimo_valido_lo_imprime(tmp_path):
    _escribir(tmp_path, BUENO)
    r = _correr(tmp_path, "minimo", "j/m")
    assert (r.returncode, r.stdout.strip()) == (0, "7")


@pytest.mark.parametrize("args", [[], ["verificar"], ["verificar", "a"], ["minimo"], ["otra", "a", "b"],
                                  ["verificar", "a", "b", "c"]])
def test_uso_incorrecto_sale_2(tmp_path, args):
    _escribir(tmp_path, BUENO)
    assert _correr(tmp_path, *args).returncode == 2


# --- el workflow real ----------------------------------------------------------

def _pasos_run():
    datos = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for jid, job in datos["jobs"].items():
        for paso in job["steps"]:
            if paso.get("run"):
                yield jid, paso["run"]


def test_el_workflow_no_conserva_pisos_literales():
    texto = WORKFLOW.read_text(encoding="utf-8")
    assert re.findall(r'grep -qE "\^\d+ passed[^"]*"', texto) == []
    assert "PISO ROTO" not in texto, "los mensajes de piso viven en ci/pisos.json"
    assert "len(cases) < 102" not in texto


def test_cada_clave_usada_existe_y_cada_clave_de_los_datos_se_usa_una_vez():
    datos = json.loads(PISOS.read_text(encoding="utf-8"))
    usos = [(modo, clave) for _, run in _pasos_run() for modo, clave in USO.findall(run)]
    pisos_usados = [c for m, c in usos if m == "verificar"]
    minimos_usados = [c for m, c in usos if m == "minimo"]
    assert sorted(pisos_usados) == sorted(datos["pisos"]), "pisos huérfanos o claves inexistentes"
    assert sorted(minimos_usados) == sorted(datos["minimos"])
    assert len(set(pisos_usados)) == len(pisos_usados), "una clave se usa en dos pasos"


def test_cada_llamada_del_workflow_sale_con_el_codigo_del_lector():
    """Un `verificar` que falla en mitad de un script con `set -e` ausente no puede
    seguir de largo: cada llamada propaga el código (`|| exit $?`) o es la última línea."""
    for jid, run in _pasos_run():
        lineas = [l.strip() for l in run.splitlines() if l.strip() and not l.strip().startswith("#")]
        for i, l in enumerate(lineas):
            if "ci/piso.py verificar" in l:
                assert l.endswith("|| exit $?") or i == len(lineas) - 1, (jid, l)


def test_el_lector_de_la_cadena_de_minimo_no_corre_con_error_ignorado():
    """El mínimo del B9 se lee en una asignación aparte: dentro de `$(...)` embebido en
    otro comando un fallo del lector quedaría tragado."""
    for jid, run in _pasos_run():
        for l in run.splitlines():
            if "ci/piso.py minimo" in l:
                assert re.match(r"\s*[A-Z_]+=\$\(python3 ci/piso\.py minimo \S+\) \|\| exit \$\?\s*$", l), l


def test_el_archivo_real_de_pisos_es_valido():
    mod = _modulo()
    datos = mod.cargar(PISOS)
    for clave in datos["pisos"]:
        mod.piso(datos, clave)
    for clave in datos["minimos"]:
        mod.minimo(datos, clave)
