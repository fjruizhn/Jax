"""Los pisos de CI viven en `ci/pisos.json`, no dentro de `.github/workflows/policy.yml`.

Por qué existe: GitHub deja de ejecutar un workflow cuyo archivo supera unos 512 000
bytes («workflow file issue»); cada PR que subía un piso le sumaba comentarios y
policy.yml ya rozaba el límite. Los números salen a un archivo de datos y el
workflow los lee en tiempo de ejecución con `.github/ci/piso.py`.

Qué fija este archivo (permanente, no depende de ninguna fotografía de master):
  * el lector falla CERRADO: sin archivo, sin parsear o sin la clave, sale con 2 y
    nunca da por bueno un piso;
  * el workflow no conserva ningún piso literal (`grep -qE "^N passed`);
  * cada clave que el workflow usa existe, y cada clave de los datos se usa una vez
    (un piso huérfano sería un piso que nadie compara).
Que ningún piso baje ni cambie de forma respecto de master lo cubre test_pisos_migracion_desde_master.py (regla genérica) y el job `pisos-no-bajan`.
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
PISO_PY = RAIZ / ".github" / "ci" / "piso.py"

USO = re.compile(r"python3 \.github/ci/piso\.py (verificar|minimo) ([^\s)]+)")


def _modulo():
    spec = importlib.util.spec_from_file_location("ci_piso", PISO_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _correr(directorio: Path, *args: str) -> subprocess.CompletedProcess:
    """Corre una copia de .github/ci/piso.py contra el ci/pisos.json que haya en `directorio`."""
    destino = directorio / ".github" / "ci"
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy(PISO_PY, destino / "piso.py")
    return subprocess.run([sys.executable, str(destino / "piso.py"), *args],
                          capture_output=True, text=True, cwd=directorio)


@pytest.fixture
def salida(tmp_path):
    f = tmp_path / "salida.txt"
    f.write_text("....\n5 passed in 0.10s\n", encoding="utf-8")
    return f


def _escribir(directorio: Path, datos) -> None:
    (directorio / "ci").mkdir(exist_ok=True)
    (directorio / "ci" / "pisos.json").write_text(
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
            if ".github/ci/piso.py verificar" in l:
                assert l.endswith("|| exit $?") or i == len(lineas) - 1, (jid, l)


def test_el_lector_de_la_cadena_de_minimo_no_corre_con_error_ignorado():
    """El mínimo del B9 se lee en una asignación aparte: dentro de `$(...)` embebido en
    otro comando un fallo del lector quedaría tragado."""
    for jid, run in _pasos_run():
        for l in run.splitlines():
            if ".github/ci/piso.py minimo" in l:
                assert re.match(r"\s*[A-Z_]+=\$\(python3 \.github/ci/piso\.py minimo \S+\) \|\| exit \$\?\s*$", l), l


def test_el_archivo_real_de_pisos_es_valido():
    mod = _modulo()
    datos = mod.cargar(PISOS)
    for clave in datos["pisos"]:
        mod.piso(datos, clave)
    for clave in datos["minimos"]:
        mod.minimo(datos, clave)


# --- gobernanza: lector y comparador dentro de la reserva de `.github/` ----------------------

def test_el_lector_vive_en_github_y_no_queda_copia_fuera_de_la_reserva():
    assert PISO_PY.is_file()
    assert not (RAIZ / "ci" / "piso.py").exists(), "una copia en ci/ queda fuera de la reserva de Fernando"
    assert [p.name for p in (RAIZ / "ci").iterdir() if p.name != "__pycache__"] == ["pisos.json"], "ci/ solo guarda datos, nunca lógica"


def test_ninguna_llamada_del_workflow_usa_un_lector_fuera_de_github():
    texto = WORKFLOW.read_text(encoding="utf-8")
    llamadas = re.findall(r"\S*piso\.py", texto)
    lectores = {ruta for ruta in llamadas if ruta.endswith("/piso.py")}
    assert lectores == {".github/ci/piso.py"}
    assert ".github/ci/comparar_pisos.py" in texto  # se extrae de base, no se ejecuta desde HEAD


JOB_COMPARADOR = "pisos-no-bajan"
COMPARAR = 'python3 -I "$GITHUB_WORKSPACE/.github/ci/comparar_pisos_base.py" refs/pisos-base/master || exit $?'


def _job_comparador():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][JOB_COMPARADOR]


def test_el_comparador_corre_aislado_y_ejecuta_el_checker_extraido_de_la_base():
    """El código del PR (conftest.py, sitecustomize, un test) corre en los otros jobs y puede
    reescribir .git/config o las refs. Este job no ejecuta nada del repo: checkout, fetch, comparar."""
    job = _job_comparador()
    assert set(job) == {"runs-on", "steps"}, "sin if:, needs, env, container, services ni continue-on-error"
    pasos = job["steps"]
    assert len(pasos) == 3
    assert set(pasos[0]) == {"uses", "with"} and pasos[0]["uses"].startswith("actions/checkout@")
    assert pasos[0]["with"] == {"persist-credentials": False}
    assert set(pasos[1]) == {"name", "run"}
    fetch = pasos[1]["run"]
    assert "git fetch --no-tags --unshallow" in fetch
    assert "git fetch --no-tags \"$REPO_URL\"" in fetch
    assert "+refs/heads/master:refs/pisos-base/master" in fetch
    assert 'PR_MERGE_REF="refs/pull/${{ github.event.pull_request.number }}/merge"' in fetch
    assert '"+${PR_MERGE_REF}:refs/pisos-candidate/merge"' in fetch
    assert "rev-parse --is-shallow-repository" in fetch and '== "false"' in fetch
    assert "git show refs/pisos-base/master:.github/ci/comparar_pisos.py" in fetch
    assert 'destino="$GITHUB_WORKSPACE/.github/ci/comparar_pisos_base.py"' in fetch
    assert 'temporal=$(mktemp "$GITHUB_WORKSPACE/.github/ci/.comparar_pisos_base.XXXXXX")' in fetch
    assert 'mv -f "$temporal" "$destino"' in fetch
    assert set(pasos[2]) == {"name", "run"} and pasos[2]["run"].strip() == COMPARAR


def test_el_job_comparador_no_instala_ni_ejecuta_codigo_del_pr():
    texto = yaml.safe_dump(_job_comparador())
    for prohibido in ("pytest", "setup-python", "npm", "import ", "conftest", "continue-on-error", "if:"):
        assert prohibido not in texto, prohibido


def test_checker_extraido_usa_ruta_profunda_y_interfaz_posicional_legacy():
    pasos = _job_comparador()["steps"]
    fetch = pasos[1]["run"]
    run = pasos[2]["run"].strip()
    ruta = "$GITHUB_WORKSPACE/.github/ci/comparar_pisos_base.py"
    assert 'git show refs/pisos-base/master:.github/ci/comparar_pisos.py > "$temporal"' in fetch
    assert 'destino="' + ruta + '"' in fetch
    assert run == f'python3 -I "{ruta}" refs/pisos-base/master || exit $?'
    assert "--head-root" not in run and "--base-ref" not in run and "--head-ref" not in run


def test_la_base_se_trae_de_la_url_fija_a_una_ref_propia_no_del_origin_configurado():
    run = _job_comparador()["steps"][1]["run"]
    assert "github.server_url" in run and "github.repository" in run
    assert "refs/pisos-base/master" in run
    assert "--depth=1" not in run and "--deepen" not in run
    assert "origin" not in run


def test_registro_de_retiros_vigente_inicia_vacio_y_con_esquema_versionado():
    registro = RAIZ / ".github" / "workflows" / "floor-retirements.json"
    datos = json.loads(registro.read_text(encoding="utf-8"))
    assert datos == {"version": 1, "grants": []}


def test_el_paso_comparador_ya_no_esta_en_el_job_de_las_pruebas():
    for jid, run in _pasos_run():
        if jid != JOB_COMPARADOR:
            assert ".github/ci/comparar_pisos.py" not in run, jid


def test_el_comparador_usa_solo_la_biblioteca_estandar():
    import ast
    arbol = ast.parse((RAIZ / ".github" / "ci" / "comparar_pisos.py").read_text(encoding="utf-8"))
    mods = {a.name.split(".")[0] for n in ast.walk(arbol) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(arbol) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods and mods <= set(sys.stdlib_module_names), mods - set(sys.stdlib_module_names)


def test_piso_py_rechaza_claves_duplicadas_en_pisos_json(tmp_path, salida):
    dup = ('{"version": 1, "minimos": {}, "pisos": {"j/x": {"patron": "^1 passed", "mensaje": "m"},'
           ' "j/x": {"patron": "^5 passed", "mensaje": "m"}}}')
    _escribir(tmp_path, dup)
    r = _correr(tmp_path, "verificar", "j/x", str(salida))
    assert r.returncode == 2 and "duplicada" in r.stderr
    dup_min = '{"version": 1, "pisos": {}, "minimos": {"j/m": 1, "j/m": 7}}'
    _escribir(tmp_path, dup_min)
    assert _correr(tmp_path, "minimo", "j/m").returncode == 2
