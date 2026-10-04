"""El comparador `.github/ci/comparar_pisos.py`: un piso de CI nunca baja.

Los pisos salieron de policy.yml a `ci/pisos.json`. Eso los sacó de la reserva de
integración de Fernando (`.github/workflows/**`, `policy/**`): una sesión podía rebajar un
piso editando un JSON. El lector y este comparador viven en `.github/`, y el comparador
(paso obligatorio de policy.yml) falla si el head baja un piso respecto de la PUNTA de
origin/master. Cada regla tiene aquí su caso ROJO y las formas de patrón que existen hoy
tienen su caso VERDE; la base ilegible, el JSON inválido y los tipos inesperados fallan cerrado.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / ".github" / "ci" / "comparar_pisos.py"

_spec = importlib.util.spec_from_file_location("comparar_pisos", SCRIPT)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def estado(pisos=None, minimos=None):
    return {"pisos": pisos if pisos is not None else {}, "minimos": minimos if minimos is not None else {}}


def piso(patron, archivo="/tmp/a"):
    return {"patron": patron, "archivo": archivo}


BASE = estado({
    "j/pasa": piso("^5 passed", "/tmp/p"),
    "j/en": piso("^6 passed in ", "/tmp/e"),
    "j/en2": piso("^2 passed in", "/tmp/e2"),
    "j/salta": piso("^53 passed, 2 skipped", "/tmp/s"),
}, {"j/casos": 102})


def head_con(**cambios):
    h = copy.deepcopy(BASE)
    for clave, valor in cambios.items():
        seccion, nombre = clave.split("__", 1)
        nombre = nombre.replace("_", "/", 1)
        if valor is None:
            del h[seccion][nombre]
        else:
            h[seccion][nombre] = valor
    return h


# --- verdes: una por cada forma de patrón que existe hoy -------------------------

def test_igual_es_verde():
    assert cp.comparar(BASE, copy.deepcopy(BASE)) == []


@pytest.mark.parametrize("clave,patron", [
    ("j/pasa", "^9 passed"),
    ("j/en", "^7 passed in "),
    ("j/en2", "^3 passed in"),
    ("j/salta", "^60 passed, 2 skipped"),
    ("j/salta", "^53 passed, 1 skipped"),   # bajar M está permitido
    ("j/salta", "^53 passed, 0 skipped"),
])
def test_subir_n_o_bajar_m_es_verde(clave, patron):
    h = copy.deepcopy(BASE)
    h["pisos"][clave]["patron"] = patron
    assert cp.comparar(BASE, h) == []


def test_clave_nueva_y_minimo_mayor_y_mensaje_libre_son_verdes():
    h = copy.deepcopy(BASE)
    h["pisos"]["j/nueva"] = piso("^1 passed")
    h["minimos"]["j/casos"] = 103
    h["minimos"]["j/otro"] = 5
    assert cp.comparar(BASE, h) == []


# --- rojos: uno por regla ----------------------------------------------------------

def _rojo(h, fragmento):
    errores = cp.comparar(BASE, h)
    assert errores and any(fragmento in e for e in errores), errores


def _con_patron(clave, patron):
    h = copy.deepcopy(BASE)
    h["pisos"][clave]["patron"] = patron
    return h


@pytest.mark.parametrize("clave,patron", [("j/pasa", "^4 passed"), ("j/en", "^5 passed in "),
                                          ("j/en2", "^1 passed in"), ("j/salta", "^52 passed, 2 skipped")])
def test_rojo_n_baja(clave, patron):
    _rojo(_con_patron(clave, patron), "N baja")


def test_rojo_m_sube():
    _rojo(_con_patron("j/salta", "^53 passed, 3 skipped"), "M (skipped) sube")


@pytest.mark.parametrize("patron", [
    "^5[0-9] passed",          # comodín en N
    "5 passed",                # sin ancla
    "^5 failed",               # otra palabra
    "^5 Passed",
    "^5 passed|^1 passed",     # alternativas
    "^(5|6) passed",
    "^5 passed in ",           # otra forma conocida: no es solo subir N
    "^5 passed, 0 skipped",
    "^5 passed.*",
])
def test_rojo_el_patron_cambia_de_forma(patron):
    errores = None
    try:
        errores = cp.comparar(BASE, _con_patron("j/pasa", patron))
    except cp.PisosError:
        return  # forma que el parser no reconoce: también es rojo (falla cerrado)
    assert errores and any("cambia de forma" in e for e in errores), errores


def test_rojo_la_forma_in_con_y_sin_espacio_no_se_intercambian():
    _rojo(_con_patron("j/en", "^6 passed in"), "cambia de forma")
    _rojo(_con_patron("j/en2", "^2 passed in "), "cambia de forma")


def test_rojo_desaparece_una_clave():
    _rojo(head_con(pisos__j_pasa=None), "el piso desaparece")


def test_rojo_cambia_el_archivo_temporal():
    h = copy.deepcopy(BASE)
    h["pisos"]["j/pasa"]["archivo"] = "/tmp/otro"
    _rojo(h, "cambia el archivo")
    h["pisos"]["j/pasa"]["archivo"] = None  # la llamada del workflow ya no existe
    _rojo(h, "cambia el archivo")


@pytest.mark.parametrize("patron", ["^5 passed ", "^ 5 passed", "^05 passed", "", "^5 passed, 2 skipped, 1 xfailed"])
def test_rojo_forma_no_reconocida_en_el_head(patron):
    with pytest.raises(cp.PisosError):
        cp.comparar(BASE, _con_patron("j/pasa", patron))


def test_forma_no_reconocida_en_una_clave_nueva_tambien_falla():
    h = copy.deepcopy(BASE)
    h["pisos"]["j/nueva"] = piso("^.* passed")
    with pytest.raises(cp.PisosError):
        cp.comparar(BASE, h)


def test_forma_no_reconocida_en_la_base_falla_cerrado():
    base = copy.deepcopy(BASE)
    base["pisos"]["j/pasa"]["patron"] = "^5[0-9] passed"
    with pytest.raises(cp.PisosError):
        cp.comparar(base, copy.deepcopy(BASE))


def test_rojo_el_minimo_baja():
    _rojo(head_con(minimos__j_casos=101), "el mínimo baja")


@pytest.mark.parametrize("valor", ["102", 102.0, True, [102], {"min": 102}])
def test_rojo_el_minimo_cambia_de_forma(valor):
    _rojo(head_con(minimos__j_casos=valor), "cambia de forma")


def test_rojo_desaparece_el_minimo():
    _rojo(head_con(minimos__j_casos=None), "el mínimo desaparece")


def test_rojo_un_minimo_nuevo_con_forma_rara():
    h = copy.deepcopy(BASE)
    h["minimos"]["j/nuevo"] = "3"
    _rojo(h, "no es un entero")


# --- falla cerrada: tipos inesperados al armar el estado ------------------------------

WORKFLOW_NUEVO = """\
jobs:
  j:
    steps:
      - run: |
          python3 .github/ci/piso.py verificar j/pasa /tmp/p || exit $?
"""


@pytest.mark.parametrize("datos", [
    "{no json", "", "[]", "null", '{"pisos": [], "minimos": {}}', '{"pisos": {}, "minimos": []}',
    '{"pisos": {}}', '{"pisos": {"j/pasa": "^5 passed"}, "minimos": {}}',
])
def test_json_o_tipos_invalidos_fallan_cerrado(datos):
    with pytest.raises(cp.PisosError):
        cp.armar_nuevo(datos, WORKFLOW_NUEVO, "head")


@pytest.mark.parametrize("patron", [5, None, ["^5 passed"], {"a": 1}])
def test_patron_con_tipo_inesperado_falla_cerrado(patron):
    s = cp.armar_nuevo(json.dumps({"pisos": {"j/pasa": {"patron": patron}}, "minimos": {}}), WORKFLOW_NUEVO, "head")
    with pytest.raises(cp.PisosError):
        cp.comparar(BASE, s)


def test_dos_llamadas_a_la_misma_clave_fallan_cerrado():
    with pytest.raises(cp.PisosError):
        cp.armar_nuevo('{"pisos": {}, "minimos": {}}', WORKFLOW_NUEVO + WORKFLOW_NUEVO, "head")


# --- el extractor del workflow viejo -----------------------------------------------------

VIEJO = """\
jobs:
  uno:
    steps:
      - run: |
          python -m pytest | tee /tmp/a
          # comentario
          grep -qE "^5 passed" /tmp/a || {
            echo "PISO ROTO: cinco."; exit 1; }
  dos:
    steps:
      - run: |
          grep -qE "^53 passed, 2 skipped" /tmp/b || { echo "PISO ROTO: x"; exit 1; }
          grep -qE "^2 passed in " /tmp/c || {
            echo "PISO ROTO: y"; exit 1; }
  memory-b9-regression:
    steps:
      - run: |
          python - <<'PY'
          if len(cases) < 102 or incomplete:
              raise SystemExit('x')
          PY
"""


def test_el_extractor_lee_los_pisos_de_un_policy_yml_con_greps_literales():
    e = cp.extraer_de_workflow_viejo(VIEJO)
    assert e["pisos"] == {
        "uno/a": piso("^5 passed", "/tmp/a"),
        "dos/b": piso("^53 passed, 2 skipped", "/tmp/b"),
        "dos/c": piso("^2 passed in ", "/tmp/c"),
    }
    assert e["minimos"] == {"memory-b9-regression/casos": 102}


def test_el_extractor_sin_pisos_falla_cerrado():
    with pytest.raises(cp.PisosError):
        cp.extraer_de_workflow_viejo("jobs:\n  uno:\n    steps:\n      - run: echo hola\n")


# --- el comparador de punta a punta, con repositorios git reales -----------------------------

def _git(raiz, *args):
    r = subprocess.run(["git", "-C", str(raiz), "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r


def _escribir(raiz, ruta, contenido):
    p = raiz / ruta
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contenido, encoding="utf-8")


DATOS_NUEVOS = {"version": 1, "minimos": {"j/casos": 102}, "pisos": {
    "j/pasa": {"patron": "^5 passed", "mensaje": "m"},
    "j/salta": {"patron": "^53 passed, 2 skipped", "mensaje": "m"}}}

WORKFLOW_BASE_NUEVO = """\
jobs:
  j:
    steps:
      - run: |
          python3 .github/ci/piso.py verificar j/pasa /tmp/pasa || exit $?
          python3 .github/ci/piso.py verificar j/salta /tmp/salta || exit $?
          MINIMO_CASOS=$(python3 .github/ci/piso.py minimo j/casos) || exit $?
"""

WORKFLOW_BASE_VIEJO = """\
jobs:
  j:
    steps:
      - run: |
          grep -qE "^5 passed" /tmp/pasa || {
            echo "PISO ROTO: p"; exit 1; }
          grep -qE "^53 passed, 2 skipped" /tmp/salta || { echo "PISO ROTO: s"; exit 1; }
          python - <<'PY'
          if len(cases) < 102 or incomplete:
              pass
          PY
"""


@pytest.fixture
def repo(tmp_path):
    """Repo git con el comparador copiado a .github/ci/ y `origin/master` apuntando a la base."""
    _git(tmp_path, "init", "-q", "-b", "master")
    (tmp_path / ".github" / "ci").mkdir(parents=True)
    shutil.copy(SCRIPT, tmp_path / ".github" / "ci" / "comparar_pisos.py")
    return tmp_path


def _base(repo, workflow, datos=None):
    _escribir(repo, ".github/workflows/policy.yml", workflow)
    if datos is not None:
        _escribir(repo, "ci/pisos.json", datos if isinstance(datos, str) else json.dumps(datos))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")


def _correr(repo, *args):
    return subprocess.run([sys.executable, str(repo / ".github" / "ci" / "comparar_pisos.py"), *args],
                          capture_output=True, text=True, cwd=repo)


def _head(repo, workflow, datos):
    _escribir(repo, ".github/workflows/policy.yml", workflow)
    if datos is None:
        (repo / "ci" / "pisos.json").unlink()
    else:
        _escribir(repo, "ci/pisos.json", datos if isinstance(datos, str) else json.dumps(datos))


def test_base_con_datos_head_igual_es_verde(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    r = _correr(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "modo datos" in r.stdout


def test_base_con_datos_head_que_baja_un_piso_es_rojo(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    h = copy.deepcopy(DATOS_NUEVOS)
    h["pisos"]["j/pasa"]["patron"] = "^4 passed"
    _head(repo, WORKFLOW_BASE_NUEVO, h)
    r = _correr(repo)
    assert r.returncode == 1 and "N baja" in r.stdout


def test_bajar_m_es_la_unica_excepcion_a_solo_subir_n_y_esta_documentada():
    """Menos saltadas endurece el piso: se permite a propósito (ver docs/ci/pisos.md)."""
    h = copy.deepcopy(BASE)
    h["pisos"]["j/salta"]["patron"] = "^53 passed, 0 skipped"
    assert cp.comparar(BASE, h) == []
    assert "única excepción" in (RAIZ / "docs" / "ci" / "pisos.md").read_text(encoding="utf-8")
    assert "única excepción" in cp.__doc__


def test_el_head_borra_pisos_json_y_la_base_lo_tiene_es_rojo(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    _head(repo, WORKFLOW_BASE_NUEVO, None)
    r = _correr(repo)
    assert r.returncode != 0
    assert "pisos.json" in r.stderr  # no hay arranque posible: la base tiene datos


def test_cambiar_el_archivo_de_salida_en_el_workflow_del_head_es_rojo(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    _head(repo, WORKFLOW_BASE_NUEVO.replace("j/pasa /tmp/pasa", "j/pasa /tmp/vacio"), DATOS_NUEVOS)
    r = _correr(repo)
    assert r.returncode == 1 and "cambia el archivo" in r.stdout


def test_arranque_verde_head_que_copia_los_pisos_de_master(repo):
    _base(repo, WORKFLOW_BASE_VIEJO)
    _head(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    r = _correr(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "modo arranque" in r.stdout


def test_arranque_rojo_head_que_baja_un_piso_respecto_del_policy_yml_base(repo):
    _base(repo, WORKFLOW_BASE_VIEJO)
    h = copy.deepcopy(DATOS_NUEVOS)
    h["pisos"]["j/salta"]["patron"] = "^52 passed, 2 skipped"
    _head(repo, WORKFLOW_BASE_NUEVO, h)
    r = _correr(repo)
    assert r.returncode == 1 and "N baja" in r.stdout


def test_arranque_rojo_head_que_baja_el_minimo_del_b9(repo):
    _base(repo, WORKFLOW_BASE_VIEJO)
    h = copy.deepcopy(DATOS_NUEVOS)
    h["minimos"]["j/casos"] = 100
    _head(repo, WORKFLOW_BASE_NUEVO, h)
    r = _correr(repo)
    assert r.returncode == 1 and "el mínimo baja" in r.stdout


def test_arranque_rojo_head_que_pierde_un_piso(repo):
    _base(repo, WORKFLOW_BASE_VIEJO)
    h = copy.deepcopy(DATOS_NUEVOS)
    del h["pisos"]["j/salta"]
    _head(repo, WORKFLOW_BASE_NUEVO, h)
    assert _correr(repo).returncode == 1


def test_base_sin_pisos_json_y_sin_pisos_extraibles_falla_cerrado(repo):
    _base(repo, "jobs:\n  j:\n    steps:\n      - run: echo sin pisos\n")
    _head(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    r = _correr(repo)
    assert r.returncode == 2
    assert "no se pudo extraer ningún piso" in r.stderr


def test_base_ilegible_ref_inexistente_falla_cerrado(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    r = _correr(repo, "origin/no-existe")
    assert r.returncode == 2 and "falla cerrado" in r.stderr


def test_base_con_json_invalido_falla_cerrado(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, "{roto")
    r = _correr(repo)
    assert r.returncode == 2 and "JSON" in r.stderr


def test_head_con_json_invalido_falla_cerrado(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    _head(repo, WORKFLOW_BASE_NUEVO, "{roto")
    r = _correr(repo)
    assert r.returncode == 2 and "JSON" in r.stderr


def test_head_con_tipo_inesperado_en_un_piso_falla_cerrado(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    h = copy.deepcopy(DATOS_NUEVOS)
    h["pisos"]["j/pasa"] = "^5 passed"
    _head(repo, WORKFLOW_BASE_NUEVO, h)
    assert _correr(repo).returncode == 2


def test_base_sin_el_workflow_falla_cerrado(repo):
    _escribir(repo, "ci/pisos.json", json.dumps(DATOS_NUEVOS))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
    _escribir(repo, ".github/workflows/policy.yml", WORKFLOW_BASE_NUEVO)
    assert _correr(repo).returncode == 2


DUP_HEAD = ('{"version": 1, "minimos": {}, "pisos": {"j/pasa": {"patron": "^1 passed", "mensaje": "m"},'
            ' "j/pasa": {"patron": "^5 passed", "mensaje": "m"}}}')


def test_json_con_clave_duplicada_falla_cerrado_aunque_el_ultimo_valor_no_rebaje():
    with pytest.raises(cp.PisosError, match="duplicada"):
        cp.armar_nuevo(DUP_HEAD, WORKFLOW_NUEVO, "head")
    with pytest.raises(cp.PisosError, match="duplicada"):
        cp.armar_nuevo('{"pisos": {}, "minimos": {"j/casos": 1, "j/casos": 102}}', WORKFLOW_NUEVO, "head")


def test_head_con_clave_duplicada_falla_cerrado_de_punta_a_punta(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DATOS_NUEVOS)
    _head(repo, WORKFLOW_BASE_NUEVO, DUP_HEAD)
    r = _correr(repo)
    assert r.returncode == 2 and "duplicada" in r.stderr


def test_base_con_clave_duplicada_falla_cerrado_de_punta_a_punta(repo):
    _base(repo, WORKFLOW_BASE_NUEVO, DUP_HEAD)
    r = _correr(repo)
    assert r.returncode == 2 and "duplicada" in r.stderr


def test_ref_base_por_defecto_es_la_ref_propia_del_job():
    assert cp.REF_BASE == "refs/pisos-base/master"


def test_argumentos_de_mas_fallan(repo):
    assert _correr(repo, "a", "b").returncode == 2
