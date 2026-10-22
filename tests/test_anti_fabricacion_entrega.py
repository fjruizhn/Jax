"""verificar_entrega.py: el detector de estructura sobre una rama de Qwen Code.

Códigos de salida: 0 limpio, 1 marcado, 2 no pudo correr. Un 2 no es un 0: si el
detector no pudo mirar, nadie debe leer «limpio». Repos y ramas reales (git en
un directorio temporal); el esquema de la base se inyecta, no se consulta.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

from jax.anti_fabricacion.whitelist import EsquemaDB

_RUTA = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "anti_fabricacion" / "verificar_entrega.py"
_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("verificar_entrega", _RUTA)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _g(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True, env=_ENV).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _g(r, "init", "-q", "-b", "main")
    (r / "a.py").write_text("X = 1\n")
    (r / "docs").mkdir()
    (r / "docs" / "x.md").write_text("hola\n")
    _g(r, "add", "-A")
    _g(r, "commit", "-q", "-m", "base")
    _g(r, "checkout", "-q", "-b", "rama")
    (r / "a.py").write_text("X = 2\n\ndef nueva_funcion():\n    pass\n")
    (r / "b.py").write_text("Y = 1\n")
    _g(r, "add", "-A")
    _g(r, "commit", "-q", "-m", "entrega")
    return r


def _esquema_falso():
    async def huella():
        return "H"

    async def esquema():
        return EsquemaDB(columnas=(("jacobs_steps", "status"),))
    return (huella, esquema)


def _correr(mod, repo, informe_texto, tmp_path, *, base="main", db=None, informe=True):
    ruta = tmp_path / "informe.md"
    if informe:
        ruta.write_text(informe_texto, encoding="utf-8")
    return mod.main(["--repo", str(repo), "--base", base, "--informe", str(ruta)],
                    fuentes_db=db or _esquema_falso())


def test_informe_que_solo_cita_lo_real_y_lo_propio_da_0(mod, repo, tmp_path, capsys):
    assert _correr(mod, repo, "Modifiqué `a.py` (agregué `nueva_funcion`) y creé `b.py`.", tmp_path) == 0
    assert "limpio" in capsys.readouterr().out


def test_informe_que_cita_un_archivo_inexistente_da_1_y_lo_nombra(mod, repo, tmp_path, capsys):
    assert _correr(mod, repo, "Modifiqué `c/d.py` para arreglar el bug.", tmp_path) == 1
    assert "c/d.py" in capsys.readouterr().out


def test_diff_en_el_informe_sobre_archivo_inexistente_da_1(mod, repo, tmp_path):
    informe = "Parche:\n--- a/q.py\n+++ b/q.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert _correr(mod, repo, informe, tmp_path) == 1


def test_lo_marcado_nuevo_no_cuenta(mod, repo, tmp_path):
    assert _correr(mod, repo, "Quedaría pendiente `nuevo: z.py`.", tmp_path) == 0


def test_tabla_columna_inventada_da_1(mod, repo, tmp_path):
    assert _correr(mod, repo, "Leí `jacobs_steps.fantasma`.", tmp_path) == 1


def test_informe_inexistente_da_2(mod, repo, tmp_path):
    assert _correr(mod, repo, "", tmp_path, informe=False) == 2


def test_directorio_que_no_es_repo_da_2(mod, tmp_path):
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    assert _correr(mod, vacio, "`a.py`", tmp_path) == 2


def test_base_desconocida_da_2(mod, repo, tmp_path):
    assert _correr(mod, repo, "`a.py`", tmp_path, base="no-existe") == 2


def test_base_de_datos_caida_da_2_y_nunca_0(mod, repo, tmp_path):
    async def huella():
        raise ConnectionError("db caida")

    async def esquema():
        raise ConnectionError("db caida")
    assert _correr(mod, repo, "Modifiqué `a.py`.", tmp_path, db=(huella, esquema)) == 2


def test_un_archivo_borrado_por_la_rama_no_es_propio(mod, repo, tmp_path):
    """MINOR 11: decir que se «modificó» un archivo que la rama borró es afirmar algo falso."""
    (repo / "docs" / "y.md").write_text("otro\n")   # `docs/` sigue siendo una carpeta del repo
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "otro")
    _g(repo, "rm", "-q", "docs/x.md")
    _g(repo, "commit", "-q", "-m", "borra")
    assert _correr(mod, repo, "Modifiqué `docs/x.md`.", tmp_path) == 1
    assert _correr(mod, repo, "Modifiqué `b.py`.", tmp_path) == 0


def test_el_docstring_de_la_entrega_no_promete_simbolos():
    texto = _RUTA.read_text(encoding="utf-8").split('"""')[1]
    assert "símbolos" not in texto
