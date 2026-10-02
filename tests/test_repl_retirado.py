"""T16 (2026-10-02, decisión de Fernando D-5 del spec El Faro): el REPL se retiró.

Hallazgo F-1: el Hyde del REPL (`jax/muscles/subprocess_muscle.py`, lanzado desde
`jax/core/main.py`) corría Claude con `--allowedTools "Write,Edit,Read,Bash"` y el
token de la suscripción en su entorno. Se cierra quitando el camino entero: el REPL
interactivo, `jax --task` (que usaba `/command` de jax-platform) y ese músculo.

Estos tests fallan contra el árbol anterior al retiro. Puros: leen el árbol, no
importan nada de lo retirado. El detector se ejercita con fuentes sintéticas (un
detector que no atrapa nada da un verde que no significa nada) y un archivo que no
se puede parsear FALLA, no se salta: un error de sintaxis no puede ser una puerta.
"""
from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RETIRADOS = ("jax/core/main.py", "jax/muscles/subprocess_muscle.py")
PROHIBIDOS = {"jax.core.main", "jax.muscles.subprocess_muscle"}
FUERA = {".venv", "venv", ".git", "node_modules", "__pycache__"}
_DINAMICOS = {"import_module", "__import__"}
# Simbolos de jax.core.registro_facetas que se borraron con el REPL. Un archivo (de prueba o no)
# que los importe revienta con ImportError: en la reauditoria de T16 (BLOCK-10) uno asi, un
# test de DB, no lo vio nadie hasta el job con MariaDB.
MODULO_REGISTRO = "jax.core.registro_facetas"
SIMBOLOS_BORRADOS = {"cargar_registro", "aplicar_registro", "_camino_del_modelo",
                     "ESTADOS_INVOCABLES", "TIPO_POR_TRANSPORTE", "CLAVE_HTTP_POR_PROVIDER"}


def test_los_archivos_del_repl_ya_no_estan_en_el_arbol():
    assert [rel for rel in RETIRADOS if (RAIZ / rel).exists()] == []


def _paquete(rel: str) -> list[str]:
    partes = rel[:-3].split("/")
    return partes if partes[-1] == "__init__" and partes.pop() else partes[:-1]


def _prohibidos_en(fuente: str, rel: str) -> list[str]:
    """Los módulos prohibidos que `fuente` (el archivo `rel`, relativo a la raíz)
    importa: import/from absolutos, from relativos resueltos contra su paquete, y
    `importlib.import_module("...")`/`__import__("...")` con el nombre literal.
    Si la fuente no se puede parsear, SyntaxError: quien llama la deja subir."""
    hallados: list[str] = []
    for nodo in ast.walk(ast.parse(fuente)):
        if isinstance(nodo, ast.Import):
            nombres = [a.name for a in nodo.names]
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.level:
                pkg = _paquete(rel)
                base = ".".join(pkg[: len(pkg) - (nodo.level - 1)])
                modulo = ".".join(p for p in (base, nodo.module) if p)
            else:
                modulo = nodo.module or ""
            nombres = [modulo] + [f"{modulo}.{a.name}" for a in nodo.names]
        elif isinstance(nodo, ast.Call):
            f = nodo.func
            nombre = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            literal = nodo.args[0] if nodo.args else None
            nombres = [literal.value] if (nombre in _DINAMICOS and isinstance(literal, ast.Constant)
                                          and isinstance(literal.value, str)) else []
        else:
            continue
        hallados += [n for n in nombres if n in PROHIBIDOS]
    return hallados


def _simbolos_borrados_en(fuente: str) -> list[str]:
    """Nombres borrados de registro_facetas que `fuente` importa (`from jax.core.registro_facetas
    import x`, `from jax.core import registro_facetas` seguido de `registro_facetas.x`)."""
    hallados: list[str] = []
    arbol = ast.parse(fuente)
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and not nodo.level:
            if nodo.module == MODULO_REGISTRO:
                hallados += [a.name for a in nodo.names if a.name in SIMBOLOS_BORRADOS]
        elif isinstance(nodo, ast.Attribute) and nodo.attr in SIMBOLOS_BORRADOS:
            v = nodo.value
            if (isinstance(v, ast.Name) and v.id == "registro_facetas") or \
               (isinstance(v, ast.Attribute) and v.attr == "registro_facetas"):
                hallados.append(nodo.attr)
    return hallados


def test_nadie_importa_simbolos_borrados_de_registro_facetas():
    hallazgos = []
    for ruta in sorted(RAIZ.rglob("*.py")):
        if FUERA & set(ruta.relative_to(RAIZ).parts) or ruta == Path(__file__).resolve():
            continue
        hallazgos += [f"{ruta.relative_to(RAIZ).as_posix()}: {s}"
                      for s in _simbolos_borrados_en(ruta.read_text(encoding="utf-8"))]
    assert hallazgos == []


def test_el_detector_de_simbolos_borrados_ve_las_dos_formas():
    assert _simbolos_borrados_en("from jax.core.registro_facetas import cargar_registro\n")
    assert _simbolos_borrados_en("from jax.core import registro_facetas\nregistro_facetas.aplicar_registro(1)\n")
    assert _simbolos_borrados_en("import jax.core.registro_facetas as r\njax.core.registro_facetas._camino_del_modelo\n")
    assert not _simbolos_borrados_en("from jax.core.registro_facetas import url_del_proveedor\n")


def test_nada_importa_el_repl_ni_el_musculo_de_subproceso():
    hallazgos = []
    for ruta in sorted(RAIZ.rglob("*.py")):
        rel = ruta.relative_to(RAIZ).as_posix()
        if FUERA & set(ruta.relative_to(RAIZ).parts) or ruta == Path(__file__).resolve():
            continue
        # SyntaxError NO se atrapa: un archivo ilegible hace fallar la prueba.
        hallazgos += [f"{rel}: {m}" for m in _prohibidos_en(ruta.read_text(encoding="utf-8"), rel)]
    assert hallazgos == []


def test_el_detector_ve_cada_forma_de_importar():
    caso = _prohibidos_en
    assert caso("import jax.core.main\n", "x.py")
    assert caso("from jax.core import main\n", "x.py")
    assert caso("from jax.core.main import build_muscles\n", "x.py")
    assert caso("from jax.muscles.subprocess_muscle import SubprocessMuscle\n", "x.py")
    # relativos, desde los paquetes donde viven (jax/core/ y jax/muscles/)
    assert caso("from . import main\n", "jax/core/otro.py")
    assert caso("from .main import x\n", "jax/core/otro.py")
    assert caso("from ..core import main\n", "jax/muscles/otro.py")
    assert caso("from . import subprocess_muscle\n", "jax/muscles/__init__.py")
    # dinamicos
    assert caso("import importlib\nimportlib.import_module('jax.core.main')\n", "x.py")
    assert caso("__import__('jax.muscles.subprocess_muscle')\n", "x.py")
    # y no hay falsos positivos
    assert not caso("from jax.core import router\nfrom . import base\n", "jax/muscles/otro.py")
    assert not caso("import importlib\nimportlib.import_module('jax.core.router')\n", "x.py")


def test_un_archivo_que_no_se_puede_parsear_falla_en_vez_de_saltarse():
    import pytest
    with pytest.raises(SyntaxError):
        _prohibidos_en("def roto(:\n", "x.py")
