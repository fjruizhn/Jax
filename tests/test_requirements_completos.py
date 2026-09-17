"""E-19 (2026-09-16): requirements.txt es la fuente ÚNICA de las dependencias
de servicio, y CI instala desde ahí.

Faltaban cryptography (jax/core/crypto_secrets.py) y pyyaml
(policy/governance/loaders.py): con `pip install -r requirements.txt`, como dice
README.md, ni credential_resolver ni la gobernanza importaban. CI no lo veía
porque cada job instalaba su propia lista a mano.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_ARBOLES = ("jax", "jacobs", "las_manos", "policy")
_EXCLUIDOS = {".git", "__pycache__", ".venv", "venv", "node_modules", "tests", ".pytest_cache"}

# import de nivel superior -> distribución fijada en requirements.txt
_DISTRIBUCION = {
    "aiomysql": "aiomysql", "cryptography": "cryptography", "fastapi": "fastapi", "pymysql": "pymysql",
    "h11": "h11", "httpx": "httpx", "pydantic": "pydantic", "uvicorn": "uvicorn", "yaml": "pyyaml",
}

# Terceros que NO van en requirements.txt, cada uno con su motivo.
_FUERA_DE_REQUIREMENTS = {
    "sentence_transformers": "reranker opcional de jax/memory/db.py::_get_reranker, import perezoso con fail-soft declarado",
}


def _es_test(p: Path) -> bool:
    return p.name.startswith("test_") or p.name.endswith("_test.py")


def _recorrer(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUIDOS]
        for nombre in filenames:
            yield Path(dirpath) / nombre


def _modulos_locales() -> set[str]:
    locales = set()
    for dirpath, dirnames, filenames in os.walk(RAIZ):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUIDOS]
        locales.update(dirnames)
        locales.update(Path(f).stem for f in filenames if f.endswith(".py"))
    return locales


def _terceros_en(fuente: str, locales: set[str]) -> set[str]:
    tops = set()
    for nodo in ast.walk(ast.parse(fuente)):
        if isinstance(nodo, ast.Import):
            tops.update(a.name.split(".")[0] for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
            tops.add(nodo.module.split(".")[0])
    return {t for t in tops if t not in sys.stdlib_module_names and t not in locales}


def _archivos_de_servicio():
    for arbol in _ARBOLES:
        for p in _recorrer(RAIZ / arbol):
            if p.suffix == ".py" and not _es_test(p):
                yield p


def _requirements() -> dict[str, str]:
    fijadas = {}
    for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if linea:
            nombre, _, version = linea.partition("==")
            fijadas[nombre.strip().lower().replace("_", "-")] = version.strip()
    return fijadas


def test_todo_import_de_terceros_esta_declarado():
    locales = _modulos_locales()
    sin_declarar: dict[str, list[str]] = {}
    for p in _archivos_de_servicio():
        for tercero in _terceros_en(p.read_text(encoding="utf-8", errors="replace"), locales):
            if tercero not in _DISTRIBUCION and tercero not in _FUERA_DE_REQUIREMENTS:
                sin_declarar.setdefault(tercero, []).append(str(p.relative_to(RAIZ)))
    assert sin_declarar == {}


def test_cada_distribucion_de_servicio_esta_fijada_con_version_exacta():
    fijadas = _requirements()
    assert sorted(d for d in set(_DISTRIBUCION.values()) if not fijadas.get(d)) == []


def test_aiofiles_ya_no_se_usa_ni_se_instala():
    assert "aiofiles" not in _requirements()
    locales = _modulos_locales()
    assert [str(p.relative_to(RAIZ)) for p in _archivos_de_servicio()
            if "aiofiles" in _terceros_en(p.read_text(encoding="utf-8", errors="replace"), locales)] == []


def test_el_ci_instala_desde_requirements():
    lineas = (RAIZ / ".github" / "workflows" / "policy.yml").read_text(encoding="utf-8").splitlines()
    inicio = lineas.index("  tests-puros:")
    fin = next(i for i in range(inicio + 1, len(lineas))
               if lineas[i].startswith("  ") and not lineas[i].startswith("   ") and lineas[i].rstrip().endswith(":"))
    assert any("pip install -r requirements.txt" in l for l in lineas[inicio:fin])
    assert [l for l in lineas if "pip install" in l and "aiofiles" in l] == []


def test_el_detector_ve_un_import_de_terceros():
    fuente = "import yaml\nimport os\nfrom jacobs import store\nfrom paquete_raro.sub import x\n"
    assert _terceros_en(fuente, {"jacobs"}) == {"yaml", "paquete_raro"}
