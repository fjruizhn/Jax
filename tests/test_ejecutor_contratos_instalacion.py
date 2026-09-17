# tests/test_ejecutor_contratos_instalacion.py
"""Lo que se instala en /opt/ejecutor/lib: sólo biblioteca estándar, render
determinista del gancho y del managed-settings de la jaula."""
import ast
import json
import sys
from pathlib import Path

import pytest

from jax.ejecutor.contratos import instalacion

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rel", instalacion.INSTALABLES)
def test_lo_instalable_es_solo_biblioteca_estandar(rel):
    arbol = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos = [a.name for a in nodo.names]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module == "jax.ejecutor.contratos":
            for alias in nodo.names:
                assert f"jax/ejecutor/contratos/{alias.name}.py" in instalacion.INSTALABLES, (rel, alias.name)
            continue
        elif isinstance(nodo, ast.ImportFrom):
            modulos = [nodo.module or ""]
        else:
            continue
        for m in modulos:
            if m.startswith("jax.ejecutor.contratos."):
                assert f"jax/ejecutor/contratos/{m.rsplit('.', 1)[1]}.py" in instalacion.INSTALABLES, (rel, m)
            else:
                assert m == "__future__" or m.split(".")[0] in sys.stdlib_module_names, (rel, m)


def test_render_del_gancho():
    texto = instalacion.renderizar_gancho("/opt/ejecutor/lib", 10)
    assert "LIB='/opt/ejecutor/lib'" in texto and "timeout -s KILL '10'" in texto
    assert "@LIB@" not in texto and "@TOPE_S@" not in texto


@pytest.mark.parametrize("lib, tope", [("relativa/lib", 10), ("/opt/con espacio", 10), ("/opt/x'y", 10),
                                       ("/opt/ejecutor/lib", 0), ("/opt/ejecutor/lib", 61)])
def test_render_rechaza_valores_peligrosos(lib, tope):
    with pytest.raises(ValueError):
        instalacion.renderizar_gancho(lib, tope)


def test_managed_settings_de_la_jaula():
    doc = json.loads(instalacion.renderizar_managed_settings("/opt/ejecutor/lib", "/etc/jax-ejecutor/politica.json", 10))
    (grupo,) = doc["hooks"]["PreToolUse"]
    assert grupo["matcher"] == "*"
    assert grupo["hooks"] == [{"type": "command", "timeout": 15,
                               "command": "/opt/ejecutor/lib/gancho.sh /etc/jax-ejecutor/politica.json"}]
    assert doc["allowManagedHooksOnly"] is True and doc["disableAllHooks"] is False


def test_manifiesto_es_estable():
    assert instalacion.manifiesto({"b": b"2", "a": b"1"}) == (
        "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b  a\n"
        "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35  b\n")
