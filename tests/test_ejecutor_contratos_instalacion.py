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
_PAQUETES_PROPIOS = {"jax.ejecutor.contratos": "jax/ejecutor/contratos", "jax.core": "jax/core"}


@pytest.mark.parametrize("rel", instalacion.INSTALABLES)
def test_lo_instalable_es_solo_biblioteca_estandar(rel):
    arbol = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module in _PAQUETES_PROPIOS:
            for alias in nodo.names:
                assert f"{_PAQUETES_PROPIOS[nodo.module]}/{alias.name}.py" in instalacion.INSTALABLES, (rel, alias.name)
            continue
        if isinstance(nodo, ast.Import):
            modulos = [a.name for a in nodo.names]
        elif isinstance(nodo, ast.ImportFrom):
            modulos = [nodo.module or ""]
        else:
            continue
        for m in modulos:
            paquete, _, hoja = m.rpartition(".")
            if paquete in _PAQUETES_PROPIOS:
                assert f"{_PAQUETES_PROPIOS[paquete]}/{hoja}.py" in instalacion.INSTALABLES, (rel, m)
            else:
                assert m == "__future__" or m.split(".")[0] in sys.stdlib_module_names, (rel, m)


def test_el_freno_y_lo_que_consulta_se_instalan():
    for rel in ("jax/core/__init__.py", "jax/core/interruptor.py", "jax/ejecutor/contratos/pausa.py",
                "jax/ejecutor/contratos/freno.py"):
        assert rel in instalacion.INSTALABLES


def test_lo_que_arranca_la_unidad_del_freno_esta_en_el_manifiesto():
    """El módulo que importa ExecStart sale de la PLANTILLA, no de una lista escrita a mano:
    reinstalar desde master no puede dejar el freno root afuera del manifiesto (2026-09-17: la
    biblioteca de producción se instaló desde la rama porque el manifiesto de master no lo traía).
    Lo que ese módulo importa lo cubre test_lo_instalable_es_solo_biblioteca_estandar."""
    import re
    texto = instalacion.renderizar_unidad_freno("/opt/ejecutor/lib")
    modulos = re.findall(r"from (jax(?:\.\w+)+) import", texto)
    assert modulos, texto
    for m in modulos:
        assert m.replace(".", "/") + ".py" in instalacion.INSTALABLES, m
        partes = m.split(".")
        for i in range(1, len(partes)):
            assert "/".join(partes[:i]) + "/__init__.py" in instalacion.INSTALABLES, (m, i)
    assert len(set(instalacion.INSTALABLES)) == len(instalacion.INSTALABLES)


def test_render_de_la_unidad_del_freno():
    texto = instalacion.renderizar_unidad_freno("/opt/ejecutor/lib")
    assert "User=root" in texto and "Restart=always" in texto and "EnvironmentFile=/etc/jax/.env" in texto
    assert "sys.path.insert(0, '/opt/ejecutor/lib')" in texto and "/usr/bin/python3 -I" in texto
    assert "RuntimeDirectory=ejecutor-freno" in texto and "@LIB@" not in texto
    with pytest.raises(ValueError):
        instalacion.renderizar_unidad_freno("/opt/x'; rm -rf /")


def test_principal_escribe_la_unidad_del_freno_en_el_manifiesto(tmp_path):
    env = {"JAX_EJECUTOR_LIB": "/opt/ejecutor/lib", "JAX_EJECUTOR_GANCHO_TOPE_S": "10",
           "JAX_EJECUTOR_POLITICA": "/etc/jax-ejecutor/politica.json"}
    assert instalacion.principal([str(tmp_path)], env) == 0
    assert (tmp_path / "ejecutor-freno.service").read_text() == instalacion.renderizar_unidad_freno("/opt/ejecutor/lib")
    assert "  ejecutor-freno.service\n" in (tmp_path / "manifiesto.sha256").read_text()


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
