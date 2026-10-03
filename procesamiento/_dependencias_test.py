"""Freno de dependencias de extraccion (jax-14, 2026-10-03).

El 2026-10-03 el venv de produccion de LAS MANOS no tenia pdfplumber, openpyxl
ni python-docx y los PDF/DOCX/XLSX quedaron en `sin_extractor` en silencio.
Estas pruebas fijan: la lista sale de `requirements-archivos.txt` (no se
duplica a mano), cada paquete tiene un modulo declarado, y la falta de un
modulo se informa por NOMBRE DE PAQUETE.
"""
from __future__ import annotations

import sys

import pytest

from procesamiento import dependencias

# Estas pruebas simulan la ausencia con `sys.modules[nombre] = None`; solo
# tienen sentido si el modulo REAL esta instalado, si no el "todo instalado"
# seria falso. Mismo criterio que el resto de procesamiento/ (piso exacto en CI).
pytest.importorskip("pdfplumber")
pytest.importorskip("openpyxl")
pytest.importorskip("docx")


def test_el_mapa_cubre_cada_linea_de_requirements_archivos():
    paquetes = dependencias.paquetes_declarados()
    assert paquetes, "requirements-archivos.txt no declara nada: la lista se leyo mal"
    sin_mapa = [p for p in paquetes if p not in dependencias.MODULO_POR_PAQUETE]
    assert sin_mapa == [], (
        f"paquetes de requirements-archivos.txt sin modulo declarado en "
        f"MODULO_POR_PAQUETE: {sin_mapa}"
    )


def test_python_docx_se_importa_como_docx():
    assert dependencias.MODULO_POR_PAQUETE["python-docx"] == "docx"


def test_con_todo_instalado_no_falta_nada():
    assert dependencias.faltantes() == []
    assert dependencias.estado() == {"ok": True, "faltan": []}


@pytest.mark.parametrize("paquete,modulo", [
    ("python-docx", "docx"), ("pdfplumber", "pdfplumber"), ("openpyxl", "openpyxl"),
])
def test_un_modulo_ausente_se_informa_por_nombre_de_paquete(monkeypatch, paquete, modulo):
    monkeypatch.setitem(sys.modules, modulo, None)
    assert dependencias.faltantes() == [paquete]
    assert dependencias.estado() == {"ok": False, "faltan": [paquete]}


def test_varios_ausentes_salen_todos_y_en_orden_estable(monkeypatch):
    monkeypatch.setitem(sys.modules, "docx", None)
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    assert dependencias.faltantes() == ["openpyxl", "python-docx"]


def test_un_paquete_sin_mapa_cuenta_como_faltante_y_no_se_ignora(tmp_path):
    """Falla cerrado: una linea nueva en el requirements sin mapa no puede
    dar 'todo bien' en silencio."""
    req = tmp_path / "req.txt"
    req.write_text("openpyxl==3.1.5\npaquete-nuevo==1.0\n", encoding="utf-8")
    assert dependencias.faltantes(req) == ["paquete-nuevo"]


def test_requirements_ilegible_falla_cerrado(tmp_path):
    assert dependencias.estado(tmp_path / "no-existe.txt")["ok"] is False


def test_comentarios_lineas_vacias_y_extras_se_ignoran(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("# nota\n\nopenpyxl==3.1.5  # inline\npython-docx[extra]>=1.0\n", encoding="utf-8")
    assert dependencias.paquetes_declarados(req) == ["openpyxl", "python-docx"]
