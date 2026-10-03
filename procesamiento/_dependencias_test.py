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
    assert dependencias.estado() == {"ok": True, "faltan": [], "motivos": {}}


@pytest.mark.parametrize("paquete,modulo", [
    ("python-docx", "docx"), ("pdfplumber", "pdfplumber"), ("openpyxl", "openpyxl"),
])
def test_un_modulo_ausente_se_informa_por_nombre_de_paquete(monkeypatch, paquete, modulo):
    monkeypatch.setitem(sys.modules, modulo, None)
    assert dependencias.faltantes() == [paquete]
    estado = dependencias.estado()
    assert estado["ok"] is False and estado["faltan"] == [paquete]
    assert estado["motivos"] == {paquete: "ModuleNotFoundError"}


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
    e = dependencias.estado(tmp_path / "no-existe.txt")
    assert e["ok"] is False
    assert e["motivos"]["no-existe.txt"] == "FileNotFoundError"


def test_comentarios_lineas_vacias_y_extras_se_ignoran(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("# nota\n\nopenpyxl==3.1.5  # inline\npython-docx[extra]>=1.0\n", encoding="utf-8")
    assert dependencias.paquetes_declarados(req) == ["openpyxl", "python-docx"]


# -- MAJOR-1: estado() NUNCA lanza ------------------------------------------
def test_un_import_que_lanza_attributeerror_cuenta_como_faltante_con_su_motivo():
    real = __import__("importlib").import_module

    def importador(nombre):
        if nombre == "docx":
            raise AttributeError("modulo a medio instalar")
        return real(nombre)

    e = dependencias.estado(importador=importador)
    assert e["ok"] is False
    assert e["faltan"] == ["python-docx"]
    assert e["motivos"] == {"python-docx": "AttributeError"}


def test_un_requirements_que_no_es_utf8_da_ok_false_sin_lanzar(tmp_path):
    req = tmp_path / "req.txt"
    req.write_bytes(b"openpyxl==3.1.5\n\xff\xfe\x80 basura\n")
    e = dependencias.estado(req)
    assert e["ok"] is False
    assert e["motivos"]["req.txt"] == "UnicodeDecodeError"


# -- MINOR-1: lo que no se puede evaluar falla cerrado ----------------------
def test_un_requirements_sin_ninguna_linea_util_es_ok_false(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("# solo comentarios\n\n", encoding="utf-8")
    e = dependencias.estado(req)
    assert e["ok"] is False
    assert e["motivos"]["req.txt"] == "requirements vacío"


@pytest.mark.parametrize("linea", [
    "-r otro.txt", "-e .", "-e git+https://usuario:secreto@host/repo.git#egg=x",
    "https://host/paquete-1.0.whl", "./vendor/paquete", "/opt/wheels/paquete.whl",
    "paquete @ https://host/p.whl",
])
def test_lineas_de_opcion_url_o_ruta_local_son_desconocidas_y_fallan_cerrado(tmp_path, linea):
    req = tmp_path / "req.txt"
    req.write_text(f"openpyxl==3.1.5\n{linea}\n", encoding="utf-8")
    e = dependencias.estado(req)
    assert e["ok"] is False
    assert e["faltan"] == ["linea-no-reconocida:2"]
    assert "secreto" not in repr(e), "la linea cruda no puede filtrarse al estado (va a log y Telegram)"


def test_un_marcador_falso_excluye_la_linea_si_packaging_esta(tmp_path, monkeypatch):
    pytest.importorskip("packaging.markers")
    req = tmp_path / "req.txt"
    req.write_text('pdfplumber==0.11.10; python_version < "3"\nopenpyxl==3.1.5\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    assert dependencias.estado(req) == {"ok": True, "faltan": [], "motivos": {}}


def test_un_marcador_verdadero_exige_la_linea(tmp_path, monkeypatch):
    pytest.importorskip("packaging.markers")
    req = tmp_path / "req.txt"
    req.write_text('pdfplumber==0.11.10; python_version >= "3"\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    assert dependencias.estado(req)["faltan"] == ["pdfplumber"]


def test_sin_packaging_la_linea_con_marcador_se_exige_igual(tmp_path, monkeypatch):
    req = tmp_path / "req.txt"
    req.write_text('pdfplumber==0.11.10; python_version < "3"\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "packaging.markers", None)
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    assert dependencias.estado(req)["faltan"] == ["pdfplumber"]


# -- MAJOR-2a: freno por tipo ------------------------------------------------
def test_el_mapa_de_extensiones_cubre_cada_paquete_del_mapa_de_modulos():
    assert set(dependencias.EXTENSIONES_POR_PAQUETE) == set(dependencias.MODULO_POR_PAQUETE)
    assert dependencias.EXTENSIONES_POR_PAQUETE["pdfplumber"] == {".pdf"}
    assert dependencias.EXTENSIONES_POR_PAQUETE["openpyxl"] == {".xlsx", ".xlsm"}
    assert dependencias.EXTENSIONES_POR_PAQUETE["python-docx"] == {".docx"}


def test_un_lote_de_imagenes_no_se_frena_por_pdfplumber():
    assert dependencias.lote_afectado(["pdfplumber"], ["a.png", "b.JPG"]) is False


def test_un_lote_con_un_pdf_se_frena_por_pdfplumber_aunque_traiga_imagenes():
    assert dependencias.lote_afectado(["pdfplumber"], ["a.png", "B.PDF"]) is True


def test_un_paquete_faltante_sin_mapa_frena_todos_los_tipos():
    assert dependencias.lote_afectado(["linea-no-reconocida:2"], ["a.png"]) is True
    assert dependencias.tipos_de(["linea-no-reconocida:2"]) == ["*"]


def test_tipos_de_junta_las_extensiones_de_los_paquetes_faltantes():
    assert dependencias.tipos_de(["python-docx", "openpyxl"]) == [".docx", ".xlsm", ".xlsx"]


def test_un_lote_vacio_no_se_frena():
    assert dependencias.lote_afectado(["pdfplumber"], []) is False


# -- Jax#338: Pillow decodifica toda imagen antes del OCR ----------------------
def test_pillow_esta_declarado_con_version_fijada_y_mapeado_a_las_imagenes():
    declaradas = dependencias.paquetes_declarados()
    assert "pillow" in declaradas
    lineas = dependencias.REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    assert any(l.strip().startswith("pillow==") for l in lineas)
    assert dependencias.MODULO_POR_PAQUETE["pillow"] == "PIL"
    assert dependencias.EXTENSIONES_POR_PAQUETE["pillow"] >= {".png", ".jpg", ".jpeg", ".tif", ".bmp", ".webp"}


def test_sin_pillow_se_frena_un_lote_de_imagenes_pero_no_uno_de_pdf():
    assert dependencias.lote_afectado(["pillow"], ["a.JPG"]) is True
    assert dependencias.lote_afectado(["pillow"], ["a.pdf", "b.docx"]) is False
