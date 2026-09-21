"""El defecto que este extractor existe para evitar, reproducido con un archivo
sintético: LibreOffice convierte UNA hoja de seis, calladito."""
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from procesamiento.extractores import excel


def _libro_de_seis_hojas(destino: Path) -> Path:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for i, nombre in enumerate(
        ["2025 JUNIO", "2024", "2023", "2022", "consolidado Bac", "CONSOLIDADO"]
    ):
        ws = wb.create_sheet(title=nombre)
        ws["A1"] = "ACTIVOS"
        ws["B1"] = 100 * (i + 1)
        ws["A2"] = "PASIVOS"
        ws["B2"] = 40 * (i + 1)
    wb.save(destino)
    return destino


def test_un_libro_de_seis_hojas_produce_SEIS_extractos(tmp_path: Path):
    origen = _libro_de_seis_hojas(tmp_path / "eeff.xlsx")
    r = excel.extraer(origen)
    assert r.estado == "ok"
    assert len(r.salidas) == 6, (
        f"se perdieron hojas: {sorted(r.salidas)}. "
        "Este es exactamente el defecto de LibreOffice medido el 2026-09-20."
    )
    assert r.detalle["hojas"] == 6
    assert r.detalle["hojas_extraidas"] == 6


def test_el_contenido_conserva_filas_y_columnas(tmp_path: Path):
    origen = _libro_de_seis_hojas(tmp_path / "eeff.xlsx")
    r = excel.extraer(origen)
    primera = r.salidas[sorted(r.salidas)[0]]
    assert "ACTIVOS,100" in primera.replace("\r\n", "\n")
    assert "PASIVOS,40" in primera.replace("\r\n", "\n")


def test_un_archivo_que_no_es_excel_da_error_sin_extracto(tmp_path: Path):
    malo = tmp_path / "no-es.xlsx"
    malo.write_bytes(b"esto no es un zip")
    r = excel.extraer(malo)
    assert r.estado == "error"
    assert r.salidas == {}
    assert "razon" in r.detalle


def test_un_libro_con_TODAS_las_hojas_en_blanco_da_error_sin_reventar(tmp_path: Path):
    """Resultado exige que un 'ok'/'parcial' tenga al menos una salida con
    contenido (tras .strip()). Un libro cuyas hojas están todas vacías
    produciría puros CSV vacíos -- eso ahora levanta ValueError dentro de
    Resultado si el extractor intenta devolver 'ok'. El extractor tiene que
    detectarlo ANTES y devolver 'error' con una razón, no reventar."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for nombre in ["Hoja A", "Hoja B"]:
        wb.create_sheet(title=nombre)
    origen = tmp_path / "todo-en-blanco.xlsx"
    wb.save(origen)

    r = excel.extraer(origen)

    assert r.estado == "error"
    assert r.salidas == {}
    assert "razon" in r.detalle
    assert "blanco" in r.detalle["razon"] or "vacia" in r.detalle["razon"] or "vacías" in r.detalle["razon"]


def test_una_hoja_en_blanco_entre_varias_con_datos_es_un_archivo_valido(tmp_path: Path):
    """El caso legítimo que NO se rechaza: una hoja en blanco entre varias
    con contenido es un archivo válido y tiene que salir 'ok', no 'error'."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws1 = wb.create_sheet(title="con datos")
    ws1["A1"] = "ACTIVOS"
    ws1["B1"] = 100
    wb.create_sheet(title="en blanco")
    ws3 = wb.create_sheet(title="tambien con datos")
    ws3["A1"] = "PASIVOS"
    ws3["B1"] = 40
    origen = tmp_path / "una-hoja-blanco.xlsx"
    wb.save(origen)

    r = excel.extraer(origen)

    assert r.estado == "ok"
    assert len(r.salidas) == 3
    assert r.detalle["hojas"] == 3
    assert r.detalle["hojas_extraidas"] == 3
