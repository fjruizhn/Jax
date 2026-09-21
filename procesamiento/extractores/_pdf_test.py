"""PDF nativo -> Markdown, con las tablas conservadas -- y la compuerta que
decide si un PDF necesita OCR.

pdftotext NO sirve: destruye las tablas, y en un expediente financiero la
tabla ES el documento. pdfplumber conserva filas y columnas.

Los fixtures son PDF 1.4 escritos a mano, byte a byte (sin dependencias de
generación). El ruling del controlador (task-4-brief.md) preveía que
pdfminer podía rechazar un PDF sin tabla `xref` y pedía recurrir a
LibreOffice en ese caso -- verificado contra pdfplumber==0.11.10 el
2026-09-21: los cuatro fixtures de abajo (texto simple, sin capa de texto,
tabla con líneas vectoriales, híbrido de dos páginas) parsean sin problema
con el modo de recuperación de pdfminer.six, así que no hace falta LibreOffice
acá -- una dependencia externa menos en CI.
"""
from pathlib import Path

import pytest

pdfplumber = pytest.importorskip("pdfplumber")

from procesamiento.extractores import pdf


def _pdf_una_pagina(destino: Path, stream: bytes) -> Path:
    """Arma un PDF de UNA página con el content stream dado."""
    partes = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n",
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        b"5 0 obj<</Length " + str(len(stream)).encode() + b">>stream\n",
        stream,
        b"endstream endobj\n",
        b"trailer<</Root 1 0 R>>\n",
    ]
    destino.write_bytes(b"".join(partes))
    return destino


def _pdf_con_texto(destino: Path) -> Path:
    """PDF mínimo con capa de texto, escrito a mano (sin dependencias)."""
    stream = b"BT /F1 12 Tf 20 100 Td (ACTIVOS TOTALES 1234) Tj ET\n"
    return _pdf_una_pagina(destino, stream)


def _pdf_sin_texto(destino: Path) -> Path:
    """PDF de una página sin ningún content stream -- lo que produce un
    escaneo puesto en un PDF sin pasar por OCR: sólo la imagen, cero texto."""
    partes = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n",
        b"trailer<</Root 1 0 R>>\n",
    ]
    destino.write_bytes(b"".join(partes))
    return destino


def _pdf_con_tabla(destino: Path) -> Path:
    """PDF de una página con una tabla REAL: líneas vectoriales formando una
    grilla de 2x2, más texto dentro de cada celda -- es lo que hace que
    `pagina.extract_tables()` de pdfplumber la reconozca como tabla (su
    estrategia por defecto detecta líneas dibujadas, no separa por espacios).
    """
    stream = (
        b"1 w\n"
        b"20 180 m 180 180 l S\n"
        b"20 150 m 180 150 l S\n"
        b"20 120 m 180 120 l S\n"
        b"20 180 m 20 120 l S\n"
        b"100 180 m 100 120 l S\n"
        b"180 180 m 180 120 l S\n"
        b"BT /F1 10 Tf 30 160 Td (Cuenta) Tj ET\n"
        b"BT /F1 10 Tf 110 160 Td (2025) Tj ET\n"
        b"BT /F1 10 Tf 30 130 Td (Pasivos) Tj ET\n"
        b"BT /F1 10 Tf 110 130 Td (450) Tj ET\n"
    )
    return _pdf_una_pagina(destino, stream)


def _pdf_hibrido(destino: Path) -> Path:
    """Dos páginas: la primera con texto real (suficiente para superar
    MINIMO_CARACTERES del documento entero), la segunda sin ningún content
    stream -- una página que es pura imagen mezclada con una que sí es
    nativa. Es el caso del brief: "una página con texto, otra que es
    imagen"."""
    stream1 = (
        b"BT /F1 12 Tf 20 100 Td (ACTIVOS TOTALES 1234) Tj ET\n"
        b"BT /F1 12 Tf 20 80 Td (PASIVO Y PATRIMONIO 5678) Tj ET\n"
    )
    partes = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R 6 0 R]/Count 2>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n",
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        b"5 0 obj<</Length " + str(len(stream1)).encode() + b">>stream\n",
        stream1,
        b"endstream endobj\n",
        # página 2: sin /Contents -- ninguna capa de texto, como una imagen pura.
        b"6 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n",
        b"trailer<</Root 1 0 R>>\n",
    ]
    destino.write_bytes(b"".join(partes))
    return destino


def test_detecta_que_un_pdf_nativo_tiene_texto(tmp_path: Path):
    assert pdf.tiene_capa_de_texto(_pdf_con_texto(tmp_path / "n.pdf")) is True


def test_extrae_el_texto_del_pdf_nativo(tmp_path: Path):
    r = pdf.extraer(_pdf_con_texto(tmp_path / "n.pdf"))
    assert r.estado == "ok"
    assert "ACTIVOS TOTALES 1234" in r.salidas["texto.md"]
    assert r.detalle["paginas"] == 1


def test_un_pdf_roto_da_error_sin_extracto(tmp_path: Path):
    malo = tmp_path / "roto.pdf"
    malo.write_bytes(b"no soy un pdf")
    r = pdf.extraer(malo)
    assert r.estado == "error"
    assert r.salidas == {}


def test_un_pdf_sin_capa_de_texto_no_se_declara_ok(tmp_path: Path):
    """Un PDF de imagen pura NO se resuelve acá: se manda a OCR. Lo que NO
    puede pasar es que devuelva 'ok' con un extracto vacío."""
    vacio = _pdf_sin_texto(tmp_path / "escaneado.pdf")
    assert pdf.tiene_capa_de_texto(vacio) is False
    r = pdf.extraer(vacio)
    assert r.estado != "ok"
    assert r.salidas == {}


def test_una_tabla_real_conserva_filas_y_columnas(tmp_path: Path):
    """Aseverá el CONTENIDO de la tabla, no sólo que extraer() no explote:
    esto es lo que pdftotext destruiría."""
    r = pdf.extraer(_pdf_con_tabla(tmp_path / "tabla.pdf"))
    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "Cuenta" in md and "2025" in md
    assert "Pasivos" in md and "450" in md
    # fila y columna: "Cuenta" y "2025" en la MISMA fila de la tabla markdown.
    lineas_tabla = [l for l in md.splitlines() if l.startswith("|")]
    assert any("Cuenta" in l and "2025" in l for l in lineas_tabla)
    assert any("Pasivos" in l and "450" in l for l in lineas_tabla)
    assert r.detalle["tablas"] == 1


def test_un_pdf_hibrido_da_parcial_y_lo_dice_en_el_detalle(tmp_path: Path):
    """Página 1 con texto real, página 2 sin ninguna capa de texto (imagen
    pura). Esto NO es 'ok' -- salió MENOS de lo que el documento tiene, así
    que tiene que ser 'parcial', y el detalle tiene que decir CUÁL página
    quedó sin resolver, no callarlo."""
    r = pdf.extraer(_pdf_hibrido(tmp_path / "hibrido.pdf"))
    assert r.estado == "parcial"
    assert "ACTIVOS TOTALES 1234" in r.salidas["texto.md"]
    assert r.detalle["paginas"] == 2
    assert r.detalle["paginas_sin_texto"] == [2]
