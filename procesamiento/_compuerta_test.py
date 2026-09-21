"""La compuerta: elige extractor por tipo, y NUNCA paga el caro si el barato
alcanzó.

Task 7 (2026-09-20/21, procesamiento-archivos-nucleo). La regla es una sola
idea: un PDF CON capa de texto no paga OCR, y un PDF SIN capa de texto no
pasa por el extractor de texto. Los tests lo verifican con monkeypatch,
comprobando que el camino caro NO se invocó -- aseverar qué NO se llamó es
tan importante como qué sí.
"""
from pathlib import Path

from procesamiento import compuerta


def test_un_tipo_desconocido_dice_sin_extractor_y_no_inventa(tmp_path: Path):
    raro = tmp_path / "algo.xyz"
    raro.write_bytes(b"contenido cualquiera")
    r = compuerta.extraer(raro)
    assert r.estado == "sin_extractor"
    assert r.salidas == {}


def test_un_pdf_sin_capa_de_texto_cae_en_ocr_y_no_en_pdfplumber(
    tmp_path: Path, monkeypatch
):
    """El PDF escaneado NO se resuelve con el extractor de texto: pasa a OCR.
    Y el OCR sólo se invoca si el camino barato ya dijo que no."""
    from procesamiento.extractores import ocr, pdf
    from procesamiento.resultado import Resultado

    llamadas: list[str] = []

    def falso_tiene_texto(origen):
        llamadas.append("deteccion")
        return False

    def falso_ocr(origen, idioma="spa"):
        llamadas.append("ocr")
        return Resultado(
            estado="ok", salidas={"texto.txt": "leido por ocr"},
            detalle={}, extractor="tesseract", version="5.5.0",
        )

    def no_debe_llamarse(origen):
        raise AssertionError("pdfplumber no debe correr sobre un PDF sin texto")

    monkeypatch.setattr(pdf, "tiene_capa_de_texto", falso_tiene_texto)
    monkeypatch.setattr(pdf, "extraer", no_debe_llamarse)
    monkeypatch.setattr(ocr, "extraer", falso_ocr)

    archivo = tmp_path / "escaneado.pdf"
    archivo.write_bytes(b"%PDF-1.4 lo que sea")
    r = compuerta.extraer(archivo)

    assert r.estado == "ok"
    assert llamadas == ["deteccion", "ocr"]


def test_un_pdf_con_texto_no_paga_ocr(tmp_path: Path, monkeypatch):
    from procesamiento.extractores import ocr, pdf
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(pdf, "tiene_capa_de_texto", lambda origen: True)
    monkeypatch.setattr(
        pdf, "extraer",
        lambda origen: Resultado(
            estado="ok", salidas={"texto.md": "hola"}, detalle={},
            extractor="pdfplumber", version="0.11.10",
        ),
    )

    def no_debe_llamarse(origen, idioma="spa"):
        raise AssertionError("no se paga OCR si el PDF ya tenia texto")

    monkeypatch.setattr(ocr, "extraer", no_debe_llamarse)

    archivo = tmp_path / "nativo.pdf"
    archivo.write_bytes(b"%PDF-1.4 lo que sea")
    assert compuerta.extraer(archivo).estado == "ok"
