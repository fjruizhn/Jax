"""Elige el extractor por tipo, y NUNCA paga el caro si el barato alcanzo.

Un tipo sin extractor devuelve `sin_extractor`: el archivo se guarda igual, pero
nadie finge haberlo leido. Es la diferencia entre un sistema honesto y uno que
hace inventar al modelo (Principio VIII).

La unica regla de esta pieza: cada capa corre SOLO si la anterior no alcanzo.
Un PDF va primero a `pdf.tiene_capa_de_texto` (deteccion barata, determinista,
sin modelos) -- si tiene capa de texto, `pdf.extraer` resuelve todo; si NO,
el texto no esta ahi para sacar y se paga OCR. `ocr.extraer` (desde la ronda
1 de OCR, task-5-6-hallazgos.md) ya acepta PDF ademas de imagen -- rasteriza
con `pdftoppm` por dentro y reporta por pagina -- asi que no hace falta que
la compuerta distinga PDF escaneado de imagen suelta para ese camino.
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.extractores import excel, ocr, pdf, word
from procesamiento.resultado import Resultado

IMAGENES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EXCEL = {".xlsx", ".xlsm"}
WORD = {".docx"}


def extraer(origen: Path) -> Resultado:
    sufijo = Path(origen).suffix.lower()

    if sufijo in EXCEL:
        return excel.extraer(origen)
    if sufijo in WORD:
        return word.extraer(origen)
    if sufijo in IMAGENES:
        return ocr.extraer(origen)
    if sufijo == ".pdf":
        # Compuerta: el camino barato primero. OCR solo si no hay capa de texto.
        if pdf.tiene_capa_de_texto(origen):
            return pdf.extraer(origen)
        return ocr.extraer(origen)

    return Resultado(
        estado="sin_extractor", salidas={}, extractor="ninguno", version="-",
        detalle={"razon": f"no hay extractor para '{sufijo or 'sin extension'}'"},
    )
