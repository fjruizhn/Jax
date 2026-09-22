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

Ronda de arreglo (2026-09-21, task-7-hallazgos.md, I-3): un PDF nativo real
renombrado `.png` saltaba la compuerta entera -- iba derecho a OCR sin pasar
nunca por `tiene_capa_de_texto`, pagando OCR pudiendo leerlo gratis, sin
dejar rastro. Se rutea por CONTENIDO (primeros bytes) cuando el contenido es
decisivo (PDF, OLE2, imagen); la extension queda de respaldo cuando el
contenido no dice nada -- un ZIP (`PK\x03\x04`) es ambiguo por si solo entre
`.xlsx` y `.docx`, así que ahí la extension SIGUE siendo quien decide, igual
que siempre. Cuando el contenido decisivo no coincide con la extension, se
anota en `detalle["extension_enganosa"]` -- información que el dueño del
archivo va a querer, y cuesta cero.
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.extractores import excel, ocr, pdf, word
from procesamiento.resultado import Resultado

IMAGENES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EXCEL = {".xlsx", ".xlsm"}
WORD = {".docx"}

_FIRMA_OLE2 = bytes.fromhex("D0CF11E0A1B11AE1")

# Firma -> tipo decisivo. PK\x03\x04 (zip) NO entra acá a propósito: es el
# contenedor de .xlsx Y .docx por igual, así que por sí solo no decide nada
# -- la extensión sigue siendo quien distingue cuál de los dos es.
_FIRMAS_IMAGEN = (b"\x89PNG", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM")


def _tipo_por_contenido(origen: Path) -> str | None:
    """'pdf' / 'ole2' / 'imagen' según los primeros bytes, o `None` si el
    contenido no da una señal decisiva (zip -- xlsx y docx son el mismo
    contenedor -- o un formato no reconocido): en ese caso la extensión
    decide sola, como siempre."""
    try:
        with open(origen, "rb") as fh:
            cabecera = fh.read(16)
    except OSError:
        return None
    if cabecera.startswith(b"%PDF"):
        return "pdf"
    if cabecera[:8] == _FIRMA_OLE2:
        return "ole2"
    if cabecera.startswith(_FIRMAS_IMAGEN) or (
        cabecera[:4] == b"RIFF" and cabecera[8:12] == b"WEBP"
    ):
        return "imagen"
    return None


def _con_extension_enganosa(r: Resultado, sufijo: str, contenido: str) -> Resultado:
    detalle = {
        **dict(r.detalle),
        "extension_enganosa": {"nombre": sufijo or "(sin extension)", "contenido": contenido},
    }
    return Resultado(
        estado=r.estado, salidas=dict(r.salidas), detalle=detalle,
        extractor=r.extractor, version=r.version,
    )


def _pdf_o_ocr(origen: Path) -> Resultado:
    """Decide entre `pdf.extraer` y `ocr.extraer` para un PDF -- la MISMA
    decisión se necesitaba en dos ramas de `extraer()` (contenido decisivo
    y respaldo por extensión) y estaba duplicada ahí, cada una con su
    propia copia de la lógica.

    C-2 (final-hallazgos.md, ronda de cierre): sin `pdfplumber` instalado,
    `pdf.tiene_capa_de_texto()` ahora deja escapar `ModuleNotFoundError`
    (ver su docstring) en vez de tragarlo como "sin texto" -- acá se
    captura para resolver con `pdf.extraer(origen)`, que YA sabe convertir
    esa misma falta de dependencia en `sin_extractor` (mismo camino que un
    `.docx` sin `python-docx`). Antes de este arreglo, esa misma falta de
    dependencia rendía "sin texto" -> se rutea a OCR -> `estado='ok'` con
    las cifras DESTRUIDAS (medido: 8/8 cifras con pdfplumber, 0/8 sin él),
    sin que `detalle` dijera una palabra de la dependencia faltante -- un
    fallo de despliegue produciendo cifras falsas etiquetadas 'ok'."""
    try:
        tiene_texto = pdf.tiene_capa_de_texto(origen)
    except ModuleNotFoundError:
        return pdf.extraer(origen)
    return pdf.extraer(origen) if tiene_texto else ocr.extraer(origen)


def extraer(origen: Path) -> Resultado:
    origen = Path(origen)
    sufijo = origen.suffix.lower()
    tipo = _tipo_por_contenido(origen)

    if tipo == "ole2":
        r = Resultado(
            estado="sin_extractor", salidas={}, extractor="ninguno", version="-",
            detalle={
                "razon": "binario de Office antiguo (formato OLE2); fuera "
                "de alcance de esta fase",
            },
        )
        return _con_extension_enganosa(r, sufijo, "ole2") if sufijo in EXCEL | WORD else r

    if tipo == "pdf":
        r = _pdf_o_ocr(origen)
        return r if sufijo == ".pdf" else _con_extension_enganosa(r, sufijo, "pdf")

    if tipo == "imagen":
        r = ocr.extraer(origen)
        return r if sufijo in IMAGENES else _con_extension_enganosa(r, sufijo, "imagen")

    # Contenido no decisivo (zip ambiguo entre xlsx/docx, o formato no
    # reconocido): la extensión decide sola.
    if sufijo in EXCEL:
        return excel.extraer(origen)
    if sufijo in WORD:
        return word.extraer(origen)
    if sufijo in IMAGENES:
        return ocr.extraer(origen)
    if sufijo == ".pdf":
        return _pdf_o_ocr(origen)

    return Resultado(
        estado="sin_extractor", salidas={}, extractor="ninguno", version="-",
        detalle={"razon": f"no hay extractor para '{sufijo or 'sin extension'}'"},
    )
