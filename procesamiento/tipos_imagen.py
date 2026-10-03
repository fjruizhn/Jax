"""Extensiones de imagen: UNA sola lista para la compuerta y el freno de
dependencias (Jax#338 ronda 17).

Antes habia dos copias a mano -- `compuerta.IMAGENES` y el mapa de `pillow` en
`dependencias` -- y las dos se olvidaron de `.gif`: el OCR acepta una imagen
por su FIRMA (PNG, JPEG, TIFF, BMP, GIF, WebP, ver `ocr._FIRMAS_IMAGEN`), y
sin Pillow el freno dejaba entrar un lote con `sello.gif` que no se podia
procesar.

Ronda 18: aca vive tambien la decision del tipo por CONTENIDO (firmas de
imagen, `%PDF`, OLE2), la misma para el OCR, la compuerta y el freno de
dependencias, que ahora decide como la compuerta: por la firma cuando el
archivo se puede leer, por la extension como respaldo.

Modulo liviano a proposito: sin imports. `dependencias` lo usa para decidir
que tipos frena cuando FALTA Pillow, asi que no puede importar Pillow, ni
`ocr` (que exige JAX_WORKSPACE_DIR al importarse). Por eso la lista esta
escrita aca, y la verifican contra los datos `_compuerta_test.py` y
`_dependencias_test.py`: toda extension que Pillow registra para un formato
cuya firma `ocr.tipo_por_cabecera` acepta tiene que estar en esta lista.
"""
from __future__ import annotations

EXTENSIONES_IMAGEN: frozenset[str] = frozenset({
    ".png", ".apng",                            # PNG
    ".jpg", ".jpeg", ".jpe", ".jfif", ".mpo",   # JPEG (MPO: fotos de iPhone)
    ".tif", ".tiff",                            # TIFF
    ".bmp",                                     # BMP
    ".gif",                                     # GIF
    ".webp",                                    # WebP
})


FIRMA_PDF = b"%PDF"
FIRMA_OLE2 = bytes.fromhex("D0CF11E0A1B11AE1")   # binario de Office antiguo (.xls/.doc)

# S-1: tesseract interpreta una entrada que no es imagen como LISTA DE RUTAS.
# Antes de llamarlo se exigen los bytes magicos de un formato de imagen (vale
# cualquiera de ellos, no solo el de la extension).
FIRMAS_IMAGEN = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM",
    b"GIF87a", b"GIF89a",
)


def tiene_firma_de_imagen(cabecera: bytes) -> bool:
    """Bytes magicos de PNG, JPEG, TIFF, BMP, GIF o WebP (`RIFF....WEBP`)."""
    return cabecera.startswith(FIRMAS_IMAGEN) or (
        cabecera[:4] == b"RIFF" and cabecera[8:12] == b"WEBP"
    )


def tipo_por_cabecera(cabecera: bytes, sufijo: str = "") -> str | None:
    """UNICA fuente de verdad del tipo por CONTENIDO (la usan `ocr`,
    `compuerta`, `ingesta` y el freno de dependencias): `"imagen"` si hay una
    firma de imagen valida (manda: un PNG con metadata `%PDF` es una imagen);
    `"pdf"` si el contenido EMPIEZA con `%PDF`, o si `%PDF` aparece desplazado
    (hasta 1024 bytes: el estandar tolera basura antes) Y la extension es
    `.pdf` -- un .txt, .csv o .eml que solo menciona `%PDF` no es un PDF;
    `None` si el contenido no decide. Un ZIP (`PK\\x03\\x04`: xlsx/docx) nunca
    es PDF."""
    if tiene_firma_de_imagen(cabecera[:16]):
        return "imagen"
    if cabecera.startswith(b"PK\x03\x04"):
        return None
    if cabecera.startswith(FIRMA_PDF):
        return "pdf"
    if sufijo.lower() == ".pdf" and FIRMA_PDF in cabecera[:1024]:
        return "pdf"
    return None


def tipo_por_contenido(cabecera: bytes, sufijo: str = "") -> str | None:
    """La decision de la compuerta sobre los primeros bytes: `"ole2"` (binario
    de Office antiguo, sin extractor), `"pdf"`, `"imagen"`, o `None` si el
    contenido no decide (zip ambiguo entre xlsx y docx, o un formato no
    reconocido): entonces decide la extension."""
    if cabecera[:8] == FIRMA_OLE2:
        return "ole2"
    return tipo_por_cabecera(cabecera, sufijo)
