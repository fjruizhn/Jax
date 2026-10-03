"""Extensiones de imagen: UNA sola lista para la compuerta y el freno de
dependencias (Jax#338 ronda 17).

Antes habia dos copias a mano -- `compuerta.IMAGENES` y el mapa de `pillow` en
`dependencias` -- y las dos se olvidaron de `.gif`: el OCR acepta una imagen
por su FIRMA (PNG, JPEG, TIFF, BMP, GIF, WebP, ver `ocr._FIRMAS_IMAGEN`), y
sin Pillow el freno dejaba entrar un lote con `sello.gif` que no se podia
procesar.

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
