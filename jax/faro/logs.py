"""Logging del servicio: lo configura EL SERVICIO, antes de construir el primer `MCPServer`.

`MCPServer.__init__` llama a `logging.basicConfig(handlers=[RichHandler(...)])` del SDK: si el proceso
llega ahi sin logging configurado, el SDK decide formato y destino por el servicio. `asegurar_logging`
se ejecuta antes (en `ServidorPuerto.__aenter__`) y deja un handler propio en la raiz; con la raiz
ya configurada, el `basicConfig` del SDK no hace nada. No toca una configuracion que ya exista.
"""
from __future__ import annotations

import logging
import sys

FORMATO = "%(asctime)s %(levelname)s %(name)s %(message)s"


def asegurar_logging(nivel: int = logging.INFO) -> None:
    raiz = logging.getLogger()
    if raiz.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(FORMATO))
    raiz.addHandler(h)
    raiz.setLevel(nivel)
