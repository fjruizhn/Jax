"""Configuración de servicio que sale del entorno (/etc/jax/.env), validada.

E-21 / E-22 (2026-09-16): URLs y rutas de servicio (LAS MANOS, Ollama, el venv
de la voz, el directorio de documentos) vivían escritas en el código. Salen de
variables de entorno. Sin la variable, o con un valor que no sirve, el proceso
NO arranca con un default silencioso: levanta EntornoInvalido nombrando la
variable. Un servicio apuntado a un host equivocado no se ve hasta que falla
delante de alguien; uno que no arranca se ve en el journal.

Un solo archivo real. las_manos/config_entorno.py es symlink (Jacobs y LAS
MANOS lo importan bare); el REPL, los workers y jax-platform (vía
jax.memory.db) lo importan como jax.core.config_entorno.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit


class EntornoInvalido(RuntimeError):
    """Falta una variable de entorno de servicio o su valor no sirve."""


def _valor(nombre: str) -> str:
    valor = os.environ.get(nombre, "").strip()
    if not valor:
        raise EntornoInvalido(
            f"{nombre} no está seteada: agregala a /etc/jax/.env (sin default silencioso)."
        )
    return valor


def url_requerida(nombre: str) -> str:
    """URL http(s) con host, sin barra final."""
    valor = _valor(nombre)
    partes = urlsplit(valor)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise EntornoInvalido(f"{nombre}={valor!r} no es una URL http(s) con host.")
    return valor.rstrip("/")


def ruta_absoluta_requerida(nombre: str) -> Path:
    valor = _valor(nombre)
    ruta = Path(valor)
    if not ruta.is_absolute():
        raise EntornoInvalido(f"{nombre}={valor!r} tiene que ser una ruta absoluta.")
    return ruta
