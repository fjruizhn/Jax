# jax/ejecutor/contratos/formato.py
"""Formato de máquina neutro de los contratos.

Misma gramática que `jax/ejecutor/herramientas.py`: `clave=valor` separados por un
espacio; el valor es un literal JSON ASCII. Sin idioma: lo lee una máquina (o el
modelo) y lo rotula el frontend con i18n. Sólo biblioteca estándar (se instala en
/opt/ejecutor/lib).
"""
from __future__ import annotations

import json


def valor(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=True)
    if isinstance(v, (tuple, list)):
        return "[" + ",".join(valor(x) for x in v) + "]"
    return json.dumps(repr(v), ensure_ascii=True)


def campos(pares) -> str:
    return " ".join(f"{clave}={valor(v)}" for clave, v in pares)
