"""Barrido de la entrega (spec 2026-09-28 §3.3.2): secretos en lo agregado y tope por archivo."""
from __future__ import annotations

import re
from pathlib import Path

from jax.ejecutor.codigo.diff import Cambio
from jax.ejecutor.codigo.reglas_diff import Violacion

_SECRETOS = re.compile(r"(github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}|sk-ant-[a-z0-9]+-[A-Za-z0-9_-]{20,}|"
                       r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def barrer(cambios: tuple[Cambio, ...], clon: Path, *, tope_bytes: int) -> tuple[Violacion, ...]:
    v: list[Violacion] = []
    for c in cambios:
        if c.estado == "D":
            continue
        for linea in c.agregadas:
            if _SECRETOS.search(linea):
                v.append(Violacion("secretos", c.ruta, "patrón de credencial en una línea agregada"))
                break
        ruta = clon / c.ruta
        if ruta.is_file() and ruta.stat().st_size > tope_bytes:
            v.append(Violacion("tamano", c.ruta, f"{ruta.stat().st_size} > {tope_bytes} bytes"))
    return tuple(v)
