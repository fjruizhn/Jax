"""Barrido de la entrega (spec 2026-09-28 §3.3.2): secretos en lo agregado y tope por archivo.

Dos funciones PURAS. Quien las llama mide en el ESPEJO (`entrega`), nunca en el clon de Qwen:
- `lineas_con_secretos` recibe cambios ya parseados -- los del diff neto y los de CADA commit
  intermedio (un secreto agregado y luego quitado dentro de la rama llegaría a GitHub en el
  historial).
- `tamanos_excedidos` recibe tamaños ya medidos con `git cat-file -s` sobre los blobs de la rama
  de la misión: un `stat` en el árbol del clon mide lo que Qwen dejó en disco, no lo que commiteó,
  y `is_file()` sigue enlaces."""
from __future__ import annotations

import re
from collections.abc import Mapping

from jax.ejecutor.codigo.diff import Cambio
from jax.ejecutor.codigo.reglas_diff import Violacion

_SECRETOS = re.compile(r"(github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}|sk-ant-[a-z0-9]+-[A-Za-z0-9_-]{20,}|"
                       r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def lineas_con_secretos(cambios: tuple[Cambio, ...]) -> tuple[Violacion, ...]:
    """Una violación por archivo con al menos una línea agregada que parece una credencial. El
    detalle no lleva la línea: va a la bitácora."""
    v: list[Violacion] = []
    for c in cambios:
        if any(_SECRETOS.search(linea) for linea in c.agregadas):
            v.append(Violacion("secretos", c.ruta, "patrón de credencial en una línea agregada"))
    return tuple(v)


def tamanos_excedidos(tamanos: Mapping[str, int], *, tope_bytes: int) -> tuple[Violacion, ...]:
    return tuple(Violacion("tamano", ruta, f"{n} > {tope_bytes} bytes")
                 for ruta, n in sorted(tamanos.items()) if n > tope_bytes)


def tapar_secretos(texto: str) -> str:
    """Para lo que sale hacia la bitácora (rutas y detalles de violaciones)."""
    return _SECRETOS.sub("***", texto)
