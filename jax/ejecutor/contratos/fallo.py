# jax/ejecutor/contratos/fallo.py
"""Un contrato que no está vivo. Lo que junta el arranque (plan 6) para negarse."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fallo:
    contrato: str
    codigo: str
    datos: tuple = ()
