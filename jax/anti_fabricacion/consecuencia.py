"""Consecuencias (esqueleto de la interfaz; la implementación viene en el commit siguiente)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .estructura import Veredicto


class Modo(str, Enum):
    DURO = "duro"
    SOMBRA = "sombra"


@dataclass(frozen=True)
class Decision:
    modo: Modo
    fallar: bool = False
    error: str | None = None
    avisar: bool = False
    guardar: dict = field(default_factory=dict)


def aplicar(veredicto: Veredicto, modo: Modo) -> Decision:
    return Decision(modo=modo)


async def leer_modo(conectar=None) -> Modo:
    return Modo.DURO
