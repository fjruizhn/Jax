"""Detector de estructura (esqueleto de la interfaz; la implementación viene en el commit siguiente)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Sequence

from .whitelist import Whitelist

LIMPIO = "limpio"
MARCADO = "marcado"
NO_VERIFICABLE = "no_verificable"
NO_CORRIO = "no_corrio"


@dataclass(frozen=True)
class Item:
    candidato: str
    tipo: str
    canal: str
    clase: str  # "sin_respaldo" | "heredado"
    motivo: str


@dataclass(frozen=True)
class Veredicto:
    estado: str
    items: tuple[Item, ...] = ()
    propuestos: tuple[str, ...] = ()
    hipotesis: tuple[str, ...] = ()
    no_verificables: tuple[str, ...] = ()
    fuera_de_alcance: tuple[str, ...] = ()
    motivo: str = ""

    def como_dict(self) -> dict:
        return {"estado": self.estado}


def verificar(
    salida: str,
    wl: Whitelist,
    corpus_confiable: str | Sequence[str],
    propias: Collection[str],
    *,
    previos: Sequence[str] = (),
    schema: str = "",
) -> Veredicto:
    return Veredicto(estado=LIMPIO)


def no_corrio(motivo: str) -> Veredicto:
    return Veredicto(estado=LIMPIO, motivo=motivo)
