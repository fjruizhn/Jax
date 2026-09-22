"""Reglas del descarte de pipelines detenidos (spec 2026-09-22-descartar-pipelines).

Puras, sin I/O: qué estado puede pasar a cuál. La escritura (compare-and-set)
vive en store.pipeline_transicion_descarte; quién puede pedirla, en
jax-platform. Ninguna transición borra filas.
"""
from __future__ import annotations

from jacobs.models import PipelineStatus

TRANSICIONES: dict[str, frozenset[PipelineStatus]] = {
    "discard": frozenset({PipelineStatus.aborted, PipelineStatus.expired}),
    "recover": frozenset({PipelineStatus.discarded}),
    "hide": frozenset({PipelineStatus.discarded}),
    "restore": frozenset({PipelineStatus.hidden}),
}

#: A qué puede volver un recuperado: exactamente lo que se podía descartar.
_PREVIOS_VALIDOS = TRANSICIONES["discard"]


class EstadoPrevioInvalido(ValueError):
    """`status_previo` no es un estado del que se pueda haber descartado.
    Fail-closed: no se adivina a qué volver."""


def destino_de(accion: str, status_previo: str | None) -> PipelineStatus:
    if accion == "discard":
        return PipelineStatus.discarded
    if accion == "hide":
        return PipelineStatus.hidden
    if accion == "restore":
        return PipelineStatus.discarded
    if accion == "recover":
        try:
            previo = PipelineStatus(status_previo)
        except ValueError:
            raise EstadoPrevioInvalido(repr(status_previo)) from None
        if previo not in _PREVIOS_VALIDOS:
            raise EstadoPrevioInvalido(repr(status_previo))
        return previo
    raise KeyError(accion)
