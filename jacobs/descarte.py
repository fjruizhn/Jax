"""Reglas del descarte de pipelines detenidos (spec 2026-09-22-descartar-pipelines).

Puras, sin I/O: qué estado puede pasar a cuál. La escritura (compare-and-set)
vive en store.pipeline_transicion_descarte; quién puede pedirla, en
jax-platform. Ninguna transición borra filas.

Fix round 1 (2026-09-22, Ruling 7, I-1): `validar_transicion` es la regla que
`store.pipeline_transicion_descarte` consulta ANTES de tocar la base. Sin
ella, un llamador que mandara `discard` desde `running` liberaba el cupo de
un pipeline que sigue ejecutando y lo dejaba huérfano para siempre -- el
compare-and-set por `epoca`/`status` no alcanza porque valida CONTRA QUÉ
estaba el pipeline, no si esa acción tenía permitido partir de ahí.
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


class TransicionDescarteInvalida(ValueError):
    """La transición pedida no está permitida (fix round 1, Ruling 7):
    `desde` no es un estado del que se pueda hacer esta `accion`, o `a` no
    es el destino correcto. Fail-closed -- se levanta ANTES de que
    `store.pipeline_transicion_descarte` toque la base."""


def validar_transicion(accion: str, desde: PipelineStatus, a: PipelineStatus) -> None:
    """Valida una transición ANTES del compare-and-set: quien escribe no
    puede confiar en que el llamador mandó `desde`/`a` correctos.

    `discard`/`hide`/`restore` tienen un único destino fijo: `a` tiene que
    ser exactamente `destino_de(accion, None)`.

    `recover` no tiene un destino fijo -- acá sólo se exige que `a` sea UNO
    de los estados de los que se puede haber descartado (mismo conjunto que
    `TRANSICIONES["discard"]`). Que `a` coincida con el `status_previo` REAL
    de esa fila (no cualquier estado previo válido en general) lo garantiza
    el propio `WHERE` del compare-and-set en `store.py`
    (`status_previo=%s` con `a.value`), no esta función: acá no hay I/O, así
    que no hay forma de conocer la fila."""
    permitidos = TRANSICIONES[accion]  # KeyError si accion no existe: error de contrato
    if desde not in permitidos:
        raise TransicionDescarteInvalida(
            f"{accion}: no se puede transicionar desde {desde.value!r} "
            f"(permitidos: {sorted(s.value for s in permitidos)})")
    if accion == "recover":
        if a not in _PREVIOS_VALIDOS:
            raise TransicionDescarteInvalida(
                f"recover: destino {a.value!r} no es un estado previo válido "
                f"(válidos: {sorted(s.value for s in _PREVIOS_VALIDOS)})")
        return
    esperado = destino_de(accion, None)
    if a is not esperado:
        raise TransicionDescarteInvalida(
            f"{accion}: destino {a.value!r} no coincide con el fijo {esperado.value!r}")
