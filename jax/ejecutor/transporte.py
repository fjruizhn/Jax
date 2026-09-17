"""Transporte del Ejecutor (Fase 2): lo que no cita, no sale.

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §2.1, §2.3,
§5 y V4. DECISIÓN §2.0: el Ejecutor no escribe prosa; lo que ve la persona lo
arma `cita.presentar` sobre las afirmaciones respaldadas.

- Las salidas crudas se entregan SIEMPRE (piso §2.3), también truncadas y
  también sin ninguna afirmación.
- Sólo sale una afirmación con veredicto EXACTAMENTE `respaldada` (lista de
  permitidos: un estado desconocido descarta). Las demás van a `descartadas`
  con su estado y su motivo.
- FAIL-CLOSED (V4): si verificar falla -- o no se pueden leer las
  afirmaciones -- no sale NINGUNA, ni las que ya habían pasado. Un verificador
  que se cayó a mitad de turno no deja sano nada de ese turno.

Puro: sin red, sin E/S.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from jax.ejecutor import cita

VERIFICADOR_CAIDO = "verificador_caido"


@dataclass(frozen=True)
class Descartada:
    afirmacion: object
    estado: str
    # El `Motivo` del veredicto (código y datos), o `VERIFICADOR_CAIDO` con el
    # tipo del error. Sin prosa: la frase la pone el frontend.
    motivo: cita.Motivo


@dataclass(frozen=True)
class Entrega:
    crudas: tuple
    respaldadas: tuple
    descartadas: tuple


def entregar(afirmaciones, capturas) -> Entrega:
    crudas = tuple(capturas)
    leidas = []
    respaldadas = []
    descartadas = []
    try:
        for afirmacion in afirmaciones:
            leidas.append(afirmacion)
            veredicto = cita.verificar(afirmacion, crudas)
            if veredicto.estado == cita.RESPALDADA:
                # Lo que se entrega lleva la línea como la imprimió la máquina,
                # no como la copió el modelo (difieren sólo en espacios).
                respaldadas.append(replace(afirmacion, linea=veredicto.linea_capturada))
            else:
                descartadas.append(Descartada(afirmacion, veredicto.estado, veredicto.motivo))
    except Exception as error:  # fail-soft: el turno entrega las crudas; fail-CLOSED para las afirmaciones: ninguna sale si el verificador o la lectura fallan
        # Sólo el TIPO del error: el mensaje puede traer cualquier cosa.
        motivo = cita.Motivo(VERIFICADOR_CAIDO, (("error", type(error).__name__),))
        return Entrega(crudas, (), tuple(Descartada(a, VERIFICADOR_CAIDO, motivo) for a in leidas))
    return Entrega(crudas, tuple(respaldadas), tuple(descartadas))


def presentar(entrega: Entrega) -> list[cita.Presentacion]:
    """Lo que ve la persona de cada afirmación respaldada, armado por el sistema.
    Estructura sin rótulos: los pone el frontend (ver `cita.presentar`)."""
    return [cita.presentar(a) for a in entrega.respaldadas]
