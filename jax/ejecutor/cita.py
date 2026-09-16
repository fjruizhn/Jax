"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.3.

Puro a propósito: sin red, sin E/S, sin reloj, sólo biblioteca estándar.
Verificar una cita es una BÚSQUEDA DE SUBCADENA, no un juicio sobre la
verdad -- por eso no hay nada que calibrar y no existe el falso positivo
por umbral mal puesto.

Garantiza PROCEDENCIA, no CORRECCIÓN: una afirmación puede citar una línea
real y aun así concluir mal a partir de ella (riesgo 2 del spec).
"""
from __future__ import annotations

from dataclasses import dataclass

RESPALDADA = "respaldada"
SIN_RESPALDO = "sin_respaldo"
FUENTE_TRUNCADA = "fuente_truncada"
FUENTE_INEXISTENTE = "fuente_inexistente"


@dataclass(frozen=True)
class Captura:
    maquina: str
    comando: str
    salida: str
    stderr: str
    truncada: bool


@dataclass(frozen=True)
class Afirmacion:
    maquina: str
    texto: str
    comando: str
    linea: str


@dataclass(frozen=True)
class Veredicto:
    estado: str
    motivo: str


def normalizar(linea: str) -> str:
    """Recorta los laterales y colapsa espacios internos. NADA MÁS.

    No toca mayúsculas, ni puntuación, ni números: `131.074` no puede
    coincidir con `131.072` (invención real de U3, tarea 3) y `Docker` no
    puede coincidir con `docker`.
    """
    return " ".join(linea.split())


def verificar(afirmacion: Afirmacion, capturas) -> Veredicto:
    """¿La línea citada está literal en la salida de ese comando, en esa máquina?

    Una captura respalda sólo si coinciden MÁQUINA y COMANDO: un `free -h` de
    otra máquina no dice nada de ésta. La línea puede estar en stdout o en
    stderr, cada flujo por separado.

    Si el comando se corrió más de una vez, respalda cualquier captura
    COMPLETA que tenga la línea. Una captura truncada no respalda nada.
    """
    aguja = normalizar(afirmacion.linea)
    # Una cita vacía no cita nada: `""` es igual a cualquier línea en blanco
    # de la salida y dejaría pasar cualquier afirmación inventada.
    if not aguja:
        return Veredicto(SIN_RESPALDO, "la afirmación no cita ninguna línea")
    # Lo mismo con la máquina: vacía no identifica nada, y `"" == ""` dejaría
    # que una captura sin procedencia respalde una afirmación sin procedencia.
    if not afirmacion.maquina.strip():
        return Veredicto(SIN_RESPALDO, "la afirmación no dice de qué máquina viene")
    se_corrio = False
    alguna_truncada = False
    for captura in capturas:
        if (captura.maquina, captura.comando) != (afirmacion.maquina, afirmacion.comando):
            continue
        se_corrio = True
        # El truncado se mira ANTES que el contenido: si la salida vino
        # cortada, dar por bueno lo que sí llegó es exactamente el error de
        # la tarea 9 (2 KB leídos de 85,9 KB).
        if captura.truncada:
            alguna_truncada = True
            continue
        # stdout y stderr se recorren POR SEPARADO. Pegarlos en un solo texto
        # fabrica, si stdout no termina en salto de línea, una línea que la
        # máquina nunca imprimió en ningún flujo.
        for flujo in (captura.salida, captura.stderr):
            for linea in flujo.splitlines():
                if normalizar(linea) == aguja:
                    return Veredicto(RESPALDADA, "")
    if alguna_truncada:
        return Veredicto(FUENTE_TRUNCADA,
                         f"la salida de {afirmacion.comando!r} en "
                         f"{afirmacion.maquina!r} vino truncada")
    if se_corrio:
        return Veredicto(SIN_RESPALDO,
                         f"la línea citada no está en la salida de "
                         f"{afirmacion.comando!r} en {afirmacion.maquina!r}")
    return Veredicto(FUENTE_INEXISTENTE,
                     f"no se corrió el comando {afirmacion.comando!r} "
                     f"en {afirmacion.maquina!r}")
