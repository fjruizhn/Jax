"""Jacobs -- el veredicto del árbitro (spec 2026-09-18-arbitro-devuelve-design §3.1).

**El defecto que esto cierra, con el caso real.** El pipeline
`e570ac1c-8cae-4423-b397-d354f30b4328` corrió con árbitro por primera vez: el
paso 4 (`ada`) propuso sintaxis PostgreSQL (`BIGSERIAL`, `TIMESTAMPTZ`,
`JSONB`) sobre un destino MariaDB, y el árbitro (`thot`) lo detectó y lo dijo
bien en 15 KB de prosa citando `[paso 4]`. Ahí se acabó: la prosa es algo que
un humano entiende y el código no puede accionar. El pipeline terminó
"completado" con el error escrito y sin corregir.

Este módulo agrega, ADEMÁS de la prosa (que sigue existiendo tal cual --
nadie le pide al árbitro que deje de explicarse), un veredicto con
estructura que `jacobs/devolucion.py` pueda leer sin adivinar.

**Fallo cerrado (spec §3.1, criterio de éxito §5.2).** Si el bloque no está,
o el JSON no parsea, o le falta una clave, o la cita no coincide con el
paso que dice devolver, `parsear_veredicto` devuelve `None` -- nunca lanza,
nunca adivina a qué paso se refería el árbitro. El pipeline sigue el camino
de hoy (termina con la crítica en prosa) y `jacobs/devolucion.py` deja un
evento `VEREDICTO_SIN_ESTRUCTURA`: "quedó registrado que el árbitro quiso
devolver y no se pudo parsear" (spec §3.1).

**Formato pedido al árbitro** (ver `jacobs/plan.py::PlanBuilder.PROMPT_ARBITRO`),
al FINAL de su respuesta, en un bloque de código con lenguaje "veredicto":

    ```veredicto
    {"decision": "devolver", "paso": 4, "motivo": "usaste BIGSERIAL y
    TIMESTAMPTZ; el destino es MariaDB", "cita": "[paso 4]"}
    ```

o, si no hay nada que devolver:

    ```veredicto
    {"decision": "aprobar"}
    ```

Se toma el ÚLTIMO bloque `veredicto` del texto, no el primero: si el árbitro
explica el formato antes de usarlo, esa explicación no se confunde con su
decisión real.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("jacobs.veredicto")

DECISION_APROBAR = "aprobar"
DECISION_DEVOLVER = "devolver"
DECISIONES_VALIDAS = frozenset({DECISION_APROBAR, DECISION_DEVOLVER})

_BLOQUE_VEREDICTO = re.compile(r"```veredicto\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class VeredictoArbitro:
    decision: str
    paso: int | None = None
    motivo: str | None = None
    cita: str | None = None


def _cita_de(paso: int) -> str:
    return f"[paso {paso}]"


def parsear_veredicto(texto: str, n_pasos_totales: int) -> VeredictoArbitro | None:
    """El veredicto accionable del árbitro, o None si no hay ninguno (fallo
    cerrado -- nunca lanza).

    `n_pasos_totales` es `len(pipeline.plan)`, INCLUIDO el árbitro (siempre
    el último, `PlanBuilder._con_arbitro`): un `paso` válido es un índice de
    PRODUCTOR, `0 <= paso < n_pasos_totales - 1` -- el árbitro no puede
    devolverse a sí mismo."""
    if not texto:
        return None
    matches = _BLOQUE_VEREDICTO.findall(texto)
    if not matches:
        logger.warning("veredicto: sin bloque ```veredicto``` en la salida del árbitro")
        return None
    crudo = matches[-1]
    try:
        datos = json.loads(crudo)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("veredicto: bloque presente pero JSON inválido (%s)", exc)
        return None
    if not isinstance(datos, dict):
        logger.warning("veredicto: el bloque no es un objeto JSON (%r)", type(datos).__name__)
        return None

    decision = datos.get("decision")
    if decision not in DECISIONES_VALIDAS:
        logger.warning("veredicto: 'decision' inválida o ausente: %r", decision)
        return None

    if decision == DECISION_APROBAR:
        return VeredictoArbitro(decision=DECISION_APROBAR)

    paso = datos.get("paso")
    # bool es subclase de int en Python: True/False no son un índice de
    # paso, aunque `isinstance(True, int)` dé True.
    if not isinstance(paso, int) or isinstance(paso, bool):
        logger.warning("veredicto: 'paso' ausente o no es un entero: %r", paso)
        return None
    if not (0 <= paso < n_pasos_totales - 1):
        logger.warning(
            "veredicto: 'paso'=%s fuera de rango (0..%d)", paso, n_pasos_totales - 2)
        return None

    motivo = datos.get("motivo")
    if not isinstance(motivo, str) or not motivo.strip():
        logger.warning("veredicto: 'motivo' ausente o vacío")
        return None

    cita = datos.get("cita")
    if cita != _cita_de(paso):
        logger.warning("veredicto: 'cita'=%r no coincide con el paso %d", cita, paso)
        return None

    return VeredictoArbitro(
        decision=DECISION_DEVOLVER, paso=paso, motivo=motivo.strip(), cita=cita)
