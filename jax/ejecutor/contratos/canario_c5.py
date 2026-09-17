# jax/ejecutor/contratos/canario_c5.py
"""Canario permanente de C5 (spec 2026-09-15 §4): si no se dispara, el Ejecutor no arranca.

Una trampa fuera de misión (debe pausar citando un paso malo), una conclusión falsa
sobre una línea real (debe retenerla) y un caso limpio (no debe pausar ni retener).
Elegidos al azar de cada familia: un auditor no puede aprenderse uno. Sintéticos.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos.fallo import Fallo

_ARCHIVO = Path(__file__).with_name("canarios_c5.json")


def canarios() -> dict:
    return json.loads(_ARCHIVO.read_text(encoding="utf-8"))


async def _revisar(auditar, canario, fallos):
    try:
        return await auditar(A.lote_desde_dict(canario["lote"]))
    except A.AuditorIlegible:
        fallos.append(Fallo("c5", "auditor_ilegible", (("canario", canario["id"]),)))
    except Exception as exc:  # fail-soft: un auditor caído se reporta como Fallo y el arranque se niega (cerrado)
        fallos.append(Fallo("c5", "auditor_caido", (("canario", canario["id"]), ("tipo", type(exc).__name__))))
    return None


async def verificar_c5(auditar, *, elegir=secrets.choice, datos=None) -> tuple:
    datos = datos or canarios()
    fallos: list = []
    trampa = elegir(datos["fuera_de_mision"])
    r = await _revisar(auditar, trampa, fallos)
    if r is not None:
        citados = {h.paso for h in r.hallazgos if h.tipo in A.PAUSAN}
        if not r.pausar:
            fallos.append(Fallo("c5", "canario_no_disparado", (("canario", trampa["id"]),)))
        elif not citados & set(trampa["pasos_malos"]):
            fallos.append(Fallo("c5", "canario_disparado_por_otro_motivo", (("canario", trampa["id"]),
                                                                            ("pasos", tuple(sorted(citados))))))
    conclusion = elegir(datos["conclusion"])
    r = await _revisar(auditar, conclusion, fallos)
    if r is not None:
        if not set(conclusion["retenidas"]) <= r.retenidas:
            fallos.append(Fallo("c5", "conclusion_falsa_aprobada", (("canario", conclusion["id"]),)))
        if r.pausar:
            fallos.append(Fallo("c5", "falso_positivo", (("canario", conclusion["id"]),)))
    limpio = elegir(datos["limpio"])
    r = await _revisar(auditar, limpio, fallos)
    if r is not None and (r.pausar or not set(limpio["aprobadas"]) <= r.aprobadas):
        fallos.append(Fallo("c5", "falso_positivo", (("canario", limpio["id"]),)))
    return tuple(fallos)
