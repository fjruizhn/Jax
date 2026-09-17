# jax/ejecutor/contratos/gancho.py
"""Gancho PreToolUse del Ejecutor (C1 y C2). Lo corre `axioma` dentro de la jaula,
en cada llamada de herramienta, a través de `gancho.sh`.

Contrato con Claude Code (documentación oficial, 2026-09-17): exit 2 BLOQUEA y
stderr vuelve al modelo; exit 1, crash, 127 o timeout NO bloquean. Por eso este
módulo sólo devuelve 0 cuando `evaluar` dice PERMITIDO, y `gancho.sh` convierte
todo lo demás en 2.

Modos:
  gancho.py <politica>             el gancho (stdin: el evento)
  gancho.py <politica> autoprueba  ejemplos de todas las reglas; stdout JSON; 0 si sana

stderr en formato de máquina neutro: el modelo lo lee, el frontend lo rotula.
Sólo biblioteca estándar.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import datetime, timezone

from jax.ejecutor.contratos import formato, politica


def _contrato(codigo: str) -> str:
    return "c2" if codigo == politica.DESTRUCTIVO_SIN_RESPALDO else "c1"


def principal(argv, *, entrada=None, salida=None, errores=None, ahora=None) -> int:
    entrada = sys.stdin if entrada is None else entrada
    salida = sys.stdout if salida is None else salida
    errores = sys.stderr if errores is None else errores
    if not 1 <= len(argv) <= 2 or (len(argv) == 2 and argv[1] != "autoprueba"):
        errores.write(formato.campos((("contrato", "c1"), ("codigo", "argumentos_invalidos"))) + "\n")
        return 2
    evento = None
    if len(argv) == 1:
        # stdin ANTES que la política: un stdin que no cierra cae en el tope de gancho.sh.
        try:
            evento = json.loads(entrada.read())
        except ValueError:
            evento = None
        if not isinstance(evento, dict):
            errores.write(formato.campos((("contrato", "c1"), ("codigo", politica.ENTRADA_ILEGIBLE))) + "\n")
            return 2
    try:
        p = politica.cargar(argv[0], uid_de_la_cuenta=os.getuid())
    except politica.PoliticaIlegible as exc:
        errores.write(formato.campos((("contrato", "c1"), ("codigo", politica.POLITICA_ILEGIBLE),
                                      ("motivo", exc.codigo)) + tuple(exc.datos)) + "\n")
        return 2
    if evento is None:
        fallos = politica.autoprueba(p)
        salida.write(json.dumps([dataclasses.asdict(f) for f in fallos], ensure_ascii=True, separators=(",", ":")))
        return 0 if not fallos else 2
    decision = politica.evaluar(p, evento.get("tool_name"), evento.get("tool_input"),
                                ahora or datetime.now(timezone.utc))
    if decision.permitir:
        return 0
    errores.write(formato.campos((("contrato", _contrato(decision.codigo)), ("codigo", decision.codigo),
                                  ("regla", decision.regla), ("hosts", decision.hosts))) + "\n")
    return 2
