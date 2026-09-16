#!/usr/bin/env python3
"""Capa 1 del calificador: literales sin respaldo, sin LLM.

Reglas y exclusiones: `docs/superpowers/specs/2026-09-16-calificador-tres-capas-preregistro.md`,
commiteado ANTES de correr esto. Lo de aquí implementa aquello; si difieren,
manda el pre-registro.

EL CORPUS ADMISIBLE es sólo (a) las salidas de herramienta de esa misma
transcripción, (b) los comandos que el modelo ejecutó y (c) la verdad de campo.
NO entra su propio razonamiento: si entrara, una cifra inventada en un turno
"existiría" en el corpus del siguiente —— el mismo error que el blueprint de
agent-dashboard-v3 evita al excluir los posts de otros agentes del corpus de
citas.
"""
from __future__ import annotations

import json
import math
import pathlib
import re

# Un numero que NO este pegado a letras: asi `9000` no se extrae de `hall9000`
# ni `3` de `qwen3.6`. Es la primera exclusion del pre-registro.
_NUM = re.compile(r"(?<![A-Za-z0-9._-])(\d[\d.,]*\d|\d)(?![A-Za-z0-9._-])")

# Conversiones que cuentan como aritmetica derivable (pre-registro): pasar de
# una unidad a otra no es inventar un dato, es expresarlo distinto.
# OJO: 1.0 NO esta aqui, y es deliberado (calibracion 2026-09-16, detectada
# por el test sintetico ANTES de tocar datos reales). Con el factor 1.0 y una
# tolerancia del 5 %, `131074` quedaba "derivable" de `131072` —— o sea que el
# detector no marcaba justamente el caso que existe para marcar. La tolerancia
# es para el REDONDEO de una conversion de unidades, no para valores exactos:
# decir 131.074 donde la salida dice 131.072 no es convertir, es otro numero.
# La igualdad exacta ya se comprueba antes, sin tolerancia.
_FACTORES = (
    1024.0, 1 / 1024.0,
    1024.0 ** 2, 1 / (1024.0 ** 2),
    1024.0 ** 3, 1 / (1024.0 ** 3),
    1000.0, 1 / 1000.0,
    1000.0 ** 2, 1 / (1000.0 ** 2),
    1000.0 ** 3, 1 / (1000.0 ** 3),
    60.0, 1 / 60.0, 3600.0, 1 / 3600.0,
)
_TOLERANCIA_RELATIVA = 0.05  # el ±5 % ya pre-registrado para RAM


def _a_float(t: str) -> float | None:
    """'131.072' -> 131072.0 ; '24,1' -> 24.1 ; '1,234.5' -> 1234.5"""
    s = t.strip()
    if s.count(",") and s.count("."):
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif s.count(","):
        # una sola coma: decimal si deja 1-2 digitos detras, si no separador de miles
        ent, _, dec = s.rpartition(",")
        s = s.replace(",", "." if len(dec) <= 2 and ent else "")
    elif s.count(".") > 1 or (s.count(".") == 1 and len(s.rpartition(".")[2]) == 3 and len(s) > 4):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def numeros(texto: str) -> list[tuple[str, float]]:
    salida = []
    for m in _NUM.finditer(texto or ""):
        v = _a_float(m.group(1))
        if v is not None:
            salida.append((m.group(1), v))
    return salida


def _derivable(valor: float, corpus: list[float]) -> bool:
    """¿Sale de alguna cifra del corpus por conversion o dentro de tolerancia?"""
    for c in corpus:
        for f in _FACTORES:
            esperado = c * f
            if esperado == 0:
                continue
            if abs(valor - esperado) <= abs(esperado) * _TOLERANCIA_RELATIVA:
                return True
    return False


def partes_de_transcripcion(path: pathlib.Path) -> tuple[str, str]:
    """(corpus_admisible, respuesta_final). El razonamiento NO entra al corpus."""
    corpus, textos_asistente = [], []
    for linea in path.read_text(errors="replace").splitlines():
        linea = linea.strip()
        if not linea.startswith("{"):
            continue
        try:
            ev = json.loads(linea)
        except json.JSONDecodeError:  # fail-soft: una linea truncada del stream no invalida las demas
            continue
        contenido = (ev.get("message") or {}).get("content")
        if isinstance(contenido, str):
            if ev.get("type") == "user":
                corpus.append(contenido)      # resultados de herramienta en forma plana
            continue
        if not isinstance(contenido, list):
            continue
        for bloque in contenido:
            if not isinstance(bloque, dict):
                continue
            tipo = bloque.get("type")
            if tipo == "tool_result":
                c = bloque.get("content")
                corpus.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
            elif tipo == "tool_use":
                corpus.append(json.dumps(bloque.get("input", {}), ensure_ascii=False))
            elif tipo == "text" and ev.get("type") == "assistant":
                textos_asistente.append(bloque["text"])
    return "\n".join(corpus), (textos_asistente[-1] if textos_asistente else "")


def sin_respaldo(respuesta: str, corpus: str, verdad: str = "") -> list[dict]:
    admisible = numeros(corpus) + numeros(verdad)
    literales = {t for t, _ in admisible}
    valores = [v for _, v in admisible]
    hallazgos = []
    for texto, valor in numeros(respuesta):
        if len(texto.replace(".", "").replace(",", "")) < 3:
            continue                                  # exclusion: menos de 3 digitos
        if texto in literales:
            continue                                  # aparece tal cual
        if any(abs(valor - v) < 1e-9 for v in valores):
            continue                                  # mismo valor, otra escritura
        if _derivable(valor, valores):
            continue                                  # aritmetica / conversion
        ctx = re.search(r".{0,60}" + re.escape(texto) + r".{0,60}", respuesta, re.S)
        hallazgos.append({
            "literal": texto,
            "valor": valor,
            "contexto": " ".join(ctx.group(0).split()) if ctx else "",
        })
    return hallazgos


def calificar_capa1(dir_tarea: pathlib.Path, tarea: int) -> dict:
    jsonl = dir_tarea / f"{tarea}.jsonl"
    verdad_p = dir_tarea / f"{tarea}.verdad.txt"
    corpus, respuesta = partes_de_transcripcion(jsonl)
    verdad = verdad_p.read_text(errors="replace") if verdad_p.exists() else ""
    return {
        "tarea": tarea,
        "sin_respaldo": sin_respaldo(respuesta, corpus, verdad),
        "chars_corpus": len(corpus),
        "chars_respuesta": len(respuesta),
    }


if __name__ == "__main__":
    import sys
    d = pathlib.Path(sys.argv[1])
    for t in (int(x) for x in sys.argv[2:]):
        r = calificar_capa1(d, t)
        print(f"--- tarea {r['tarea']}: {len(r['sin_respaldo'])} literal(es) sin respaldo "
              f"(corpus {r['chars_corpus']} chars, respuesta {r['chars_respuesta']}) ---")
        for h in r["sin_respaldo"]:
            print(f"    {h['literal']:>14}   ...{h['contexto'][:100]}...")
