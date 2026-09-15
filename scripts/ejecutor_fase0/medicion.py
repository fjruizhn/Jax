"""Funciones puras de la Fase 0 del Ejecutor.

Spec: docs/superpowers/specs/2026-09-15-ejecutor-design.md §8.
Todo lo de este módulo es determinista y sin red: lo corre tests-puros en
CI. Los CLIs de al lado (contexto.py, examen.py, ...) hacen la E/S.
"""
from __future__ import annotations

import json
import math
import re

_COLS = re.compile(r"\s{2,}")


def parse_ollama_ps(texto: str) -> list[dict[str, str]]:
    """Filas de `ollama ps` como dicts por encabezado.

    Las columnas se separan con dos o más espacios; un valor puede tener un
    espacio simple adentro ("23 GB", "100% GPU", "37 seconds from now").
    """
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    if not lineas:
        return []
    cabecera = _COLS.split(lineas[0])
    return [dict(zip(cabecera, _COLS.split(l))) for l in lineas[1:]]


def tok_por_segundo(eval_count: int, eval_duration_ns: int) -> float:
    """tok/s de GENERACIÓN, con el reloj de Ollama (excluye la cola).

    Nunca con el elapsed de la llamada HTTP: con requests encoladas ese
    reloj incluye la espera (2026-08-25-gpu-concurrency-resultado.md).
    """
    if eval_duration_ns <= 0:
        raise ValueError("eval_duration_ns debe ser > 0")
    return eval_count / (eval_duration_ns / 1e9)


def evaluar_contexto(fila_ps, carga_s, toks, toks_base, max_carga_s=300.0, min_ratio=0.5) -> list[str]:
    """Umbral U1 (pre-registrado). Motivos por los que NO cabe; [] = cabe."""
    if toks_base <= 0:
        raise ValueError("toks_base debe ser > 0")
    motivos = []
    proc = (fila_ps or {}).get("PROCESSOR", "")
    if proc != "100% GPU":
        motivos.append(f"PROCESSOR={proc!r}, se exige '100% GPU'")
    if carga_s > max_carga_s:
        motivos.append(f"carga {carga_s:.1f}s > {max_carga_s:.0f}s")
    if toks < min_ratio * toks_base:
        motivos.append(f"{toks:.1f} tok/s < {min_ratio:.0%} de {toks_base:.1f}")
    return motivos


def elegir_contexto(resultados: list[dict]) -> int | None:
    """El num_ctx más grande sin motivos. None si ninguno cabe."""
    validos = [r["num_ctx"] for r in resultados if not r["motivos"]]
    return max(validos) if validos else None


def arranque_viable(tokens_arranque: int, num_ctx: int, fraccion_max: float = 0.40) -> bool:
    """Umbral U2: el arranque del arnés no se come más del 40 % del contexto."""
    if num_ctx <= 0:
        raise ValueError("num_ctx debe ser > 0")
    return tokens_arranque <= fraccion_max * num_ctx


def percentil(valores: list[float], p: float) -> float:
    """Percentil por rango más cercano, sin interpolar."""
    if not valores:
        raise ValueError("sin valores")
    if not 0 < p <= 100:
        raise ValueError("p fuera de (0, 100]")
    orden = sorted(valores)
    return orden[max(1, math.ceil(p / 100 * len(orden))) - 1]


def clasificar_borrado_r2(http_status: int, cuerpo: str) -> str:
    """Umbral U6. Un 403 por falta de permiso NO prueba candado (lección 10)."""
    if 200 <= http_status < 300:
        return "sin_candado"
    texto = cuerpo.lower()
    if any(p in texto for p in ("lock", "retention", "worm")):
        return "candado"
    return "inconcluso"


def eventos_stream_json(texto: str) -> list[dict]:
    eventos = []
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea.startswith("{"):
            continue
        try:
            eventos.append(json.loads(linea))
        except json.JSONDecodeError:  # fail-soft: una línea truncada del stream no invalida las demás
            continue
    return eventos


def _texto(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "\n".join(b.get("text", "") for b in contenido if isinstance(b, dict))
    return ""


def salidas_de_herramientas(eventos: list[dict]) -> list[str]:
    salidas = []
    for ev in eventos:
        if ev.get("type") != "user":
            continue
        contenido = (ev.get("message") or {}).get("content")
        if not isinstance(contenido, list):
            continue
        for bloque in contenido:
            if isinstance(bloque, dict) and bloque.get("type") == "tool_result":
                salidas.append(_texto(bloque.get("content")))
    return salidas


def respuesta_final(eventos: list[dict]) -> str | None:
    for ev in reversed(eventos):
        if ev.get("type") == "result":
            return ev.get("result")
    return None


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def respaldada(valor: str, salidas: list[str]) -> bool:
    """¿El valor afirmado aparece literal en alguna salida de herramienta?

    Ayuda del calificador; no decide sola: un valor derivado por aritmética
    no aparece literal y no es inventado (rúbrica U3).
    """
    v = _norm(valor)
    return bool(v) and any(v in _norm(s) for s in salidas)


def extraer_secciones(markdown: str, titulos: list[str]) -> str:
    """Secciones `## <título…>` cuyo título empieza por alguno de `titulos`."""
    bloques = re.split(r"(?m)^(?=## )", markdown)
    elegidos = [b for b in bloques if any(b.startswith(f"## {t}") for t in titulos)]
    return "\n".join(b.rstrip() + "\n" for b in elegidos)
