#!/usr/bin/env python3
"""Mide la semilla de capability.min_output_tokens (spec 2026-09-17 §4.4).

SOLO LEE. Máximo de tokens de salida de corridas COMPLETADAS por capability,
redondeado hacia arriba a múltiplo de 1024, de dos fuentes:
  1. Motor Registry: las_manos/logs/motor_jobs.jsonl, `_usage.completion_tokens`
     del ÚLTIMO registro por job_id (estado final) cuando ese estado final es
     `completed` -- una línea `completed` de un job cuyo estado final es OTRO
     no cuenta (mismo criterio que jacobs/reaper.py::_load_terminal_motor_jobs
     y el script de plan P). Incluye razonamiento: es lo que el tope tiene
     que dejar pasar.
  2. Transportes HTTP directos de Jacobs: axioma_usage.tokens_out
     (request_type='pipeline') unido a jacobs_steps completados por faceta y
     ventana de tiempo. Una fila de axioma_usage cuya ventana calza con pasos
     de MÁS de una capability es AMBIGUA (no se sabe cuál la generó) y se
     excluye, no se le atribuye a ninguna -- mismo criterio que P.

Una capability sin corridas medibles queda en 0 y se declara.

Consulta de UNA vez, fuera del camino caliente: su EXPLAIN se registra al
correrla (Task 13, Step 6 del plan) y un scan se acepta por ser un script
puntual.

REFERENCIA: plan P (jax-platform), Task 2, commit d79b0f9 -- Task 13 del
plan J copia su medición sin repetirla (Ruling R2 del ledger de este plan).
Este script sigue el MISMO método de P, no una aproximación propia:

  - COLLATE: axioma_usage.facet quedó en utf8mb4_uca1400_ai_ci y
    jacobs_steps.facet en utf8mb4_unicode_ci (esquemas creados en momentos
    distintos) -- el JOIN por facet falla contra producción con 1267
    "Illegal mix of collations" sin `COLLATE utf8mb4_uca1400_ai_ci`. Las
    claves de faceta son ASCII en minúscula, así que la igualdad da lo mismo
    con cualquiera de las dos collations.
  - Ventana: `FLOOR(s.started_at) AND CEIL(s.finished_at) + HOLGURA_S`,
    holgura SÓLO del lado derecho (created_at es TIMESTAMP, segundos
    enteros; started_at/finished_at son DOUBLE). Ninguna resta del lado
    izquierdo: eso ensancharía la ventana más allá de lo que P midió y
    subiría la probabilidad de fila ambigua sin necesidad.
  - Exclusión de filas ambiguas: `maximos_http()` (abajo) NO agrupa con
    `MAX()+GROUP BY` en SQL -- trae `(id, capability, tokens_out)` fila por
    fila y excluye en Python las filas cuyo `id` de axioma_usage cae en la
    ventana de pasos de MÁS de una capability. Sin esta exclusión, dos pasos
    de la misma faceta que se solapan (p.ej. un `reconcile` que termina justo
    antes de que arranque un `file_write`) pueden compartir la MISMA fila de
    uso, y un `MAX()+GROUP BY` se la atribuye a las dos -- inflando la
    capability equivocada, no la que la generó.
  - Sesión de sólo lectura real: `SET SESSION TRANSACTION READ ONLY` +
    `START TRANSACTION READ ONLY` antes de consultar, `ROLLBACK` al final
    (nunca commit) -- el propio MariaDB rechaza cualquier escritura dentro
    de esa sesión, no sólo la ausencia de sentencias de escritura en el
    código.

Uso (con GO, contra producción, solo lectura):
  set -a; source /etc/jax/.env; set +a
  PYTHONPATH=.:las_manos python scripts/medir_min_output_tokens.py --jobs las_manos/logs/motor_jobs.jsonl

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MULTIPLO = 1024
HOLGURA_S = 5  # axioma_usage.created_at es TIMESTAMP (segundos enteros)

_SQL_HTTP_DIRECTO = (
    "SELECT u.id, s.capability, u.tokens_out FROM jacobs_steps s "
    "JOIN axioma_usage u ON u.facet = s.facet COLLATE utf8mb4_uca1400_ai_ci AND u.request_type = 'pipeline' "
    " AND UNIX_TIMESTAMP(u.created_at) BETWEEN FLOOR(s.started_at) AND CEIL(s.finished_at) + %s "
    "WHERE s.status = 'completed' AND s.started_at IS NOT NULL AND s.finished_at IS NOT NULL"
)


def redondear(n: int) -> int:
    if n <= 0:
        return 0
    return math.ceil(n / MULTIPLO) * MULTIPLO


def maximos_de_jobs(lineas: Iterable[str]) -> tuple[dict[str, int], int]:
    """Máximo por capability de jobs COMPLETADOS. Antes de medir, se queda
    con el ÚLTIMO registro por job_id (estado final): una línea `completed`
    de un job cuyo estado final es OTRO no cuenta (mismo criterio que
    jacobs/reaper.py::_load_terminal_motor_jobs y el `corridas_de_motor` del
    script de plan P, jax-platform Task 2, commit d79b0f9). Una línea sin
    `job_id` no se puede deduplicar contra nada: cuenta por sí misma."""
    ultimo: dict[object, dict] = {}
    rotas = 0
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        try:
            job = json.loads(linea)
        except json.JSONDecodeError:
            rotas += 1
            continue
        clave = job.get("job_id")
        if clave is None:
            clave = object()  # sin job_id: no deduplica, cuenta por sí misma
        ultimo[clave] = job

    maximos: dict[str, int] = {}
    for job in ultimo.values():
        if job.get("status") != "completed":
            continue
        tokens = (job.get("_usage") or {}).get("completion_tokens")
        capability = job.get("capability")
        if not capability or not isinstance(tokens, int):
            continue
        maximos[capability] = max(maximos.get(capability, 0), tokens)
    return maximos, rotas


def maximos_http(filas: Iterable[tuple[Any, str, int]]) -> tuple[dict[str, int], list[Any]]:
    """De las filas `(id, capability, tokens_out)` de `_SQL_HTTP_DIRECTO`:
    excluye las filas cuyo `id` de axioma_usage cae en la ventana de pasos de
    MÁS de una capability (ambiguas -- no se sabe cuál las generó) y las de
    `tokens_out` en 0 (nada que medir). El máximo por capability es el mayor
    de las filas que quedan. Mismo criterio que el script de plan P
    (jax-platform Task 2, commit d79b0f9, post-procesamiento de
    `SQL_PASOS_HTTP`)."""
    caps_por_fila: dict[Any, set[str]] = {}
    tokens_por_fila: dict[Any, int] = {}
    for fila_id, capability, tokens_out in filas:
        caps_por_fila.setdefault(fila_id, set()).add(capability)
        tokens_por_fila[fila_id] = tokens_out

    ambiguas = sorted(i for i, caps in caps_por_fila.items() if len(caps) > 1)
    maximos: dict[str, int] = {}
    for fila_id, caps in caps_por_fila.items():
        if len(caps) == 1 and tokens_por_fila[fila_id]:
            capability = next(iter(caps))
            maximos[capability] = max(maximos.get(capability, 0), tokens_por_fila[fila_id])
    return maximos, ambiguas


def combinar(*fuentes: dict[str, int]) -> dict[str, int]:
    salida: dict[str, int] = {}
    for fuente in fuentes:
        for capability, valor in fuente.items():
            salida[capability] = max(salida.get(capability, 0), valor)
    return salida


async def _maximos_http_directo() -> tuple[dict[str, int], list[Any], list[str]]:
    from jacobs import store
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SET SESSION TRANSACTION READ ONLY")
            await cur.execute("START TRANSACTION READ ONLY")
            # ROLLBACK en su propio finally (ronda de arreglo 2, observación
            # 3): si cualquiera de las SELECT de acá abajo revienta, la
            # sesión de sólo lectura se cierra igual -- no queda colgada en
            # el servidor. conn.close() sigue viniendo DESPUÉS (finally
            # exterior); la excepción original se relanza sin tragarse.
            try:
                await cur.execute(_SQL_HTTP_DIRECTO, (HOLGURA_S,))
                filas = await cur.fetchall()
                await cur.execute("SELECT `key` FROM capability ORDER BY `key`")
                capabilities = [r[0] for r in await cur.fetchall()]
            finally:
                await cur.execute("ROLLBACK")
    finally:
        conn.close()
    maximos, ambiguas = maximos_http(filas)
    return maximos, ambiguas, capabilities


def main() -> int:
    parser = argparse.ArgumentParser(description="Mide la semilla de capability.min_output_tokens (solo lee).")
    parser.add_argument("--jobs", required=True, type=Path, help="ruta a motor_jobs.jsonl")
    args = parser.parse_args()

    with args.jobs.open(encoding="utf-8") as fh:
        de_jobs, rotas = maximos_de_jobs(fh)
    de_http, ambiguas, capabilities = asyncio.run(_maximos_http_directo())
    medidos = combinar(de_jobs, de_http)

    print(f"motor_jobs.jsonl: {len(de_jobs)} capabilities medidas, {rotas} líneas ilegibles")
    print(f"axioma_usage + jacobs_steps: {len(de_http)} capabilities medidas")
    print(f"filas de uso ambiguas excluidas: {ambiguas}")
    print("| capability | max tokens medidos | min_output_tokens | fuente |")
    print("|---|---|---|---|")
    for cap in capabilities:
        fuentes = [n for n, d in (("motor_jobs", de_jobs), ("http_directo", de_http)) if cap in d]
        fuente = ", ".join(fuentes) or "sin corridas medibles"
        print(f"| {cap} | {medidos.get(cap, 0)} | {redondear(medidos.get(cap, 0))} | {fuente} |")
    print()
    for cap in capabilities:
        print(f"UPDATE capability SET min_output_tokens={redondear(medidos.get(cap, 0))} WHERE `key`='{cap}';")
    return 0


if __name__ == "__main__":
    sys.exit(main())
