#!/usr/bin/env python3
"""Mide la semilla de capability.min_output_tokens (spec 2026-09-17 §4.4).

SOLO LEE. Máximo de tokens de salida de corridas COMPLETADAS por capability,
redondeado hacia arriba a múltiplo de 1024, de dos fuentes:
  1. Motor Registry: las_manos/logs/motor_jobs.jsonl, `_usage.completion_tokens`
     de los jobs `completed` (incluye razonamiento: es lo que el tope tiene que
     dejar pasar).
  2. Transportes HTTP directos de Jacobs: axioma_usage.tokens_out
     (request_type='pipeline') unido a jacobs_steps completados por faceta y
     ventana de tiempo. Si dos pasos de la misma faceta se solapan, gana el
     mayor: cota superior, que es la dirección segura para un mínimo.
Una capability sin corridas medibles queda en 0 y se declara.

Consulta de UNA vez, fuera del camino caliente: su EXPLAIN se registra al
correrla (Task 13, Step 6 del plan) y un scan se acepta por ser un script
puntual.

NOTA (Ruling del plan P, jax-platform Task 2, commit d79b0f9 -- Task 13 del
plan J la copia sin repetir la medición, ver Ruling R2 del ledger): el JOIN
por facet entre axioma_usage y jacobs_steps falla contra producción con
1267 "Illegal mix of collations" porque axioma_usage.facet quedó en
utf8mb4_uca1400_ai_ci y jacobs_steps.facet en utf8mb4_unicode_ci (esquemas
creados en momentos distintos). Se agrega `COLLATE utf8mb4_uca1400_ai_ci` al
JOIN -- las claves de faceta son ASCII en minúscula, así que la igualdad da
lo mismo con cualquiera de las dos collations. Verificado por
test_medir_min_output_tokens.py::test_el_join_http_directo_lleva_la_collate_del_esquema_real.
La unión por faceta + ventana de tiempo puede inflar un máximo si dos pasos
de la misma faceta se solapan: se acepta porque el lado seguro de un MÍNIMO
es sobreestimar, nunca subestimar (mismo criterio de P).

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

MULTIPLO = 1024
HOLGURA_S = 5  # axioma_usage.created_at es TIMESTAMP (segundos enteros)

_SQL_HTTP_DIRECTO = (
    "SELECT s.capability, MAX(u.tokens_out) FROM jacobs_steps s "
    "JOIN axioma_usage u ON u.facet = s.facet COLLATE utf8mb4_uca1400_ai_ci AND u.request_type = 'pipeline' "
    " AND UNIX_TIMESTAMP(u.created_at) BETWEEN s.started_at - %s AND s.finished_at + %s "
    "WHERE s.status = 'completed' AND s.started_at IS NOT NULL AND s.finished_at IS NOT NULL "
    "GROUP BY s.capability"
)


def redondear(n: int) -> int:
    if n <= 0:
        return 0
    return math.ceil(n / MULTIPLO) * MULTIPLO


def maximos_de_jobs(lineas: Iterable[str]) -> tuple[dict[str, int], int]:
    maximos: dict[str, int] = {}
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
        if job.get("status") != "completed":
            continue
        tokens = (job.get("_usage") or {}).get("completion_tokens")
        capability = job.get("capability")
        if not capability or not isinstance(tokens, int):
            continue
        maximos[capability] = max(maximos.get(capability, 0), tokens)
    return maximos, rotas


def combinar(*fuentes: dict[str, int]) -> dict[str, int]:
    salida: dict[str, int] = {}
    for fuente in fuentes:
        for capability, valor in fuente.items():
            salida[capability] = max(salida.get(capability, 0), valor)
    return salida


async def _maximos_http_directo() -> tuple[dict[str, int], list[str]]:
    from jacobs import store
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_SQL_HTTP_DIRECTO, (HOLGURA_S, HOLGURA_S))
            maximos = {cap: int(valor or 0) for cap, valor in await cur.fetchall()}
            await cur.execute("SELECT `key` FROM capability ORDER BY `key`")
            capabilities = [r[0] for r in await cur.fetchall()]
    finally:
        conn.close()
    return maximos, capabilities


def main() -> int:
    parser = argparse.ArgumentParser(description="Mide la semilla de capability.min_output_tokens (solo lee).")
    parser.add_argument("--jobs", required=True, type=Path, help="ruta a motor_jobs.jsonl")
    args = parser.parse_args()

    with args.jobs.open(encoding="utf-8") as fh:
        de_jobs, rotas = maximos_de_jobs(fh)
    de_http, capabilities = asyncio.run(_maximos_http_directo())
    medidos = combinar(de_jobs, de_http)

    print(f"motor_jobs.jsonl: {len(de_jobs)} capabilities medidas, {rotas} líneas ilegibles")
    print(f"axioma_usage + jacobs_steps: {len(de_http)} capabilities medidas")
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
