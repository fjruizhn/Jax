#!/usr/bin/env python3
"""Prueba de concurrencia de GPU -- item DEUDA.md "GPU_SEMAPHORE no cubre a
Jacobs". Mide si N inferencias simultaneas contra Ollama local degradan
tok/s, encolan, o revientan por VRAM en la AMD Radeon AI PRO R9700 (32GB,
card0 en rocm-smi), antes de decidir si GPU_SEMAPHORE necesita volverse
cross-proceso.

Corre con:
  python3 scripts/gpu_concurrency_probe.py --model <modelo> --concurrency 1 2 3 --repeats 3

TRES CORRECCIONES sobre el diseño original del plan (2026-08-25), cada una
porque la version anterior medía otra cosa:

1. TOK/S SOLO NO ALCANZA -- hay que medir el WALL-CLOCK de la ronda.
   `OLLAMA_NUM_PARALLEL` está en 1 en este servidor (confirmado en el log de
   arranque de ollama). Con Ollama serializando, dos requests concurrentes
   se ENCOLAN: cada una reporta su propio `eval_count/elapsed` casi intacto,
   así que **tok/s por request se mantiene plano y la degradación da ~0%** --
   y un lector apurado concluye "no hay contención, no hace falta semáforo".
   La conclusión sería correcta por el motivo equivocado: no es que la GPU
   aguante, es que nadie la usa en paralelo. Lo que distingue los dos casos
   es el tiempo de pared de la ronda: si N requests tardan ~N veces lo que
   una, están encoladas.
2. LA VRAM SE MUESTREA DURANTE, NO ANTES Y DESPUES. El pico ocurre mientras
   corren las inferencias; medir `after` cuando ya terminaron puede dar
   delta ~0 y no dice nada del pico real.
3. TOK/S SE CALCULA CON `eval_duration`, EL RELOJ DE GENERACION DE OLLAMA,
   y no con el `elapsed` de la llamada HTTP. Con requests encoladas
   `elapsed` incluye LA ESPERA EN COLA, asi que `eval_count/elapsed` cae
   aunque la GPU genere exactamente igual de rapido -- y el criterio de
   "degradacion > 25%" del umbral se disparia por encolamiento, que es justo
   lo que ese umbral separa. Se reportan las DOS: `gen` (generacion pura, la
   que decide) y `e2e` (extremo a extremo, lo que siente el llamador).
4. SE REGISTRA EL ENTORNO QUE DECIDE EL RESULTADO (modelo, num_parallel,
   modelos residentes). Un numero sin el entorno en que se produjo no es
   evidencia reutilizable: el veredicto depende de `num_parallel` tanto como
   de la GPU.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time

import httpx

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"
PROMPT = "Contá del 1 al 30, un numero por linea, sin texto extra."
GPU_CARD_KEY = "card0"   # AMD Radeon AI PRO R9700, 32GB -- confirmado con rocm-smi
VRAM_SAMPLE_SECONDS = 0.2


def _vram_used_bytes() -> int:
    out = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return int(json.loads(out.stdout)[GPU_CARD_KEY]["VRAM Total Used Memory (B)"])


def _num_parallel_del_log() -> str:
    """Lo que Ollama eligió, NO lo que la variable de entorno dice.

    Sin `OLLAMA_NUM_PARALLEL` seteada, Ollama elige solo segun la memoria
    disponible. Preguntarle al entorno daria 'no seteada' y no responderia
    nada; el log de arranque tiene el valor efectivo."""
    try:
        out = subprocess.run(
            ["journalctl", "-u", "ollama", "--no-pager", "-n", "5000"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception as exc:                       # noqa: BLE001
        return f"indeterminado ({type(exc).__name__})"
    for linea in reversed(out.splitlines()):
        if "OLLAMA_NUM_PARALLEL" in linea:
            trozo = linea.split("OLLAMA_NUM_PARALLEL:")[-1].split()[0]
            return trozo.strip(" \"',")
    return "indeterminado (sin linea en el journal)"


async def _residentes() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/ps")
        return [f"{m['name']} ({round(m['size'] / 1e9, 1)}GB)"
                for m in r.json().get("models", [])]
    except Exception as exc:                       # noqa: BLE001
        return [f"indeterminado ({type(exc).__name__})"]


async def _muestrear_vram(parar: asyncio.Event) -> int:
    """Máximo de VRAM usada MIENTRAS corre la ronda (ver corrección 2)."""
    pico = 0
    while not parar.is_set():
        try:
            pico = max(pico, await asyncio.to_thread(_vram_used_bytes))
        except Exception:                          # noqa: BLE001
            pass                                   # fail-soft: es telemetría,
                                                   # no el resultado
        try:
            await asyncio.wait_for(parar.wait(), timeout=VRAM_SAMPLE_SECONDS)
        except asyncio.TimeoutError:
            pass
    return pico


async def _one_inference(model: str) -> dict:
    payload = {"model": model,
               "messages": [{"role": "user", "content": PROMPT}],
               "stream": False}
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc),
                "elapsed": time.monotonic() - start, "tok_s": 0.0}
    elapsed = time.monotonic() - start
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}",
                "elapsed": elapsed, "tok_s": 0.0}
    data = resp.json()
    tokens_out = data.get("eval_count", 0)
    # `eval_duration` (ns) es el reloj de GENERACION de Ollama: excluye la
    # espera en cola y la carga del modelo. Es el unico que responde "genera
    # mas lento cuando hay concurrencia?" -- ver correccion 3.
    eval_ns = data.get("eval_duration", 0) or 0
    return {"ok": True, "elapsed": elapsed, "tokens_out": tokens_out,
            "tok_s_e2e": tokens_out / elapsed if elapsed > 0 else 0.0,
            "tok_s_gen": tokens_out / (eval_ns / 1e9) if eval_ns else 0.0}


async def _run_round(model: str, concurrency: int) -> dict:
    parar = asyncio.Event()
    muestreo = asyncio.create_task(_muestrear_vram(parar))
    vram_antes = _vram_used_bytes()

    inicio = time.monotonic()
    results = await asyncio.gather(*[_one_inference(model)
                                     for _ in range(concurrency)])
    wall_clock = time.monotonic() - inicio

    parar.set()
    pico_vram = await muestreo
    oks = [r for r in results if r["ok"]]
    return {
        "concurrency": concurrency,
        "wall_clock": wall_clock,
        "errors": [r["error"] for r in results if not r["ok"]],
        "pico_vram_bytes": max(pico_vram, vram_antes),
        "vram_delta_bytes": max(pico_vram, vram_antes) - vram_antes,
        "tok_s_gen_per_request": [r["tok_s_gen"] for r in oks],
        "tok_s_e2e_per_request": [r["tok_s_e2e"] for r in oks],
        "elapsed_per_request": [r["elapsed"] for r in oks],
    }


async def main_async(model: str, concurrencies: list[int], repeats: int) -> list[dict]:
    print(f"Modelo:            {model}")
    print(f"OLLAMA_NUM_PARALLEL efectivo: {_num_parallel_del_log()}")
    print(f"Residentes en VRAM: {', '.join(await _residentes()) or '(ninguno)'}")
    print(f"Repeticiones por nivel: {repeats}\n")

    summaries: list[dict] = []
    baseline_tok_s = None
    baseline_wall = None
    baseline_e2e = None
    for c in concurrencies:
        rounds = [await _run_round(model, c) for _ in range(repeats)]
        gen = [t for r in rounds for t in r["tok_s_gen_per_request"]]
        e2e = [t for r in rounds for t in r["tok_s_e2e_per_request"]]
        total_errors = sum(len(r["errors"]) for r in rounds)
        median_tok_s = statistics.median(gen) if gen else 0.0
        median_e2e = statistics.median(e2e) if e2e else 0.0
        median_wall = statistics.median([r["wall_clock"] for r in rounds])
        if c == 1:
            baseline_tok_s, baseline_wall = median_tok_s, median_wall
            baseline_e2e = median_e2e
        degradation_pct = (round(100 * (1 - median_tok_s / baseline_tok_s), 1)
                           if baseline_tok_s else 0.0)
        degradation_e2e_pct = (round(100 * (1 - median_e2e / baseline_e2e), 1)
                               if baseline_e2e else 0.0)
        # El discriminante de la corrección 1: ~1.0 = de verdad en paralelo;
        # ~N = encolado (Ollama serializó y no medimos contención de GPU).
        factor_wall = (round(median_wall / baseline_wall, 2)
                       if baseline_wall else 0.0)
        summary = {
            "concurrency": c,
            "tok_s_gen_median": round(median_tok_s, 1),
            "degradation_pct": degradation_pct,
            "tok_s_e2e_median": round(median_e2e, 1),
            "degradation_e2e_pct": degradation_e2e_pct,
            "wall_clock_median_s": round(median_wall, 1),
            "factor_wall_vs_baseline": factor_wall,
            "errors": total_errors,
            "pico_vram_mb": round(max(r["pico_vram_bytes"] for r in rounds)
                                  / (1024 * 1024), 1),
            "delta_vram_max_mb": round(max(r["vram_delta_bytes"] for r in rounds)
                                       / (1024 * 1024), 1),
        }
        summaries.append(summary)
        print(f"concurrency={c}: tok/s GEN={summary['tok_s_gen_median']} "
              f"(degrad={summary['degradation_pct']}%) "
              f"tok/s e2e={summary['tok_s_e2e_median']} "
              f"(degrad={summary['degradation_e2e_pct']}%) "
              f"wall={summary['wall_clock_median_s']}s "
              f"(x{summary['factor_wall_vs_baseline']} vs baseline) "
              f"errores={summary['errors']} "
              f"pico_vram={summary['pico_vram_mb']}MB "
              f"delta_vram={summary['delta_vram_max_mb']}MB")
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    # Sin default de modelo A PROPOSITO: el modelo primario de jax_local sale
    # de `facet_binding` y cambia (CLAUDE.md). Un default hardcodeado aca
    # mediria el modelo de otra epoca sin que nadie lo note -- el plan
    # original traia `qwen3-coder:30b`, que para el 2026-08-28 ya no era el
    # de produccion (qwen3.6:35b-a3b-q4_K_M).
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main_async(args.model, args.concurrency, args.repeats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
