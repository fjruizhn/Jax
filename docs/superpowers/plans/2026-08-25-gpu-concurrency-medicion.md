# GPU_SEMAPHORE — Medición de Concurrencia Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determinar con evidencia real (no supuesta) si las 3 rutas que le pegan a Ollama local sin pasar por `GPU_SEMAPHORE` (`jax/muscles/ollama_muscle.py`, `jacobs/plan.py::_llm_plan`, `jacobs/executor.py::_invoke_ollama`) necesitan exclusión mutua cross-proceso. `DEUDA.md` NO se toca en este plan (política explícita del usuario, 2026-08-25) — el veredicto queda en un archivo de evidencia propio, y el comentario falso de `GPU_SEMAPHORE` se corrige en código sin importar el resultado.

**Architecture:** Script de carga (`scripts/gpu_concurrency_probe.py`) que dispara N inferencias concurrentes contra Ollama nativo (`/api/chat`) con el mismo modelo, mide tok/s por request y VRAM usada (vía `rocm-smi --json`), y compara contra el baseline de concurrencia=1. El veredicto se documenta con el resultado real, no de antemano.

**Tech Stack:** Python 3.12, `httpx`, `asyncio`, `rocm-smi` (ya instalado, confirmado: `/usr/bin/rocm-smi`), Ollama 0.24 corriendo en `localhost:11434` con modelo `qwen3-coder:30b` (confirmado con `ollama list`, 18GB, ya descargado).

**Spec:** `DEUDA.md` (raíz de `/home/fruiz/jax`), sección "Bloquea trabajo", ítem `GPU_SEMAPHORE no cubre a Jacobs`.

## Global Constraints

- No se toca código de producción en este plan (Task 3 es la única excepción: corregir un comentario falso y dos referencias de línea desactualizadas — cero cambio de comportamiento).
- El script de medición corre contra el Ollama real de hall9000 — coordinar con cualquier trabajo en curso que use la GPU antes de correrlo (puede degradar temporalmente la latencia de otros usos).
- Decisión de cierre/reapertura se basa en el número medido, no en intuición: umbral acordado en Task 2 antes de correr la medición real (no elegido post-hoc para justificar un resultado).

---

### Task 1: Script de medición de concurrencia

**Files:**
- Create: `scripts/gpu_concurrency_probe.py`

**Interfaces:**
- Produce: función `main()` ejecutable como script, imprime una tabla de `concurrency=N: tok/s mediana=X degradación=Y% errores=Z` a stdout.

- [ ] **Step 1: Verificar el formato real de `rocm-smi --json` (ya verificado en la investigación, confirmar de nuevo en vivo antes de codear)**

Run: `rocm-smi --showmeminfo vram --json 2>/dev/null`
Expected: `{"card0": {"VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": "..."}, "card1": {...}}` — `card0` es la AMD Radeon AI PRO R9700 (32GB real), `card1` es la iGPU de 2GB. El WARNING de "low-power state" va a stderr, no contamina el JSON de stdout.

- [ ] **Step 2: Escribir el script completo**

```python
#!/usr/bin/env python3
"""
Prueba de concurrencia de GPU -- item DEUDA.md "GPU_SEMAPHORE no cubre a
Jacobs". Mide si 2-3 inferencias simultaneas contra Ollama local degradan
tok/s u OOM en la AMD Radeon AI PRO R9700 (32GB, card0 en rocm-smi), antes
de decidir si GPU_SEMAPHORE necesita volverse cross-proceso.

Corre con:
  python3 scripts/gpu_concurrency_probe.py --model qwen3-coder:30b --concurrency 1 2 3 --repeats 3

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

OLLAMA_URL = "http://localhost:11434/api/chat"
PROMPT = "Contá del 1 al 30, un numero por linea, sin texto extra."
GPU_CARD_KEY = "card0"  # AMD Radeon AI PRO R9700, 32GB -- confirmado con rocm-smi


def _vram_used_bytes() -> int:
    out = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    data = json.loads(out.stdout)
    return int(data[GPU_CARD_KEY]["VRAM Total Used Memory (B)"])


async def _one_inference(model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
    }
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc), "elapsed": time.monotonic() - start, "tok_s": 0.0}
    elapsed = time.monotonic() - start
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}", "elapsed": elapsed, "tok_s": 0.0}
    data = resp.json()
    tokens_out = data.get("eval_count", 0)
    return {
        "ok": True,
        "elapsed": elapsed,
        "tokens_out": tokens_out,
        "tok_s": tokens_out / elapsed if elapsed > 0 else 0.0,
    }


async def _run_round(model: str, concurrency: int) -> dict:
    vram_before = _vram_used_bytes()
    results = await asyncio.gather(*[_one_inference(model) for _ in range(concurrency)])
    vram_after = _vram_used_bytes()
    errors = [r for r in results if not r["ok"]]
    oks = [r for r in results if r["ok"]]
    return {
        "concurrency": concurrency,
        "errors": [r["error"] for r in errors],
        "vram_delta_bytes": vram_after - vram_before,
        "tok_s_per_request": [r["tok_s"] for r in oks],
    }


async def main_async(model: str, concurrencies: list[int], repeats: int) -> list[dict]:
    print(f"Modelo: {model} | repeticiones por nivel de concurrencia: {repeats}\n")
    summaries = []
    baseline_tok_s = None
    for c in concurrencies:
        rounds = [await _run_round(model, c) for _ in range(repeats)]
        all_tok_s = [t for r in rounds for t in r["tok_s_per_request"]]
        total_errors = sum(len(r["errors"]) for r in rounds)
        median_tok_s = statistics.median(all_tok_s) if all_tok_s else 0.0
        if c == 1:
            baseline_tok_s = median_tok_s
        degradation_pct = (
            round(100 * (1 - median_tok_s / baseline_tok_s), 1)
            if baseline_tok_s else 0.0
        )
        max_vram_delta = max((r["vram_delta_bytes"] for r in rounds), default=0)
        summary = {
            "concurrency": c,
            "tok_s_median": round(median_tok_s, 1),
            "degradation_pct": degradation_pct,
            "errors": total_errors,
            "max_vram_delta_mb": round(max_vram_delta / (1024 * 1024), 1),
        }
        summaries.append(summary)
        print(
            f"concurrency={c}: tok/s mediana={summary['tok_s_median']} "
            f"degradacion vs baseline={summary['degradation_pct']}% "
            f"errores={summary['errors']} "
            f"delta_vram_max={summary['max_vram_delta_mb']}MB"
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main_async(args.model, args.concurrency, args.repeats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Prueba de humo — correr con concurrency=1 solamente, confirmar que el script funciona antes de la corrida completa**

Run: `python3 scripts/gpu_concurrency_probe.py --model qwen3-coder:30b --concurrency 1 --repeats 1`
Expected: una línea `concurrency=1: tok/s mediana=<numero> degradacion vs baseline=0.0% errores=0 delta_vram_max=<numero>MB` — sin excepciones, sin errores de conexión.

- [ ] **Step 4: Commit**

```bash
cd /home/fruiz/jax
git add scripts/gpu_concurrency_probe.py
git commit -m "feat(scripts): agrega probe de concurrencia de GPU para medir necesidad real de GPU_SEMAPHORE cross-proceso"
```

---

### Task 2: Definir el umbral de decisión (antes de medir)

**Files:**
- Create: `docs/superpowers/specs/2026-08-25-gpu-concurrency-umbral.md`

- [ ] **Step 1: Escribir el criterio de decisión, ANTES de ver el resultado real**

```markdown
# Umbral de decisión — GPU_SEMAPHORE cross-proceso

Definido antes de correr la medición real (2026-08-25), para no elegir el
umbral después de ver el número y justificar el resultado que se prefiera.

**Se considera que SÍ hace falta exclusión mutua cross-proceso (dejar
constancia en el archivo de resultado de que el fix queda pendiente,
fuera de este plan) si, en concurrency=2 o concurrency=3 (mediana de 3
corridas cada uno):**

- degradación de tok/s por request > 25% respecto al baseline de
  concurrency=1, O
- cualquier error HTTP / timeout / respuesta inválida de Ollama que no
  ocurra en concurrency=1, O
- delta de VRAM en una sola corrida > 90% de los 32GB totales (riesgo real
  de OOM bajo carga real de producción, no solo del probe).

**Si ninguna de las tres se cumple:** se documenta con los números reales
en `docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md` (no
en DEUDA.md — el usuario decide si y cómo lo refleja ahí), y se corrige
el comentario falso en `ollama_muscle.py:36` ("compartido por toda carga
GPU futura" -> ya no es cierto, y con este resultado tampoco haría falta
que lo fuera).
```

- [ ] **Step 2: Commit**

```bash
cd /home/fruiz/jax
git add docs/superpowers/specs/2026-08-25-gpu-concurrency-umbral.md
git commit -m "docs: fija el umbral de decisión de GPU_SEMAPHORE antes de medir"
```

---

### Task 3: Correr la medición real y decidir

**Files:**
- Create: `docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md`
- Modify: `jax/muscles/ollama_muscle.py:35-37` (comentario)
- Modify: `jacobs/plan.py:567-572` (comentario, línea desactualizada en referencia cruzada)
- Modify: `jacobs/executor.py:360-365` (comentario, línea desactualizada en referencia cruzada)

**Interfaces:**
- Consume: `scripts/gpu_concurrency_probe.py` (Task 1), umbral de `docs/superpowers/specs/2026-08-25-gpu-concurrency-umbral.md` (Task 2).

- [ ] **Step 1: Correr la medición completa**

Run: `python3 scripts/gpu_concurrency_probe.py --model qwen3-coder:30b --concurrency 1 2 3 --repeats 3 2>&1 | tee /tmp/gpu_concurrency_result.txt`
Expected: tabla completa de 3 líneas (concurrency=1,2,3) con tok/s mediana, % degradación, errores y delta VRAM. Verificar contra el umbral de Task 2.

- [ ] **Step 2: Registrar el veredicto (NO en `DEUDA.md` — política explícita del usuario, 2026-08-25: nada va a `DEUDA.md` en este plan). Escribir el resultado en un archivo de evidencia propio.**

Crear `docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md` con el resultado real de Step 1 (números pegados, no `<...>`):

```markdown
# Resultado de la medición de concurrencia de GPU — 2026-08-25

Umbral acordado (ver 2026-08-25-gpu-concurrency-umbral.md): >25%
degradación de tok/s, cualquier error nuevo, o >90% de VRAM en
concurrency=2/3 respecto a concurrency=1.

Medido con `scripts/gpu_concurrency_probe.py --model qwen3-coder:30b
--concurrency 1 2 3 --repeats 3`:

| concurrency | tok/s mediana | degradación vs baseline | errores | delta VRAM máx |
|---|---|---|---|---|
| 1 | <numero real> | 0% | <numero real> | <numero real> |
| 2 | <numero real> | <numero real>% | <numero real> | <numero real> |
| 3 | <numero real> | <numero real>% | <numero real> | <numero real> |

**Veredicto:** <"por debajo del umbral, no hace falta exclusión mutua
cross-proceso" O "supera el umbral, hace falta lock cross-proceso — ver
sección de fix evaluado abajo">.

<Si supera el umbral, agregar acá: "Fix evaluado, no implementado en
este plan: lock de archivo cross-proceso (fcntl.flock, ej.
/run/jax/gpu.lock, envuelto en asyncio.to_thread) usado por los 3 call
sites reales (ollama_muscle.py, jacobs/plan.py::_llm_plan,
jacobs/executor.py::_invoke_ollama) más
las_manos/motor_registry/worker.py::_call_http_openai_compat (solo
transporte 'ollama'). Implementación queda fuera de este plan — requiere
decisión de scope aparte con el usuario.">
```

Esto NO reemplaza ni edita ningún ítem de `DEUDA.md` — el usuario decide si y cómo lo refleja ahí.

- [ ] **Step 3: Corregir el comentario falso en `ollama_muscle.py` (incondicional — el comentario es falso hoy, sin importar el resultado de la medición)**

En `jax/muscles/ollama_muscle.py:35-37`, reemplazar:
```python
# Semaforo global de GPU: UNA inferencia local a la vez en hall9000.
# Compartido a nivel de modulo para que cualquier carga GPU futura lo respete.
GPU_SEMAPHORE = asyncio.Semaphore(1)
```
Por:
```python
# Semaforo de GPU -- UNA inferencia local a la vez DENTRO DE ESTE PROCESO.
# Es un asyncio.Semaphore de proceso, NO cross-proceso: Jacobs (proceso
# jax-las-manos) tiene 3 caminos propios a Ollama que no pasan por acá
# (ver jacobs/plan.py::_llm_plan, jacobs/executor.py::_invoke_ollama,
# las_manos/motor_registry/worker.py) -- medido 2026-08-25
# (scripts/gpu_concurrency_probe.py), ver docs/superpowers/specs/
# 2026-08-25-gpu-concurrency-resultado.md para el veredicto.
GPU_SEMAPHORE = asyncio.Semaphore(1)
```

- [ ] **Step 4: Corregir las referencias de línea desactualizadas en los comentarios cruzados**

En `jacobs/plan.py:567-572`, el comentario actual dice `OJO: GPU_SEMAPHORE (jax/muscles/ollama_muscle.py:37)` — esa línea sigue siendo correcta (verificado en Step 3 de este task, la constante no se movió). No requiere cambio de número de línea, pero agregar referencia a la medición:
```python
        # OJO: GPU_SEMAPHORE (jax/muscles/ollama_muscle.py:37) es un
        # asyncio.Semaphore de PROCESO del REPL de JAX -- esta llamada corre
        # en el proceso de jax-las-manos (Jacobs) y le pega a Ollama directo
        # por httpx, sin pasar por ese semáforo. No hay exclusión mutua real
        # entre el REPL y Jacobs para el acceso a la GPU (verificado
        # 2026-08-19, sonda T0.a/T1 de latencia de _llm_plan; medido de
        # nuevo 2026-08-25, ver DEUDA.md para el veredicto).
```

Mismo agregado en `jacobs/executor.py:360-365` (el bloque que empieza `OJO: GPU_SEMAPHORE...` dentro del docstring de `_invoke_ollama`).

- [ ] **Step 5: Commit (DEUDA.md no se toca — solo el resultado de medición y las correcciones de código/comentario)**

```bash
cd /home/fruiz/jax
git add docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md jax/muscles/ollama_muscle.py jacobs/plan.py jacobs/executor.py
git commit -m "docs: registra resultado de medición de concurrencia de GPU, corrige comentario falso de GPU_SEMAPHORE y referencias de línea desactualizadas"
```

---

## Self-Review

- **Cobertura del spec:** el ítem de `DEUDA.md` pide decidir si el gap "bloquea trabajo" — cubierto por Task 2 (umbral fijado antes) + Task 3 (medición real + registro de evidencia en un archivo propio, sin tocar `DEUDA.md` por política explícita del usuario).
- **Sin placeholders reales:** los `<...>` en Step 2 son marcadores explícitos a rellenar con el número medido en Step 1 del mismo task — no son "TBD" de diseño, es el propio mecanismo de "pegar el resultado real" que pide la Regla del Carpintero (medir antes de decidir).
- **Consistencia:** `GPU_CARD_KEY = "card0"` en el script coincide con el `rocm-smi --json` verificado en vivo (Step 1 de Task 1).
