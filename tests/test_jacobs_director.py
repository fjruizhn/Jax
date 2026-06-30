#!/usr/bin/env python3
"""
Test de regresión — Jacobs Director (wave scheduler).

Verifica la lógica de olas SIN tocar DB ni red (mocks). Cubre:
  1. fan-out por especialidad (paralelo intra-ola)
  2. compatibilidad: pipeline lineal → olas de 1 (idéntico a secuencial)
  3. resume con steps ya completos
  4. ciclo → declara, no cuelga
  5. orden temporal: auditor espera a sus deps
  6. fallo sin skip_on_fail aborta; con skip_on_fail continúa
  7. clean-room: detecta auditor con facet de un dep

Uso: python tests/test_jacobs_director.py
Salida esperada: ALL PASS

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import sys


# --- Réplica aislada de _compute_waves (debe coincidir con executor.py) ---
def compute_waves(plan, done):
    pending = {s["idx"] for s in plan if s["idx"] not in done}
    deps_by_idx = {s["idx"]: set(s.get("deps") or []) for s in plan}
    waves, satisfied = [], set(done)
    while pending:
        ready = sorted(i for i in pending if deps_by_idx.get(i, set()) <= satisfied)
        if not ready:
            break
        waves.append(ready)
        for i in ready:
            pending.discard(i)
            satisfied.add(i)
    return waves


# --- Réplica de _check_cleanroom ---
_AUDIT_CAPS = {"validate_consistency", "critique", "review", "audit"}

def check_cleanroom(plan):
    by_idx = {s["idx"]: s for s in plan}
    out = []
    for s in plan:
        if s.get("cap") not in _AUDIT_CAPS:
            continue
        for dep in s.get("deps", []):
            d = by_idx.get(dep)
            if d and d.get("facet") == s.get("facet"):
                out.append((s["idx"], dep, s["facet"]))
    return out


PASS, FAIL = "✅", "❌"
results = []

def check(name, cond):
    results.append((name, cond))
    print(f"  {PASS if cond else FAIL} {name}")


# --- 1. fan-out ---
plan_fanout = [
    {"idx": 0, "deps": []},
    {"idx": 1, "deps": [0]}, {"idx": 2, "deps": [0]}, {"idx": 3, "deps": [0]},
    {"idx": 4, "deps": [1, 2, 3]},
    {"idx": 5, "deps": [0, 1, 2, 3, 4]},
]
w = compute_waves(plan_fanout, set())
check("fan-out: ola 1 corre [1,2,3] en paralelo", w[1] == [1, 2, 3])
check("fan-out: auditor [4] en su propia ola", w[2] == [4])

# --- 2. compatibilidad lineal ---
plan_lin = [{"idx": i, "deps": [i - 1] if i else []} for i in range(6)]
wl = compute_waves(plan_lin, set())
check("lineal: 6 olas de tamaño 1 (sin regresión)", all(len(x) == 1 for x in wl) and len(wl) == 6)

# --- 3. resume ---
wr = compute_waves(plan_fanout, {0, 1, 2})
check("resume: arranca en [3] con 0,1,2 hechos", wr[0] == [3])

# --- 4. ciclo ---
plan_cycle = [{"idx": 0, "deps": [1]}, {"idx": 1, "deps": [0]}]
wc = compute_waves(plan_cycle, set())
check("ciclo: no cuelga, devuelve vacío", wc == [])

# --- 5. orden temporal ---
order = []
async def fake_run(i):
    order.append(("s", i)); await asyncio.sleep(0.01); order.append(("e", i)); return True
async def run_waves(plan):
    for wave in compute_waves(plan, set()):
        await asyncio.gather(*[fake_run(i) for i in wave])
asyncio.run(run_waves(plan_fanout))
s4 = order.index(("s", 4))
check("temporal: auditor espera a 1,2,3", all(order.index(("e", k)) < s4 for k in (1, 2, 3)))

# --- 6. fallo / skip_on_fail (lógica de abort) ---
def wave_aborts(results_map, skip_map, wave):
    return [i for i in wave if not results_map[i] and not skip_map.get(i)]
check("fallo sin skip aborta", wave_aborts({1: True, 2: False}, {}, [1, 2]) == [2])
check("fallo con skip continúa", wave_aborts({1: True, 2: False}, {2: True}, [1, 2]) == [])

# --- 7. clean-room ---
plan_bad = [
    {"idx": 0, "deps": [], "facet": "ada", "cap": "design"},
    {"idx": 1, "deps": [0], "facet": "ada", "cap": "critique"},  # ada audita ada → mal
]
plan_good = [
    {"idx": 0, "deps": [], "facet": "ada", "cap": "design"},
    {"idx": 1, "deps": [0], "facet": "thot", "cap": "critique"},  # thot audita ada → bien
]
check("clean-room: detecta auditor mismo-facet", len(check_cleanroom(plan_bad)) == 1)
check("clean-room: acepta auditor independiente", len(check_cleanroom(plan_good)) == 0)


# --- resumen ---
print()
ok = sum(1 for _, c in results if c)
total = len(results)
if ok == total:
    print(f"=== ALL PASS ({ok}/{total}) ===")
    sys.exit(0)
else:
    print(f"=== {total - ok} FALLO(S) ({ok}/{total}) ===")
    for n, c in results:
        if not c:
            print(f"    FALLÓ: {n}")
    sys.exit(1)
