#!/usr/bin/env python3
"""
Test de regresión — Jacobs Director (wave scheduler).

Suite pytest. Verifica la lógica de olas SIN tocar DB ni red, con réplicas
autocontenidas que deben coincidir con executor._compute_waves y
plan._check_cleanroom (alcance idéntico al script original — NO importa
jacobs.executor real). Cubre 10 casos:

  1. fan-out: ola 1 corre [1,2,3] en paralelo
  2. fan-out: auditor [4] en su propia ola
  3. lineal: 6 olas de tamaño 1 (sin regresión)
  4. resume: arranca en [3] con 0,1,2 hechos
  5. ciclo: no cuelga, devuelve vacío
  6. temporal: auditor espera a sus deps (1,2,3)
  7. fallo sin skip_on_fail aborta
  8. fallo con skip_on_fail continúa
  9. clean-room: detecta auditor mismo-facet
 10. clean-room: acepta auditor independiente

Uso: python -m pytest tests/test_jacobs_director.py -v

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio


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


# --- Plan de fan-out compartido por varios casos ---
PLAN_FANOUT = [
    {"idx": 0, "deps": []},
    {"idx": 1, "deps": [0]}, {"idx": 2, "deps": [0]}, {"idx": 3, "deps": [0]},
    {"idx": 4, "deps": [1, 2, 3]},
    {"idx": 5, "deps": [0, 1, 2, 3, 4]},
]


# --- 1. fan-out: ola 1 corre [1,2,3] en paralelo ---
def test_fanout_ola1_corre_123_en_paralelo():
    w = compute_waves(PLAN_FANOUT, set())
    assert w[1] == [1, 2, 3]


# --- 2. fan-out: auditor [4] en su propia ola ---
def test_fanout_auditor_4_en_su_propia_ola():
    w = compute_waves(PLAN_FANOUT, set())
    assert w[2] == [4]


# --- 3. lineal: 6 olas de tamaño 1 (sin regresión) ---
def test_lineal_6_olas_de_tamano_1_sin_regresion():
    plan_lin = [{"idx": i, "deps": [i - 1] if i else []} for i in range(6)]
    wl = compute_waves(plan_lin, set())
    assert all(len(x) == 1 for x in wl) and len(wl) == 6


# --- 4. resume: arranca en [3] con 0,1,2 hechos ---
def test_resume_arranca_en_3_con_012_hechos():
    wr = compute_waves(PLAN_FANOUT, {0, 1, 2})
    assert wr[0] == [3]


# --- 5. ciclo: no cuelga, devuelve vacío ---
def test_ciclo_no_cuelga_devuelve_vacio():
    plan_cycle = [{"idx": 0, "deps": [1]}, {"idx": 1, "deps": [0]}]
    wc = compute_waves(plan_cycle, set())
    assert wc == []


# --- 6. temporal: auditor espera a sus deps (1,2,3) ---
def test_temporal_auditor_espera_a_123():
    order = []

    async def fake_run(i):
        order.append(("s", i))
        await asyncio.sleep(0.01)
        order.append(("e", i))
        return True

    async def run_waves(plan):
        for wave in compute_waves(plan, set()):
            await asyncio.gather(*[fake_run(i) for i in wave])

    asyncio.run(run_waves(PLAN_FANOUT))
    s4 = order.index(("s", 4))
    assert all(order.index(("e", k)) < s4 for k in (1, 2, 3))


# --- 7/8. fallo / skip_on_fail (lógica de abort) ---
def _wave_aborts(results_map, skip_map, wave):
    return [i for i in wave if not results_map[i] and not skip_map.get(i)]


def test_fallo_sin_skip_aborta():
    assert _wave_aborts({1: True, 2: False}, {}, [1, 2]) == [2]


def test_fallo_con_skip_continua():
    assert _wave_aborts({1: True, 2: False}, {2: True}, [1, 2]) == []


# --- 9. clean-room: detecta auditor mismo-facet ---
def test_cleanroom_detecta_auditor_mismo_facet():
    plan_bad = [
        {"idx": 0, "deps": [], "facet": "ada", "cap": "design"},
        {"idx": 1, "deps": [0], "facet": "ada", "cap": "critique"},  # ada audita ada → mal
    ]
    assert len(check_cleanroom(plan_bad)) == 1


# --- 10. clean-room: acepta auditor independiente ---
def test_cleanroom_acepta_auditor_independiente():
    plan_good = [
        {"idx": 0, "deps": [], "facet": "ada", "cap": "design"},
        {"idx": 1, "deps": [0], "facet": "thot", "cap": "critique"},  # thot audita ada → bien
    ]
    assert len(check_cleanroom(plan_good)) == 0
