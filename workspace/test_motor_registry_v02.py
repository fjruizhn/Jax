#!/usr/bin/env python3
"""
Pruebas del Motor Registry v0.2 — LAS MANOS.

Prueba 1: dispatch real a Kimi (POST /motor/dispatch)
Prueba 2: polling hasta completar (GET /motor/job/{job_id})
Prueba 3: kill switch (/etc/jax/PAUSE)

Corre directamente sin curl — usa urllib de la stdlib.
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:7777"
PAUSE_PATH = "/etc/jax/PAUSE"

DISPATCH_PAYLOAD = {
    "caller": "hyde",
    "capability": "refactor",
    "prompt": "refactoriza el archivo de prueba: def suma(a, b): return a + b"
}


def http_get(path: str) -> dict:
    url = BASE_URL + path
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def http_post(path: str, payload: dict) -> dict:
    url = BASE_URL + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def run_sudo(cmd: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(["sudo"] + cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ─────────────────────────────────────────
#  /health
# ─────────────────────────────────────────
print("=" * 60)
print("HEALTH CHECK")
print("=" * 60)
try:
    h = http_get("/health")
    print(f"Respuesta: {json.dumps(h, indent=2, ensure_ascii=False)}")
    ks = h.get("kill_switch_active", "?")
    if h.get("status") == "alive":
        print(f"{PASS} — servidor vivo. kill_switch_active={ks}")
    else:
        print(f"{FAIL} — respuesta inesperada")
        sys.exit(1)
except Exception as e:
    print(f"{FAIL} — /health falló: {e}")
    sys.exit(1)


# ─────────────────────────────────────────
#  PRUEBA 1 — dispatch real
# ─────────────────────────────────────────
print()
print("=" * 60)
print("PRUEBA 1 — POST /motor/dispatch (dispatch a Kimi)")
print("=" * 60)
try:
    resp1 = http_post("/motor/dispatch", DISPATCH_PAYLOAD)
    print(f"Respuesta dispatch:\n{json.dumps(resp1, indent=2, ensure_ascii=False)}")
    job_id = resp1.get("job_id")
    dispatch_status = resp1.get("status")
    motor = resp1.get("motor")
    if job_id and dispatch_status in ("pending", "running"):
        print(f"{PASS} — job_id={job_id}, status={dispatch_status}, motor={motor}")
    elif dispatch_status == "rejected":
        print(f"{FAIL} — dispatch rechazado: {resp1.get('rejected_reason')}")
        sys.exit(1)
    else:
        print(f"{FAIL} — status inesperado: {dispatch_status}")
        sys.exit(1)
except Exception as e:
    print(f"{FAIL} — dispatch falló: {e}")
    sys.exit(1)


# ─────────────────────────────────────────
#  PRUEBA 2 — polling hasta completar
# ─────────────────────────────────────────
print()
print("=" * 60)
print(f"PRUEBA 2 — GET /motor/job/{job_id} (polling hasta completar)")
print("=" * 60)

MAX_WAIT = 120
INTERVAL = 5
poll_start = time.time()
final_view = None

for i in range(1, MAX_WAIT // INTERVAL + 1):
    try:
        view = http_get(f"/motor/job/{job_id}")
        status = view.get("status")
        elapsed = int(time.time() - poll_start)
        print(f"  Poll {i:2d} ({elapsed:3d}s): status={status}")
        if status not in ("pending", "running"):
            final_view = view
            break
    except Exception as e:
        print(f"  Poll {i:2d}: error — {e}")
    time.sleep(INTERVAL)

if final_view is None:
    print(f"{FAIL} — timeout sin resultado después de {MAX_WAIT}s")
else:
    print()
    print(f"Respuesta final:\n{json.dumps(final_view, indent=2, ensure_ascii=False)}")
    print()

    status = final_view.get("status")
    result_summary = final_view.get("result_summary")
    has_reasoning = "reasoning_content" in final_view

    checks = []

    # status = completed
    if status == "completed":
        checks.append((True, f"status=completed"))
    else:
        checks.append((False, f"status={status} (esperado: completed)"))

    # result_summary es texto real de Kimi (no el stub)
    if result_summary and "[STUB" not in result_summary:
        checks.append((True, f"result_summary contiene texto real de Kimi ({len(result_summary)} chars)"))
    elif result_summary and "[STUB" in result_summary:
        checks.append((False, f"result_summary contiene stub v0.1: '{result_summary[:80]}'"))
    else:
        checks.append((False, f"result_summary ausente o vacío"))

    # reasoning_content NO debe aparecer
    if not has_reasoning:
        checks.append((True, "reasoning_content NO aparece en la respuesta JSON (correcto)"))
    else:
        checks.append((False, "reasoning_content APARECE en la respuesta — fuga de datos internos"))

    all_pass_2 = all(ok for ok, _ in checks)
    for ok, msg in checks:
        tag = PASS if ok else FAIL
        print(f"  {tag} — {msg}")

    if all_pass_2:
        print(f"\n{PASS} PRUEBA 2 completa")
    else:
        print(f"\n{FAIL} PRUEBA 2 con errores")


# ─────────────────────────────────────────
#  PRUEBA 3 — kill switch
# ─────────────────────────────────────────
print()
print("=" * 60)
print("PRUEBA 3 — Kill switch (/etc/jax/PAUSE)")
print("=" * 60)

# 3a — asegurar que PAUSE no existe
rc, out, err = run_sudo(["rm", "-f", PAUSE_PATH])
print(f"  sudo rm -f {PAUSE_PATH} → rc={rc}")

# 3b — crear PAUSE
rc, out, err = run_sudo(["mkdir", "-p", "/etc/jax"])
print(f"  sudo mkdir -p /etc/jax → rc={rc}")
rc, out, err = run_sudo(["touch", PAUSE_PATH])
print(f"  sudo touch {PAUSE_PATH} → rc={rc}")

# Verificar que existe
import os
pause_exists = os.path.exists(PAUSE_PATH)
print(f"  PAUSE existe: {pause_exists}")

# 3c — POST dispatch con kill switch activo
print()
print("  POST /motor/dispatch con PAUSE activo...")
try:
    resp3 = http_post("/motor/dispatch", DISPATCH_PAYLOAD)
    print(f"  Respuesta dispatch:\n  {json.dumps(resp3, indent=2, ensure_ascii=False)}")
    job_id3 = resp3.get("job_id")
    dispatch_status3 = resp3.get("status")
    if job_id3:
        print(f"  job_id={job_id3}, status={dispatch_status3}")
    else:
        print(f"  {FAIL} — no se obtuvo job_id")
except Exception as e:
    print(f"  {FAIL} — dispatch falló: {e}")
    job_id3 = None

# 3d — GET job status (esperar 3s a que el worker procese)
ks_status = None
ks_error = None
if job_id3:
    print()
    print(f"  Esperando 3s para que el worker procese...")
    time.sleep(3)
    try:
        view3 = http_get(f"/motor/job/{job_id3}")
        print(f"  Respuesta GET job:\n  {json.dumps(view3, indent=2, ensure_ascii=False)}")
        ks_status = view3.get("status")
        ks_error = view3.get("error", "")
    except Exception as e:
        print(f"  {FAIL} — GET job falló: {e}")

# 3e — verificar
print()
checks3 = []
if ks_status == "failed":
    checks3.append((True, f"status=failed (correcto — kill switch bloqueó el job)"))
else:
    checks3.append((False, f"status={ks_status} (esperado: failed)"))

if ks_error and "killed_by_switch" in ks_error:
    checks3.append((True, f"error contiene 'killed_by_switch': '{ks_error[:100]}'"))
else:
    checks3.append((False, f"error NO contiene 'killed_by_switch': '{ks_error}'"))

for ok, msg in checks3:
    tag = PASS if ok else FAIL
    print(f"  {tag} — {msg}")

# 3f — eliminar PAUSE
rc, out, err = run_sudo(["rm", PAUSE_PATH])
print(f"\n  sudo rm {PAUSE_PATH} → rc={rc}")
pause_gone = not os.path.exists(PAUSE_PATH)
print(f"  PAUSE eliminado: {pause_gone}")

all_pass_3 = all(ok for ok, _ in checks3)
if all_pass_3:
    print(f"\n{PASS} PRUEBA 3 completa")
else:
    print(f"\n{FAIL} PRUEBA 3 con errores")


# ─────────────────────────────────────────
#  RESUMEN FINAL
# ─────────────────────────────────────────
print()
print("=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
