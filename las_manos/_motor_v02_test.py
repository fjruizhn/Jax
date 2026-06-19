#!/usr/bin/env python3
"""
Motor Registry v0.2 — pruebas de integración.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_motor_v02_test.py

Prueba 1 — dispatch real a Kimi
Prueba 2 — polling hasta completed, verifica result_summary real y que
           reasoning_content NO aparezca en GET /motor/job/{id}
Prueba 3 — kill switch: PAUSE activo → status=failed, error contiene killed_by_switch

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

LAS_MANOS_DIR = "/home/fruiz/jax/las_manos"
BASE_URL = "http://127.0.0.1:7777"
PAUSE_PATH = "/etc/jax/PAUSE"
ENV_FILE = "/etc/jax/.env"
UVICORN = f"{LAS_MANOS_DIR}/.venv/bin/uvicorn"
LOG_FILE = f"{LAS_MANOS_DIR}/logs/uvicorn.out"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    except Exception as e:
        print(f"Warning: no se pudo leer {path}: {e}")
    return env


def http_get(path: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(BASE_URL + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_post(path: str, payload: dict, timeout: int = 15) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------------------
# Carga de credenciales
# ---------------------------------------------------------------------------

env_vars = load_env(ENV_FILE)
kimi_key = env_vars.get("KIMI_API_KEY", "")
if not kimi_key:
    print(f"{FAIL} KIMI_API_KEY no encontrada en {ENV_FILE}")
    sys.exit(1)
print(f"KIMI_API_KEY: presente ({len(kimi_key)} chars)")

# ---------------------------------------------------------------------------
# Iniciar servidor si no está corriendo
# ---------------------------------------------------------------------------

server_proc = None
try:
    h = http_get("/health", timeout=3)
    if h.get("status") == "alive":
        print(f"Servidor ya corriendo. kill_switch_active={h.get('kill_switch_active')}")
except Exception:
    print("Servidor no responde — arrancando...")
    proc_env = {**os.environ, **env_vars}
    log_fh = open(LOG_FILE, "a")
    server_proc = subprocess.Popen(
        [UVICORN, "server:app", "--host", "127.0.0.1", "--port", "7777"],
        env=proc_env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=LAS_MANOS_DIR,
    )
    for _ in range(12):
        time.sleep(1)
        try:
            h = http_get("/health", timeout=2)
            if h.get("status") == "alive":
                print(f"Servidor levantó (pid={server_proc.pid}). kill_switch_active={h.get('kill_switch_active')}")
                break
        except Exception:
            pass
    else:
        print(f"{FAIL} Servidor no levantó en 12s")
        sys.exit(1)


def stop_server() -> None:
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


DISPATCH_PAYLOAD = {
    "caller": "hyde",
    "capability": "refactor",
    "prompt": "refactoriza el archivo de prueba: def suma(a, b): return a + b",
}

# ---------------------------------------------------------------------------
# PRUEBA 1 — dispatch
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("PRUEBA 1 — POST /motor/dispatch")
print("=" * 60)

try:
    resp1 = http_post("/motor/dispatch", DISPATCH_PAYLOAD)
    print(json.dumps(resp1, indent=2, ensure_ascii=False))
    job_id = resp1.get("job_id")
    dispatch_status = resp1.get("status")
    motor = resp1.get("motor")
    if job_id and dispatch_status in ("pending", "running"):
        print(f"{PASS} job_id={job_id}, status={dispatch_status}, motor={motor}")
    elif dispatch_status == "rejected":
        print(f"{FAIL} dispatch rechazado: {resp1.get('rejected_reason')}")
        stop_server()
        sys.exit(1)
    else:
        print(f"{FAIL} status inesperado: {dispatch_status}")
        stop_server()
        sys.exit(1)
except Exception as e:
    print(f"{FAIL} dispatch falló: {e}")
    stop_server()
    sys.exit(1)

# ---------------------------------------------------------------------------
# PRUEBA 2 — polling hasta completar
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print(f"PRUEBA 2 — GET /motor/job/{job_id} (polling, máx 120s)")
print("=" * 60)

final_view = None
for i in range(1, 25):
    try:
        view = http_get(f"/motor/job/{job_id}")
        st = view.get("status")
        elapsed = (i - 1) * 5
        print(f"  Poll {i:2d} ({elapsed:3d}s): status={st}")
        if st not in ("pending", "running"):
            final_view = view
            break
    except Exception as e:
        print(f"  Poll {i}: error — {e}")
    time.sleep(5)

p2_ok = False
if final_view is None:
    print(f"{FAIL} Timeout después de 120s")
else:
    print(f"\nRespuesta final:\n{json.dumps(final_view, indent=2, ensure_ascii=False)}")

    st = final_view.get("status")
    rs = final_view.get("result_summary") or ""
    has_reasoning = "reasoning_content" in final_view

    checks2 = [
        (st == "completed",               f"status={st}"),
        (bool(rs) and "[STUB" not in rs,  f"result_summary real ({len(rs)} chars)"),
        (not has_reasoning,               "reasoning_content NO aparece en GET response"),
    ]

    print()
    for ok, msg in checks2:
        print(f"  {PASS if ok else FAIL} — {msg}")

    p2_ok = all(ok for ok, _ in checks2)
    print(f"\n{PASS if p2_ok else FAIL} PRUEBA 2")

# ---------------------------------------------------------------------------
# PRUEBA 3 — kill switch
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("PRUEBA 3 — Kill switch (/etc/jax/PAUSE)")
print("=" * 60)

# Limpiar PAUSE previo
if os.path.exists(PAUSE_PATH):
    subprocess.run(["sudo", "rm", "-f", PAUSE_PATH], check=False)
    print(f"PAUSE previo eliminado")

# Crear PAUSE
ks_ok = False
try:
    subprocess.run(["sudo", "mkdir", "-p", "/etc/jax"], check=True)
    subprocess.run(["sudo", "touch", PAUSE_PATH], check=True)
    print(f"PAUSE creado. Existe: {os.path.exists(PAUSE_PATH)}")
except Exception as e:
    print(f"{FAIL} No se pudo crear PAUSE: {e}")
    stop_server()
    sys.exit(1)

# Dispatch con kill switch activo
try:
    resp3 = http_post("/motor/dispatch", DISPATCH_PAYLOAD)
    print(f"\nDispatch:\n{json.dumps(resp3, indent=2, ensure_ascii=False)}")
    job_id3 = resp3.get("job_id")
except Exception as e:
    print(f"{FAIL} dispatch con PAUSE activo falló: {e}")
    subprocess.run(["sudo", "rm", "-f", PAUSE_PATH], check=False)
    stop_server()
    sys.exit(1)

# Esperar 3s y verificar
time.sleep(3)
ks_status = None
ks_error = ""
try:
    view3 = http_get(f"/motor/job/{job_id3}")
    print(f"\nGET job:\n{json.dumps(view3, indent=2, ensure_ascii=False)}")
    ks_status = view3.get("status")
    ks_error = view3.get("error") or ""
except Exception as e:
    print(f"{FAIL} GET job con PAUSE falló: {e}")

checks3 = [
    (ks_status == "failed",                       f"status={ks_status} (esperado: failed)"),
    ("killed_by_switch" in ks_error,              f"error contiene 'killed_by_switch': '{ks_error}'"),
]

print()
for ok, msg in checks3:
    print(f"  {PASS if ok else FAIL} — {msg}")

ks_ok = all(ok for ok, _ in checks3)
print(f"\n{PASS if ks_ok else FAIL} PRUEBA 3")

# Limpiar PAUSE
subprocess.run(["sudo", "rm", "-f", PAUSE_PATH], check=False)
print(f"PAUSE eliminado. Existe: {os.path.exists(PAUSE_PATH)}")

# ---------------------------------------------------------------------------
# Detener servidor si lo arrancamos nosotros
# ---------------------------------------------------------------------------
stop_server()

# ---------------------------------------------------------------------------
# RESUMEN
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("RESUMEN FINAL — Motor Registry v0.2")
print("=" * 60)
p1_ok = bool(job_id)
print(f"  Prueba 1 (dispatch):       {PASS if p1_ok else FAIL}")
print(f"  Prueba 2 (Kimi real):      {PASS if p2_ok else FAIL}")
print(f"  Prueba 3 (kill switch):    {PASS if ks_ok else FAIL}")
all_pass = p1_ok and p2_ok and ks_ok
print()
print(f"  {'TODO OK — listo para commit' if all_pass else 'HAY FALLOS — revisar arriba'}")
