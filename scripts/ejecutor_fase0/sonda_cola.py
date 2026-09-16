#!/usr/bin/env python3
"""Fase 0 · U5: espera en la cola de Ollama de una petición tipo Mesa.

Directo a Ollama (NUNCA /api/chat de la Mesa: ensuciaría la memoria).
Mismo modelo que usa el Ejecutor, para medir cola y no recargas.
espera = wall - (load + prompt_eval + eval), con los relojes de Ollama.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from medicion import percentil  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--intervalo", type=float, default=20.0)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    # El append ciego mezcla corridas sin avisar: el 2026-09-15 una corrida
    # interrumpida y la siguiente acabaron en el mismo archivo, y el resumen
    # impreso (calculado sólo con las muestras en memoria) dio 25,9 s cuando
    # el conjunto real daba 62,7 s. Negarse a escribir sobre un archivo que ya
    # existe obliga a nombrar cada corrida y hace imposible ese falso verde.
    salida = pathlib.Path(a.salida).expanduser()
    if salida.exists():
        sys.exit(f"ERROR: {salida} ya existe. Usá un nombre nuevo: mezclar dos "
                 f"corridas en un archivo falsea el percentil.")
    esperas, walls = [], []
    for i in range(a.n):
        cuerpo = {"model": a.modelo, "stream": False, "keep_alive": -1, "options": {"num_predict": 8},
                  "messages": [{"role": "user", "content": "di ok"}]}
        t0 = time.monotonic()
        req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        wall = time.monotonic() - t0
        ollama_s = (d.get("load_duration", 0) + d.get("prompt_eval_duration", 0) + d.get("eval_duration", 0)) / 1e9
        esperas.append(max(0.0, wall - ollama_s))
        walls.append(wall)
        with salida.open("a") as f:
            f.write(json.dumps({"i": i, "wall": wall, "ollama_s": ollama_s, "espera": esperas[-1]}) + "\n")
        time.sleep(a.intervalo)
    print(json.dumps({"p50_espera": percentil(esperas, 50), "p95_espera": percentil(esperas, 95),
                      "p95_wall": percentil(walls, 95)}))


if __name__ == "__main__":
    main()
