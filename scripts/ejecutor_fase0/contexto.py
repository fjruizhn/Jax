#!/usr/bin/env python3
"""Fase 0 · U1: ¿cuánto contexto de Qwen cabe en la R9700?

Uso:
  python3 scripts/ejecutor_fase0/contexto.py --modelo qwen3.6:35b-a3b-q4_K_M \
      --ctx 32768 65536 131072 --salida ~/ejecutor-fase0/resultados

Corta jax_local mientras corre (gate G1). Al final RESTAURA producción y lo
verifica: 32768 (lo decide Ollama por VRAM: OLLAMA_CONTEXT_LENGTH=0) y
keep_alive -1 (lo pone jax-platform/backend/api/chat.py:673).
"""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from medicion import elegir_contexto, evaluar_contexto, parse_ollama_ps, tok_por_segundo  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"
PROMPT = "Escribe los números del 1 al 300 en palabras, separados por comas."


def _post(ruta, cuerpo, timeout):
    req = urllib.request.Request(OLLAMA + ruta, data=json.dumps(cuerpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _fila(modelo):
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True).stdout
    return next((f for f in parse_ollama_ps(ps) if f.get("NAME") == modelo), None)


def _descargar(modelo):
    _post("/api/generate", {"model": modelo, "keep_alive": 0}, 60)
    for _ in range(60):
        if _fila(modelo) is None:
            return
        time.sleep(1)
    raise RuntimeError(f"{modelo} no se descargó en 60 s")


def medir(modelo, num_ctx):
    _descargar(modelo)
    r = _post("/api/generate", {"model": modelo, "prompt": PROMPT, "stream": False,
                                "keep_alive": "10m",
                                "options": {"num_ctx": num_ctx, "num_predict": 512}}, 900)
    return {"num_ctx": num_ctx, "carga_s": r["load_duration"] / 1e9,
            "toks": tok_por_segundo(r["eval_count"], r["eval_duration"]),
            "ps": _fila(modelo), "error": None}


def restaurar(modelo):
    _descargar(modelo)
    _post("/api/chat", {"model": modelo, "stream": False, "keep_alive": -1,
                        "messages": [{"role": "user", "content": "ok"}]}, 900)
    fila = _fila(modelo) or {}
    return fila.get("CONTEXT") == "32768" and fila.get("UNTIL") == "Forever", fila


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    dir_ = pathlib.Path(a.salida).expanduser()
    dir_.mkdir(parents=True, exist_ok=True)
    resultados = []
    try:
        for ctx in sorted(a.ctx):
            try:
                resultados.append(medir(a.modelo, ctx))
            except Exception as e:  # fail-soft: un contexto que no carga es un resultado (no cabe), no un fallo de la medición
                resultados.append({"num_ctx": ctx, "carga_s": float("inf"), "toks": 0.0,
                                   "ps": None, "error": repr(e)})
        base = resultados[0]["toks"]
        for r in resultados:
            r["motivos"] = evaluar_contexto(r["ps"], r["carga_s"], r["toks"], base)
            if r["error"]:
                r["motivos"].append(f"error: {r['error']}")
        with (dir_ / "contexto.jsonl").open("a") as f:
            for r in resultados:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elegido = elegir_contexto(resultados)
        (dir_ / "decision_contexto.json").write_text(json.dumps(
            {"num_ctx": elegido, "t": datetime.datetime.now().isoformat()}))
        print("elegido:", elegido)
    finally:
        ok, fila = restaurar(a.modelo)
        print("restaurado:", ok, fila)
        if not ok:
            sys.exit(2)


if __name__ == "__main__":
    main()
