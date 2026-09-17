#!/usr/bin/env python3
"""E-24 (2026-09-16): medición antes/después del cliente HTTP compartido.

  embedding  el camino caliente REAL: MemoryDB.get_embedding contra el Ollama
             de hall9000, con el código del checkout que esté en PYTHONPATH
             (master para "antes", la rama para "después").
  aislado    el costo de abrir un cliente por llamada sin Ollama en el medio:
             un servidor HTTP/1.1 local con keep-alive que responde al
             instante, `por-llamada` contra `compartido`.

Solo lee: no escribe en la DB ni en disco. Percentiles con el mismo método que
el arnés de carga (scripts/load_test.py::percentil).

Uso:
  PYTHONPATH=<checkout> python scripts/medir_cliente_http_compartido.py embedding -c 1 -n 200
  python scripts/medir_cliente_http_compartido.py aislado --variante por-llamada -c 10 -n 2000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import percentil  # noqa: E402


async def medir(llamada, concurrencia: int, peticiones: int) -> dict:
    semaforo = asyncio.Semaphore(concurrencia)
    latencias: list[float] = []
    errores = 0

    async def una() -> None:
        nonlocal errores
        async with semaforo:
            t0 = time.perf_counter()
            try:
                ok = await llamada()
            except Exception:  # fail-soft: en una medición el error se CUENTA en `errores`, que sale en el resultado; no se esconde ni corta la corrida
                errores += 1
                return
            if ok:
                latencias.append(time.perf_counter() - t0)
            else:
                errores += 1

    inicio = time.perf_counter()
    await asyncio.gather(*(una() for _ in range(peticiones)))
    segundos = time.perf_counter() - inicio

    def ms(valor):
        return None if valor is None else round(valor * 1000, 2)

    return {
        "concurrencia": concurrencia, "peticiones": peticiones, "errores": errores,
        "rps": round(peticiones / segundos, 1),
        "p50_ms": ms(percentil(latencias, 50)), "p95_ms": ms(percentil(latencias, 95)),
        "p99_ms": ms(percentil(latencias, 99)),
    }


async def modo_embedding(concurrencia: int, peticiones: int) -> dict:
    from jax.memory.db import MemoryDB
    memoria = MemoryDB()
    texto = "medición E-24: cliente HTTP compartido en la memoria de JAX"

    async def llamada() -> bool:
        return await memoria.get_embedding(texto) is not None

    try:
        return await medir(llamada, concurrencia, peticiones)
    finally:
        await memoria.close()


class _Rapido(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Cabeceras y cuerpo en UN solo write (flush al final de handle_one_request).
    # Con el default (wbufsize=0) salen en dos segmentos chicos: Nagle retiene el
    # segundo hasta el ACK retardado del cliente y cada petición por keep-alive
    # paga ~40 ms que son del servidor de prueba, no del cliente. Medido
    # 2026-09-16: compartido p50 41,97 ms con dos writes, 0,37 ms con uno.
    wbufsize = -1

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        cuerpo = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def log_message(self, *args):
        return


async def modo_aislado(variante: str, concurrencia: int, peticiones: int) -> dict:
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), _Rapido)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{servidor.server_address[1]}/"
    compartido = httpx.AsyncClient() if variante == "compartido" else None

    async def llamada() -> bool:
        if compartido is not None:
            respuesta = await compartido.post(url, json={}, timeout=10)
        else:
            async with httpx.AsyncClient(timeout=10) as propio:
                respuesta = await propio.post(url, json={})
        return respuesta.status_code == 200

    try:
        return await medir(llamada, concurrencia, peticiones)
    finally:
        if compartido is not None:
            await compartido.aclose()
        servidor.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description="E-24: medición del cliente HTTP compartido")
    ap.add_argument("modo", choices=("embedding", "aislado"))
    ap.add_argument("--variante", choices=("por-llamada", "compartido"), default="compartido")
    ap.add_argument("-c", "--concurrencia", type=int, default=1)
    ap.add_argument("-n", "--peticiones", type=int, default=200)
    args = ap.parse_args()
    if args.modo == "embedding":
        resultado = asyncio.run(modo_embedding(args.concurrencia, args.peticiones))
    else:
        resultado = asyncio.run(modo_aislado(args.variante, args.concurrencia, args.peticiones))
    resultado["modo"] = args.modo if args.modo == "embedding" else f"aislado/{args.variante}"
    print(json.dumps(resultado, ensure_ascii=False))
    return 0 if resultado["errores"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
