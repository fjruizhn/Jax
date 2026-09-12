#!/usr/bin/env python3
"""Arnes de prueba de carga — politica 4 de LAS CUATRO DEL RENDIMIENTO.

POR QUE EXISTE. La politica dice que nada se lanza sin medir bajo carga, y el
2026-09-11 no habia NINGUNA herramienta instalada en hall9000 (ni k6, ni wrk,
ni ab, ni locust). Una politica sin herramienta es una intencion. Esto usa solo
httpx + asyncio, que ya estan, asi que corre hoy y en cualquier maquina del
ecosistema sin instalar nada ni agregar un repo de apt.

QUE MIDE, Y POR QUE ESAS TRES COSAS:
  - **rps**: cuanto aguanta. Cuenta TODAS las peticiones, fallidas incluidas:
    un servidor que rechaza rapido tiene un rps altisimo y sirve para nada.
  - **p95 / p99**: lo que sufre el usuario de la cola. El promedio esconde
    exactamente el caso que hace que alguien se queje.
  - **tasa de error**: sin esto, los otros dos numeros mienten.

COMO SE USA (el punto de degradacion se busca subiendo la concurrencia):
  python3 scripts/load_test.py --url http://127.0.0.1:8080/api/health -c 1 -n 200
  python3 scripts/load_test.py --url ... -c 10 -n 500
  python3 scripts/load_test.py --url ... -c 50 -n 1000

El resultado se escribe en la Biblioteca del proyecto CON FECHA: una prueba de
carga es VERDAD OPERACIONAL y caduca cuando cambia el esquema, el volumen de
datos o la infraestructura.

ADVERTENCIA: correrlo contra produccion ES trafico de produccion. Contra un
endpoint que escribe, escribe de verdad.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Optional


def percentil(valores: list[float], p: float) -> Optional[float]:
    """Percentil p (0-100) por el metodo del rango mas cercano.

    Devuelve None con lista vacia, y no 0.0: un p95 de 0 ms se lee como
    "buenisimo" cuando lo que paso es que no se midio nada.
    """
    if not valores:
        return None
    ordenados = sorted(valores)
    # Rango mas cercano: ceil(p/100 * N), 1-based. Con N=100 y p=95 da el
    # valor 95 exacto -- el off-by-one aca es el error clasico del arnes casero.
    k = max(1, -(-int(round(p * len(ordenados) / 100.0 * 1000)) // 1000))
    k = min(k, len(ordenados))
    return ordenados[k - 1]


def resumen(latencias_ok: list[float], n_errores: int, segundos: float) -> dict:
    """Arma el resumen. Las latencias son SOLO de las peticiones que salieron
    bien -- una que fallo en 2 ms no puede bajar el p95 y hacer pasar por sano
    a un servidor que rechaza la mitad del trafico."""
    total = len(latencias_ok) + n_errores
    return {
        "peticiones": total,
        "ok": len(latencias_ok),
        "errores": n_errores,
        "tasa_error": (n_errores / total) if total else 0.0,
        "segundos": round(segundos, 3),
        "rps": round(total / segundos, 2) if segundos > 0 else None,
        "p50_ms": _ms(percentil(latencias_ok, 50)),
        "p95_ms": _ms(percentil(latencias_ok, 95)),
        "p99_ms": _ms(percentil(latencias_ok, 99)),
        "max_ms": _ms(max(latencias_ok)) if latencias_ok else None,
    }


def _ms(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 2)


async def _correr(url: str, metodo: str, concurrencia: int, peticiones: int,
                  cuerpo: Optional[str], cabeceras: dict, timeout: float) -> dict:
    import httpx

    latencias: list[float] = []
    errores = 0
    sem = asyncio.Semaphore(concurrencia)

    async with httpx.AsyncClient(timeout=timeout) as cliente:
        async def una():
            nonlocal errores
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await cliente.request(metodo, url, content=cuerpo, headers=cabeceras)
                    ms = (time.perf_counter() - t0) * 1000
                    # 5xx es un error del servidor bajo carga, no una respuesta:
                    # contarlo como exito es el modo mas facil de publicar un
                    # p95 bonito de una app que se esta cayendo.
                    if r.status_code >= 500:
                        errores += 1
                    else:
                        latencias.append(ms)
                except Exception:
                    errores += 1

        inicio = time.perf_counter()
        await asyncio.gather(*[una() for _ in range(peticiones)])
        transcurrido = time.perf_counter() - inicio

    return resumen(latencias, errores, transcurrido)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prueba de carga minima (politica 4)")
    ap.add_argument("--url", required=True)
    ap.add_argument("-X", "--metodo", default="GET")
    ap.add_argument("-c", "--concurrencia", type=int, default=10)
    ap.add_argument("-n", "--peticiones", type=int, default=200)
    ap.add_argument("--cuerpo", default=None)
    ap.add_argument("-H", "--cabecera", action="append", default=[],
                    help="'Nombre: valor', repetible")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()

    cabeceras = {}
    for h in a.cabecera:
        k, _, v = h.partition(":")
        cabeceras[k.strip()] = v.strip()

    r = asyncio.run(_correr(a.url, a.metodo, a.concurrencia, a.peticiones,
                            a.cuerpo, cabeceras, a.timeout))
    r["url"] = a.url
    r["concurrencia"] = a.concurrencia
    r["fecha"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(json.dumps(r, indent=2, ensure_ascii=False))
    # Exit 1 si hubo CUALQUIER error: un load test que termina en verde con
    # peticiones fallidas no sirve como gate de lanzamiento.
    return 1 if r["errores"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
