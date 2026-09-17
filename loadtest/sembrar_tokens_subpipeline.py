"""Siembra un padre `running` con un paso de Ada `running` y N tokens, para la
carga del contrato de sub-pipelines (frente F, 2026-09-16).

Solo jax_memory_test (la barrera vive en jacobs/_arnes_ada.py).

  sembrar --n 60000 --salida DIR/tokens.json   -> tokens.json (lista) y tokens.json.padre
  cerrar  --padre PIPELINE_ID                  -> deja el padre en completed

Los tokens son secretos de prueba: la salida va a un directorio temporal, nunca
al repo, y se borra al terminar la medición.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from jacobs import _arnes_ada as ada
from jacobs import store, subpipelines


async def sembrar(n: int, salida: str, lote: int) -> None:
    await store.init_tables()
    padre, paso = await ada.padre_en_ejecucion()
    tokens: list[str] = []
    while len(tokens) < n:
        k = min(lote, n - len(tokens))
        tokens += await asyncio.gather(*[
            subpipelines.emitir_token_subpipeline(padre, paso) for _ in range(k)
        ])
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(tokens, f)
    with open(salida + ".padre", "w", encoding="utf-8") as f:
        f.write(padre)
    print(f"padre={padre} tokens={len(tokens)} salida={salida}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="accion", required=True)
    s = sub.add_parser("sembrar")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--salida", required=True)
    s.add_argument("--lote", type=int, default=50)
    c = sub.add_parser("cerrar")
    c.add_argument("--padre", required=True)
    args = p.parse_args()
    if args.accion == "sembrar":
        asyncio.run(_y_cerrar_pool(sembrar(args.n, args.salida, args.lote)))
    else:
        asyncio.run(_y_cerrar_pool(ada.cerrar(args.padre)))


async def _y_cerrar_pool(coro) -> None:
    try:
        await coro
    finally:
        await store.cerrar_pool()


if __name__ == "__main__":
    main()
