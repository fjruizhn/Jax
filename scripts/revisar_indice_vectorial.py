#!/usr/bin/env python3
"""Revisa -- y si se pide, repara -- los indices vectoriales de la base.

Un indice HNSW puede quedar devolviendo MENOS filas de las que hay, sin error y
para siempre. La causa medida es el borrado en cascada; el detalle esta en
`jax/memory/indice_vectorial.py` y en
`docs/runbooks/indice-vectorial-envenenado.md`.

USO
    set -a; . <(sudo -n cat /etc/jax/.env); set +a
    python3 scripts/revisar_indice_vectorial.py                  # solo mira
    python3 scripts/revisar_indice_vectorial.py --reparar-de-verdad

Por omision NO REPARA: mira e informa. Reparar es DDL y toma un lock sobre la
tabla, asi que se escribe entero y a proposito, igual que el barredor de bases.

SALIDA
    0  todos los indices sanos
    1  al menos uno miente (o se reparo, y se dice cual)
    2  no se pudo revisar (sin conexion, sin credenciales)

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiomysql  # noqa: E402

from jax.core.db_connect_config import db_connect_timeout_seconds  # noqa: E402
from jax.memory import indice_vectorial as iv  # noqa: E402


async def _conn():
    return await aiomysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.environ["JAX_DB_NAME"], autocommit=True,
        connect_timeout=db_connect_timeout_seconds())


async def _correr(reparar: bool, muestra: int) -> int:
    conn = await _conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()")
            base = (await cur.fetchone())[0]
            informes = await iv.revisar_todos(cur, muestra)
            if not informes:
                print(f"{base}: no hay indices vectoriales.")
                return 0

            print(f"{base}:")
            for inf in informes:
                print(f"  {inf}")

            # La caché: el otro modo de fallo, y este SÍ se ve venir. Cuando
            # los vectores no entran, `/grupos` deja de ser determinista bajo
            # carga (medido a 9.000 hechos el 2026-09-20).
            ocupaciones = await iv.ocupacion_de_cache(cur)
            for ocup in ocupaciones:
                print(f"  {ocup}")
            apretadas = [o for o in ocupaciones if not o.holgada]

            rotos = [i for i in informes if not i.sano]
            if apretadas and not rotos:
                print(f"\n{len(apretadas)} índice(s) con la caché al límite. No está "
                      "roto todavía, pero bajo carga las respuestas empiezan a variar "
                      "entre corridas. Subí `mhnsw_max_cache_size` -- y hacelo DOS "
                      "veces: `SET GLOBAL` (surte efecto ya) y el `conf.d` de la "
                      "MariaDB (sobrevive al reinicio). Ver el runbook.")
                return 1
            if not rotos:
                return 0

            if not reparar:
                print(f"\n{len(rotos)} indice(s) MIENTEN. No se reparo nada: "
                      f"agregá --reparar-de-verdad.")
                return 1

            print("\nreparando (DDL: toma un lock sobre la tabla):")
            for inf in rotos:
                definicion = await iv.reparar_uno(cur, inf.tabla, inf.indice, inf.columna)
                despues = await iv.revisar_uno(cur, inf.tabla, inf.indice, inf.columna, muestra)
                print(f"  {inf.tabla}.{inf.indice}: {definicion}")
                print(f"    -> {despues}")
                if not despues.sano:
                    print("    SIGUE ROTO despues de reconstruir. No sigas solo: mirá el runbook.")
                    return 1
            return 1
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reparar-de-verdad", action="store_true", dest="reparar",
                   help="reconstruye los indices que mienten (DDL, toma lock)")
    p.add_argument("--muestra", type=int, default=iv.MUESTRA,
                   help=f"filas de la muestra (default {iv.MUESTRA})")
    args = p.parse_args(argv)
    if args.muestra < 1:
        raise SystemExit("--muestra tiene que ser al menos 1")
    if not os.getenv("JAX_DB_HOST") or not os.getenv("JAX_DB_NAME"):
        print("faltan JAX_DB_HOST/JAX_DB_NAME: sourceá /etc/jax/.env", file=sys.stderr)
        return 2
    return asyncio.run(_correr(args.reparar, args.muestra))


if __name__ == "__main__":
    raise SystemExit(main())
