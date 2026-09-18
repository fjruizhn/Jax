#!/usr/bin/env python3
"""Borra las bases de test por sesión (`jax_memory_test_<sufijo>`) que ya
nadie usa.

Desde la decisión del 2026-09-17 cada sesión corre la suite contra su propia
base (ver `base_de_test.py`). Esas bases se ACUMULAN: nadie las borra al
cerrar la terminal, y un disco lleno de bases muertas es la forma lenta de
volver al problema que esto arregla.

Qué NO toca, nunca:
  * `jax_memory` -- producción.
  * `jax_memory_test` -- la base compartida, la que usa el CI y cualquier
    rama que no exporte sufijo.
Las dos exclusiones están puestas dos veces a propósito (el filtro de
candidatas y `_es_borrable()`), y `tests/test_base_por_sesion.py` las
ejercita: la Regla del carpintero aplicada a un `DROP DATABASE`.

Por omisión NO BORRA: lista lo que borraría (`--dry-run` es el default).
Para borrar de verdad hay que escribir `--borrar-de-verdad`.

USO
    set -a; . <(sudo -n cat /etc/jax/.env); set +a
    python3 scripts/limpiar_bases_de_test.py --dias 7
    python3 scripts/limpiar_bases_de_test.py --dias 7 --borrar-de-verdad

"Más viejas que N días" se mide con la fecha de creación de la base --
`information_schema.TABLES.CREATE_TIME` más reciente de sus tablas, que es lo
que MariaDB expone (no hay CREATE_TIME de esquema). Una base sin tablas se
toma por vacía y se considera vieja: no hay nada que perder en ella.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jax.core.db_connect_config import db_connect_timeout_seconds  # noqa: E402

from base_de_test import (  # noqa: E402
    BASE_COMPARTIDA,
    BASE_DE_PRODUCCION,
    es_base_de_test,
)

#: Las bases que este script no borra NUNCA, diga lo que diga la línea de
#: comandos: producción y la base de tests compartida.
INTOCABLES = frozenset({BASE_DE_PRODUCCION, BASE_COMPARTIDA})

SEGUNDOS_POR_DIA = 86400


def _es_borrable(nombre: str) -> bool:
    """¿Se puede borrar `nombre`? Sólo una base `jax_memory_test_<sufijo>`
    con sufijo válido, y nunca una intocable.

    El chequeo de INTOCABLES va PRIMERO y es explícito aunque
    `es_base_de_test()` ya devuelva True para la compartida: el que lee este
    archivo tiene que ver el freno sin seguir la cadena de llamadas.
    """
    if nombre in INTOCABLES:
        return False
    if not es_base_de_test(nombre):
        return False
    return nombre != BASE_COMPARTIDA


def _parametros_de_conexion() -> dict:
    faltantes = [v for v in ("JAX_DB_HOST", "JAX_DB_USER", "JAX_DB_PASSWORD")
                 if not os.environ.get(v)]
    if faltantes:
        raise SystemExit(
            f"faltan variables de conexión: {', '.join(faltantes)}. "
            f"Sourceá /etc/jax/.env (set -a; . <(sudo -n cat /etc/jax/.env); set +a)."
        )
    return {
        "host": os.environ["JAX_DB_HOST"],
        "port": int(os.environ.get("JAX_DB_PORT", "3306")),
        "user": os.environ["JAX_DB_USER"],
        "password": os.environ["JAX_DB_PASSWORD"],
    }


async def _candidatas(dias: int) -> list[tuple[str, float]]:
    """Las bases borrables y la edad en días de su tabla más nueva."""
    import aiomysql

    # connect_timeout escrito en la llamada: sin él una DB colgada deja el
    # limpiador esperando para siempre, y el tripwire del árbol lo exige así
    # (kwarg en el AST, no dentro de un `**dict`).
    conn = await aiomysql.connect(
        autocommit=True, connect_timeout=db_connect_timeout_seconds(),
        **_parametros_de_conexion())
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                "WHERE SCHEMA_NAME LIKE %s ORDER BY SCHEMA_NAME",
                (f"{BASE_COMPARTIDA}\\_%",))
            nombres = [f[0] for f in await cur.fetchall() if _es_borrable(f[0])]
            ahora = time.time()
            viejas = []
            for nombre in nombres:
                await cur.execute(
                    "SELECT UNIX_TIMESTAMP(MAX(GREATEST(CREATE_TIME, "
                    "COALESCE(UPDATE_TIME, CREATE_TIME)))) "
                    "FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s",
                    (nombre,))
                fila = await cur.fetchone()
                ultimo = fila[0] if fila and fila[0] else 0
                edad = (ahora - float(ultimo)) / SEGUNDOS_POR_DIA if ultimo else float("inf")
                if edad >= dias:
                    viejas.append((nombre, edad))
            return viejas
    finally:
        conn.close()


async def _borrar(nombres: list[str]) -> None:
    import aiomysql

    # connect_timeout escrito en la llamada: sin él una DB colgada deja el
    # limpiador esperando para siempre, y el tripwire del árbol lo exige así
    # (kwarg en el AST, no dentro de un `**dict`).
    conn = await aiomysql.connect(
        autocommit=True, connect_timeout=db_connect_timeout_seconds(),
        **_parametros_de_conexion())
    try:
        async with conn.cursor() as cur:
            for nombre in nombres:
                # Tercer control del mismo freno, pegado al DROP: lo que se
                # ejecuta es esta línea, no la intención de dos funciones más
                # arriba.
                if not _es_borrable(nombre):
                    raise RuntimeError(f"me negué a borrar {nombre!r}")
                await cur.execute(f"DROP DATABASE `{nombre}`")
                print(f"  borrada: {nombre}")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dias", type=int, default=7,
                   help="borra las bases sin actividad hace N días o más (default: 7)")
    p.add_argument("--borrar-de-verdad", action="store_true",
                   help="BORRA. Sin este flag sólo lista (dry-run).")
    args = p.parse_args(argv)
    if args.dias < 0:
        raise SystemExit("--dias no puede ser negativo")

    viejas = asyncio.run(_candidatas(args.dias))
    if not viejas:
        print(f"no hay bases {BASE_COMPARTIDA}_* de {args.dias} día(s) o más.")
        return 0

    print(f"{len(viejas)} base(s) de {args.dias} día(s) o más:")
    for nombre, edad in viejas:
        edad_txt = "vacía" if edad == float("inf") else f"{edad:.1f} días"
        print(f"  {nombre}  ({edad_txt})")
    if not args.borrar_de_verdad:
        print("\nDRY-RUN: no se borró nada. Agregá --borrar-de-verdad para borrarlas.")
        return 0
    print("\nborrando:")
    asyncio.run(_borrar([n for n, _ in viejas]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
