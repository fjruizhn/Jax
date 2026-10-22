#!/usr/bin/env python3
"""Siembra de despliegue de `axioma_config` `anti_fabricacion.modo`.

    sembrar_modo.py              # simula: imprime el SQL, no toca nada
    sembrar_modo.py --verificar  # 0 si la clave existe con 'sombra' o 'duro'; 1 si falta o es inválida
    sembrar_modo.py --aplicar    # INSERT IGNORE de 'sombra' (nunca pisa un valor existente)

ORDEN DE ACTIVACIÓN (docs/runbooks/anti-fabricacion-activacion.md): sembrar `sombra` ANTES de
reiniciar LAS MANOS con el código del detector. Sin la clave, el modo es `duro` por omisión y
cada falso positivo falla un job. Pasar a `duro` lo decide Fernando tras U5; este script NUNCA
escribe `duro` ni actualiza una fila existente.

`--aplicar` escribe en la base de la conexión (JAX_DB_*): lo corre una persona con GO, nunca
una prueba ni una sesión sola. Las pruebas inyectan `conectar`.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from jax.anti_fabricacion import consecuencia  # noqa: E402

VALOR_INICIAL = "sombra"
SQL_SEMBRAR = "INSERT IGNORE INTO axioma_config (config_key, config_value) VALUES (%s, %s)"


async def _leer(conectar) -> str | None:
    conn = await conectar()
    try:
        async with conn.cursor() as cur:
            await cur.execute(consecuencia.SQL_MODO, (consecuencia.CLAVE_MODO,))
            fila = await cur.fetchone()
    finally:
        conn.close()
    return None if not fila or fila[0] is None else str(fila[0]).strip().lower()


async def _aplicar(conectar) -> None:
    conn = await conectar()
    try:
        async with conn.cursor() as cur:
            await cur.execute(SQL_SEMBRAR, (consecuencia.CLAVE_MODO, VALOR_INICIAL))
        commit = getattr(conn, "commit", None)
        if commit is not None:
            await commit()
    finally:
        conn.close()


def main(argv=None, *, conectar=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--aplicar", action="store_true", help="INSERT IGNORE de 'sombra' (no pisa nada)")
    g.add_argument("--verificar", action="store_true", help="comprueba que la clave existe y es válida")
    a = ap.parse_args(argv)
    conectar = conectar or consecuencia._db_conn
    if a.verificar:
        valor = asyncio.run(_leer(conectar))
        if valor in (consecuencia.Modo.SOMBRA.value, consecuencia.Modo.DURO.value):
            print(f"{consecuencia.CLAVE_MODO}={valor}")
            return 0
        print(f"{consecuencia.CLAVE_MODO} falta o es inválida ({valor!r}): sembrar antes de reiniciar LAS MANOS",
              file=sys.stderr)
        return 1
    if a.aplicar:
        asyncio.run(_aplicar(conectar))
        print(f"sembrado (INSERT IGNORE): {consecuencia.CLAVE_MODO}={VALOR_INICIAL}")
        return 0
    print("SIMULACIÓN (no se escribe nada). Con --aplicar se ejecutaría:")
    print(SQL_SEMBRAR.replace("%s", "'{}'").format(consecuencia.CLAVE_MODO, VALOR_INICIAL) + ";")
    return 0


if __name__ == "__main__":
    sys.exit(main())
