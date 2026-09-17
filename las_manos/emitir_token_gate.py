"""
LAS MANOS — Emite un token de aprobación humana (human gate).

Es la ÚNICA forma de emitir un token: no hay ruta HTTP (antes del 2026-09-17
`POST /human_gate/token` se lo daba a cualquier proceso local). Necesita las
credenciales de la base, que viven en /etc/jax/.env: un proceso que no puede
leer ese archivo (la jaula de Hyde, la cuenta `axioma`) no puede emitir.

Uso (hall9000, como fruiz):
  set -a; . /etc/jax/.env; set +a
  /home/fruiz/jax/las_manos/.venv/bin/python /home/fruiz/jax/las_manos/emitir_token_gate.py

Imprime el token (una sola vez: en la base queda su sha256) y su vencimiento.
El token va en `approval_token` del sobre de /execute o en `human_gate_token`
de /motor/dispatch, y sirve para UNA operación.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import getpass
import sys
import time
import tomllib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from jacobs import store  # noqa: E402
import human_gate  # noqa: E402


async def _emitir() -> human_gate.TokenEmitido:
    with open(BASE_DIR / "config.toml", "rb") as f:
        ttl = human_gate.ttl_segundos(tomllib.load(f)["human_gate"])
    try:
        return await human_gate.emitir_token_gate(getpass.getuser(), ttl)
    finally:
        await store.cerrar_pool()


def main() -> int:
    emitido = asyncio.run(_emitir())
    vence = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(emitido.vence_at))
    print(emitido.token)
    print(f"vence: {vence} (un solo uso)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
