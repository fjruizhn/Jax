"""Puente HTTP mínimo que corre dentro de la jaula.

El proceso recibe solamente los dos archivos bind de su ejecución. La conexión
al proveedor la realiza el proxy fuera de la jaula; este módulo no conoce
credenciales ni destinos externos.
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def construir_argv_puente(ruta_socket: str, ruta_token: str, motor: str) -> list[str]:
    if motor not in {"claude", "kimi", "codex"}:
        raise ValueError("motor sin puente S-1")
    return ["/usr/bin/python3", "-m", "jax.faro.puente_http", "--socket", ruta_socket,
            "--token-file", ruta_token, "--motor", motor]


async def _copiar(origen: asyncio.StreamReader, destino: asyncio.StreamWriter) -> None:
    while datos := await origen.read(65536):
        destino.write(datos)
        await destino.drain()


async def _correr(socket_path: str, token_path: str) -> int:
    try:
        with open(token_path, encoding="utf-8") as f:
            token = f.read().strip()
        if not token:
            return 2
        reader, writer = await asyncio.open_unix_connection(socket_path)
    except OSError:
        return 2
    writer.write(b"FARO-TOKEN " + token.encode() + b"\n")
    del token
    loop = asyncio.get_running_loop()
    stdin = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stdin), sys.stdin.buffer)
    transport, protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout.buffer)
    stdout = asyncio.StreamWriter(transport, protocol, None, loop)
    izquierda = asyncio.create_task(_copiar(stdin, writer))
    derecha = asyncio.create_task(_copiar(reader, stdout))
    try:
        await asyncio.wait({izquierda, derecha}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for tarea in (izquierda, derecha):
            tarea.cancel()
        await asyncio.gather(izquierda, derecha, return_exceptions=True)
        writer.close()
        stdout.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--motor", required=True)
    args = parser.parse_args(argv)
    if args.motor not in {"claude", "kimi", "codex"}:
        return 2
    return asyncio.run(_correr(args.socket, args.token_file))


if __name__ == "__main__":
    raise SystemExit(main())
