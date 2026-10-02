"""El rele del Puerto: stdio de un lado, el socket Unix `/run/faro/<run_id>.sock` del otro.

Es lo unico del Faro que viaja DENTRO de la jaula del motor (spec §3 «Transporte»), asi que es
minimo a proposito: SOLO biblioteca estandar (una prueba lo vigila: no importa `mcp`, ni
pydantic, ni nada pesado), sin logica de protocolo (copia bytes en los dos sentidos) y SIN
identidad: quien pide lo dice el socket (las credenciales del par), no lo que pase por aqui.

    python -m jax.faro.relay --socket /run/faro/<run_id>.sock

Termina con 0 cuando cualquiera de los dos extremos cierra; con 2 si no puede conectar. Por
stdout solo sale protocolo; los mensajes propios van a stderr.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

_TROZO = 65536


async def _copiar(origen: asyncio.StreamReader, destino: asyncio.StreamWriter) -> None:
    while True:
        datos = await origen.read(_TROZO)
        if not datos:
            return
        destino.write(datos)
        await destino.drain()


async def _correr(ruta: str) -> int:
    try:
        lector_sock, escritor_sock = await asyncio.open_unix_connection(ruta)
    except OSError as exc:
        print(f"faro-relay: no se pudo conectar a {ruta}: {exc}", file=sys.stderr)
        return 2
    bucle = asyncio.get_running_loop()
    lector_in = asyncio.StreamReader()
    await bucle.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(lector_in), sys.stdin.buffer)
    transporte, protocolo = await bucle.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout.buffer)
    escritor_out = asyncio.StreamWriter(transporte, protocolo, None, bucle)
    ida = asyncio.create_task(_copiar(lector_in, escritor_sock))
    vuelta = asyncio.create_task(_copiar(lector_sock, escritor_out))
    try:
        await asyncio.wait({ida, vuelta}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in (ida, vuelta):
            t.cancel()
        await asyncio.gather(ida, vuelta, return_exceptions=True)
        escritor_sock.close()
        escritor_out.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jax.faro.relay", description="Rele stdio <-> socket Unix del Puerto del Faro")
    ap.add_argument("--socket", required=True, help="ruta del socket Unix del run (/run/faro/<run_id>.sock)")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(_correr(args.socket))
    except (BrokenPipeError, ConnectionResetError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
